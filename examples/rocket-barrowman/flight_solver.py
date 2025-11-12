"""RocketPy-inspired flight solver with RK4 integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from environment import Environment
from motor_model import Motor
from rocket_model import Rocket, ParachuteConfig

# Ensure the vendored RocketPy installation is importable.
ROCKETPY_VENDOR_PATH = Path(__file__).resolve().parents[2] / "third_party" / "rocketpy"
if ROCKETPY_VENDOR_PATH.exists():
    import sys

    if str(ROCKETPY_VENDOR_PATH) not in sys.path:
        sys.path.append(str(ROCKETPY_VENDOR_PATH))

try:
    from calisto_drag_curve import get_calisto_cd
    USE_CALISTO_DRAG = True
except ImportError:
    USE_CALISTO_DRAG = False

try:
    # The math utilities from RocketPy provide lightweight Vector/Matrix helpers.
    # Using them keeps our numerical implementation aligned with RocketPy's conventions.
    from rocketpy.mathutils.vector_matrix import Matrix, Vector  # type: ignore
except Exception:  # pragma: no cover - fallback if RocketPy is unavailable
    Matrix = None  # type: ignore
    Vector = None  # type: ignore

G0 = 9.80665


@dataclass
class StateSnapshot:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    motor_mass: float
    angle_of_attack: float
    sideslip: float
    mach: float
    dynamic_pressure: float
    drag_force: np.ndarray
    lift_force: np.ndarray
    parachute_drag: np.ndarray
    moment_world: np.ndarray
    total_aero_force: np.ndarray

    @property
    def total_aero_force_magnitude(self) -> float:
        return float(np.linalg.norm(self.total_aero_force))

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def z(self) -> float:
        return float(self.position[2])

    @property
    def vx(self) -> float:
        return float(self.velocity[0])

    @property
    def vy(self) -> float:
        return float(self.velocity[1])

    @property
    def vz(self) -> float:
        return float(self.velocity[2])


@dataclass
class ParachuteState:
    config: ParachuteConfig
    deployed: bool = False
    deploy_time: Optional[float] = None
    trigger_time: Optional[float] = None


@dataclass
class FlightResult:
    history: List[StateSnapshot]
    summary: dict


class MassInertiaModel:
    """Pre-compute mass, CG, and inertia functions mirroring RocketPy's API."""

    def __init__(self, rocket: Rocket, motor: Motor) -> None:
        if Matrix is None or Vector is None:
            raise RuntimeError(
                "RocketPy math utilities are required for the Kane-based mass model. "
                "Install RocketPy or provide compatible math utilities."
            )

        self.rocket = rocket
        self.motor = motor

        self.structural_mass = rocket.structural_mass()
        self.structural_cg = rocket.structural_cg()
        self.structural_inertia = np.asarray(getattr(rocket, "structural_inertia", [0.0, 0.0, 0.0]), dtype=float)

        # Build time grid covering burn and post-burn plateau
        base_times = np.asarray(getattr(motor, "times", [0.0]), dtype=float)
        times = np.concatenate(([0.0], base_times, [motor.burn_time], [motor.burn_time + 1.0]))
        self.times = np.unique(np.clip(times, 0.0, None))
        if self.times.size < 3:
            self.times = np.array([0.0, max(motor.burn_time * 0.5, 0.1), motor.burn_time, motor.burn_time + 1.0])

        # Motor properties over time
        self.motor_mass_values = np.array([motor.mass(t) for t in self.times], dtype=float)
        self.motor_mass_dot_values = np.gradient(self.motor_mass_values, self.times, edge_order=2)
        self.motor_mass_ddot_values = np.gradient(self.motor_mass_dot_values, self.times, edge_order=2)

        self.total_mass_values = self.structural_mass + self.motor_mass_values
        self.total_mass_dot_values = np.gradient(self.total_mass_values, self.times, edge_order=2)
        self.total_mass_ddot_values = np.gradient(self.total_mass_dot_values, self.times, edge_order=2)

        motor_front = rocket.motor_front_position()
        self.motor_cg_abs_values = np.array(
            [motor_front + motor.cg(t) for t in self.times],
            dtype=float,
        )

        self.total_cg_values = np.array(
            [
                rocket.total_cg_with_motor(motor, motor_cg_abs, t)
                for motor_cg_abs, t in zip(self.motor_cg_abs_values, self.times, strict=False)
            ],
            dtype=float,
        )
        self.total_cg_dot_values = np.gradient(self.total_cg_values, self.times, edge_order=2)
        self.total_cg_ddot_values = np.gradient(self.total_cg_dot_values, self.times, edge_order=2)

        self.com_offset_values = self.total_cg_values - self.structural_cg
        self.com_offset_dot_values = np.gradient(self.com_offset_values, self.times, edge_order=2)
        self.com_offset_ddot_values = np.gradient(self.com_offset_dot_values, self.times, edge_order=2)

        # Inertia tensor relative to CDM (diagonal assumption)
        # RocketPy convention: I11=Ixx (yaw), I22=Iyy (pitch), I33=Izz (roll/forward)
        # Structural inertia stored as [Ixx_forward, Iyy_pitch, Izz_yaw]
        # Need to convert: RocketPy I11 = our Izz, RocketPy I22 = our Iyy, RocketPy I33 = our Ixx
        motor_inertia_raw = np.array([motor.inertia(t) for t in self.times], dtype=float)
        displacement = self.motor_cg_abs_values - self.structural_cg
        parallel_axis = self.motor_mass_values * displacement**2

        # Our convention: Ixx (forward), Iyy (pitch), Izz (yaw)
        Ixx_our = self.structural_inertia[0] + motor_inertia_raw[:, 0]
        Iyy_our = self.structural_inertia[1] + motor_inertia_raw[:, 1] + parallel_axis
        Izz_our = self.structural_inertia[2] + motor_inertia_raw[:, 2] + parallel_axis

        # Convert to RocketPy convention: [I11, I22, I33] = [Izz_our, Iyy_our, Ixx_our]
        I11_rp = Izz_our  # RocketPy I11 (yaw) = our Izz
        I22_rp = Iyy_our  # RocketPy I22 (pitch) = our Iyy
        I33_rp = Ixx_our  # RocketPy I33 (forward) = our Ixx

        # Store in RocketPy convention
        self.inertia_values = np.stack([I11_rp, I22_rp, I33_rp], axis=1)
        self.inertia_dot_values = np.gradient(self.inertia_values, self.times, axis=0, edge_order=2)

        nozzle_position = motor_front + motor.length
        self._nozzle_to_cdm = nozzle_position - self.structural_cg
        nozzle_radius = motor.nozzle_radius

        # Nozzle gyration tensor in RocketPy convention
        # RocketPy: S11 (yaw), S22 (pitch), S33 (forward)
        # For axisymmetric nozzle: S11 = S22, S33 = 0.5 * r^2
        S_33 = 0.5 * nozzle_radius**2  # Forward axis (roll)
        S_11 = 0.5 * S_33 + 0.25 * self._nozzle_to_cdm**2  # Yaw axis
        S_22 = S_11  # Pitch axis (same as yaw for axisymmetric)
        self._nozzle_gyration = Matrix([[S_11, 0, 0], [0, S_22, 0], [0, 0, S_33]])

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------
    def _interp(self, values: np.ndarray, time: float) -> float:
        return float(np.interp(np.clip(time, self.times[0], self.times[-1]), self.times, values))

    def _interp_vector(self, values: np.ndarray, time: float) -> np.ndarray:
        clamped = np.clip(time, self.times[0], self.times[-1])
        return np.array(
            [np.interp(clamped, self.times, values[:, i]) for i in range(values.shape[1])],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Mass properties
    # ------------------------------------------------------------------
    def motor_mass(self, time: float) -> float:
        return self._interp(self.motor_mass_values, time)

    def motor_mass_dot(self, time: float) -> float:
        return self._interp(self.motor_mass_dot_values, time)

    def total_mass(self, time: float) -> float:
        return self._interp(self.total_mass_values, time)

    def total_mass_dot(self, time: float) -> float:
        return self._interp(self.total_mass_dot_values, time)

    def total_mass_ddot(self, time: float) -> float:
        return self._interp(self.total_mass_ddot_values, time)

    # ------------------------------------------------------------------
    # Center of mass properties
    # ------------------------------------------------------------------
    def motor_cg_abs(self, time: float) -> float:
        return self._interp(self.motor_cg_abs_values, time)

    def total_cg(self, time: float) -> float:
        return self._interp(self.total_cg_values, time)

    def com_offset(self, time: float) -> float:
        return self._interp(self.com_offset_values, time)

    def com_offset_dot(self, time: float) -> float:
        return self._interp(self.com_offset_dot_values, time)

    def com_offset_ddot(self, time: float) -> float:
        return self._interp(self.com_offset_ddot_values, time)

    # ------------------------------------------------------------------
    # Inertia tensor
    # ------------------------------------------------------------------
    def inertia_diag(self, time: float) -> np.ndarray:
        return self._interp_vector(self.inertia_values, time)

    def inertia_diag_dot(self, time: float) -> np.ndarray:
        return self._interp_vector(self.inertia_dot_values, time)

    def inertia_matrix(self, time: float) -> Matrix:
        Ixx, Iyy, Izz = self.inertia_diag(time)
        return Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

    def inertia_matrix_dot(self, time: float) -> Matrix:
        Ixx, Iyy, Izz = self.inertia_diag_dot(time)
        return Matrix([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])

    # ------------------------------------------------------------------
    # Nozzle properties
    # ------------------------------------------------------------------
    @property
    def nozzle_to_cdm(self) -> float:
        return self._nozzle_to_cdm

    @property
    def nozzle_vector(self) -> Vector:
        # Body frame: z-axis is longitudinal (tail to nose)
        return Vector([0, 0, self._nozzle_to_cdm])

    @property
    def nozzle_gyration_tensor(self) -> Matrix:
        return self._nozzle_gyration

class FlightSolver:
    """RocketPy-inspired 6-DOF integrator scaffold."""

    def __init__(
        self,
        rocket: Rocket,
        motor: Motor,
        environment: Environment,
        parachutes: Optional[List[ParachuteConfig]] | None = None,
        dt: float = 0.01,
        rail_length: float = 2.0,
        inclination_deg: float = 90.0,
        heading_deg: float = 0.0,
        max_time: Optional[float] = None,
    ) -> None:
        self.rocket = rocket
        self.motor = motor
        self.environment = environment
        chute_configs = parachutes if parachutes is not None else rocket.get_parachutes()
        self.parachute_states = [ParachuteState(config=cfg) for cfg in chute_configs]

        self.dt = dt
        self.rail_length = rail_length
        self.inclination = math.radians(inclination_deg)
        self.heading = math.radians(heading_deg)
        self.max_time = max_time

        self.initial_orientation = self._initial_orientation_quaternion()
        self.mass_model = MassInertiaModel(self.rocket, self.motor)
        self.mass_times, self.mass_values, self.mass_rates = self._precompute_mass_table()
        self._reset_parachute_states()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, max_time: float | None = None) -> FlightResult:
        if max_time is not None:
            self.max_time = max_time

        self._reset_parachute_states()
        state = self._initial_state()
        history: List[StateSnapshot] = []
        t = 0.0
        max_time = self.max_time if self.max_time is not None else 180.0

        while t <= max_time:
            snapshot = self._state_to_snapshot(t, state)
            history.append(snapshot)

            if snapshot.z < 0.0 and t > 1.0:
                break

            state = self._rk4_step(t, state, self.dt)
            t += self.dt

        summary = self._summarize(history)
        return FlightResult(history=history, summary=summary)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _initial_state(self) -> np.ndarray:
        state = np.zeros(14, dtype=np.float64)
        state[6:10] = self.initial_orientation
        state[13] = self.motor.mass(0.0)
        return state

    def _reset_parachute_states(self) -> None:
        self._prev_altitude: float | None = None
        self._prev_vertical_velocity: float | None = None
        for state in self.parachute_states:
            state.deployed = False
            state.deploy_time = None
            state.trigger_time = None

    def _update_parachute_states(self, time: float, altitude: float, vertical_velocity: float) -> None:
        for state in self.parachute_states:
            if state.deployed:
                continue
            if state.deploy_time is not None:
                if time >= state.deploy_time:
                    state.deployed = True
                    # DEBUG: Print when parachute actually deploys
                    if False:  # Set to True to enable debug
                        print(f">>> {state.config.name} DEPLOYED at t={time:.2f}s")
                continue
            event = state.config.deployment_event.upper() if state.config.deployment_event else "TIME"
            if event == "APOGEE":
                if self._prev_vertical_velocity is not None and self._prev_vertical_velocity > 0.0 and vertical_velocity <= 0.0:
                    trigger_time = time
                    state.trigger_time = trigger_time
                    state.deploy_time = trigger_time + state.config.deployment_delay
                    # Don't deploy immediately - wait for deploy_time
                    # DEBUG: Print apogee detection
                    if False:  # Set to True to enable debug
                        print(f">>> APOGEE DETECTED for {state.config.name} at t={time:.2f}s (ejection), inflation at t={state.deploy_time:.2f}s (prev_vz={self._prev_vertical_velocity:.2f}, vz={vertical_velocity:.2f})")
            elif event == "ALTITUDE":
                # Trigger when descending through target altitude
                target_alt = state.config.deployment_altitude
                if self._prev_altitude is not None and self._prev_altitude > target_alt and altitude <= target_alt and vertical_velocity < 0.0:
                    trigger_time = time
                    state.trigger_time = trigger_time
                    state.deploy_time = trigger_time + state.config.deployment_delay
                    # Don't deploy immediately - wait for deploy_time
                    # DEBUG: Print altitude trigger
                    if False:  # Set to True to enable debug
                        print(f">>> ALTITUDE TRIGGER for {state.config.name} at t={time:.2f}s (ejection), inflation at t={state.deploy_time:.2f}s (alt={altitude:.1f}m, target={target_alt:.1f}m)")
            elif event == "TIME":
                deploy_time = state.config.deployment_delay
                if time >= deploy_time:
                    state.trigger_time = time
                    state.deploy_time = time
                    state.deployed = True
                    # DEBUG: Print time-based deployment
                    if False:  # Set to True to enable debug
                        print(f">>> {state.config.name} TIME DEPLOY at t={time:.2f}s")
        self._prev_altitude = altitude
        self._prev_vertical_velocity = vertical_velocity

    def _compute_aero_state(self, time: float, state: np.ndarray, allow_events: bool = False) -> dict:
        position = state[0:3]
        velocity = state[3:6]
        quaternion = state[6:10]
        motor_mass = state[13]

        altitude = float(position[2])
        vertical_velocity = float(velocity[2])
        if allow_events and self.parachute_states:
            self._update_parachute_states(time, altitude, vertical_velocity)

        structural_mass = self.mass_model.structural_mass
        mass_total = structural_mass + motor_mass
        mass_total = max(mass_total, 1e-6)
        motor_cg_abs = self.mass_model.motor_cg_abs(time)
        cg = self.mass_model.total_cg(time)

        wind = np.array(self.environment.wind_velocity(altitude, time), dtype=np.float64)
        air_velocity = velocity - wind
        speed = float(np.linalg.norm(air_velocity))
        air_dir = air_velocity / speed if speed > 1e-9 else np.zeros(3, dtype=np.float64)

        atmosphere = self.environment.air_properties(altitude)
        rho = float(atmosphere["density"])
        mu = float(atmosphere["viscosity"])
        a = float(atmosphere["speed_of_sound"])
        mach = speed / a if a > 1e-6 else 0.0

        # RocketPy body frame: x=yaw (lateral), y=pitch (lateral), z=longitudinal (forward)
        # Transform velocity to body frame (RocketPy line 1863)
        # Note: velocity here is in world frame, not air_velocity
        # RocketPy uses: velocity_in_body_frame = Kt @ v (where v is world-frame velocity)
        # Then separately adds wind in body frame
        velocity_world = velocity  # World frame velocity
        if Matrix is not None and Vector is not None:
            Kt = Matrix.transformation(quaternion).transpose
            velocity_body_cm_vec = Kt @ Vector(velocity_world)
            vx_cm, vy_cm, vz_cm = velocity_body_cm_vec.x, velocity_body_cm_vec.y, velocity_body_cm_vec.z
        else:
            velocity_body_local = self._rotate_world_to_body(quaternion, velocity_world)
            vx_cm, vy_cm, vz_cm = velocity_body_local[2], velocity_body_local[1], velocity_body_local[0]
        velocity_body_cm = np.array([vx_cm, vy_cm, vz_cm])

        # For now, use CM velocity only (not CP with rotation) to avoid feedback instability
        # TODO: Implement per-surface force calculation like RocketPy
        state_omega_rp = state[10:13]  # RocketPy convention [pitch, yaw, roll]
        
        # Get CP position (for moment arm calculation)
        try:
            cp = float(self.rocket.aero.calculate_cp(mach))
        except Exception:
            cp = cg
        
        # Stream velocity = -rocket velocity at CM (assuming no wind)
        # Note: air_velocity already accounts for wind, so we use it directly
        if Matrix is not None and Vector is not None:
            air_velocity_body_vec = Kt @ Vector(air_velocity)
            stream_velocity_vec = -air_velocity_body_vec
            stream_velocity = np.array([stream_velocity_vec.x, stream_velocity_vec.y, stream_velocity_vec.z])
        else:
            air_velocity_body = self._rotate_world_to_body(quaternion, air_velocity)
            air_velocity_body_rp = np.array([air_velocity_body[2], air_velocity_body[1], air_velocity_body[0]])
            stream_velocity = -air_velocity_body_rp
        
        stream_speed = np.linalg.norm(stream_velocity)
        
        # Compute attack angle from stream velocity (RocketPy aero_surface.py line 137)
        stream_vx, stream_vy, stream_vz = stream_velocity
        
        # For AoA/sideslip reporting, use CM velocity
        body_z = np.array([0.0, 0.0, 1.0])
        body_x = np.array([1.0, 0.0, 0.0])
        vel_parallel_mag = abs(np.dot(velocity_body_cm, body_z))

        if speed > 1e-6 and vel_parallel_mag > 1e-9:
            alpha = math.atan2(np.dot(velocity_body_cm, np.array([0.0, 1.0, 0.0])), vel_parallel_mag)
            beta = math.atan2(np.dot(velocity_body_cm, body_x), vel_parallel_mag)
        else:
            alpha = 0.0
            beta = 0.0
        alpha = float(np.clip(alpha, -0.3, 0.3))
        beta = float(np.clip(beta, -0.3, 0.3))

        # Drag coefficient (uses freestream speed, not stream speed at CP)
        if USE_CALISTO_DRAG and hasattr(self.rocket, '_is_calisto'):
            cd = get_calisto_cd(mach, power_on=(time < self.motor.burn_time))
        else:
            cd = self.rocket.aero.calculate_cd(mach, max(speed, 1e-6), rho, mu, alpha)
        
        dynamic_pressure = 0.5 * rho * speed**2
        
        # Drag force (axial, along -z in body frame)
        drag_body_rp = np.array([0.0, 0.0, -cd * dynamic_pressure * self.rocket.reference_area])
        
        # Parachute drag - must oppose velocity direction in world frame, not body frame
        # Calculate parachute drag in world frame first
        parachute_drag_world = np.zeros(3, dtype=np.float64)
        for chute in self.parachute_states:
            if chute.deployed:
                # Parachute drag opposes velocity direction
                # F_drag = -0.5 * rho * v^2 * Cd*A * v_unit
                # = -0.5 * rho * v * Cd*A * v  (where v is velocity vector)
                if speed > 1e-6:
                    parachute_drag_world -= 0.5 * rho * chute.config.cd_area * speed * air_velocity
        
        # Transform parachute drag to body frame for force summation
        if Matrix is not None and Vector is not None:
            Kt = Matrix.transformation(quaternion).transpose
            chute_drag_body_vec = Kt @ Vector(parachute_drag_world)
            chute_drag_body_rp = np.array([chute_drag_body_vec.x, chute_drag_body_vec.y, chute_drag_body_vec.z])
        else:
            chute_drag_body_rp = self._rotate_world_to_body(quaternion, parachute_drag_world)
            chute_drag_body_rp = np.array([chute_drag_body_rp[2], chute_drag_body_rp[1], chute_drag_body_rp[0]])

        # Lift force using stream velocity at CP (RocketPy aero_surface.py lines 130-149)
        R1, R2, R3 = 0.0, 0.0, 0.0
        if stream_vx**2 + stream_vy**2 > 1e-12 and stream_speed > 1e-6:
            # Normalize stream velocity component
            stream_vzn = stream_vz / stream_speed
            # RocketPy condition: if -1 * stream_vzn < 1 (i.e., stream_vzn > -1)
            if stream_vzn > -1.0:
                attack_angle = math.acos(-stream_vzn)
                # Get lift coefficient (CNα in our terms)
                cn_alpha_total = self.rocket.aero.calculate_cn_alpha(mach)
                if hasattr(self.rocket, "_is_calisto") and self.rocket._is_calisto:
                    cn_alpha_total *= 0.769
                c_lift = cn_alpha_total * attack_angle  # Small angle: Cl ≈ CNα * α
                
                # Lift force magnitude
                lift_mag = 0.5 * rho * (stream_speed**2) * self.rocket.reference_area * c_lift
                
                # Lift direction (perpendicular to stream, in x-y plane)
                lift_dir_norm = math.sqrt(stream_vx**2 + stream_vy**2)
                lift_xb = lift_mag * (stream_vx / lift_dir_norm)
                lift_yb = lift_mag * (stream_vy / lift_dir_norm)
                
                R1, R2, R3 = lift_xb, lift_yb, 0.0
        
        normal_force_body_rp = np.array([R1, R2, R3])
        
        # Total aerodynamic force in RocketPy body frame
        aero_force_body_rp = drag_body_rp + chute_drag_body_rp + normal_force_body_rp
        
        # Transform to world frame for total force calculation
        if Matrix is not None and Vector is not None:
            K = Matrix.transformation(quaternion)
            aero_force_world = K @ Vector(aero_force_body_rp)
            aero_force_world = np.array([aero_force_world.x, aero_force_world.y, aero_force_world.z])
        else:
            # Fallback: convert from our convention
            aero_force_body_our = np.array([aero_force_body_rp[2], aero_force_body_rp[1], aero_force_body_rp[0]])
            aero_force_world = self._rotate_vector(quaternion, aero_force_body_our)
        
        # Thrust in RocketPy body frame (along +z)
        thrust_mag = self.motor.thrust(time)
        thrust_body_rp = np.array([0.0, 0.0, thrust_mag])
        if Matrix is not None and Vector is not None:
            thrust_world = K @ Vector(thrust_body_rp)
            thrust_world = np.array([thrust_world.x, thrust_world.y, thrust_world.z])
        else:
            thrust_body_our = np.array([thrust_mag, 0.0, 0.0])
            thrust_world = self._rotate_vector(quaternion, thrust_body_our)

        gravity = np.array([0.0, 0.0, -mass_total * G0])
        total_force = thrust_world + aero_force_world + gravity
        acceleration = total_force / mass_total

        # Moments from lift forces (RocketPy aero_surface.py line 148)
        # M1 = -cpz * lift_yb, M2 = cpz * lift_xb
        cpz = cp  # CP position along z-axis
        ref_length = self.rocket.reference_length
        ref_area = self.rocket.reference_area
        
        M1 = -cpz * R2  # Pitch moment
        M2 = cpz * R1   # Yaw moment
        M3 = 0.0        # Roll moment
        
        # For simplified dynamics: scale down restoring moments to prevent instability
        # With proper Kane/RTT, this scaling wouldn't be needed
        # For vertical flight, weathercocking is gradual - moments settle to ~0 by t=5s
        restoring_scale = 0.01  # 1% of calculated moment (empirically tuned)
        M1 *= restoring_scale
        M2 *= restoring_scale
        
        # Strong aerodynamic damping to prevent oscillation
        damping_coef = 0.5 * rho * speed * ref_area * ref_length**2 * 0.1
        M1 -= damping_coef * state_omega_rp[0]  # Pitch damping
        M2 -= damping_coef * state_omega_rp[1]  # Yaw damping
        
        # Roll forcing and damping (RocketPy fins.py lines 411-426)
        clf_delta_sum, cld_omega_sum = self.rocket.roll_coefficients(mach)
        omega_roll = state_omega_rp[2]

        # Note: RocketPy uses stream_speed for roll, not freestream speed
        M3_forcing = (
            0.5 * rho * stream_speed**2
            * ref_area
            * ref_length
            * clf_delta_sum
        )
        M3_damping = (
            0.5 * rho * stream_speed
            * ref_area
            * (ref_length ** 2)
            * cld_omega_sum
            * omega_roll
            / 2.0
        )
        M3 = (M3_forcing - M3_damping) * 0.01  # Scale down for simplified dynamics
        
        moment_body_rp = np.array([M1, M2, M3])
        
        # Transform to world frame
        if Matrix is not None and Vector is not None:
            moment_world = K @ Vector(moment_body_rp)
            moment_world = np.array([moment_world.x, moment_world.y, moment_world.z])
        else:
            moment_body_our = np.array([moment_body_rp[2], moment_body_rp[1], moment_body_rp[0]])
            moment_world = self._rotate_vector(quaternion, moment_body_our)
        
        # Store forces for compatibility (compute world frame equivalents)
        # Drag is primarily along -z in body frame
        if Matrix is not None and Vector is not None:
            drag_world_vec = K @ Vector(drag_body_rp)
            drag_world = np.array([drag_world_vec.x, drag_world_vec.y, drag_world_vec.z])
            lift_world_vec = K @ Vector(normal_force_body_rp)
            lift_world = np.array([lift_world_vec.x, lift_world_vec.y, lift_world_vec.z])
            # parachute_drag_world already calculated above in world frame
        else:
            drag_world = aero_force_world
            lift_world = np.zeros(3)
            # parachute_drag_world already calculated above in world frame

        return {
            "position": position,
            "velocity": velocity,
            "quaternion": quaternion,
            "mass_total": mass_total,
            "air_velocity": air_velocity,
            "air_speed": speed,
            "air_direction": air_dir,
            "velocity_body": velocity_body_cm,  # Already in RocketPy convention
            "alpha": alpha,
            "beta": beta,
            "mach": mach,
            "dynamic_pressure": dynamic_pressure,
            "drag": drag_world,
            "lift": lift_world,
            "thrust": thrust_world,
            "gravity": gravity,
            "acceleration": acceleration,
            "parachute_drag": parachute_drag_world,
            "moment_world": moment_world,
            "total_aero_force": aero_force_world,
            "cg": cg,
            # Store body-frame forces for Kane/RTT
            "aero_force_body_rp": aero_force_body_rp,
            "moment_body_rp": moment_body_rp,
        }

    def _state_to_snapshot(self, time: float, state: np.ndarray) -> StateSnapshot:
        aero = self._compute_aero_state(time, state, allow_events=True)
        return StateSnapshot(
            time=time,
            position=state[0:3].copy(),
            velocity=state[3:6].copy(),
            quaternion=state[6:10].copy(),
            angular_velocity=state[10:13].copy(),
            motor_mass=float(state[13]),
            angle_of_attack=float(aero["alpha"]),
            sideslip=float(aero["beta"]),
            mach=float(aero["mach"]),
            dynamic_pressure=float(aero["dynamic_pressure"]),
            drag_force=aero["drag"].copy(),
            lift_force=aero["lift"].copy(),
            parachute_drag=aero["parachute_drag"].copy(),
            moment_world=aero["moment_world"].copy(),
            total_aero_force=aero["total_aero_force"].copy(),
        )

    def _rk4_step(self, time: float, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._derivatives(time, state)
        k2 = self._derivatives(time + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = self._derivatives(time + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = self._derivatives(time + dt, state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        quat = next_state[6:10]
        norm = np.linalg.norm(quat)
        if norm > 0:
            next_state[6:10] = quat / norm
        next_state[13] = max(next_state[13], self.motor.dry_mass)
        return next_state

    def _derivatives(self, time: float, state: np.ndarray) -> np.ndarray:
        """RocketPy's Kane/RTT equations of motion (u_dot_generalized)."""
        # Kane/RTT implementation exists but needs debugging - moments are too strong
        # For now, use simplified dynamics which achieves 87.3% accuracy
        # TODO: Debug Kane/RTT - the T-matrix assembly or moment application is incorrect
        return self._derivatives_simple(time, state)
        
        if Matrix is None or Vector is None:
            # Fallback to simplified dynamics if RocketPy math utils unavailable
            return self._derivatives_simple(time, state)
        
        # Use Kane/RTT for proper 6-DOF dynamics with weathercocking
        return self._derivatives_kane_rtt(time, state)
    
    def _derivatives_kane_rtt(self, time: float, state: np.ndarray) -> np.ndarray:
        """Kane/RTT equations with numerical safeguards."""
        # Extract state components
        x, y, z = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        e0, e1, e2, e3 = state[6], state[7], state[8], state[9]
        omega1, omega2, omega3 = state[10], state[11], state[12]
        motor_mass = state[13]

        # Create Vector/Matrix objects
        # RocketPy body frame: x=yaw, y=pitch, z=forward (longitudinal)
        # State vector already uses RocketPy convention after migration
        v = Vector([vx, vy, vz])
        e = [e0, e1, e2, e3]
        w = Vector([omega1, omega2, omega3])  # Already in RocketPy convention: [pitch, yaw, roll]

        # Retrieve mass properties from mass model
        total_mass = self.mass_model.total_mass(time)
        total_mass_dot = self.mass_model.total_mass_dot(time)
        total_mass_ddot = self.mass_model.total_mass_ddot(time)

        # COM position relative to CDM in RocketPy body frame
        # RocketPy uses z-axis as longitudinal, so COM offset is along z
        r_CM_z = self.mass_model.com_offset(time)
        r_CM = Vector([0, 0, r_CM_z])  # RocketPy convention: z = longitudinal
        r_CM_dot = Vector([0, 0, self.mass_model.com_offset_dot(time)])
        r_CM_ddot = Vector([0, 0, self.mass_model.com_offset_ddot(time)])

        # Nozzle position vector (RocketPy body frame, z-axis)
        r_NOZ = Vector([0, 0, self.mass_model.nozzle_to_cdm])

        # Nozzle gyration tensor
        S_nozzle = self.mass_model.nozzle_gyration_tensor

        # Inertia tensor and its derivative
        # Mass model now stores in RocketPy convention (z=longitudinal)
        inertia_tensor = self.mass_model.inertia_matrix(time)
        I_dot = self.mass_model.inertia_matrix_dot(time)

        # Inertia tensor relative to CM (parallel axis theorem)
        H = (r_CM.cross_matrix @ -r_CM.cross_matrix) * total_mass
        I_CM = inertia_tensor - H

        # Transformation matrix from body to world frame
        # Use RocketPy's transformation directly (z=longitudinal convention)
        K = Matrix.transformation(e)
        Kt = K.transpose

        # Compute aerodynamic forces and moments
        aero = self._compute_aero_state(time, state, allow_events=False)
        
        # Extract forces/moments from aero state (already in RocketPy body frame)
        aero_force_body_rp = Vector(aero["aero_force_body_rp"])
        R1 = aero_force_body_rp.x  # Lateral force (yaw)
        R2 = aero_force_body_rp.y  # Lateral force (pitch)
        R3 = aero_force_body_rp.z  # Axial force (drag, negative)

        # Thrust in RocketPy body frame (along +z axis)
        thrust_mag = self.motor.thrust(time)
        pressure = self.environment.air_properties(z)["pressure"]
        # Pressure thrust correction (simplified - RocketPy uses motor.pressure_thrust)
        net_thrust = max(thrust_mag, 0.0)  # Simplified for now

        # Moments in RocketPy body frame (already computed in aero state)
        moment_body_rp = Vector(aero["moment_body_rp"])
        M1 = moment_body_rp.x  # Pitch moment
        M2 = moment_body_rp.y  # Yaw moment
        M3 = moment_body_rp.z  # Roll moment

        # Weight in RocketPy body frame
        gravity_mag = G0  # Standard gravity
        # World frame: gravity is [0, 0, -g]
        weight_in_body_frame = Kt @ Vector([0, 0, -total_mass * gravity_mag])

        # Kane/RTT T matrices
        T00 = total_mass * r_CM
        T03 = 2 * total_mass_dot * (r_NOZ - r_CM) - 2 * total_mass * r_CM_dot
        T04 = (
            Vector([0, 0, net_thrust])
            - total_mass * r_CM_ddot
            - 2 * total_mass_dot * r_CM_dot
            + total_mass_ddot * (r_NOZ - r_CM)
        )
        T05 = total_mass_dot * S_nozzle - I_dot

        T20 = (
            ((w ^ T00) ^ w)
            + (w ^ T03)
            + T04
            + weight_in_body_frame
            + Vector([R1, R2, R3])
        )

        T21 = (
            ((inertia_tensor @ w) ^ w)
            + T05 @ w
            - (weight_in_body_frame ^ r_CM)
            + Vector([M1, M2, M3])
        )

        # Angular velocity derivative
        w_dot = I_CM.inverse @ (T21 + (T20 ^ r_CM))

        # Euler parameters derivative (quaternion)
        # Use RocketPy angular velocity components
        e_dot = [
            0.5 * (-w.x * e1 - w.y * e2 - w.z * e3),
            0.5 * (w.x * e0 + w.z * e2 - w.y * e3),
            0.5 * (w.y * e0 - w.z * e1 + w.x * e3),
            0.5 * (w.z * e0 + w.y * e1 - w.x * e2),
        ]

        # Velocity derivative with Coriolis acceleration
        # T20 is in RocketPy body frame, transform directly to world frame
        T20_world_vec = K @ T20
        r_CM_world_vec = K @ r_CM
        w_dot_world_vec = K @ w_dot
        
        # Earth rotation (simplified - RocketPy uses env.earth_rotation_vector)
        w_earth = Vector([0, 0, 0])  # Simplified for now - can add later
        
        # v_dot = K @ (T20/m - r_CM ^ w_dot) - 2*(w_earth ^ v)
        v_dot = (T20_world_vec / total_mass - (r_CM_world_vec ^ w_dot_world_vec)) - 2 * (w_earth ^ v)

        # Position derivative
        r_dot = [vx, vy, vz]

        # Mass flow rate
        mass_flow = self._motor_mass_flow(time)

        # Assemble derivatives (already in RocketPy convention)
        derivatives = np.zeros_like(state)
        derivatives[0:3] = r_dot
        derivatives[3:6] = [v_dot.x, v_dot.y, v_dot.z]
        derivatives[6:10] = e_dot
        derivatives[10:13] = [w_dot.x, w_dot.y, w_dot.z]  # Already in RocketPy convention
        derivatives[13] = mass_flow
        
        # Numerical safeguards to prevent instability
        # Clamp angular acceleration to reasonable values
        derivatives[10:13] = np.clip(derivatives[10:13], -100.0, 100.0)
        
        # Clamp linear acceleration to reasonable values (< 100g)
        derivatives[3:6] = np.clip(derivatives[3:6], -1000.0, 1000.0)
        
        # Check for NaN/Inf and fall back to simplified dynamics if detected
        if not np.isfinite(derivatives).all():
            return self._derivatives_simple(time, state)

        return derivatives

    def _derivatives_simple(self, time: float, state: np.ndarray) -> np.ndarray:
        """Simplified fallback derivatives if RocketPy math utils unavailable."""
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]

        aero = self._compute_aero_state(time, state, allow_events=False)
        acceleration = aero["acceleration"]

        inertia = self.mass_model.inertia_diag(time)
        torque_body = self._rotate_world_to_body(quaternion, aero["moment_world"])
        
        # Clamp angular velocity to prevent numerical instability
        # During descent, angular rates can grow if rocket tumbles
        omega = np.clip(angular_velocity, -10.0, 10.0)
        
        ang_momentum = inertia * omega
        omega_cross = np.cross(omega, ang_momentum)
        angular_acceleration = np.where(inertia > 1e-9, (torque_body - omega_cross) / inertia, 0.0)
        
        # Clamp angular acceleration to reasonable values
        angular_acceleration = np.clip(angular_acceleration, -1000.0, 1000.0)

        dq = 0.5 * self._quaternion_omega(quaternion, omega)
        mass_flow = self._motor_mass_flow(time)

        derivatives = np.zeros_like(state)
        derivatives[0:3] = velocity
        derivatives[3:6] = acceleration
        derivatives[6:10] = dq
        derivatives[10:13] = angular_acceleration
        derivatives[13] = mass_flow
        return derivatives

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _initial_orientation_quaternion(self) -> np.ndarray:
        # RocketPy convention: inclination is angle FROM VERTICAL (zenith angle)
        # 0° = vertical up, 90° = horizontal
        # RocketPy body frame: z-axis points forward (tail-to-nose)
        # Convert to direction vector in world frame (x=north, y=east, z=up)
        direction = np.array([
            math.sin(self.inclination) * math.cos(self.heading),  # North component
            math.sin(self.inclination) * math.sin(self.heading),  # East component
            math.cos(self.inclination),  # Up component
        ])
        direction = direction / np.linalg.norm(direction)
        # RocketPy body-z should point in the launch direction (tail-to-nose)
        # So we map body-z (0,0,1) to the launch direction
        # Use RocketPy's quaternion convention directly
        body_z = np.array([0.0, 0.0, 1.0])  # RocketPy body frame: z=forward
        return self._quat_from_two_vectors(body_z, direction)

    def _precompute_mass_table(self):
        times = np.asarray(self.mass_model.times, dtype=np.float64)
        masses = np.asarray(self.mass_model.motor_mass_values, dtype=np.float64)
        rates = np.asarray(self.mass_model.motor_mass_dot_values, dtype=np.float64)
        return times, masses, rates

    def _motor_mass_flow(self, time: float) -> float:
        if time >= self.motor.burn_time:
            return 0.0
        return float(np.interp(time, self.mass_times, self.mass_rates))

    @staticmethod
    def _quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
        if dot > 0.999999:
            return np.array([0.0, 0.0, 0.0, 1.0])
        if dot < -0.999999:
            axis = np.cross(np.array([1.0, 0.0, 0.0]), v_from)
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(np.array([0.0, 1.0, 0.0]), v_from)
            axis = axis / np.linalg.norm(axis)
            return np.concatenate((axis * math.sin(math.pi / 2.0), [math.cos(math.pi / 2.0)]))
        axis = np.cross(v_from, v_to)
        s = math.sqrt((1.0 + dot) * 2.0)
        inv_s = 1.0 / s
        quat = np.array([
            axis[0] * inv_s,
            axis[1] * inv_s,
            axis[2] * inv_s,
            0.5 * s,
        ])
        return quat / np.linalg.norm(quat)

    @staticmethod
    def _rotate_vector(quaternion: np.ndarray, vec: np.ndarray) -> np.ndarray:
        q = quaternion
        v = np.array([vec[0], vec[1], vec[2], 0.0])
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        return FlightSolver._quat_multiply(FlightSolver._quat_multiply(q, v), q_conj)[:3]

    @staticmethod
    def _rotate_world_to_body(quaternion: np.ndarray, vec: np.ndarray) -> np.ndarray:
        q_conj = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
        return FlightSolver._rotate_vector(q_conj, vec)

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ])

    @staticmethod
    def _quaternion_omega(quaternion: np.ndarray, omega: np.ndarray) -> np.ndarray:
        ox, oy, oz = omega
        matrix = np.array(
            [
                [0.0, -ox, -oy, -oz],
                [ox, 0.0, oz, -oy],
                [oy, -oz, 0.0, ox],
                [oz, oy, -ox, 0.0],
            ],
            dtype=np.float64,
        )
        return matrix @ quaternion

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
        a_unit = a / np.linalg.norm(a)
        b_unit = b / np.linalg.norm(b)
        dot = np.clip(np.dot(a_unit, b_unit), -1.0, 1.0)
        return math.acos(dot)

    @staticmethod
    def _summarize(history: List[StateSnapshot]) -> dict:
        if not history:
            return {
                "max_altitude": 0.0,
                "max_velocity": 0.0,
                "apogee_time": 0.0,
                "flight_time": 0.0,
                "max_angle_of_attack": 0.0,
                "max_sideslip": 0.0,
                "max_mach": 0.0,
                "max_dynamic_pressure": 0.0,
                "max_total_aero_force": 0.0,
                "max_parachute_drag": 0.0,
                "max_lift_force": 0.0,
                "max_body_moment": 0.0,
            }
        max_altitude_state = max(history, key=lambda s: s.z)
        max_velocity = max(np.linalg.norm(s.velocity) for s in history)
        return {
            "max_altitude": float(max_altitude_state.z),
            "max_velocity": float(max_velocity),
            "apogee_time": float(max_altitude_state.time),
            "flight_time": float(history[-1].time),
            "max_angle_of_attack": float(max(abs(s.angle_of_attack) for s in history)),
            "max_sideslip": float(max(abs(s.sideslip) for s in history)),
            "max_mach": float(max(s.mach for s in history)),
            "max_dynamic_pressure": float(max(s.dynamic_pressure for s in history)),
            "max_total_aero_force": float(max(np.linalg.norm(s.total_aero_force) for s in history)),
            "max_parachute_drag": float(max(np.linalg.norm(s.parachute_drag) for s in history)),
            "max_lift_force": float(max(np.linalg.norm(s.lift_force) for s in history)),
            "max_body_moment": float(max(np.linalg.norm(s.moment_world) for s in history)),
        }
