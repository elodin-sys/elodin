#!/usr/bin/env python3
"""
Calisto Rocket Simulation - Elodin Integration
Runs the RocketPy-compatible 6-DOF solver and visualizes in Elodin editor

Usage:
    # From elodin root directory, in nix shell:
    nix develop
    python3 examples/rocket-barrowman/main.py
"""

import sys
import os
import typing as ty
from dataclasses import dataclass, field

# Check for visualize mode BEFORE importing elodin (which has its own CLI)
VISUALIZE_MODE = os.getenv("ELODIN_VISUALIZE", "false").lower() == "true"

import numpy as np
import jax
import jax.numpy as jnp

# Add the rocket-barrowman directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    import elodin as el
except ImportError:
    print("ERROR: elodin not found. Make sure you're in the nix shell:")
    print("  cd /home/kush-mahajan/elodin")
    print("  nix develop")
    print("  python3 examples/rocket-barrowman/main.py")
    sys.exit(1)

# Import our rocket simulation
from environment import Environment
from motor_model import Motor
from rocket_model import Rocket as RocketModel
from flight_solver import FlightSolver, FlightResult
from calisto_builder import build_calisto

# Try to import mesh renderer for Elodin visualization
try:
    from mesh_renderer import generate_rocket_glb_from_solver, generate_elodin_assets, TRIMESH_AVAILABLE
except ImportError:
    TRIMESH_AVAILABLE = False
    generate_rocket_glb_from_solver = None
    generate_elodin_assets = None


def main():
    """Run Calisto rocket simulation and visualize in Elodin."""

    print("\n" + "=" * 70)
    print("CALISTO ROCKET SIMULATION - ELODIN")
    print("=" * 70)

    # Build Calisto rocket
    rocket_raw, motor_raw = build_calisto()
    rocket = RocketModel(rocket_raw)
    motor = Motor.from_openrocket(motor_raw)

    print(f"\n✓ Calisto loaded:")
    print(f"  Dry mass: {rocket.dry_mass:.3f} kg")
    print(f"  Dry CG: {rocket.dry_cg:.3f} m")
    print(f"  Reference diameter: {rocket.reference_diameter * 1000:.1f} mm")
    print(f"  Parachutes: {len(rocket.parachutes)}")

    # Environment (Spaceport America: elevation=1400m, no wind)
    env = Environment(elevation=1400.0)

    # Run simulation
    print(f"\nRunning simulation...")
    solver = FlightSolver(
        rocket=rocket,
        motor=motor,
        environment=env,
        rail_length=5.2,  # meters
        inclination_deg=5.0,  # degrees from vertical (0 = straight up, 90 = horizontal)
        heading_deg=0.0,  # degrees (north)
    )

    result = solver.run(max_time=600.0)  # 10 minutes max, will stop at impact

    # Print summary
    if len(result.history) == 0:
        print(f"\n❌ Simulation failed - no history recorded")
        return

    max_alt = max(s.z for s in result.history)
    apogee_state = next((s for s in result.history if s.z == max_alt), result.history[-1])
    apogee_time = apogee_state.time
    max_v = max(np.linalg.norm(s.velocity) for s in result.history)

    print(f"\n✓ Simulation complete:")
    print(f"  Max altitude: {max_alt:.1f} m (AGL)")
    print(f"  Apogee time: {apogee_time:.2f} s")
    print(f"  Max velocity: {max_v:.1f} m/s")
    print(f"  History length: {len(result.history)} snapshots")

    # Debug: print first few states
    print(f"\nFirst 3 states:")
    for i, s in enumerate(result.history[:3]):
        print(f"  t={s.time:.2f}s: pos={s.position}, z={s.z:.2f}m")

    visualize_in_elodin(result, solver)


def visualize_in_elodin(result: FlightResult, solver: FlightSolver) -> None:
    print(f"\nConverting to Elodin format and launching editor...")
    
    # Generate rocket mesh for Elodin visualization
    if TRIMESH_AVAILABLE and generate_rocket_glb_from_solver:
        try:
            # Generate in the elodin/assets directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            assets_dir = os.path.join(script_dir, "..", "..", "assets")
            os.makedirs(assets_dir, exist_ok=True)
            generate_rocket_glb_from_solver(solver, assets_dir)
        except Exception as e:
            print(f"⚠ Could not generate rocket mesh: {e}")
            print("  (Elodin will use existing rocket.glb in assets)")

    history = result.history
    times_np = np.array([s.time for s in history])
    dt = float(times_np[1] - times_np[0]) if len(times_np) > 1 else 0.01

    # Sanitize data - replace NaN/Inf with safe values
    def sanitize_array(arr, default=0.0):
        """Replace NaN and Inf with safe values."""
        arr = np.nan_to_num(arr, nan=default, posinf=1e6, neginf=-1e6)
        return arr
    
    def sanitize_quaternion(q):
        """Ensure quaternion is valid (normalized, no NaN)."""
        q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        return q / norm

    # Extract and sanitize arrays
    positions_raw = np.array([s.position for s in history])
    velocities_raw = np.array([s.velocity for s in history])
    quaternions_raw = np.array([sanitize_quaternion(s.quaternion) for s in history])
    angular_velocities_raw = np.array([s.angular_velocity for s in history])
    aero_forces_raw = np.array([s.total_aero_force for s in history])
    
    # Clamp unreasonable values
    positions_raw = sanitize_array(positions_raw)
    velocities_raw = sanitize_array(velocities_raw)
    velocities_raw = np.clip(velocities_raw, -2000, 2000)  # Max ~Mach 6
    angular_velocities_raw = sanitize_array(angular_velocities_raw)
    angular_velocities_raw = np.clip(angular_velocities_raw, -50, 50)  # Max 50 rad/s
    aero_forces_raw = sanitize_array(aero_forces_raw)
    aero_forces_raw = np.clip(aero_forces_raw, -1e6, 1e6)

    positions = jnp.asarray(positions_raw)
    velocities = jnp.asarray(velocities_raw)
    quaternions = jnp.asarray(quaternions_raw)
    angular_velocities = jnp.asarray(angular_velocities_raw)
    aero_forces = jnp.asarray(aero_forces_raw)

    altitudes = positions[:, 2]
    downrange = jnp.linalg.norm(positions[:, :2], axis=1)
    speed = jnp.linalg.norm(velocities, axis=1)

    # Calculate acceleration from velocity differences
    # For first point, use forward difference; for others, use backward difference
    if len(velocities) > 1:
        accel_vec = (velocities[1:] - velocities[:-1]) / dt
        # Prepend first acceleration (forward difference) to match array length
        accel_vec = jnp.concatenate((accel_vec[:1], accel_vec), axis=0)
    else:
        # Single point: zero acceleration
        accel_vec = jnp.zeros((1, 3))
    # Clamp acceleration to reasonable values (< 100g = 981 m/s²)
    accel_vec = jnp.clip(accel_vec, -1000, 1000)
    accel_mag = jnp.nan_to_num(jnp.linalg.norm(accel_vec, axis=1), nan=0.0, posinf=1000)

    # Sanitize all scalar arrays
    machs = jnp.nan_to_num(jnp.asarray([s.mach for s in history]), nan=0.0, posinf=10.0, neginf=0.0)
    aoas_raw = sanitize_array(np.array([s.angle_of_attack for s in history]))
    aoas = jnp.asarray(np.clip(aoas_raw, -180, 180))
    dynamic_pressures_raw = sanitize_array(np.array([s.dynamic_pressure for s in history]))
    dynamic_pressures = jnp.asarray(np.clip(dynamic_pressures_raw, 0, 1e6))
    aero_force_mag = jnp.nan_to_num(jnp.linalg.norm(aero_forces, axis=1), nan=0.0, posinf=1e6)

    def quat_to_euler_deg(q: jnp.ndarray) -> jnp.ndarray:
        x, y, z, w = q
        # Ensure quaternion is normalized
        norm = jnp.sqrt(x*x + y*y + z*z + w*w)
        norm = jnp.where(norm < 1e-6, 1.0, norm)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = jnp.arctan2(sinr_cosp, cosr_cosp)

        sinp = jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = jnp.arcsin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = jnp.arctan2(siny_cosp, cosy_cosp)

        euler = jnp.rad2deg(jnp.array([roll, pitch, yaw]))
        # Clamp to reasonable range
        return jnp.clip(euler, -360.0, 360.0)

    euler_deg = jax.vmap(quat_to_euler_deg)(quaternions)
    # Replace any remaining NaN with 0
    euler_deg = jnp.nan_to_num(euler_deg, nan=0.0)

    angular_rates_deg = jnp.rad2deg(angular_velocities)
    angular_rates_deg = jnp.clip(angular_rates_deg, -3000.0, 3000.0)  # Reasonable angular rate limit

    Altitude = ty.Annotated[jax.Array, el.Component("altitude_m", el.ComponentType.F64)]
    Downrange = ty.Annotated[jax.Array, el.Component("downrange_m", el.ComponentType.F64)]
    Speed = ty.Annotated[jax.Array, el.Component("speed_ms", el.ComponentType.F64)]
    Accel = ty.Annotated[jax.Array, el.Component("accel_ms2", el.ComponentType.F64)]
    MachComp = ty.Annotated[jax.Array, el.Component("mach", el.ComponentType.F64)]
    AngleOfAttackComp = ty.Annotated[
        jax.Array, el.Component("angle_of_attack_deg", el.ComponentType.F64)
    ]
    DynamicPressureComp = ty.Annotated[
        jax.Array, el.Component("dynamic_pressure_pa", el.ComponentType.F64)
    ]
    AeroForceMag = ty.Annotated[jax.Array, el.Component("aero_force_n", el.ComponentType.F64)]
    EulerAngles = ty.Annotated[
        jax.Array,
        el.Component(
            "euler_deg",
            el.ComponentType(el.PrimitiveType.F64, (3,)),
            metadata={"element_names": "roll,pitch,yaw"},
        ),
    ]
    BodyRates = ty.Annotated[
        jax.Array,
        el.Component(
            "angular_rates_deg",
            el.ComponentType(el.PrimitiveType.F64, (3,)),
            metadata={"element_names": "p,q,r"},
        ),
    ]

    @el.dataclass
    class PlaybackTelemetry(el.Archetype):
        altitude: Altitude = field(default_factory=lambda: jnp.float64(0.0))
        downrange: Downrange = field(default_factory=lambda: jnp.float64(0.0))
        speed: Speed = field(default_factory=lambda: jnp.float64(0.0))
        accel_ms2: Accel = field(default_factory=lambda: jnp.float64(0.0))
        mach: MachComp = field(default_factory=lambda: jnp.float64(0.0))
        angle_of_attack: AngleOfAttackComp = field(default_factory=lambda: jnp.float64(0.0))
        dynamic_pressure: DynamicPressureComp = field(default_factory=lambda: jnp.float64(0.0))
        aero_force: AeroForceMag = field(default_factory=lambda: jnp.float64(0.0))
        euler: EulerAngles = field(default_factory=lambda: jnp.zeros(3))
        angular_rates: BodyRates = field(default_factory=lambda: jnp.zeros(3))

    world = el.World()
    mass0 = solver.mass_model.total_mass(0.0)
    inertia_diag0 = solver.mass_model.inertia_diag(0.0)

    # Ensure initial position is at origin (ground level)
    initial_pos = positions[0]
    if initial_pos[2] < 0:
        initial_pos = initial_pos.copy()
        initial_pos[2] = 0.0  # Start at ground level
    
    # Ensure initial quaternion is valid
    initial_quat = quaternions[0]
    quat_norm = np.linalg.norm(initial_quat)
    if quat_norm < 1e-6:
        initial_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity
    else:
        initial_quat = initial_quat / quat_norm
    
    rocket_entity = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    linear=jnp.asarray(initial_pos),
                    angular=el.Quaternion.from_array(initial_quat),
                ),
                inertia=el.SpatialInertia(
                    mass=mass0,
                    inertia=inertia_diag0,
                ),
            ),
            PlaybackTelemetry(),
        ],
        name="rocket",
    )

    # Get absolute paths to assets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    elodin_root = os.path.join(script_dir, "..", "..")
    assets_dir = os.path.abspath(os.path.join(elodin_root, "assets"))
    compass_path = os.path.join(assets_dir, "compass.glb")
    rocket_path = os.path.join(assets_dir, "rocket.glb")
    
    schematic = f"""
    tabs {{
        hsplit share=0.65 name="Flight View" {{
            viewport name=Viewport pos="rocket.world_pos + (0,0,0,0, 6,3,2)" look_at="rocket.world_pos" hdr=#true show_grid=#true active=#true
            vsplit share=0.45 {{
                graph "rocket.altitude_m,rocket.downrange_m" name="Altitude / Downrange"
                graph "rocket.speed_ms" name="Speed"
                graph "rocket.dynamic_pressure_pa,rocket.mach,rocket.aero_force_n" name="Aero Loads"
            }}
        }}
        hsplit share=0.35 name="Orientation" {{
            graph "rocket.euler_deg[0],rocket.euler_deg[1],rocket.euler_deg[2]" name="Euler Angles"
            graph "rocket.angular_rates_deg[0],rocket.angular_rates_deg[1],rocket.angular_rates_deg[2]" name="Body Rates"
        }}
    }}

    object_3d "(0,0,0,1)" {{
        glb path="{compass_path}"
    }}
    object_3d rocket.world_pos {{
        glb path="{rocket_path}"
    }}
    line_3d rocket.world_pos line_width=2.0 perspective=#false {{
        color 255 223 0
    }}
    """

    # Pass schematic directly to editor - this avoids file watcher triggering reload loops
    if hasattr(world, "schematic"):
        world.schematic(schematic, "rocket.kdl")
    else:
        # Fallback: write to file if schematic() method not available
        with open("rocket.kdl", "w") as f:
            f.write(schematic)

    frame_count = positions.shape[0]
    frame_max = float(frame_count - 1)

    def frame_index(tick_value: jnp.ndarray) -> jnp.ndarray:
        idx = jnp.clip(jnp.floor(tick_value), 0.0, frame_max)
        return idx.astype(jnp.int32)

    @el.system
    def playback_world_pos(
        tick: el.Query[el.SimulationTick],
        pos_q: el.Query[el.WorldPos],
    ) -> el.Query[el.WorldPos]:
        idx = frame_index(tick[0])
        transform = el.SpatialTransform(
            linear=positions[idx],
            angular=el.Quaternion.from_array(quaternions[idx]),
        )
        return pos_q.map(el.WorldPos, lambda _: transform)

    @el.system
    def playback_world_vel(
        tick: el.Query[el.SimulationTick],
        vel_q: el.Query[el.WorldVel],
    ) -> el.Query[el.WorldVel]:
        idx = frame_index(tick[0])
        motion = el.SpatialMotion(
            linear=velocities[idx],
            angular=angular_velocities[idx],
        )
        return vel_q.map(el.WorldVel, lambda _: motion)

    @el.system
    def playback_altitude(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Altitude],
    ) -> el.Query[Altitude]:
        idx = frame_index(tick[0])
        return comp.map(Altitude, lambda _: jnp.float64(altitudes[idx]))

    @el.system
    def playback_downrange(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Downrange],
    ) -> el.Query[Downrange]:
        idx = frame_index(tick[0])
        return comp.map(Downrange, lambda _: jnp.float64(downrange[idx]))

    @el.system
    def playback_speed(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Speed],
    ) -> el.Query[Speed]:
        idx = frame_index(tick[0])
        return comp.map(Speed, lambda _: jnp.float64(speed[idx]))

    @el.system
    def playback_accel(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[Accel],
    ) -> el.Query[Accel]:
        idx = frame_index(tick[0])
        return comp.map(Accel, lambda _: jnp.float64(accel_mag[idx]))

    @el.system
    def playback_mach(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[MachComp],
    ) -> el.Query[MachComp]:
        idx = frame_index(tick[0])
        return comp.map(MachComp, lambda _: jnp.float64(machs[idx]))

    @el.system
    def playback_aoa(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AngleOfAttackComp],
    ) -> el.Query[AngleOfAttackComp]:
        idx = frame_index(tick[0])
        return comp.map(AngleOfAttackComp, lambda _: jnp.float64(aoas[idx]))

    @el.system
    def playback_dynamic_pressure(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[DynamicPressureComp],
    ) -> el.Query[DynamicPressureComp]:
        idx = frame_index(tick[0])
        return comp.map(DynamicPressureComp, lambda _: jnp.float64(dynamic_pressures[idx]))

    @el.system
    def playback_aero_force(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[AeroForceMag],
    ) -> el.Query[AeroForceMag]:
        idx = frame_index(tick[0])
        return comp.map(AeroForceMag, lambda _: jnp.float64(aero_force_mag[idx]))

    @el.system
    def playback_euler(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[EulerAngles],
    ) -> el.Query[EulerAngles]:
        idx = frame_index(tick[0])
        return comp.map(EulerAngles, lambda _: euler_deg[idx])

    @el.system
    def playback_body_rates(
        tick: el.Query[el.SimulationTick],
        comp: el.Query[BodyRates],
    ) -> el.Query[BodyRates]:
        idx = frame_index(tick[0])
        return comp.map(BodyRates, lambda _: angular_rates_deg[idx])

    playback_system = (
        playback_world_pos
        | playback_world_vel
        | playback_altitude
        | playback_downrange
        | playback_speed
        | playback_accel
        | playback_mach
        | playback_aoa
        | playback_dynamic_pressure
        | playback_aero_force
        | playback_euler
        | playback_body_rates
    )

    print(f"\n✓ Schematic: rocket.kdl")
    print(f"✓ Launching Elodin editor...")

    world.run(
        playback_system,
        sim_time_step=dt,
        run_time_step=dt,
        default_playback_speed=1.0,
        max_ticks=frame_count,
    )

    print(f"\n{'=' * 70}")
    print(f"Simulation + visualization complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    import tempfile

    # Check if we should load saved visualization data
    # This happens when:
    # 1. ELODIN_VISUALIZE env var is set, OR
    # 2. Pickle files exist (meaning Streamlit saved simulation data)
    temp_dir = Path(tempfile.gettempdir()) / "elodin_rocket_sim"
    result_file = temp_dir / "simulation_result.pkl"
    solver_file = temp_dir / "solver.pkl"

    should_visualize = VISUALIZE_MODE or (result_file.exists() and solver_file.exists())

    if should_visualize:
        # Load saved simulation data from Streamlit
        if not result_file.exists() or not solver_file.exists():
            print("ERROR: Simulation data not found. Please run a simulation in Streamlit first.")
            print(f"   Result file: {result_file} (exists: {result_file.exists()})")
            print(f"   Solver file: {solver_file} (exists: {solver_file.exists()})")
            sys.exit(1)

        with open(result_file, "rb") as f:
            result_data = pickle.load(f)
        with open(solver_file, "rb") as f:
            solver_data = pickle.load(f)

        # Reconstruct FlightResult from dict
        from flight_solver import FlightResult, StateSnapshot
        import numpy as np

        history = []
        for s_data in result_data["history"]:
            # Get all required fields, with defaults for optional ones
            history.append(
                StateSnapshot(
                    time=s_data["time"],
                    position=np.array(s_data["position"]),
                    velocity=np.array(s_data["velocity"]),
                    quaternion=np.array(s_data["quaternion"]),
                    angular_velocity=np.array(s_data["angular_velocity"]),
                    motor_mass=s_data["motor_mass"],
                    angle_of_attack=s_data["angle_of_attack"],
                    sideslip=s_data["sideslip"],
                    mach=s_data.get("mach", 0.0),
                    dynamic_pressure=s_data.get("dynamic_pressure", 0.0),
                    drag_force=np.array(s_data.get("drag_force", [0.0, 0.0, 0.0])),
                    lift_force=np.array(s_data.get("lift_force", [0.0, 0.0, 0.0])),
                    parachute_drag=np.array(s_data.get("parachute_drag", [0.0, 0.0, 0.0])),
                    moment_world=np.array(s_data.get("moment_world", [0.0, 0.0, 0.0])),
                    total_aero_force=np.array(s_data["total_aero_force"]),
                )
            )

        # Create FlightResult with proper structure
        result = FlightResult(
            history=history,
            summary={
                "max_altitude": result_data.get("max_altitude", 0.0),
                "max_velocity": result_data.get("max_velocity", 0.0),
                "apogee_time": result_data.get("apogee_time", 0.0),
                "landing_time": result_data.get("landing_time", 0.0),
            },
        )

        # Reconstruct minimal solver for visualization
        from rocket_model import Rocket
        from motor_model import Motor
        from environment import Environment
        from flight_solver import MassInertiaModel

        # Create minimal MassInertiaModel for visualization
        class MinimalMassModel:
            def __init__(self, mass_model_data):
                self.times = np.array(mass_model_data.get("times", [0.0, 1.0]))
                self.total_mass_values = np.array(
                    mass_model_data.get("total_mass_values", [0.0, 0.0])
                )
                self.inertia_values = np.array(
                    mass_model_data.get("inertia_values", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                )

            def total_mass(self, time: float) -> float:
                """Interpolate total mass at given time."""
                if len(self.times) == 0:
                    return 0.0
                clamped = np.clip(time, self.times[0], self.times[-1])
                return float(np.interp(clamped, self.times, self.total_mass_values))

            def inertia_diag(self, time: float) -> np.ndarray:
                """Interpolate inertia diagonal at given time."""
                if len(self.times) == 0 or self.inertia_values.shape[0] == 0:
                    return np.array([0.0, 0.0, 0.0])
                clamped = np.clip(time, self.times[0], self.times[-1])
                return np.array(
                    [
                        np.interp(clamped, self.times, self.inertia_values[:, i])
                        for i in range(self.inertia_values.shape[1])
                    ],
                    dtype=float,
                )

        # Create minimal objects for visualization
        class MinimalSolver:
            def __init__(self, solver_data):
                self.rocket = type(
                    "obj",
                    (object,),
                    {
                        "dry_mass": solver_data["rocket"]["dry_mass"],
                        "dry_cg": solver_data["rocket"]["dry_cg"],
                        "reference_diameter": solver_data["rocket"]["reference_diameter"],
                    },
                )()
                self.motor = type(
                    "obj",
                    (object,),
                    {
                        "total_mass": solver_data["motor"]["total_mass"],
                        "propellant_mass": solver_data["motor"]["propellant_mass"],
                    },
                )()
                self.environment = type(
                    "obj",
                    (object,),
                    {
                        "elevation": solver_data["environment"]["elevation"],
                    },
                )()
                # Add mass_model
                if "mass_model" in solver_data:
                    self.mass_model = MinimalMassModel(solver_data["mass_model"])
                else:
                    # Fallback: create minimal mass model with default values
                    dry_mass = solver_data["rocket"].get("dry_mass", 0.0)
                    motor_mass = solver_data["motor"].get("total_mass", 0.0)
                    total_mass = dry_mass + motor_mass
                    self.mass_model = MinimalMassModel(
                        {
                            "times": [0.0, 1.0],
                            "total_mass_values": [total_mass, total_mass],
                            "inertia_values": [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]],  # Default inertia
                        }
                    )

        solver = MinimalSolver(solver_data)

        visualize_in_elodin(result, solver)
    else:
        main()
