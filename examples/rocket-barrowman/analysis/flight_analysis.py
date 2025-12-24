"""
Aerospace-Grade Flight Analysis Module

Computes comprehensive flight dynamics metrics:
- Stability derivatives (static and dynamic)
- First and second order flight terms
- Aerodynamic coefficients
- Performance metrics
- Safety margins
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from core import FlightResult, Matrix, Vector


@dataclass
class StabilityDerivatives:
    """Stability and control derivatives."""

    # Longitudinal stability
    C_L_alpha: float  # Lift curve slope
    C_D_alpha: float  # Drag due to angle of attack
    C_m_alpha: float  # Pitching moment coefficient (static stability)
    C_m_q: float  # Pitch damping
    C_m_alphadot: float  # Pitch rate due to alpha rate

    # Lateral-directional stability
    C_Y_beta: float  # Side force due to sideslip
    C_l_beta: float  # Roll moment due to sideslip (dihedral effect)
    C_l_p: float  # Roll damping
    C_n_beta: float  # Yaw moment due to sideslip (weathercock stability)
    C_n_r: float  # Yaw damping
    C_n_p: float  # Cross-coupling (yaw due to roll)

    # Control derivatives
    C_m_delta: float  # Pitch control effectiveness
    C_l_delta: float  # Roll control effectiveness
    C_n_delta: float  # Yaw control effectiveness

    # Damping derivatives
    C_L_q: float  # Lift due to pitch rate
    C_D_q: float  # Drag due to pitch rate


@dataclass
class FlightMetrics:
    """Comprehensive flight performance metrics."""

    # Stability metrics
    static_margin: float  # Static margin (calibers)
    stability_derivatives: StabilityDerivatives

    # Performance metrics
    max_q: float  # Maximum dynamic pressure (Max-Q)
    max_q_time: float
    max_g: float  # Maximum acceleration (g's)
    max_g_time: float
    max_mach: float
    max_mach_time: float

    # Flight phases
    boost_phase_duration: float
    coast_phase_duration: float
    descent_phase_duration: float

    # Energy metrics
    total_energy: np.ndarray
    kinetic_energy: np.ndarray
    potential_energy: np.ndarray

    # Aerodynamic metrics
    lift_to_drag_ratio: np.ndarray
    drag_coefficient: np.ndarray
    lift_coefficient: np.ndarray

    # Safety metrics
    min_stability_cal: float  # Minimum stability margin during flight
    max_angle_of_attack: float
    max_sideslip: float


class FlightAnalyzer:
    """Analyzes flight data to compute aerospace-grade metrics."""

    def __init__(self, result: FlightResult, solver):
        self.result = result
        self.solver = solver
        self.history = result.history

    def compute_all_metrics(self) -> FlightMetrics:
        """Compute comprehensive flight metrics."""
        times = np.array([s.time for s in self.history])
        positions = np.array([s.position for s in self.history])
        velocities = np.array([s.velocity for s in self.history])
        aoas = np.array([s.angle_of_attack for s in self.history])
        sideslips = np.array([s.sideslip for s in self.history])
        machs = np.array([s.mach for s in self.history])
        dynamic_pressures = np.array([s.dynamic_pressure for s in self.history])
        drag_forces = np.array([s.drag_force for s in self.history])
        lift_forces = np.array([s.lift_force for s in self.history])
        moments = np.array([s.moment_world for s in self.history])
        angular_velocities = np.array([s.angular_velocity for s in self.history])

        # Find apogee - truncate aero analysis here since chutes deploy and airframe separates
        apogee_idx = np.argmax(positions[:, 2])
        
        # Compute stability derivatives ONLY up to apogee
        # After apogee, chutes deploy and Barrowman aero model doesn't apply
        stability = self._compute_stability_derivatives(
            times[:apogee_idx + 1],
            aoas[:apogee_idx + 1],
            sideslips[:apogee_idx + 1],
            machs[:apogee_idx + 1],
            dynamic_pressures[:apogee_idx + 1],
            drag_forces[:apogee_idx + 1],
            lift_forces[:apogee_idx + 1],
            moments[:apogee_idx + 1],
            angular_velocities[:apogee_idx + 1],
        )

        # Performance metrics
        max_q_idx = np.argmax(dynamic_pressures)
        max_q = dynamic_pressures[max_q_idx]
        max_q_time = times[max_q_idx]

        # Compute accelerations
        if len(velocities) > 1:
            dt = times[1] - times[0] if len(times) > 1 else 0.01
            accelerations = np.diff(velocities, axis=0) / dt
            accel_mags = np.linalg.norm(accelerations, axis=1)
            g_forces = accel_mags / 9.80665
            max_g_idx = np.argmax(g_forces)
            max_g = g_forces[max_g_idx]
            max_g_time = times[max_g_idx + 1]
        else:
            max_g = 0.0
            max_g_time = 0.0

        max_mach_idx = np.argmax(machs)
        max_mach = machs[max_mach_idx]
        max_mach_time = times[max_mach_idx]

        # Flight phases (apogee_idx already computed above)
        motor_burn_time = self.solver.motor.burn_time
        boost_phase = motor_burn_time
        apogee_time = times[apogee_idx]
        coast_phase = apogee_time - boost_phase
        descent_phase = times[-1] - apogee_time

        # Energy metrics
        masses = np.array([self.solver.mass_model.total_mass(t) for t in times])
        kinetic_energy = 0.5 * masses * np.linalg.norm(velocities, axis=1) ** 2
        potential_energy = masses * 9.80665 * positions[:, 2]
        total_energy = kinetic_energy + potential_energy

        # Aerodynamic coefficients
        qs = dynamic_pressures
        ref_area = np.pi * (self.solver.rocket.reference_diameter / 2) ** 2

        # Avoid division by zero
        qs_safe = np.where(qs > 1e-3, qs, 1e-3)
        drag_coeff = np.where(
            qs_safe > 1e-3, 2 * np.linalg.norm(drag_forces, axis=1) / (qs_safe * ref_area), 0.0
        )
        lift_coeff = np.where(
            qs_safe > 1e-3, 2 * np.linalg.norm(lift_forces, axis=1) / (qs_safe * ref_area), 0.0
        )

        # L/D ratio
        drag_mags = np.linalg.norm(drag_forces, axis=1)
        lift_mags = np.linalg.norm(lift_forces, axis=1)
        ld_ratio = np.where(drag_mags > 1e-3, lift_mags / drag_mags, 0.0)

        # Stability margin (simplified - would need CG/CP data)
        # Estimate from moment coefficient
        moment_mags = np.linalg.norm(moments, axis=1)
        np.where(
            qs_safe > 1e-3,
            moment_mags / (qs_safe * ref_area * self.solver.rocket.reference_diameter),
            0.0,
        )
        # Static margin estimate (calibers) - time-varying based on CG shift
        static_margin = self._estimate_static_margin()
        min_stability = (
            float(np.min(static_margin))
            if isinstance(static_margin, np.ndarray)
            else float(static_margin)
        )
        # Return average static margin for summary, but keep array for time-series plots
        avg_static_margin = (
            float(np.mean(static_margin))
            if isinstance(static_margin, np.ndarray)
            else float(static_margin)
        )

        return FlightMetrics(
            static_margin=avg_static_margin,
            stability_derivatives=stability,
            max_q=max_q,
            max_q_time=max_q_time,
            max_g=max_g,
            max_g_time=max_g_time,
            max_mach=max_mach,
            max_mach_time=max_mach_time,
            boost_phase_duration=boost_phase,
            coast_phase_duration=coast_phase,
            descent_phase_duration=descent_phase,
            total_energy=total_energy,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            lift_to_drag_ratio=ld_ratio,
            drag_coefficient=drag_coeff,
            lift_coefficient=lift_coeff,
            min_stability_cal=min_stability,
            max_angle_of_attack=np.max(np.abs(aoas)),
            max_sideslip=np.max(np.abs(sideslips)),
        )

    def _estimate_static_margin(self) -> np.ndarray:
        """Estimate static margin from rocket geometry - time-varying based on CG shift."""
        # Static margin ≈ (CP - CG) / diameter
        # CG varies with motor burn, so static margin changes during flight
        # CP varies slightly with Mach number, so calculate it at each time step
        try:
            rocket = self.solver.rocket
            ref_diameter = rocket.reference_diameter

            if ref_diameter <= 0:
                # Fallback if diameter is invalid
                times = np.array([s.time for s in self.history])
                return np.full_like(times, 1.5)

            times = np.array([s.time for s in self.history])
            machs = np.array([s.mach for s in self.history])

            # Calculate CP at each time step (varies with Mach)
            # CP is typically calculated from nose tip
            try:
                cp_positions = np.array([float(rocket.aero.calculate_cp(mach)) for mach in machs])
            except Exception:
                # Fallback: estimate CP position (typically ~60% of body length from nose)
                try:
                    total_length = getattr(rocket, "reference_length", 2.0)
                    if total_length <= 0:
                        total_length = 2.0
                except Exception:
                    total_length = 2.0  # Default estimate
                cp_positions = np.full_like(times, total_length * 0.6)

            # Calculate CG at each time step (accounts for propellant burn)
            # CG is also from nose tip
            cg_positions = np.array([self.solver.mass_model.total_cg(t) for t in times])

            # Static margin in calibers: (CP - CG) / diameter
            # Both CP and CG should be from the same reference (nose tip)
            # CP and CG are both measured from nose tip, so difference is correct
            static_margin = (cp_positions - cg_positions) / ref_diameter

            # Sanity check: static margin should be reasonable (0.5-5 calibers typically)
            # If it's negative, rocket is unstable
            # If it's > 10 calibers, something is wrong with CP or CG calculation
            static_margin = np.clip(static_margin, -5.0, 10.0)  # Allow negative to show instability

            return static_margin
        except Exception:
            # Fallback: return constant stable margin
            times = np.array([s.time for s in self.history])
            return np.full_like(times, 1.5)  # Default stable margin

    def _compute_stability_derivatives(
        self,
        times: np.ndarray,
        aoas: np.ndarray,
        sideslips: np.ndarray,
        machs: np.ndarray,
        qs: np.ndarray,
        drag_forces: np.ndarray,
        lift_forces: np.ndarray,
        moments: np.ndarray,
        angular_velocities: np.ndarray,
    ) -> StabilityDerivatives:
        """
        Compute stability derivatives from flight data.
        
        Note: Data should be truncated at apogee since after chute deployment
        and airframe separation, the Barrowman aerodynamic model no longer applies.
        """

        # Reference values
        ref_area = np.pi * (self.solver.rocket.reference_diameter / 2) ** 2
        ref_length = self.solver.rocket.reference_diameter

        # Normalize forces and moments
        qs_safe = np.where(qs > 1e-3, qs, 1e-3)

        # Lift and drag coefficients
        lift_mags = np.linalg.norm(lift_forces, axis=1)
        drag_mags = np.linalg.norm(drag_forces, axis=1)
        C_L = 2 * lift_mags / (qs_safe * ref_area)
        C_D = 2 * drag_mags / (qs_safe * ref_area)

        # Transform moments from world frame to body frame
        # Moments are stored in world frame, need body frame for proper analysis
        moments_body = np.zeros_like(moments)
        
        # Only use history up to the length of truncated arrays
        n_samples = len(moments)
        velocities_world = np.array([self.history[i].velocity for i in range(n_samples)])
        speeds = np.linalg.norm(velocities_world, axis=1)

        for i in range(n_samples):
            snapshot = self.history[i]
            quaternion = snapshot.quaternion
            # Transformation matrix from world to body (transpose of body to world)
            Kt = Matrix.transformation(quaternion).transpose
            moment_world_vec = Vector(moments[i])
            moment_body_vec = Kt @ moment_world_vec
            moments_body[i] = np.array([moment_body_vec.x, moment_body_vec.y, moment_body_vec.z])

        # Moment components in body frame (RocketPy convention: [pitch, yaw, roll] = [M_y, M_z, M_x])
        M_pitch = moments_body[:, 1]  # Pitch moment (M_y in body frame)
        M_yaw = moments_body[:, 2]  # Yaw moment (M_z in body frame)
        M_roll = moments_body[:, 0]  # Roll moment (M_x in body frame)

        # Normalize moments to coefficients
        C_m_pitch = 2 * M_pitch / (qs_safe * ref_area * ref_length)
        C_m_yaw = 2 * M_yaw / (qs_safe * ref_area * ref_length)
        C_m_roll = 2 * M_roll / (qs_safe * ref_area * ref_length)
        moment_mags = np.linalg.norm(moments_body, axis=1)
        C_m_mag = moment_mags / (qs_safe * ref_area * ref_length)

        # Angular rates (p, q, r in body frame)
        # Angular velocities are already in body frame (RocketPy convention: [pitch, yaw, roll])
        p = angular_velocities[:, 2]  # Roll rate (M_x)
        q = angular_velocities[:, 0]  # Pitch rate (M_y)
        r = angular_velocities[:, 1]  # Yaw rate (M_z)

        # Compute alpha rate (dalpha/dt) for C_m_alphadot
        if len(aoas) > 1:
            dt = times[1] - times[0] if len(times) > 1 else 0.01
            alpha_rate = np.diff(aoas) / dt
            alpha_rate = np.concatenate([[alpha_rate[0]], alpha_rate])  # Pad first value
        else:
            alpha_rate = np.zeros_like(aoas)

        # Normalize angular rates (non-dimensionalize)
        q_normalized = q * ref_length / (2 * speeds + 1e-3)
        r_normalized = r * ref_length / (2 * speeds + 1e-3)
        p_normalized = p * ref_length / (2 * speeds + 1e-3)
        alpha_rate_normalized = alpha_rate * ref_length / (2 * speeds + 1e-3)

        # Compute derivatives using finite differences
        # C_L_alpha: dC_L/dalpha
        valid_indices = np.where((np.abs(aoas) > 0.01) & (qs_safe > 1e-3))[0]
        if len(valid_indices) > 10:
            # Use linear regression for better estimate
            aoas_valid = aoas[valid_indices]
            C_L_valid = C_L[valid_indices]
            # Filter out outliers
            mask = np.abs(aoas_valid) < np.radians(30)  # Reasonable AOA range
            if np.sum(mask) > 5:
                C_L_alpha = np.polyfit(aoas_valid[mask], C_L_valid[mask], 1)[0]
            else:
                C_L_alpha = 2.0  # Typical value
        else:
            C_L_alpha = 2.0  # Default

        # C_m_alpha: pitching moment coefficient (static stability)
        # Negative C_m_alpha = stable (nose-down moment with positive AOA)
        if len(valid_indices) > 10:
            aoas_valid = aoas[valid_indices]
            C_m_valid = C_m_mag[valid_indices]
            mask = np.abs(aoas_valid) < np.radians(30)
            if np.sum(mask) > 5:
                C_m_alpha = np.polyfit(aoas_valid[mask], C_m_valid[mask], 1)[0]
            else:
                C_m_alpha = -0.5  # Stable default
        else:
            C_m_alpha = -0.5

        # C_m_q: pitch damping (dC_m/dq) - pitch moment vs pitch rate
        valid_q = np.where(
            (np.abs(q_normalized) > 1e-4) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_q) > 10:
            C_m_q = np.polyfit(q_normalized[valid_q], C_m_pitch[valid_q], 1)[0]
        else:
            # Fallback: estimate from geometry if available
            C_m_q = -10.0  # Typical damping

        # C_D_alpha: Drag due to AOA (dC_D/dalpha)
        valid_aoa = np.where(
            (np.abs(aoas) > 0.01) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(30))
        )[0]
        if len(valid_aoa) > 10:
            aoas_valid = aoas[valid_aoa]
            C_D_valid = C_D[valid_aoa]
            # Filter outliers
            mask = np.abs(aoas_valid) < np.radians(20)
            if np.sum(mask) > 5:
                C_D_alpha = np.polyfit(aoas_valid[mask], C_D_valid[mask], 1)[0]
            else:
                C_D_alpha = 0.1  # Small positive value typical
        else:
            C_D_alpha = 0.1

        # C_m_alphadot: Alpha rate damping (dC_m/dalphadot)
        valid_alphadot = np.where(
            (np.abs(alpha_rate_normalized) > 1e-4)
            & (qs_safe > 1e-3)
            & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_alphadot) > 10:
            C_m_alphadot = np.polyfit(
                alpha_rate_normalized[valid_alphadot], C_m_pitch[valid_alphadot], 1
            )[0]
        else:
            C_m_alphadot = -1.0  # Typical damping value

        # C_L_q: Lift due to pitch rate (dC_L/dq)
        valid_lift_q = np.where(
            (np.abs(q_normalized) > 1e-4) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_lift_q) > 10:
            C_L_q = np.polyfit(q_normalized[valid_lift_q], C_L[valid_lift_q], 1)[0]
        else:
            C_L_q = 1.0  # Typical positive value

        # C_D_q: Drag due to pitch rate (dC_D/dq)
        if len(valid_lift_q) > 10:
            C_D_q = np.polyfit(q_normalized[valid_lift_q], C_D[valid_lift_q], 1)[0]
        else:
            C_D_q = 0.0  # Usually small

        # Lateral-directional derivatives from sideslip and moments
        # C_Y_beta: Side force due to sideslip (dC_Y/dbeta)
        # Side force is the y-component of lift force in body frame
        lift_forces_body = np.zeros_like(lift_forces)
        for i, snapshot in enumerate(self.history):
            quaternion = snapshot.quaternion
            Kt = Matrix.transformation(quaternion).transpose
            lift_world_vec = Vector(lift_forces[i])
            lift_body_vec = Kt @ lift_world_vec
            lift_forces_body[i] = np.array([lift_body_vec.x, lift_body_vec.y, lift_body_vec.z])

        # Side force coefficient (C_Y) - lateral force in body frame
        side_force = lift_forces_body[:, 0]  # X-component (lateral)
        C_Y = 2 * side_force / (qs_safe * ref_area)

        valid_beta = np.where(
            (np.abs(sideslips) > 0.01) & (qs_safe > 1e-3) & (np.abs(sideslips) < np.radians(30))
        )[0]
        if len(valid_beta) > 10:
            sideslips_valid = sideslips[valid_beta]
            C_Y_valid = C_Y[valid_beta]
            mask = np.abs(sideslips_valid) < np.radians(20)
            if np.sum(mask) > 5:
                C_Y_beta = np.polyfit(sideslips_valid[mask], C_Y_valid[mask], 1)[0]
            else:
                C_Y_beta = -0.1  # Typical negative (side force opposes sideslip)
        else:
            C_Y_beta = -0.1

        # C_l_beta: Roll moment due to sideslip (dihedral effect)
        valid_roll_beta = np.where(
            (np.abs(sideslips) > 0.01) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_roll_beta) > 10:
            sideslips_valid = sideslips[valid_roll_beta]
            C_m_roll_valid = C_m_roll[valid_roll_beta]
            mask = np.abs(sideslips_valid) < np.radians(20)
            if np.sum(mask) > 5:
                C_l_beta = np.polyfit(sideslips_valid[mask], C_m_roll_valid[mask], 1)[0]
            else:
                C_l_beta = -0.05  # Typical negative (dihedral effect)
        else:
            C_l_beta = -0.05

        # C_l_p: Roll damping (dC_l/dp)
        valid_roll_p = np.where(
            (np.abs(p_normalized) > 1e-4) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_roll_p) > 10:
            C_l_p = np.polyfit(p_normalized[valid_roll_p], C_m_roll[valid_roll_p], 1)[0]
        else:
            C_l_p = -0.5  # Typical damping

        # C_n_beta: Yaw moment due to sideslip (weathercock stability)
        valid_yaw_beta = np.where(
            (np.abs(sideslips) > 0.01) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_yaw_beta) > 10:
            sideslips_valid = sideslips[valid_yaw_beta]
            C_m_yaw_valid = C_m_yaw[valid_yaw_beta]
            mask = np.abs(sideslips_valid) < np.radians(20)
            if np.sum(mask) > 5:
                C_n_beta = np.polyfit(sideslips_valid[mask], C_m_yaw_valid[mask], 1)[0]
            else:
                C_n_beta = 0.1  # Typical positive (weathercock stability)
        else:
            C_n_beta = 0.1

        # C_n_r: Yaw damping (dC_n/dr)
        valid_yaw_r = np.where(
            (np.abs(r_normalized) > 1e-4) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_yaw_r) > 10:
            C_n_r = np.polyfit(r_normalized[valid_yaw_r], C_m_yaw[valid_yaw_r], 1)[0]
        else:
            C_n_r = -0.5  # Typical damping

        # C_n_p: Cross-coupling (yaw moment due to roll rate)
        valid_cross = np.where(
            (np.abs(p_normalized) > 1e-4) & (qs_safe > 1e-3) & (np.abs(aoas) < np.radians(10))
        )[0]
        if len(valid_cross) > 10:
            C_n_p = np.polyfit(p_normalized[valid_cross], C_m_yaw[valid_cross], 1)[0]
        else:
            C_n_p = 0.0  # Usually small for axisymmetric rockets

        # Control derivatives (not available from flight data without control surfaces)
        C_m_delta = 0.0  # Would need control surface deflection data
        C_l_delta = 0.0
        C_n_delta = 0.0

        return StabilityDerivatives(
            C_L_alpha=C_L_alpha,
            C_D_alpha=C_D_alpha,
            C_m_alpha=C_m_alpha,
            C_m_q=C_m_q,
            C_m_alphadot=C_m_alphadot,
            C_Y_beta=C_Y_beta,
            C_l_beta=C_l_beta,
            C_l_p=C_l_p,
            C_n_beta=C_n_beta,
            C_n_r=C_n_r,
            C_n_p=C_n_p,
            C_m_delta=C_m_delta,
            C_l_delta=C_l_delta,
            C_n_delta=C_n_delta,
            C_L_q=C_L_q,
            C_D_q=C_D_q,
        )

    def compute_first_order_terms(self) -> Dict[str, np.ndarray]:
        """Compute first-order flight dynamics terms."""
        times = np.array([s.time for s in self.history])
        velocities = np.array([s.velocity for s in self.history])
        angular_velocities = np.array([s.angular_velocity for s in self.history])
        aoas = np.array([s.angle_of_attack for s in self.history])
        sideslips = np.array([s.sideslip for s in self.history])

        # First derivatives (rates)
        if len(times) > 1:
            dt = times[1] - times[0]
            velocity_rate = np.diff(velocities, axis=0) / dt
            angular_rate_rate = np.diff(angular_velocities, axis=0) / dt
            aoa_rate = np.diff(aoas) / dt
            sideslip_rate = np.diff(sideslips) / dt

            # Pad to match original length
            velocity_rate = np.vstack([velocity_rate[0:1], velocity_rate])
            angular_rate_rate = np.vstack([angular_rate_rate[0:1], angular_rate_rate])
            aoa_rate = np.concatenate([[aoa_rate[0]], aoa_rate])
            sideslip_rate = np.concatenate([[sideslip_rate[0]], sideslip_rate])
        else:
            velocity_rate = np.zeros_like(velocities)
            angular_rate_rate = np.zeros_like(angular_velocities)
            aoa_rate = np.zeros_like(aoas)
            sideslip_rate = np.zeros_like(sideslips)

        return {
            "velocity_rate": velocity_rate,
            "angular_rate_rate": angular_rate_rate,
            "aoa_rate": aoa_rate,
            "sideslip_rate": sideslip_rate,
            "acceleration": velocity_rate,
            "angular_acceleration": angular_rate_rate,
        }

    def compute_second_order_terms(self) -> Dict[str, np.ndarray]:
        """Compute second-order flight dynamics terms."""
        first_order = self.compute_first_order_terms()
        times = np.array([s.time for s in self.history])

        if len(times) > 2:
            dt = times[1] - times[0]
            # Second derivatives (jerk, angular jerk)
            jerk = np.diff(first_order["acceleration"], axis=0) / dt
            angular_jerk = np.diff(first_order["angular_acceleration"], axis=0) / dt
            aoa_accel = np.diff(first_order["aoa_rate"]) / dt
            sideslip_accel = np.diff(first_order["sideslip_rate"]) / dt

            # Pad
            jerk = np.vstack([jerk[0:1], jerk])
            angular_jerk = np.vstack([angular_jerk[0:1], angular_jerk])
            aoa_accel = np.concatenate([[aoa_accel[0]], aoa_accel])
            sideslip_accel = np.concatenate([[sideslip_accel[0]], sideslip_accel])
        else:
            jerk = np.zeros((len(times), 3))
            angular_jerk = np.zeros((len(times), 3))
            aoa_accel = np.zeros_like(times)
            sideslip_accel = np.zeros_like(times)

        return {
            "jerk": jerk,
            "angular_jerk": angular_jerk,
            "aoa_acceleration": aoa_accel,
            "sideslip_acceleration": sideslip_accel,
        }


def compute_flight_phases(
    result: FlightResult, motor_burn_time: float
) -> Dict[str, Tuple[int, int]]:
    """Identify flight phase boundaries."""
    times = np.array([s.time for s in result.history])
    altitudes = np.array([s.position[2] for s in result.history])

    # Boost phase: t=0 to motor burnout
    boost_end = np.searchsorted(times, motor_burn_time)

    # Coast phase: motor burnout to apogee
    apogee_idx = np.argmax(altitudes)
    coast_end = apogee_idx

    # Descent phase: apogee to end
    descent_start = apogee_idx

    return {
        "boost": (0, boost_end),
        "coast": (boost_end, coast_end),
        "descent": (descent_start, len(times)),
    }
