"""
BDX RC Jet Configuration Module

Contains aircraft parameters, aerodynamic coefficients, and simulation settings
following the BDX_Simulation_Whitepaper.md design.
"""

import typing as ty
from dataclasses import dataclass

import elodin as el
import numpy as np
from numpy._typing import NDArray


@dataclass
class AeroCoefficients:
    """Longitudinal and lateral-directional stability derivatives."""

    # Longitudinal coefficients (Section 3.3 of whitepaper)
    # NOTE: Trimmed for level flight at 70 m/s
    # Required CL = W/(q*S) = 186/(3001*0.75) = 0.083
    C_L0: float = 0.083  # Exactly trimmed for level flight
    C_Lalpha: float = 0.5  # Very gentle lift slope for stability
    C_Lq: float = 1.0  # Low pitch-lift coupling
    C_Lde: float = 0.1  # Low elevator-lift coupling

    C_D0: float = 0.025  # Parasite drag
    C_Dde: float = 0.02  # Control surface drag (/rad)
    k: float = 0.045  # Induced drag factor

    C_m0: float = 0.0  # Trim pitching moment (assuming trimmed)
    C_malpha: float = -0.5  # Moderate static stability
    C_mq: float = -20.0  # Very strong pitch damping - prevents oscillation
    C_mde: float = +0.5  # Moderate elevator control

    # Lateral-directional coefficients
    C_Ybeta: float = -0.5  # Side force due to sideslip (/rad)
    C_Yp: float = 0.0  # Side force due to roll rate (/rad)
    C_Yr: float = 0.3  # Side force due to yaw rate (/rad)
    C_Yda: float = 0.0  # Aileron side force (/rad)
    C_Ydr: float = 0.15  # Rudder side force (/rad)

    C_lbeta: float = -0.08  # Dihedral effect (/rad)
    C_lp: float = -0.5  # Roll damping (/rad)
    C_lr: float = 0.1  # Roll due to yaw (/rad)
    C_lda: float = 0.15  # Aileron control power (/rad)
    C_ldr: float = 0.01  # Rudder-roll coupling (/rad)

    C_nbeta: float = 0.1  # Weathercock stability (/rad)
    C_np: float = -0.03  # Adverse yaw (/rad)
    C_nr: float = -0.15  # Yaw damping (/rad)
    C_nda: float = -0.01  # Aileron adverse yaw (/rad)
    C_ndr: float = -0.1  # Rudder control power (/rad)

    # Stall parameters
    alpha_stall: float = 15.0  # Stall angle (degrees)


@dataclass
class PropulsionParams:
    """Turbine engine parameters."""

    max_thrust: float = 200.0  # Maximum thrust (N) - P200 class turbine
    spool_tau: float = 0.3  # Spool time constant (s) - faster for RC application
    thrust_a1: float = 0.2  # Linear thrust coefficient
    thrust_a2: float = 0.8  # Quadratic thrust coefficient
    idle_spool: float = 0.2  # Idle spool speed (normalized)


@dataclass
class ActuatorParams:
    """Control surface servo parameters."""

    servo_tau: float = 0.05  # Time constant (s)
    max_deflection_deg: float = 25.0  # Max deflection (degrees)
    max_rudder_deflection_deg: float = 30.0  # Rudder max deflection (degrees)
    max_rate_deg_s: float = 400.0  # Max deflection rate (deg/s)
    max_rudder_rate_deg_s: float = 350.0  # Rudder max rate (deg/s)


@dataclass
class BDXConfig:
    """Complete BDX aircraft configuration."""

    _GLOBAL: ty.ClassVar[ty.Optional["BDXConfig"]] = None

    # Geometry (Section 1.3 of whitepaper)
    wingspan: float = 2.65  # m
    wing_area: float = 0.75  # m²
    mean_chord: float = 0.30  # m
    aspect_ratio: float = 9.36  # b²/S
    oswald_efficiency: float = 0.8  # Oswald efficiency factor

    # Mass properties (Section 7.1)
    mass: float = 19.0  # kg
    Ixx: float = 0.8  # kg·m² (roll inertia)
    Iyy: float = 2.5  # kg·m² (pitch inertia)
    Izz: float = 3.0  # kg·m² (yaw inertia)
    Ixz: float = 0.1  # kg·m² (cross-product of inertia)

    # Aerodynamic coefficients
    aero: AeroCoefficients = None

    # Propulsion
    propulsion: PropulsionParams = None

    # Actuators
    actuators: ActuatorParams = None

    # Initial conditions - start in fast cruise (easier to stabilize)
    initial_speed: float = 70.0  # m/s (fast cruise - more stable)
    initial_altitude: float = 50.0  # m (higher altitude)
    initial_pitch_deg: float = 0.0  # degrees (level on runway)
    initial_roll_deg: float = 0.0  # degrees
    initial_yaw_deg: float = 0.0  # degrees (aligned with runway)

    # Simulation parameters
    dt: float = 1.0 / 120.0  # 120 Hz simulation rate
    simulation_time: float = 180.0  # Total simulation time (s) - 3 minutes for full pattern

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.aero is None:
            self.aero = AeroCoefficients()
        if self.propulsion is None:
            self.propulsion = PropulsionParams()
        if self.actuators is None:
            self.actuators = ActuatorParams()

    @property
    def total_ticks(self) -> int:
        """Total number of simulation ticks."""
        return int(self.simulation_time / self.dt)

    @property
    def spatial_inertia(self) -> el.SpatialInertia:
        """Convert to Elodin spatial inertia."""
        return el.SpatialInertia(
            mass=self.mass,
            inertia=np.array([self.Ixx, self.Iyy, self.Izz]),
        )

    @property
    def initial_attitude(self) -> el.Quaternion:
        """Convert initial Euler angles to quaternion."""
        roll = np.deg2rad(self.initial_roll_deg)
        pitch = np.deg2rad(self.initial_pitch_deg)
        yaw = np.deg2rad(self.initial_yaw_deg)

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return el.Quaternion(np.array([x, y, z, w]))

    @property
    def initial_velocity_body(self) -> NDArray[np.float64]:
        """Initial velocity in body frame (forward flight)."""
        # Start with forward velocity along body X-axis
        return np.array([self.initial_speed, 0.0, 0.0])

    @classmethod
    @property
    def GLOBAL(cls) -> "BDXConfig":
        """Get global configuration instance."""
        if cls._GLOBAL is None:
            raise ValueError("No global BDXConfig set. Call set_as_global() first.")
        return cls._GLOBAL

    def set_as_global(self):
        """Set this instance as the global configuration."""
        BDXConfig._GLOBAL = self


# Default configuration instance
DEFAULT_CONFIG = BDXConfig()
