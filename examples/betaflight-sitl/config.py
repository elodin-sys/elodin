"""
Drone Configuration for Betaflight SITL Simulation

This module defines the physical parameters of the simulated quadcopter,
matching a typical Betaflight Quad-X configuration.

Frame Layout (Betaflight Quad-X, looking from above):
    Motor 1 (FR, CCW)    Motor 2 (BR, CW)
            \\          /
             \\  FRONT /
              \\      /
               X-----> Y
              /      \\
             /  BACK  \\
            /          \\
    Motor 4 (FL, CW)    Motor 3 (BL, CCW)

Elodin Coordinate System (ENU):
    X = Forward (East)
    Y = Left (North)  
    Z = Up

Motor Order (Betaflight):
    0: Front Right (CCW)
    1: Back Right (CW)
    2: Back Left (CCW)
    3: Front Left (CW)
"""

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Self
import numpy as np
from numpy.typing import NDArray


@dataclass
class MotorConfig:
    """Configuration for a single motor."""
    
    # Motor position relative to CoM [x, y, z] in meters
    position: NDArray[np.float64]
    
    # Thrust direction (unit vector, typically [0, 0, 1] for up)
    thrust_direction: NDArray[np.float64]
    
    # Spin direction: 1 for CCW (positive torque), -1 for CW (negative torque)
    spin_direction: float
    
    # Maximum thrust in Newtons
    max_thrust: float = 10.0
    
    # Motor time constant (how fast motor responds) in seconds
    time_constant: float = 0.02
    
    # Torque coefficient (torque = k * thrust)
    torque_coefficient: float = 0.01


@dataclass
class DroneConfig:
    """
    Complete drone configuration for Betaflight SITL simulation.
    
    Physical parameters are chosen to match a typical 5" racing quadcopter.
    """
    
    # Singleton instance
    _GLOBAL: ClassVar[Optional[Self]] = None
    
    # --- Physical Properties ---
    
    # Total mass in kg (typical 5" quad with battery)
    mass: float = 0.8
    
    # Moment of inertia diagonal [Ixx, Iyy, Izz] in kg*m^2
    # Typical values for 5" quad
    inertia_diagonal: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0025, 0.0025, 0.004])
    )
    
    # Arm length from center to motor in meters (half of motor-to-motor distance)
    arm_length: float = 0.12
    
    # --- Motor Configuration ---
    
    # Maximum thrust per motor in Newtons
    # A typical 2306 motor with 5" prop: ~1.5kg thrust = ~14.7N
    motor_max_thrust: float = 15.0
    
    # Motor time constant in seconds (response time)
    motor_time_constant: float = 0.02
    
    # Torque coefficient: reaction_torque = k * thrust
    # Determines yaw authority
    motor_torque_coeff: float = 0.012
    
    # --- Drag Properties ---
    
    # Linear drag coefficient [drag_x, drag_y, drag_z] in N/(m/s)
    linear_drag: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.3])
    )
    
    # Rotational drag coefficient [drag_roll, drag_pitch, drag_yaw] in N*m/(rad/s)
    angular_drag: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.015])
    )
    
    # --- Initial State ---
    
    # Initial position [x, y, z] in meters (ENU)
    initial_position: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.1])
    )
    
    # Initial velocity [vx, vy, vz] in m/s
    initial_velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    
    # Initial attitude as quaternion [x, y, z, w] (Elodin internal format, scalar last)
    # Identity quaternion: w=1, x=y=z=0 → [0, 0, 0, 1]
    initial_quaternion: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )
    
    # Initial angular velocity [wx, wy, wz] in rad/s
    initial_angular_velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    
    # --- Simulation Settings ---
    
    # Physics time step in seconds (1kHz default)
    sim_time_step: float = 0.001
    
    # Total simulation time in seconds
    simulation_time: float = 30.0
    
    # --- Environment ---
    
    # Gravity acceleration in m/s^2 (positive down in NED, but we use ENU so positive up)
    gravity: float = 9.81
    
    # Air density in kg/m^3 (sea level)
    air_density: float = 1.225
    
    # Ground level in meters
    ground_level: float = 0.0
    
    # --- Computed Properties ---
    
    @property
    def dt(self) -> float:
        """Physics time step."""
        return self.sim_time_step
    
    @property
    def total_sim_ticks(self) -> int:
        """Total number of simulation ticks."""
        return int(self.simulation_time / self.dt)
    
    @property
    def motor_positions(self) -> NDArray[np.float64]:
        """
        Get motor positions for Betaflight Quad-X layout.
        
        Looking from above:
            M1(FR)  M2(BR)
            M4(FL)  M3(BL)
        
        Returns:
            Array of shape (4, 3) with motor positions [x, y, z]
        """
        # 45 degree arm angles for X configuration
        d = self.arm_length * np.sqrt(2) / 2  # distance in each axis
        
        # Motor positions: [Forward, Left, Up] in ENU
        # Motor 0: Front Right (+X, -Y)
        # Motor 1: Back Right (-X, -Y)
        # Motor 2: Back Left (-X, +Y)
        # Motor 3: Front Left (+X, +Y)
        return np.array([
            [d, -d, 0.0],   # Motor 0: FR
            [-d, -d, 0.0],  # Motor 1: BR
            [-d, d, 0.0],   # Motor 2: BL
            [d, d, 0.0],    # Motor 3: FL
        ])
    
    @property
    def motor_thrust_directions(self) -> NDArray[np.float64]:
        """Thrust direction for each motor (all point up in ENU)."""
        return np.array([
            [0.0, 0.0, 1.0],  # Motor 0
            [0.0, 0.0, 1.0],  # Motor 1
            [0.0, 0.0, 1.0],  # Motor 2
            [0.0, 0.0, 1.0],  # Motor 3
        ])
    
    @property
    def motor_spin_directions(self) -> NDArray[np.float64]:
        """
        Spin direction for each motor.
        
        Betaflight Quad-X "props out":
            M1(FR): CCW = +1
            M2(BR): CW  = -1
            M3(BL): CCW = +1
            M4(FL): CW  = -1
        """
        return np.array([1.0, -1.0, 1.0, -1.0])
    
    @property
    def motor_torque_axes(self) -> NDArray[np.float64]:
        """
        Compute torque axes from motor positions and thrust directions.
        
        Torque from thrust = position × thrust_direction (cross product)
        Plus yaw torque from motor spin.
        """
        return np.cross(self.motor_positions, self.motor_thrust_directions)
    
    @property
    def hover_throttle(self) -> float:
        """
        Approximate throttle needed to hover.
        
        hover_thrust = mass * gravity
        throttle = hover_thrust / (4 * max_thrust)
        """
        hover_thrust = self.mass * self.gravity
        return hover_thrust / (4 * self.motor_max_thrust)
    
    def get_motor_config(self, index: int) -> MotorConfig:
        """Get configuration for a specific motor."""
        return MotorConfig(
            position=self.motor_positions[index],
            thrust_direction=self.motor_thrust_directions[index],
            spin_direction=self.motor_spin_directions[index],
            max_thrust=self.motor_max_thrust,
            time_constant=self.motor_time_constant,
            torque_coefficient=self.motor_torque_coeff,
        )
    
    @classmethod
    @property
    def GLOBAL(cls) -> Self:
        """Get the global configuration instance."""
        if cls._GLOBAL is None:
            raise ValueError("No global config set. Call set_as_global() first.")
        return cls._GLOBAL
    
    def set_as_global(self) -> None:
        """Set this configuration as the global instance."""
        DroneConfig._GLOBAL = self


# Pre-configured drone types
def create_5inch_racing_quad() -> DroneConfig:
    """Create configuration for a typical 5" racing quadcopter."""
    return DroneConfig(
        mass=0.65,
        inertia_diagonal=np.array([0.0020, 0.0020, 0.0035]),
        arm_length=0.11,
        motor_max_thrust=14.0,
        motor_time_constant=0.015,
        motor_torque_coeff=0.010,
        linear_drag=np.array([0.15, 0.15, 0.25]),
        angular_drag=np.array([0.008, 0.008, 0.012]),
    )


def create_3inch_cinewhoop() -> DroneConfig:
    """Create configuration for a 3" cinewhoop style quad."""
    return DroneConfig(
        mass=0.35,
        inertia_diagonal=np.array([0.0008, 0.0008, 0.0015]),
        arm_length=0.08,
        motor_max_thrust=6.0,
        motor_time_constant=0.025,
        motor_torque_coeff=0.008,
        linear_drag=np.array([0.3, 0.3, 0.4]),  # Higher drag from ducts
        angular_drag=np.array([0.015, 0.015, 0.020]),
    )


def create_7inch_long_range() -> DroneConfig:
    """Create configuration for a 7" long range quadcopter."""
    return DroneConfig(
        mass=1.2,
        inertia_diagonal=np.array([0.0045, 0.0045, 0.008]),
        arm_length=0.16,
        motor_max_thrust=18.0,
        motor_time_constant=0.030,
        motor_torque_coeff=0.015,
        linear_drag=np.array([0.2, 0.2, 0.3]),
        angular_drag=np.array([0.012, 0.012, 0.018]),
    )


# Default configuration
DEFAULT_CONFIG = DroneConfig()


if __name__ == "__main__":
    # Print configuration summary
    config = DEFAULT_CONFIG
    
    print("Betaflight SITL Drone Configuration")
    print("=" * 50)
    print(f"Mass:          {config.mass:.3f} kg")
    print(f"Arm length:    {config.arm_length:.3f} m")
    print(f"Inertia:       {config.inertia_diagonal}")
    print(f"Max thrust:    {config.motor_max_thrust:.1f} N per motor")
    print(f"Hover throttle: {config.hover_throttle:.1%}")
    print()
    print("Motor Positions (ENU):")
    for i, pos in enumerate(config.motor_positions):
        spin = "CCW" if config.motor_spin_directions[i] > 0 else "CW"
        print(f"  Motor {i}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] ({spin})")
    print()
    print("Simulation Settings:")
    print(f"  Time step:    {config.sim_time_step*1000:.1f} ms ({1/config.sim_time_step:.0f} Hz)")
    print(f"  Duration:     {config.simulation_time:.1f} s")
    print(f"  Total ticks:  {config.total_sim_ticks}")
