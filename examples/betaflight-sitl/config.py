"""
Drone Configuration for Betaflight SITL Simulation

This module defines the physical parameters of the simulated quadcopter,
matching a typical Betaflight Quad-X configuration.

Frame Layout (Betaflight Quad-X "props out", looking from above):

         FRONT
    4 (FL, CW)    2 (FR, CCW)
            \\    /
             \\  /
              \\/
              /\\
             /  \\
            /    \\
    3 (BL, CCW)    1 (BR, CW)
         BACK

Elodin Coordinate System (ENU/FLU body frame):
    X = Forward
    Y = Left
    Z = Up

IMPORTANT: Betaflight 2026.6.1+ SITL sends motor outputs in native Betaflight
order (sitl.c pwmCompleteMotorUpdate() copies motorsPwm[] directly; the old
Gazebo ArduCopterPlugin remapping was removed):

Motor Index Mapping (what we RECEIVE from SITL):
    motor[0] = BF motor 0 = Back Right (BR, CW, spin -1)
    motor[1] = BF motor 1 = Front Right (FR, CCW, spin +1)
    motor[2] = BF motor 2 = Back Left (BL, CCW, spin +1)
    motor[3] = BF motor 3 = Front Left (FL, CW, spin -1)
"""

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Self
import numpy as np
from numpy.typing import NDArray


class _classproperty:
    """Descriptor that works like @property but on the class itself (Python 3.13+)."""

    def __init__(self, func):
        self.fget = func

    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.fget(cls)


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
    linear_drag: NDArray[np.float64] = field(default_factory=lambda: np.array([0.2, 0.2, 0.3]))

    # Rotational drag coefficient [drag_roll, drag_pitch, drag_yaw] in N*m/(rad/s)
    angular_drag: NDArray[np.float64] = field(default_factory=lambda: np.array([0.01, 0.01, 0.015]))

    # --- Initial State ---

    # Initial position [x, y, z] in meters (ENU)
    initial_position: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 0.1]))

    # Initial velocity [vx, vy, vz] in m/s
    initial_velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # Initial attitude as quaternion [x, y, z, w] (Elodin internal format, scalar last)
    # Identity quaternion: w=1, x=y=z=0 → [0, 0, 0, 1]
    initial_quaternion: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0])
    )

    # Initial angular velocity [wx, wy, wz] in rad/s
    initial_angular_velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    # --- Simulation Settings ---

    # Physics/PID lockstep rate in Hz. 4kHz is the default and holds real-time;
    # higher rates (e.g. 8kHz) run slightly slower than real time because each
    # tick includes a UDP round trip.
    simulation_rate: float = 4000.0  # 250µs
    # simulation_rate: float = 8000.0  # 125µs
    # simulation_rate: float = 2000.0  # 500µs
    # simulation_rate: float = 1500.0  # 667µs
    # simulation_rate: float = 1000.0  # 1000µs

    # Total simulation time in seconds
    simulation_time: float = 15.0

    # Enable sensor noise simulation (default: True for realistic behavior)
    sensor_noise: bool = True

    # --- Sensor Update Rates (Hz) ---
    # Based on Elodin Aleph flight controller hardware specifications.
    # See README.md "Sensor Simulation Rates" section for details.

    # Gyroscope target rate (capped by the physics/PID rate)
    gyro_rate: float = 8000.0  # Requested rate; capped by the physics/PID rate

    # Accelerometer target rate (capped by the physics/PID rate)
    accel_rate: float = 4800.0

    # Barometer rate (BMP581: up to 480Hz continuous mode)
    baro_rate: float = 480.0

    # Magnetometer rate (BMM350: ~200Hz)
    mag_rate: float = 200.0

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
        return 1.0 / self.simulation_rate

    @property
    def pid_rate(self) -> float:
        """PID loop rate in Hz (inverse of time step)."""
        return self.simulation_rate

    @property
    def total_sim_ticks(self) -> int:
        """Total number of simulation ticks."""
        return int(self.simulation_time / self.dt)

    @property
    def gyro_tick_interval(self) -> int:
        """Ticks between gyro updates (1 = every tick)."""
        return max(1, round(self.pid_rate / self.gyro_rate))

    @property
    def accel_tick_interval(self) -> int:
        """Ticks between accelerometer updates."""
        return max(1, round(self.pid_rate / self.accel_rate))

    @property
    def baro_tick_interval(self) -> int:
        """Ticks between barometer updates."""
        return max(1, round(self.pid_rate / self.baro_rate))

    @property
    def mag_tick_interval(self) -> int:
        """Ticks between magnetometer updates."""
        return max(1, round(self.pid_rate / self.mag_rate))

    @property
    def motor_positions(self) -> NDArray[np.float64]:
        """
        Get motor positions for Betaflight Quad-X layout.

        Betaflight Quad-X "props out" (looking from above):
                  FRONT
            4(FL)      2(FR)
                \\    /
                 \\  /
                  \\/
                  /\\
                 /  \\
                /    \\
            3(BL)      1(BR)
                  BACK

        SITL (2026.6.1+) sends motors in native Betaflight order, so we receive:
            motor[0] = BF0 = BR (Back Right)
            motor[1] = BF1 = FR (Front Right)
            motor[2] = BF2 = BL (Back Left)
            motor[3] = BF3 = FL (Front Left)

        Returns:
            Array of shape (4, 3) with motor positions [x, y, z] in body FLU
        """
        # 45 degree arm angles for X configuration
        d = self.arm_length * np.sqrt(2) / 2  # distance in each axis

        # Motor positions: [Forward, Left, Up] in body FLU frame
        # Ordered by native Betaflight motor index (BR, FR, BL, FL)
        return np.array(
            [
                [-d, -d, 0.0],  # motor[0] = Back Right (BR)
                [d, -d, 0.0],  # motor[1] = Front Right (FR)
                [-d, d, 0.0],  # motor[2] = Back Left (BL)
                [d, d, 0.0],  # motor[3] = Front Left (FL)
            ]
        )

    @property
    def motor_thrust_directions(self) -> NDArray[np.float64]:
        """Thrust direction for each motor (all point up in ENU)."""
        return np.array(
            [
                [0.0, 0.0, 1.0],  # Motor 0
                [0.0, 0.0, 1.0],  # Motor 1
                [0.0, 0.0, 1.0],  # Motor 2
                [0.0, 0.0, 1.0],  # Motor 3
            ]
        )

    @property
    def motor_spin_directions(self) -> NDArray[np.float64]:
        """
        Spin direction for each motor (Betaflight Quad-X "props out").

        Spin direction determines yaw torque reaction:
            +1 = CCW rotation (produces +Z torque when spinning)
            -1 = CW rotation (produces -Z torque when spinning)

        Native Betaflight motor order ("props out"):
            motor[0] = BR: CW  = -1
            motor[1] = FR: CCW = +1
            motor[2] = BL: CCW = +1
            motor[3] = FL: CW  = -1
        """
        return np.array([-1.0, 1.0, 1.0, -1.0])

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

    @_classproperty
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
    print(f"  Time step:    {config.dt * 1e6:.1f} µs ({config.pid_rate:.0f} Hz)")
    print(f"  Duration:     {config.simulation_time:.1f} s")
    print(f"  Total ticks:  {config.total_sim_ticks}")
    print()
    print("Sensor Update Rates (Aleph hardware):")
    print(f"  Gyroscope:     {config.gyro_rate:.0f} Hz (every {config.gyro_tick_interval} tick)")
    print(f"  Accelerometer: {config.accel_rate:.0f} Hz (every {config.accel_tick_interval} ticks)")
    print(f"  Barometer:     {config.baro_rate:.0f} Hz (every {config.baro_tick_interval} ticks)")
    print(f"  Magnetometer:  {config.mag_rate:.0f} Hz (every {config.mag_tick_interval} ticks)")
