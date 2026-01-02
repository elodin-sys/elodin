"""
Crazyflie 2.1 Configuration

Physical parameters for the Bitcraze Crazyflie 2.1 quadcopter.
These values are used to create an accurate physics simulation.

References:
- Bitcraze Crazyflie 2.1 product page: https://store.bitcraze.io/products/crazyflie-2-1
- Crazyflie firmware source: https://github.com/bitcraze/crazyflie-firmware
"""

import typing as ty
from dataclasses import dataclass
from enum import Enum

import elodin as el
import numpy as np
from numpy._typing import NDArray


class Frame(Enum):
    """
    Crazyflie motor configuration (Quad-X layout).

    Looking down at the Crazyflie from above:

         FRONT
      M1(CW)  M2(CCW)
          \\  /
           \\/
           /\\
          /  \\
      M4(CCW) M3(CW)
         BACK

    Motor numbering follows Crazyflie firmware convention.
    """

    CRAZYFLIE_X = 0

    @property
    def motor_matrix(self) -> NDArray[np.float64]:
        """
        Returns motor mixing matrix for roll, pitch, yaw, throttle.
        Each row is [roll_factor, pitch_factor, yaw_factor, throttle_factor] for one motor.
        """
        if self == Frame.CRAZYFLIE_X:
            # Motor angles from front (45, -45, -135, 135 degrees)
            motor_angles = np.pi * np.array([0.25, -0.25, -0.75, 0.75])
            # Yaw direction: CW motors produce CCW torque reaction and vice versa
            # M1: CW, M2: CCW, M3: CW, M4: CCW
            yaw_factor = np.array([-1.0, 1.0, -1.0, 1.0])
            throttle_factor = np.ones(4)
        else:
            raise ValueError(f"Unsupported frame: {self}")

        pitch_factor = -np.sin(motor_angles)
        roll_factor = np.cos(motor_angles)

        # Normalize to [-0.5, 0.5] range
        roll_factor /= 2 * np.max(np.abs(roll_factor))
        pitch_factor /= 2 * np.max(np.abs(pitch_factor))
        yaw_factor /= 2 * np.max(np.abs(yaw_factor))

        return np.array([roll_factor, pitch_factor, yaw_factor, throttle_factor])


@dataclass
class CrazyflieConfig:
    """
    Configuration for Crazyflie 2.1 simulation.

    All values are based on the stock Crazyflie 2.1 with:
    - Stock 7x16mm coreless motors
    - Stock 45mm propellers
    - No additional decks/payload
    """

    _GLOBAL: ty.ClassVar[ty.Optional["CrazyflieConfig"]] = None

    # =========================================================================
    # Physical Properties
    # =========================================================================

    # Mass in kg (27 grams)
    mass: float = 0.027

    # Moment of inertia diagonal [Ixx, Iyy, Izz] in kg*m^2
    # Values from Crazyflie firmware / system identification
    inertia_diagonal: NDArray[np.float64] = None

    # Motor arm length (center to motor) in meters
    # Crazyflie is 92mm diagonal, so arm = 92/2 * sqrt(2)/2 ≈ 32.5mm
    arm_length: float = 0.0325

    # =========================================================================
    # Motor Properties
    # =========================================================================

    # Motor positions relative to center of mass (x, y, z) in body frame
    # Body frame: +X forward, +Y left, +Z up
    motor_positions: NDArray[np.float64] = None

    # Motor thrust directions (unit vectors pointing thrust direction)
    # All motors thrust upward in body frame
    motor_thrust_directions: NDArray[np.float64] = None

    # Motor time constant (first-order response) in seconds
    motor_time_constant: float = 0.02

    # PWM range (0-255 for Crazyflie)
    pwm_min: int = 0
    pwm_max: int = 255

    # =========================================================================
    # Powertrain Characteristics
    # =========================================================================

    # PWM to RPM mapping (affine fit: rpm = pwm_to_rpm_a + pwm_to_rpm_b * pwm)
    # These are approximate values - students will identify exact values in Lab 2
    pwm_to_rpm_a: float = 4070.0  # RPM at PWM=0 (motor deadband)
    pwm_to_rpm_b: float = 65.0  # RPM per PWM unit

    # Thrust constant: thrust = k * omega^2 (N/(rad/s)^2)
    # Students will identify this in Lab 2
    # Approximate value for 45mm props: ~1.8e-8
    thrust_constant: float = 1.8e-8

    # Torque constant: torque = kt * omega^2 (Nm/(rad/s)^2)
    # Typically ~1% of thrust constant
    torque_constant: float = 1.8e-10

    # =========================================================================
    # Simulation Settings
    # =========================================================================

    # Frame type
    frame: Frame = Frame.CRAZYFLIE_X

    # Simulation time step in seconds (500 Hz control loop)
    sim_time_step: float = 0.002

    # Fast loop time step for physics (1000 Hz)
    fast_loop_time_step: float = 0.001

    # Total simulation time in seconds
    simulation_time: float = 60.0

    # Initial position [x, y, z] in meters (ENU world frame)
    start_pos: NDArray[np.float64] = None

    # Initial attitude as euler angles [roll, pitch, yaw] in degrees
    start_euler_angles: NDArray[np.float64] = None

    # Enable sensor noise
    sensor_noise: bool = True

    # =========================================================================
    # Drag coefficients
    # =========================================================================

    # Linear drag coefficient (N/(m/s))
    drag_coefficient: float = 0.1

    def __post_init__(self):
        """Initialize default numpy arrays."""
        if self.inertia_diagonal is None:
            # Approximate values for Crazyflie 2.1 (kg*m^2)
            self.inertia_diagonal = np.array([1.4e-5, 1.4e-5, 2.2e-5])

        if self.motor_positions is None:
            # Motor positions in body frame (X forward, Y left, Z up)
            # Propeller plane is ~12mm above center of mass
            arm = self.arm_length
            prop_height = 0.012  # 12mm above CG
            self.motor_positions = np.array(
                [
                    [arm, -arm, prop_height],  # M1: front-right (CW)
                    [arm, arm, prop_height],  # M2: front-left (CCW)
                    [-arm, arm, prop_height],  # M3: back-left (CW)
                    [-arm, -arm, prop_height],  # M4: back-right (CCW)
                ]
            )

        if self.motor_thrust_directions is None:
            # All motors thrust upward in body frame
            self.motor_thrust_directions = np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        if self.start_pos is None:
            self.start_pos = np.array([0.0, 0.0, 0.1])  # 10cm above ground

        if self.start_euler_angles is None:
            self.start_euler_angles = np.array([0.0, 0.0, 0.0])

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def dt(self) -> float:
        """Main simulation time step."""
        return self.sim_time_step

    @property
    def total_sim_ticks(self) -> int:
        """Total number of simulation ticks."""
        return int(self.simulation_time / self.dt)

    @property
    def weight(self) -> float:
        """Weight in Newtons."""
        return self.mass * 9.81

    @property
    def hover_thrust_per_motor(self) -> float:
        """Thrust per motor required for hover (N)."""
        return self.weight / 4.0

    @property
    def hover_rpm(self) -> float:
        """Motor RPM required for hover."""
        omega = np.sqrt(self.hover_thrust_per_motor / self.thrust_constant)
        return omega * 60.0 / (2.0 * np.pi)

    @property
    def attitude(self) -> el.Quaternion:
        """Initial attitude as quaternion."""
        # Convert euler angles (degrees) to quaternion
        roll, pitch, yaw = np.radians(self.start_euler_angles)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return el.Quaternion(np.array([x, y, z, w]))

    @property
    def spatial_transform(self) -> el.SpatialTransform:
        """Initial spatial transform for the drone."""
        return el.SpatialTransform(
            linear=self.start_pos,
            angular=self.attitude,
        )

    @property
    def spatial_inertia(self) -> el.SpatialInertia:
        """Spatial inertia for physics simulation."""
        return el.SpatialInertia(
            mass=self.mass,
            inertia=self.inertia_diagonal,
        )

    @property
    def motor_torque_axes(self) -> NDArray[np.float64]:
        """Torque axes for each motor (cross product of position and thrust direction)."""
        return np.cross(self.motor_positions, self.motor_thrust_directions)

    @classmethod
    @property
    def GLOBAL(cls) -> "CrazyflieConfig":
        """Get the global configuration instance."""
        if cls._GLOBAL is None:
            raise ValueError("No global config set. Call config.set_as_global() first.")
        return cls._GLOBAL

    def set_as_global(self):
        """Set this configuration as the global instance."""
        CrazyflieConfig._GLOBAL = self


def create_default_config() -> CrazyflieConfig:
    """Create a default Crazyflie 2.1 configuration."""
    config = CrazyflieConfig()
    config.set_as_global()
    return config


# Module-level convenience
Config = CrazyflieConfig


if __name__ == "__main__":
    # Print configuration summary
    config = create_default_config()
    print("Crazyflie 2.1 Configuration")
    print("=" * 40)
    print(f"Mass: {config.mass * 1000:.1f} g")
    print(f"Weight: {config.weight:.4f} N")
    print(f"Arm length: {config.arm_length * 1000:.1f} mm")
    print(f"Inertia: {config.inertia_diagonal}")
    print()
    print("Hover Requirements:")
    print(f"  Thrust per motor: {config.hover_thrust_per_motor:.4f} N")
    print(f"  RPM per motor: {config.hover_rpm:.0f}")
    print()
    print("Simulation Settings:")
    print(f"  Time step: {config.sim_time_step * 1000:.1f} ms")
    print(f"  Total time: {config.simulation_time:.1f} s")
    print(f"  Sensor noise: {config.sensor_noise}")
