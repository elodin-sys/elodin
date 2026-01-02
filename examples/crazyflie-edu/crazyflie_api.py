"""
Crazyflie Firmware-Like API

This module provides a Python API that mirrors the structure of the Crazyflie
firmware's user code interface. Students write their control code using this
API, and the patterns translate directly to C firmware code.

The goal is that code written here can be ported to UserCode.c with minimal
changes (mainly syntax differences between Python and C).
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp

from config import CrazyflieConfig as Config

# =============================================================================
# Control Input Components (from GUI/Radio)
# =============================================================================

# Button states (simulated via GUI, 1.0 = pressed, 0.0 = released)
ButtonBlue = ty.Annotated[
    jax.Array,
    el.Component("button_blue", el.ComponentType.F64),
]

ButtonYellow = ty.Annotated[
    jax.Array,
    el.Component("button_yellow", el.ComponentType.F64),
]

ButtonGreen = ty.Annotated[
    jax.Array,
    el.Component("button_green", el.ComponentType.F64),
]

ButtonRed = ty.Annotated[
    jax.Array,
    el.Component("button_red", el.ComponentType.F64),
]

# Arm state (1.0 = armed, 0.0 = disarmed)
IsArmed = ty.Annotated[
    jax.Array,
    el.Component("is_armed", el.ComponentType.F64),
]


# =============================================================================
# Motor Command Component
# =============================================================================

MotorCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "priority": 100,
            "element_names": "m1,m2,m3,m4",
        },
    ),
]


# =============================================================================
# Control State Archetype
# =============================================================================


@dataclass
class ControlState(el.Archetype):
    """
    Control state archetype.

    Contains all the inputs and outputs that the student's code interacts with.
    This mirrors the data structures available in the Crazyflie firmware.
    """

    # Button inputs (read-only for student code, 1.0 = pressed, 0.0 = released)
    button_blue: ButtonBlue = field(default_factory=lambda: jnp.array(0.0))
    button_yellow: ButtonYellow = field(default_factory=lambda: jnp.array(0.0))
    button_green: ButtonGreen = field(default_factory=lambda: jnp.array(0.0))
    button_red: ButtonRed = field(default_factory=lambda: jnp.array(0.0))

    # Arm state (1.0 = armed, 0.0 = disarmed)
    is_armed: IsArmed = field(default_factory=lambda: jnp.array(0.0))

    # Motor outputs (written by student code)
    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))


# =============================================================================
# State Container for User Code
# =============================================================================


@dataclass
class CrazyflieState:
    """
    Container for all state available to user code.

    This class provides a clean interface for students to read sensor data
    and write motor commands. The structure mirrors what would be available
    in the Crazyflie firmware.

    Usage in user_code.py:
        def main_loop(state: CrazyflieState) -> None:
            # Read sensors
            gyro_x = state.gyro[0]

            # Check buttons
            if state.button_blue:
                # Set motor commands
                state.set_motors(50, 50, 50, 50)

    Attributes:
        gyro: Angular velocity [x, y, z] in rad/s (body frame)
        accel: Acceleration [x, y, z] in g units (body frame)
        is_armed: Whether the vehicle is armed
        button_blue: Blue button state
        button_yellow: Yellow button state
        button_green: Green button state
        button_red: Red button state (dead man switch in hardware)
        motor_command: Current motor commands [m1, m2, m3, m4] (0-255)
        time: Current simulation time in seconds
        dt: Time step in seconds
    """

    # Sensor readings (read-only)
    gyro: jax.Array  # rad/s, body frame [x, y, z]
    accel: jax.Array  # g units, body frame [x, y, z]

    # Control inputs (read-only)
    is_armed: bool
    button_blue: bool
    button_yellow: bool
    button_green: bool
    button_red: bool

    # Motor commands (read-write)
    motor_command: jax.Array  # PWM values 0-255

    # Timing
    time: float  # seconds
    dt: float  # time step in seconds

    def set_motors(self, m1: float, m2: float, m3: float, m4: float) -> None:
        """
        Set motor commands for all four motors.

        Args:
            m1: Motor 1 command (front-right, CW) - 0 to 255
            m2: Motor 2 command (front-left, CCW) - 0 to 255
            m3: Motor 3 command (back-left, CW) - 0 to 255
            m4: Motor 4 command (back-right, CCW) - 0 to 255

        Example:
            # Set all motors to 50% power
            state.set_motors(127, 127, 127, 127)

            # Or set individual values
            state.set_motors(100, 100, 120, 120)  # More thrust on rear
        """
        config = Config.GLOBAL
        # Clamp to valid range
        self.motor_command = jnp.array(
            [
                jnp.clip(m1, config.pwm_min, config.pwm_max),
                jnp.clip(m2, config.pwm_min, config.pwm_max),
                jnp.clip(m3, config.pwm_min, config.pwm_max),
                jnp.clip(m4, config.pwm_min, config.pwm_max),
            ]
        )

    def set_all_motors(self, command: float) -> None:
        """
        Set all motors to the same command value.

        Args:
            command: Motor command for all motors (0 to 255)

        Example:
            # Set all motors to 50
            state.set_all_motors(50)
        """
        self.set_motors(command, command, command, command)

    def motors_off(self) -> None:
        """Turn all motors off."""
        self.set_all_motors(0)

    # =========================================================================
    # Utility Functions (Students implement these in Lab 2)
    # =========================================================================

    def pwm_from_speed(self, desired_speed_rad_s: float) -> float:
        """
        Convert desired motor speed (rad/s) to PWM command.

        TODO: Students implement this in Lab 2 by fitting experimental data.

        Args:
            desired_speed_rad_s: Desired angular velocity in rad/s

        Returns:
            PWM command (0-255)
        """
        # Placeholder - students replace with their fit
        # rpm = a + b * pwm  =>  pwm = (rpm - a) / b
        # where rpm = omega * 60 / (2*pi)
        config = Config.GLOBAL
        desired_rpm = desired_speed_rad_s * 60.0 / (2.0 * jnp.pi)
        pwm = (desired_rpm - config.pwm_to_rpm_a) / config.pwm_to_rpm_b
        return jnp.clip(pwm, config.pwm_min, config.pwm_max)

    def speed_from_force(self, desired_force_n: float) -> float:
        """
        Convert desired thrust force (N) to motor speed (rad/s).

        TODO: Students implement this in Lab 2 using force rig measurements.

        Args:
            desired_force_n: Desired thrust per motor in Newtons

        Returns:
            Motor angular velocity in rad/s
        """
        # Placeholder - students replace with their identified constant
        # F = k * omega^2  =>  omega = sqrt(F / k)
        config = Config.GLOBAL
        if desired_force_n <= 0:
            return 0.0
        return jnp.sqrt(desired_force_n / config.thrust_constant)

    def pwm_from_force(self, desired_force_n: float) -> float:
        """
        Convert desired thrust force (N) directly to PWM command.

        This combines speed_from_force and pwm_from_speed.

        Args:
            desired_force_n: Desired thrust per motor in Newtons

        Returns:
            PWM command (0-255)
        """
        speed = self.speed_from_force(desired_force_n)
        return self.pwm_from_speed(speed)


# =============================================================================
# Print Status Function (for debugging)
# =============================================================================


def print_status(state: CrazyflieState) -> None:
    """
    Print current vehicle status.

    This mirrors the PrintStatus() function in the Crazyflie firmware.
    Called when the "Print info" button is pressed.
    """
    print("=" * 50)
    print("Crazyflie Status")
    print("=" * 50)
    print(f"Time: {state.time:.2f} s")
    print(f"Armed: {'YES' if state.is_armed else 'NO'}")
    print()
    print("Sensors:")
    print(f"  Gyro (rad/s):  [{state.gyro[0]:+.4f}, {state.gyro[1]:+.4f}, {state.gyro[2]:+.4f}]")
    print(f"  Accel (g):     [{state.accel[0]:+.4f}, {state.accel[1]:+.4f}, {state.accel[2]:+.4f}]")
    print()
    print("Buttons:")
    print(f"  Blue: {state.button_blue}, Yellow: {state.button_yellow}")
    print(f"  Green: {state.button_green}, Red: {state.button_red}")
    print()
    print("Motors (PWM):")
    print(
        f"  M1: {state.motor_command[0]:.0f}, M2: {state.motor_command[1]:.0f}, "
        f"M3: {state.motor_command[2]:.0f}, M4: {state.motor_command[3]:.0f}"
    )
    print("=" * 50)
