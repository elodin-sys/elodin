"""
User Code - Student Control Implementation

This file is where you write your control code for the Crazyflie.
The main_loop() function is called every control cycle (500 Hz by default).

The code you write here should be directly translatable to C code that
runs on the actual Crazyflie hardware (see firmware/UserCode.c).

INSTRUCTIONS:
=============
1. Read sensor data from state.gyro and state.accel
2. Check button states with state.button_blue, etc.
3. Set motor commands with state.set_motors() or state.set_all_motors()

SAFETY:
=======
- The vehicle must be ARMED for motors to spin
- The RED button acts as a dead man switch (must be held in hardware)
- Motors are automatically set to 0 when not armed

EXAMPLE (Lab 1):
================
To turn on motors when blue button is pressed:

    if state.is_armed and state.button_blue:
        state.set_all_motors(50)  # Low power for testing
    else:
        state.motors_off()

"""

from crazyflie_api import CrazyflieState

# =============================================================================
# Configuration Constants
# =============================================================================

# Team ID - Replace with your assigned team number
TEAM_ID = 0  # TODO: Set your team ID here!

# Motor command limit for safety during testing
MOTOR_TEST_LIMIT = 50  # Don't exceed this until you're confident!

# =============================================================================
# Utility Functions
# =============================================================================

# These coefficients are determined in Lab 2
# Students replace these placeholder values with their experimental fits

# PWM to Speed mapping: speed_rad_s = pwm_to_speed_a + pwm_to_speed_b * pwm
PWM_TO_SPEED_A = 426.0  # rad/s at PWM=0 (placeholder)
PWM_TO_SPEED_B = 6.8  # rad/s per PWM unit (placeholder)

# Thrust constant: force_N = THRUST_CONSTANT * speed_rad_s^2
THRUST_CONSTANT = 1.8e-8  # N/(rad/s)^2 (placeholder - identify in Lab 2!)


def pwm_from_speed(desired_speed_rad_s: float) -> float:
    """
    Convert desired motor speed (rad/s) to PWM command.

    Lab 2 Part 1: You will derive this from tachometer measurements.

    The relationship is: speed = a + b * pwm
    Inverting: pwm = (speed - a) / b

    Args:
        desired_speed_rad_s: Desired angular velocity in rad/s

    Returns:
        PWM command (0-255)
    """
    # Invert the linear relationship
    pwm = (desired_speed_rad_s - PWM_TO_SPEED_A) / PWM_TO_SPEED_B

    # Clamp to valid range
    return max(0, min(255, pwm))


def speed_from_force(desired_force_n: float) -> float:
    """
    Convert desired thrust force (N) to motor speed (rad/s).

    Lab 2 Part 2: You will derive THRUST_CONSTANT from force rig measurements.

    The relationship is: force = k * speed^2
    Inverting: speed = sqrt(force / k)

    Args:
        desired_force_n: Desired thrust per motor in Newtons

    Returns:
        Motor angular velocity in rad/s
    """
    if desired_force_n <= 0:
        return 0.0

    # Invert the quadratic relationship
    import math

    return math.sqrt(desired_force_n / THRUST_CONSTANT)


def pwm_from_force(desired_force_n: float) -> float:
    """
    Convert desired thrust force (N) directly to PWM command.

    This combines speed_from_force() and pwm_from_speed().

    Args:
        desired_force_n: Desired thrust per motor in Newtons

    Returns:
        PWM command (0-255)
    """
    speed = speed_from_force(desired_force_n)
    return pwm_from_speed(speed)


# =============================================================================
# Main Control Loop
# =============================================================================


def main_loop(state: CrazyflieState) -> None:
    """
    Main control loop - called every control cycle.

    This function is called at 500 Hz (every 2ms) during simulation.
    You have access to sensor data and can set motor commands.

    Available in 'state':
    ---------------------
    Sensors (read-only):
        state.gyro[0], state.gyro[1], state.gyro[2]  - Angular velocity (rad/s)
        state.accel[0], state.accel[1], state.accel[2] - Acceleration (g units)

    Buttons (read-only):
        state.button_blue   - Blue button pressed
        state.button_yellow - Yellow button pressed
        state.button_green  - Green button pressed
        state.button_red    - Red button (dead man switch)
        state.is_armed      - Vehicle is armed

    Timing (read-only):
        state.time          - Current time in seconds
        state.dt            - Time step in seconds

    Motor Control (write):
        state.set_motors(m1, m2, m3, m4)  - Set individual motor commands
        state.set_all_motors(cmd)         - Set all motors to same value
        state.motors_off()                - Turn all motors off

    Args:
        state: CrazyflieState object with sensor data and motor command access
    """
    # =========================================================================
    # YOUR CODE GOES HERE
    # =========================================================================

    # Lab 1 Example: Turn on motors when blue button is pressed
    # Uncomment and modify for your experiments

    # if state.is_armed and state.button_blue:
    #     # Set all motors to a low test value
    #     state.set_all_motors(MOTOR_TEST_LIMIT)
    # else:
    #     # Motors off when not testing
    #     state.motors_off()

    # =========================================================================
    # Default behavior: motors off
    # =========================================================================
    state.motors_off()


# =============================================================================
# Print Status (called when "Print Info" button is pressed)
# =============================================================================


def print_status(state: CrazyflieState) -> None:
    """
    Print diagnostic information.

    This is called when the "Print Info" button is pressed in the GUI.
    Add any debug information you want to display here.
    """
    print("=" * 50)
    print(f"Team ID: {TEAM_ID}")
    print(f"Time: {state.time:.2f} s")
    print(f"Armed: {'YES' if state.is_armed else 'NO'}")
    print()
    print("Sensor Readings:")
    print(f"  Gyro (rad/s):  x={state.gyro[0]:+.4f}, y={state.gyro[1]:+.4f}, z={state.gyro[2]:+.4f}")
    print(f"  Accel (g):     x={state.accel[0]:+.4f}, y={state.accel[1]:+.4f}, z={state.accel[2]:+.4f}")
    print()
    print("Motor Commands (PWM):")
    print(
        f"  M1={state.motor_command[0]:.0f}, M2={state.motor_command[1]:.0f}, "
        f"M3={state.motor_command[2]:.0f}, M4={state.motor_command[3]:.0f}"
    )
    print()
    print("Powertrain Constants (Lab 2):")
    print(f"  PWM_TO_SPEED_A = {PWM_TO_SPEED_A}")
    print(f"  PWM_TO_SPEED_B = {PWM_TO_SPEED_B}")
    print(f"  THRUST_CONSTANT = {THRUST_CONSTANT:.2e}")
    print("=" * 50)

