/**
 * user_code.c - Student Control Implementation
 *
 * This is THE file where students write their control code.
 * The same code runs in both SITL simulation and on Crazyflie hardware.
 *
 * Part of the Crazyflie Educational Labs
 *
 * INSTRUCTIONS:
 * =============
 * 1. Set your TEAM_ID below
 * 2. Implement your control logic in user_main_loop()
 * 3. Update powertrain constants from Lab 2
 *
 * SAFETY:
 * =======
 * - Vehicle must be armed for motors to spin (Q key in sim)
 * - Blue button is dead man switch (Shift key in sim - must be held)
 * - Start with low MOTOR_TEST_LIMIT until confident
 */

#include "user_code.h"
#include <math.h>
#include <stdio.h>

// =============================================================================
// Configuration Constants
// =============================================================================

// Team ID - Replace with your assigned team number
#define TEAM_ID 0  // TODO: Set your team ID!

// Motor command limit for safety during testing
// PWM range is 0-65535; start low!
#define MOTOR_TEST_LIMIT 65000  // ~15% of max

// =============================================================================
// Powertrain Constants (Lab 2)
// =============================================================================

// PWM to Speed mapping: omega = PWM_TO_SPEED_A + PWM_TO_SPEED_B * pwm
// where pwm is in 0-255 scale (we convert from 0-65535 internally)
static const float PWM_TO_SPEED_A = 426.0f;   // rad/s at PWM=0 (placeholder)
static const float PWM_TO_SPEED_B = 6.8f;     // rad/s per PWM unit (placeholder)

// Thrust constant: force = THRUST_CONSTANT * omega^2
static const float THRUST_CONSTANT = 1.8e-8f;  // N/(rad/s)^2 (placeholder)

// Vehicle parameters
static const float VEHICLE_MASS = 0.027f;  // kg (Crazyflie 2.1)
static const float GRAVITY = 9.81f;        // m/s^2

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Convert desired motor speed (rad/s) to PWM command.
 *
 * Lab 2 Part 1: Update PWM_TO_SPEED_A and PWM_TO_SPEED_B
 * with your experimental fits.
 *
 * @param desired_speed Desired angular velocity in rad/s
 * @return PWM command (0-65535)
 */
static uint16_t pwm_from_speed(float desired_speed) {
    // Invert the linear relationship: omega = a + b * pwm
    // pwm = (omega - a) / b
    float pwm_255 = (desired_speed - PWM_TO_SPEED_A) / PWM_TO_SPEED_B;
    pwm_255 = user_clampf(pwm_255, 0.0f, 255.0f);

    // Scale from 0-255 to 0-65535
    return user_clamp_pwm(pwm_255 * 257.0f);
}

/**
 * Convert desired thrust force (N) to motor speed (rad/s).
 *
 * Lab 2 Part 2: Update THRUST_CONSTANT with your experimental value.
 *
 * @param desired_force Desired thrust per motor in Newtons
 * @return Motor angular velocity in rad/s
 */
static float speed_from_force(float desired_force) {
    // Invert the quadratic relationship: F = k * omega^2
    // omega = sqrt(F / k)
    if (desired_force <= 0.0f) {
        return 0.0f;
    }
    return sqrtf(desired_force / THRUST_CONSTANT);
}

/**
 * Convert desired thrust force (N) directly to PWM command.
 *
 * @param desired_force Desired thrust per motor in Newtons
 * @return PWM command (0-65535)
 */
static uint16_t pwm_from_force(float desired_force) {
    float speed = speed_from_force(desired_force);
    return pwm_from_speed(speed);
}

/**
 * Calculate hover thrust per motor.
 *
 * @return Thrust in Newtons for each motor to hover
 */
static float hover_thrust_per_motor(void) {
    return (VEHICLE_MASS * GRAVITY) / 4.0f;
}

// Suppress unused function warnings for utility functions
// (students will use them in later labs)
__attribute__((unused)) static uint16_t _use_pwm_from_speed = 0;
__attribute__((unused)) static float _use_speed_from_force = 0;
__attribute__((unused)) static uint16_t _use_pwm_from_force = 0;
__attribute__((unused)) static float _use_hover_thrust = 0;

// =============================================================================
// User Code Implementation
// =============================================================================

/**
 * Initialize user code - called once at startup.
 */
void user_init(void) {
    printf("[UserCode] Initialized, Team ID: %d\n", TEAM_ID);
    printf("[UserCode] Control loop: %d Hz (dt=%.3f ms)\n",
           CONTROL_LOOP_HZ, CONTROL_LOOP_DT * 1000.0f);
}

/**
 * Main control loop - called every control cycle (500 Hz).
 *
 * =========================================================================
 * YOUR CONTROL CODE GOES HERE
 * =========================================================================
 *
 * Available in 'state':
 * ---------------------
 * Sensors (read-only):
 *   state->sensors.gyro.x, .y, .z  - Angular velocity (rad/s)
 *   state->sensors.accel.x, .y, .z - Acceleration (g units)
 *
 * Buttons (read-only):
 *   state->is_armed      - Vehicle is armed
 *   state->button_blue   - Blue button (dead man switch)
 *   state->button_yellow - Yellow button
 *   state->button_green  - Green button
 *   state->button_red    - Red button
 *
 * Timing (read-only):
 *   state->time          - Current time in seconds
 *   state->dt            - Time step in seconds
 *
 * Motor Control (use helper functions):
 *   user_set_motors(state, m1, m2, m3, m4)  - Set individual motors
 *   user_set_all_motors(state, pwm)         - Set all motors same
 *   user_motors_off(state)                  - Turn all motors off
 *
 * @param state Pointer to current state
 */
void user_main_loop(user_state_t* state) {
    // Read sensor data (for reference - use as needed)
    // float gyro_x = state->sensors.gyro.x;
    // float gyro_y = state->sensors.gyro.y;
    // float gyro_z = state->sensors.gyro.z;
    // float accel_z = state->sensors.accel.z;

    // =========================================================================
    // Lab 1 Example: Turn on motors when armed AND blue button is held
    // =========================================================================
    //
    // Safety pattern:
    //   - Q key toggles "armed" state
    //   - Left Shift is the "blue button" (dead man switch - must be held)
    //   - Motors only spin when BOTH conditions are met

    if (state->is_armed && state->button_blue) {
        // Set all motors to test limit
        user_set_motors(state, MOTOR_TEST_LIMIT, MOTOR_TEST_LIMIT, MOTOR_TEST_LIMIT, MOTOR_TEST_LIMIT);
    } else {
        // Motors off - either not armed or blue button not held
        user_motors_off(state);
    }

    // =========================================================================
    // Lab 2 Example: Command specific thrust (uncomment to use)
    // =========================================================================
    /*
    if (state->is_armed && state->button_blue) {
        // Command 90% of hover thrust per motor
        float thrust = 0.9f * hover_thrust_per_motor();
        uint16_t pwm = pwm_from_force(thrust);
        user_set_all_motors(state, pwm);
    } else {
        user_motors_off(state);
    }
    */
}

/**
 * Print status information - called when "Print Info" is requested.
 *
 * @param state Pointer to current state
 */
void user_print_status(const user_state_t* state) {
    printf("================================================\n");
    printf("User Code Status - Team %d\n", TEAM_ID);
    printf("================================================\n");
    printf("Time: %.2f s\n", state->time);
    printf("Armed: %s\n", state->is_armed ? "YES" : "NO");
    printf("\n");
    printf("Sensors:\n");
    printf("  Gyro (rad/s):  x=%+.4f, y=%+.4f, z=%+.4f\n",
           state->sensors.gyro.x, state->sensors.gyro.y, state->sensors.gyro.z);
    printf("  Accel (g):     x=%+.4f, y=%+.4f, z=%+.4f\n",
           state->sensors.accel.x, state->sensors.accel.y, state->sensors.accel.z);
    printf("\n");
    printf("Buttons: Blue=%d Yellow=%d Green=%d Red=%d\n",
           state->button_blue, state->button_yellow,
           state->button_green, state->button_red);
    printf("\n");
    printf("Motors (PWM):\n");
    printf("  M1=%5u  M2=%5u  M3=%5u  M4=%5u\n",
           state->motor_pwm[0], state->motor_pwm[1],
           state->motor_pwm[2], state->motor_pwm[3]);
    printf("\n");
    printf("Powertrain Constants (Lab 2):\n");
    printf("  PWM_TO_SPEED_A = %.2f rad/s\n", PWM_TO_SPEED_A);
    printf("  PWM_TO_SPEED_B = %.4f rad/s per PWM\n", PWM_TO_SPEED_B);
    printf("  THRUST_CONSTANT = %.2e N/(rad/s)^2\n", THRUST_CONSTANT);
    printf("\n");
    printf("Calculated:\n");
    printf("  Hover thrust/motor = %.4f N\n", hover_thrust_per_motor());
    printf("  Hover speed = %.1f rad/s\n", speed_from_force(hover_thrust_per_motor()));
    printf("  Hover PWM = %u\n", pwm_from_force(hover_thrust_per_motor()));
    printf("================================================\n");
}

