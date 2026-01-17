/**
 * user_code.h - Student Control Code API
 *
 * This header defines the interface for student control code.
 * The same code runs in:
 *   1. SITL simulation (compiled natively, communicates via UDP with Elodin)
 *   2. Crazyflie hardware (compiled for STM32, runs in app layer)
 *
 * Part of the Crazyflie Educational Labs
 */

#ifndef USER_CODE_H
#define USER_CODE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Constants
// =============================================================================

// PWM range (Crazyflie uses 16-bit PWM internally)
#define PWM_MIN 0
#define PWM_MAX 65535

// Control loop frequency
#define CONTROL_LOOP_HZ 500
#define CONTROL_LOOP_DT (1.0f / CONTROL_LOOP_HZ)

// =============================================================================
// Sensor Data Types
// =============================================================================

/**
 * 3-axis vector (used for gyro, accel, etc.)
 */
typedef struct {
    float x;
    float y;
    float z;
} vec3_t;

/**
 * Sensor readings from IMU
 */
typedef struct {
    vec3_t gyro;   // Angular velocity in rad/s (body frame)
    vec3_t accel;  // Acceleration in g units (body frame)
} sensors_t;

// =============================================================================
// User State Structure
// =============================================================================

/**
 * User state structure - passed to main_loop every control cycle.
 *
 * This provides read-only access to sensors and buttons,
 * and write access to motor commands.
 *
 * Usage:
 *   void user_main_loop(user_state_t* state) {
 *       // Read sensors
 *       float roll_rate = state->sensors.gyro.x;
 *
 *       // Check buttons
 *       if (state->is_armed && state->button_blue) {
 *           user_set_all_motors(state, 10000);
 *       }
 *   }
 */
typedef struct {
    // Timing
    double time;           // Current time in seconds
    float dt;              // Time step (typically 0.002s = 500Hz)

    // Sensor data (read-only)
    sensors_t sensors;

    // Control inputs (read-only)
    bool is_armed;         // Vehicle is armed (motors can spin)
    bool button_blue;      // Blue button pressed (dead man switch)
    bool button_yellow;    // Yellow button pressed
    bool button_green;     // Green button pressed
    bool button_red;       // Red button pressed

    // Motor outputs (read-write via helper functions)
    uint16_t motor_pwm[4]; // Motor PWM commands (0-65535)
} user_state_t;

// =============================================================================
// User Code Functions (implemented by student in user_code.c)
// =============================================================================

/**
 * Initialize user code - called once at startup.
 *
 * Use this to initialize any state variables your control code needs.
 */
void user_init(void);

/**
 * Main control loop - called every control cycle (500 Hz).
 *
 * This is where you implement your control logic.
 *
 * @param state Pointer to current state (sensors, buttons, motors)
 */
void user_main_loop(user_state_t* state);

/**
 * Print status - called when "Print Info" is requested.
 *
 * Use this for debugging output.
 *
 * @param state Pointer to current state
 */
void user_print_status(const user_state_t* state);

// =============================================================================
// Helper Functions (provided by the platform)
// =============================================================================

/**
 * Set all motors to the same PWM value.
 *
 * @param state Pointer to user state
 * @param pwm   PWM value (0-65535)
 */
static inline void user_set_all_motors(user_state_t* state, uint16_t pwm) {
    state->motor_pwm[0] = pwm;
    state->motor_pwm[1] = pwm;
    state->motor_pwm[2] = pwm;
    state->motor_pwm[3] = pwm;
}

/**
 * Set individual motor PWM values.
 *
 * Motor numbering (Crazyflie Quad-X, viewed from above):
 *   M1: Front-right (CW)
 *   M2: Front-left (CCW)
 *   M3: Back-left (CW)
 *   M4: Back-right (CCW)
 *
 * @param state Pointer to user state
 * @param m1    Motor 1 PWM (0-65535)
 * @param m2    Motor 2 PWM (0-65535)
 * @param m3    Motor 3 PWM (0-65535)
 * @param m4    Motor 4 PWM (0-65535)
 */
static inline void user_set_motors(user_state_t* state,
                                   uint16_t m1, uint16_t m2,
                                   uint16_t m3, uint16_t m4) {
    state->motor_pwm[0] = m1;
    state->motor_pwm[1] = m2;
    state->motor_pwm[2] = m3;
    state->motor_pwm[3] = m4;
}

/**
 * Turn all motors off.
 *
 * @param state Pointer to user state
 */
static inline void user_motors_off(user_state_t* state) {
    user_set_all_motors(state, 0);
}

/**
 * Clamp a float to a range.
 *
 * @param value Value to clamp
 * @param min   Minimum value
 * @param max   Maximum value
 * @return      Clamped value
 */
static inline float user_clampf(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * Clamp a PWM value to valid range.
 *
 * @param value Value to clamp (can be float for calculations)
 * @return      Clamped PWM value (0-65535)
 */
static inline uint16_t user_clamp_pwm(float value) {
    if (value < PWM_MIN) return PWM_MIN;
    if (value > PWM_MAX) return PWM_MAX;
    return (uint16_t)value;
}

#ifdef __cplusplus
}
#endif

#endif // USER_CODE_H

