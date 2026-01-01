/**
 * UtilityFunctions.c - Powertrain Utility Functions
 *
 * This file contains the utility functions for converting between
 * PWM commands, motor speeds, and thrust forces.
 *
 * Students update the constants in this file based on Lab 2 experiments.
 *
 * Part of the Crazyflie Educational Labs
 */

#include <math.h>
#include <stdint.h>

// =============================================================================
// Powertrain Constants - UPDATE THESE FROM YOUR EXPERIMENTS
// =============================================================================

/**
 * PWM to Speed mapping coefficients
 *
 * The relationship is: omega (rad/s) = PWM_TO_SPEED_A + PWM_TO_SPEED_B * pwm
 *
 * Where pwm is on a 0-255 scale.
 *
 * Lab 2 Part 1: Determine these from tachometer measurements
 */
static const float PWM_TO_SPEED_A = 0.0f;   // TODO: Replace with your value
static const float PWM_TO_SPEED_B = 0.0f;   // TODO: Replace with your value

/**
 * Thrust constant
 *
 * The relationship is: force (N) = THRUST_CONSTANT * omega^2 (rad/s)
 *
 * Lab 2 Part 2: Determine this from force rig measurements
 * Expected order of magnitude: ~10^-8 N/(rad/s)^2
 */
static const float THRUST_CONSTANT = 1.0e-8f;  // TODO: Replace with your value

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Convert desired motor speed (rad/s) to PWM command (0-255).
 *
 * This is the inverse of the motor speed model.
 *
 * @param desiredSpeed_rad_per_sec Desired angular velocity in rad/s
 * @return PWM command as integer (0-255)
 */
int pwmCommandFromSpeed(float desiredSpeed_rad_per_sec) {
    // Invert the linear model: omega = a + b * pwm
    // Solving for pwm: pwm = (omega - a) / b

    // Replace these with your experimental coefficients!
    float a = PWM_TO_SPEED_A;  // zeroth order term
    float b = PWM_TO_SPEED_B;  // first order term

    // Safety check - avoid division by zero
    if (b == 0.0f) {
        return 0;
    }

    float pwm = (desiredSpeed_rad_per_sec - a) / b;

    // Clamp to valid range
    if (pwm < 0.0f) return 0;
    if (pwm > 255.0f) return 255;

    return (int)pwm;
}

/**
 * Convert desired thrust force (N) to motor speed (rad/s).
 *
 * This is the inverse of the thrust model.
 *
 * @param desiredForce_N Desired thrust per motor in Newtons
 * @return Motor angular velocity in rad/s
 */
float speedFromForce(float desiredForce_N) {
    // Replace this with your determined constant!
    // Remember to add the trailing "f" for single precision!
    float propConstant = THRUST_CONSTANT;

    // Safety check - no sqrtf for negative numbers
    if (desiredForce_N <= 0.0f) {
        return 0.0f;
    }

    // Invert the quadratic model: F = k * omega^2
    // Solving for omega: omega = sqrt(F / k)
    return sqrtf(desiredForce_N / propConstant);
}

/**
 * Convert motor speed (rad/s) to thrust force (N).
 *
 * Forward model for validation.
 *
 * @param speed_rad_per_sec Motor angular velocity in rad/s
 * @return Thrust force in Newtons
 */
float forceFromSpeed(float speed_rad_per_sec) {
    return THRUST_CONSTANT * speed_rad_per_sec * speed_rad_per_sec;
}

/**
 * Convert desired thrust force (N) directly to PWM command.
 *
 * Combines speedFromForce and pwmCommandFromSpeed.
 *
 * @param desiredForce_N Desired thrust per motor in Newtons
 * @return PWM command as integer (0-255)
 */
int pwmCommandFromForce(float desiredForce_N) {
    float speed = speedFromForce(desiredForce_N);
    return pwmCommandFromSpeed(speed);
}

/**
 * Calculate hover thrust per motor.
 *
 * @param mass_kg Vehicle mass in kg
 * @return Thrust per motor in Newtons for hover
 */
float hoverThrustPerMotor(float mass_kg) {
    const float g = 9.81f;  // m/s^2
    return (mass_kg * g) / 4.0f;
}

/**
 * Calculate PWM command for hover.
 *
 * @param mass_kg Vehicle mass in kg
 * @return PWM command for hover (0-255)
 */
int hoverPwmCommand(float mass_kg) {
    float thrust = hoverThrustPerMotor(mass_kg);
    return pwmCommandFromForce(thrust);
}

