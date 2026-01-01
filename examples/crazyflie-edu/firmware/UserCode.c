/**
 * UserCode.c - Student Control Implementation
 *
 * This is the C equivalent of user_code.py from the simulation.
 * The code structure mirrors the Python version for easy porting.
 *
 * Part of the Crazyflie Educational Labs
 *
 * INSTRUCTIONS:
 * =============
 * 1. Set your TEAM_ID
 * 2. Update powertrain constants from SimLab2/HWLab2
 * 3. Implement your control logic in userCodeMainLoop()
 *
 * SAFETY:
 * =======
 * - Vehicle must be armed for motors to spin
 * - Start with MOTOR_TEST_LIMIT until confident
 * - Always have an emergency stop plan
 */

#include "UserCode.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

// =============================================================================
// Configuration Constants
// =============================================================================

// Team ID - Replace with your assigned team number
#define TEAM_ID 0  // TODO: Set your team ID!

// Motor command limit for safety (PWM 0-65535)
// Start low and increase as you gain confidence
#define MOTOR_TEST_LIMIT 10000  // ~15% of max

// PWM limits
#define PWM_MIN 0
#define PWM_MAX 65535

// =============================================================================
// Powertrain Constants (from Lab 2)
// =============================================================================

// PWM to Speed mapping: omega = PWM_TO_SPEED_A + PWM_TO_SPEED_B * pwm
// Where pwm is 0-255 scale (we convert from 0-65535 internally)
static float PWM_TO_SPEED_A = 426.0f;   // rad/s at PWM=0 (placeholder)
static float PWM_TO_SPEED_B = 6.8f;     // rad/s per PWM unit (placeholder)

// Thrust constant: force = THRUST_CONSTANT * omega^2
static float THRUST_CONSTANT = 1.8e-8f;  // N/(rad/s)^2 (placeholder)

// Vehicle parameters
static const float VEHICLE_MASS = 0.027f;  // kg
static const float GRAVITY = 9.81f;        // m/s^2

// =============================================================================
// Module State
// =============================================================================

static bool isArmed = false;

// Button states (set by external code)
static bool buttonBlue = false;
static bool buttonYellow = false;
static bool buttonGreen = false;
static bool buttonRed = false;

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Clamp a value to a range
 */
static float clampf(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * Clamp a uint16 to PWM range
 */
static uint16_t clampPwm(float value) {
    if (value < PWM_MIN) return PWM_MIN;
    if (value > PWM_MAX) return PWM_MAX;
    return (uint16_t)value;
}

/**
 * Convert desired motor speed (rad/s) to PWM command.
 *
 * Lab 2 Part 1: Update PWM_TO_SPEED_A and PWM_TO_SPEED_B
 * with your experimental fits.
 *
 * @param desiredSpeed Desired angular velocity in rad/s
 * @return PWM command (0-65535)
 */
static uint16_t pwmFromSpeed(float desiredSpeed) {
    // Invert the linear relationship: omega = a + b * pwm
    // pwm = (omega - a) / b
    float pwm255 = (desiredSpeed - PWM_TO_SPEED_A) / PWM_TO_SPEED_B;
    pwm255 = clampf(pwm255, 0.0f, 255.0f);

    // Scale from 0-255 to 0-65535
    return clampPwm(pwm255 * 257.0f);
}

/**
 * Convert desired thrust force (N) to motor speed (rad/s).
 *
 * Lab 2 Part 2: Update THRUST_CONSTANT with your experimental value.
 *
 * @param desiredForce Desired thrust per motor in Newtons
 * @return Motor angular velocity in rad/s
 */
static float speedFromForce(float desiredForce) {
    // Invert the quadratic relationship: F = k * omega^2
    // omega = sqrt(F / k)
    if (desiredForce <= 0.0f) {
        return 0.0f;
    }
    return sqrtf(desiredForce / THRUST_CONSTANT);
}

/**
 * Convert desired thrust force (N) directly to PWM command.
 *
 * @param desiredForce Desired thrust per motor in Newtons
 * @return PWM command (0-65535)
 */
static uint16_t pwmFromForce(float desiredForce) {
    float speed = speedFromForce(desiredForce);
    return pwmFromSpeed(speed);
}

/**
 * Calculate hover thrust per motor
 *
 * @return Thrust in Newtons for each motor to hover
 */
static float hoverThrustPerMotor(void) {
    return (VEHICLE_MASS * GRAVITY) / 4.0f;
}

/**
 * Set all motors to the same PWM value
 */
static void setAllMotors(uint16_t* motorPwm, uint16_t value) {
    motorPwm[0] = value;
    motorPwm[1] = value;
    motorPwm[2] = value;
    motorPwm[3] = value;
}

/**
 * Turn all motors off
 */
static void motorsOff(uint16_t* motorPwm) {
    setAllMotors(motorPwm, 0);
}

// =============================================================================
// Public API
// =============================================================================

/**
 * Initialize user code module
 */
void userCodeInit(void) {
    isArmed = false;
    buttonBlue = false;
    buttonYellow = false;
    buttonGreen = false;
    buttonRed = false;

    // Print startup message
    printf("[UserCode] Initialized, Team ID: %d\n", TEAM_ID);
}

/**
 * Check if armed
 */
bool userCodeIsArmed(void) {
    return isArmed;
}

/**
 * Set arm state
 */
void userCodeSetArmed(bool armed) {
    isArmed = armed;
    if (armed) {
        printf("[UserCode] ARMED\n");
    } else {
        printf("[UserCode] DISARMED\n");
    }
}

/**
 * Main control loop - called at 500 Hz
 *
 * This function is the heart of your control code. It runs every 2ms
 * and has access to sensor data and motor commands.
 *
 * Available sensor data:
 *   sensors->gyro.x, .y, .z  - Angular velocity (rad/s), body frame
 *   sensors->acc.x, .y, .z   - Acceleration (g units), body frame
 *
 * Motor output:
 *   motorPwm[0..3] - PWM commands (0-65535) for motors 1-4
 *
 * Button states (set externally):
 *   buttonBlue, buttonYellow, buttonGreen, buttonRed
 *
 * @param sensors Pointer to current sensor readings
 * @param motorPwm Array of 4 motor PWM values (output)
 */
void userCodeMainLoop(sensorData_t* sensors, uint16_t* motorPwm) {
    // Read sensor data (for reference - use as needed)
    float gyroX = sensors->gyro.x;
    float gyroY = sensors->gyro.y;
    float gyroZ = sensors->gyro.z;

    float accelX = sensors->acc.x;
    float accelY = sensors->acc.y;
    float accelZ = sensors->acc.z;

    // Suppress unused variable warnings
    (void)gyroX; (void)gyroY; (void)gyroZ;
    (void)accelX; (void)accelY; (void)accelZ;

    // =========================================================================
    // YOUR CODE GOES HERE
    // =========================================================================

    // Lab 1 Example: Turn on motors when blue button is pressed
    // Uncomment and modify for your experiments

    /*
    if (isArmed && buttonBlue) {
        // Set all motors to test limit
        setAllMotors(motorPwm, MOTOR_TEST_LIMIT);
    } else {
        // Motors off
        motorsOff(motorPwm);
    }
    */

    // Lab 2 Example: Command specific thrust
    /*
    if (isArmed && buttonBlue) {
        // Command 90% of hover thrust
        float thrust = 0.9f * hoverThrustPerMotor();
        uint16_t pwm = pwmFromForce(thrust);
        setAllMotors(motorPwm, pwm);
    } else {
        motorsOff(motorPwm);
    }
    */

    // =========================================================================
    // Default behavior: motors off
    // =========================================================================
    motorsOff(motorPwm);
}

/**
 * Print status information
 */
void userCodePrintStatus(void) {
    printf("================================================\n");
    printf("User Code Status\n");
    printf("================================================\n");
    printf("Team ID: %d\n", TEAM_ID);
    printf("Armed: %s\n", isArmed ? "YES" : "NO");
    printf("\n");
    printf("Powertrain Constants:\n");
    printf("  PWM_TO_SPEED_A = %.2f rad/s\n", PWM_TO_SPEED_A);
    printf("  PWM_TO_SPEED_B = %.4f rad/s per PWM\n", PWM_TO_SPEED_B);
    printf("  THRUST_CONSTANT = %.2e N/(rad/s)^2\n", THRUST_CONSTANT);
    printf("\n");
    printf("Calculated Values:\n");
    printf("  Hover thrust/motor = %.4f N\n", hoverThrustPerMotor());
    printf("  Hover speed = %.1f rad/s\n", speedFromForce(hoverThrustPerMotor()));
    printf("  Hover PWM = %u\n", pwmFromForce(hoverThrustPerMotor()));
    printf("================================================\n");
}

// =============================================================================
// Button Setters (called by external code)
// =============================================================================

void userCodeSetButtonBlue(bool pressed) { buttonBlue = pressed; }
void userCodeSetButtonYellow(bool pressed) { buttonYellow = pressed; }
void userCodeSetButtonGreen(bool pressed) { buttonGreen = pressed; }
void userCodeSetButtonRed(bool pressed) { buttonRed = pressed; }

