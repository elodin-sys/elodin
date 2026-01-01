/**
 * UserCode.h - Student Control Implementation Header
 *
 * This header defines the interface for student control code.
 * Include this in your main.c or stabilizer.c to integrate.
 *
 * Part of the Crazyflie Educational Labs
 */

#ifndef USER_CODE_H
#define USER_CODE_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Sensor data structure
 * Contains current sensor readings from the IMU
 */
typedef struct {
    struct {
        float x;  // rad/s, body frame
        float y;
        float z;
    } gyro;

    struct {
        float x;  // g units, body frame
        float y;
        float z;
    } acc;
} sensorData_t;

/**
 * Initialize user code module
 * Called once at startup
 */
void userCodeInit(void);

/**
 * Main control loop
 * Called at 500 Hz (every 2ms)
 *
 * @param sensors Pointer to current sensor readings
 * @param motorPwm Array of 4 motor PWM values (0-65535), output
 */
void userCodeMainLoop(sensorData_t* sensors, uint16_t* motorPwm);

/**
 * Print status information
 * Called when "Print Info" is requested
 */
void userCodePrintStatus(void);

/**
 * Check if armed
 * @return true if vehicle is armed and motors can spin
 */
bool userCodeIsArmed(void);

/**
 * Set arm state
 * @param armed New arm state
 */
void userCodeSetArmed(bool armed);

#endif // USER_CODE_H

