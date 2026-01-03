/**
 * app_main.c - Crazyflie App Layer Entry Point
 *
 * This file provides the Crazyflie firmware integration for user_code.c.
 * It reads sensor data from the firmware and passes it to user_main_loop().
 *
 * Usage:
 *   1. Copy user_code.c, user_code.h, and this file to your crazyflie-firmware
 *      examples directory (e.g., examples/app_my_control/)
 *   2. Create a Kbuild.in file (see below)
 *   3. Build with: make app_my_control_defconfig && make -j
 *   4. Flash with: cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
 *
 * Kbuild.in contents:
 *   obj-y += src/app_main.o
 *   obj-y += src/user_code.o
 */

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"
#include "log.h"
#include "param.h"
#include "pm.h"
#include "stabilizer_types.h"
#include "commander.h"
#include "motors.h"

// Include the user code header
// When deployed to crazyflie-firmware, user_code.h will be in the same directory
#include "user_code.h"

// =============================================================================
// Configuration
// =============================================================================

// Control loop rate (Hz)
#define CONTROL_RATE_HZ 500
#define CONTROL_PERIOD_MS (1000 / CONTROL_RATE_HZ)

// Task stack size (words)
#define APP_TASK_STACK_SIZE (2 * configMINIMAL_STACK_SIZE)
#define APP_TASK_PRIORITY 3

// =============================================================================
// Logging Variables (visible in Crazyflie client)
// =============================================================================

static float log_gyro_x;
static float log_gyro_y;
static float log_gyro_z;
static float log_accel_x;
static float log_accel_y;
static float log_accel_z;
static uint16_t log_motor_pwm[4];
static uint8_t log_is_armed;
static uint8_t log_button_blue;

// =============================================================================
// Parameters (controllable from Crazyflie client)
// =============================================================================

static uint8_t param_armed = 0;        // Set to 1 to arm
static uint8_t param_button_blue = 0;  // Dead man switch
static uint8_t param_button_yellow = 0;
static uint8_t param_button_green = 0;
static uint8_t param_button_red = 0;

// =============================================================================
// Sensor Reading
// =============================================================================

/**
 * Read current sensor data from the Crazyflie.
 */
static void read_sensors(user_state_t* state) {
    // Get current state estimate from the state estimator
    state_t cf_state;
    stateEstimatorGetState(&cf_state);

    // Get gyro data (rad/s)
    sensorData_t sensors;
    sensorsAcquire(&sensors);

    state->sensors.gyro.x = sensors.gyro.x;
    state->sensors.gyro.y = sensors.gyro.y;
    state->sensors.gyro.z = sensors.gyro.z;

    // Convert accelerometer from m/s^2 to g units
    state->sensors.accel.x = sensors.acc.x;
    state->sensors.accel.y = sensors.acc.y;
    state->sensors.accel.z = sensors.acc.z;

    // Update logging variables
    log_gyro_x = state->sensors.gyro.x;
    log_gyro_y = state->sensors.gyro.y;
    log_gyro_z = state->sensors.gyro.z;
    log_accel_x = state->sensors.accel.x;
    log_accel_y = state->sensors.accel.y;
    log_accel_z = state->sensors.accel.z;
}

/**
 * Read control inputs (from parameters).
 */
static void read_inputs(user_state_t* state) {
    state->is_armed = param_armed != 0;
    state->button_blue = param_button_blue != 0;
    state->button_yellow = param_button_yellow != 0;
    state->button_green = param_button_green != 0;
    state->button_red = param_button_red != 0;

    log_is_armed = param_armed;
    log_button_blue = param_button_blue;
}

/**
 * Apply motor PWM commands.
 *
 * Safety: Motors only spin if armed AND blue button is active.
 */
static void apply_motors(const user_state_t* state) {
    // Safety check: only apply if armed and blue button pressed
    if (state->is_armed && state->button_blue) {
        // Scale from 16-bit PWM to Crazyflie motor format (0-65535)
        motorsSetRatio(MOTOR_M1, state->motor_pwm[0]);
        motorsSetRatio(MOTOR_M2, state->motor_pwm[1]);
        motorsSetRatio(MOTOR_M3, state->motor_pwm[2]);
        motorsSetRatio(MOTOR_M4, state->motor_pwm[3]);
    } else {
        // Motors off
        motorsSetRatio(MOTOR_M1, 0);
        motorsSetRatio(MOTOR_M2, 0);
        motorsSetRatio(MOTOR_M3, 0);
        motorsSetRatio(MOTOR_M4, 0);
    }

    // Update logging
    log_motor_pwm[0] = state->motor_pwm[0];
    log_motor_pwm[1] = state->motor_pwm[1];
    log_motor_pwm[2] = state->motor_pwm[2];
    log_motor_pwm[3] = state->motor_pwm[3];
}

// =============================================================================
// App Task
// =============================================================================

static void appTask(void* param) {
    (void)param;

    DEBUG_PRINT("User Control App starting...\n");

    // Wait for system to be fully ready
    vTaskDelay(M2T(2000));

    // Initialize user code
    user_init();

    DEBUG_PRINT("User Control App initialized, entering main loop\n");

    user_state_t state;
    TickType_t lastWakeTime = xTaskGetTickCount();
    uint32_t tick = 0;

    while (1) {
        // Update timing
        state.time = (double)tick * CONTROL_LOOP_DT;
        state.dt = CONTROL_LOOP_DT;

        // Read sensor data
        read_sensors(&state);

        // Read control inputs
        read_inputs(&state);

        // Clear motor commands before user code
        state.motor_pwm[0] = 0;
        state.motor_pwm[1] = 0;
        state.motor_pwm[2] = 0;
        state.motor_pwm[3] = 0;

        // Call user control loop
        user_main_loop(&state);

        // Apply motor commands
        apply_motors(&state);

        tick++;

        // Maintain control loop timing
        vTaskDelayUntil(&lastWakeTime, M2T(CONTROL_PERIOD_MS));
    }
}

// =============================================================================
// App Entry Point
// =============================================================================

void appMain(void) {
    DEBUG_PRINT("User Control App built with Crazyflie firmware\n");

    // Create the app task
    xTaskCreate(appTask, "AppTask", APP_TASK_STACK_SIZE, NULL, APP_TASK_PRIORITY, NULL);
}

// =============================================================================
// Logging Configuration
// =============================================================================

LOG_GROUP_START(userCtrl)
LOG_ADD(LOG_FLOAT, gyroX, &log_gyro_x)
LOG_ADD(LOG_FLOAT, gyroY, &log_gyro_y)
LOG_ADD(LOG_FLOAT, gyroZ, &log_gyro_z)
LOG_ADD(LOG_FLOAT, accX, &log_accel_x)
LOG_ADD(LOG_FLOAT, accY, &log_accel_y)
LOG_ADD(LOG_FLOAT, accZ, &log_accel_z)
LOG_ADD(LOG_UINT16, pwmM1, &log_motor_pwm[0])
LOG_ADD(LOG_UINT16, pwmM2, &log_motor_pwm[1])
LOG_ADD(LOG_UINT16, pwmM3, &log_motor_pwm[2])
LOG_ADD(LOG_UINT16, pwmM4, &log_motor_pwm[3])
LOG_ADD(LOG_UINT8, armed, &log_is_armed)
LOG_ADD(LOG_UINT8, btnBlue, &log_button_blue)
LOG_GROUP_STOP(userCtrl)

// =============================================================================
// Parameter Configuration
// =============================================================================

PARAM_GROUP_START(userCtrl)
PARAM_ADD(PARAM_UINT8, armed, &param_armed)
PARAM_ADD(PARAM_UINT8, btnBlue, &param_button_blue)
PARAM_ADD(PARAM_UINT8, btnYellow, &param_button_yellow)
PARAM_ADD(PARAM_UINT8, btnGreen, &param_button_green)
PARAM_ADD(PARAM_UINT8, btnRed, &param_button_red)
PARAM_GROUP_STOP(userCtrl)

