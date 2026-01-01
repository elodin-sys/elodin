# Hardware Lab 1: Programming the Crazyflie

In this lab you will set up the Crazyflie hardware, flash firmware, and port your simulation code to run on the real vehicle.

> ⚠️ **Safety First**: The Crazyflie has spinning propellers that can cause injury. Always:
> - Keep propellers away from faces and fingers
> - Have a clear takeoff area
> - Know how to disarm (power off) quickly
> - Start with propellers removed for initial testing

**Read the "Deliverables" section before starting work.**

---

## 3.1 Hardware Overview

### 3.1.1 Crazyflie 2.1 Components

The Crazyflie 2.1 includes:
- **STM32F405** microcontroller (168 MHz ARM Cortex-M4)
- **BMI088** 6-axis IMU (accelerometer + gyroscope)
- **BMP388** barometric pressure sensor
- **nRF51822** radio (2.4 GHz, for communication with Crazyradio)
- **4x coreless DC motors** (7x16mm)
- **4x 45mm propellers**

### 3.1.2 Required Equipment

- Crazyflie 2.1 quadcopter
- Crazyradio PA USB dongle
- Micro-USB cable (for charging/flashing)
- Computer with Python 3.8+

---

## 3.2 Software Setup

### 3.2.1 Install the Crazyflie Client

The `cfclient` provides a GUI for connecting to and controlling the Crazyflie:

```bash
pip install cfclient
```

Or use the system package manager:
```bash
# Ubuntu/Debian
sudo apt install python3-cfclient

# macOS
brew install --cask crazyflie-client
```

### 3.2.2 Install cflib (Python Library)

For programmatic control:

```bash
pip install cflib
```

### 3.2.3 USB Permissions (Linux)

On Linux, add udev rules for the Crazyradio:

```bash
# Create udev rules
sudo tee /etc/udev/rules.d/99-crazyflie.rules << 'EOF'
# Crazyradio PA
SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="7777", MODE="0664", GROUP="plugdev"
# Crazyflie 2.x (USB bootloader)
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0664", GROUP="plugdev"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add yourself to plugdev group
sudo usermod -a -G plugdev $USER
# Log out and back in for group change to take effect
```

---

## 3.3 Connecting to the Crazyflie

### 3.3.1 Power On

1. Insert a charged battery into the Crazyflie
2. The blue LEDs should blink during startup
3. When ready, the front-left LED shows battery status

### 3.3.2 Using cfclient

1. Plug in the Crazyradio PA USB dongle
2. Launch cfclient: `cfclient`
3. Click "Scan" to find your Crazyflie
4. Select it and click "Connect"

You should see:
- Green connection indicator
- Sensor data updating
- Battery voltage displayed

### 3.3.3 Basic Controls

With a gamepad connected:
- Left stick: Throttle (up/down) and Yaw (left/right)
- Right stick: Pitch (up/down) and Roll (left/right)

**Test without propellers first!**

---

## 3.4 Crazyflie Firmware

### 3.4.1 Firmware Overview

The Crazyflie runs open-source firmware written in C:
- Repository: https://github.com/bitcraze/crazyflie-firmware

The firmware structure:
```
crazyflie-firmware/
├── src/
│   ├── modules/        # Core modules (estimator, controller, etc.)
│   ├── drivers/        # Hardware drivers
│   ├── hal/            # Hardware abstraction layer
│   └── deck/           # Expansion deck drivers
├── Makefile
└── ...
```

### 3.4.2 Clone the Firmware

```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
cd crazyflie-firmware
```

### 3.4.3 Build Environment

Install the ARM toolchain:

```bash
# Ubuntu/Debian
sudo apt install gcc-arm-none-eabi

# macOS
brew install --cask gcc-arm-embedded
```

Build the firmware:

```bash
make
```

This produces `cf2.bin` in the build directory.

### 3.4.4 Flash Firmware

Put the Crazyflie in bootloader mode:
1. Power off the Crazyflie
2. Hold down the power button for 3+ seconds until blue LEDs flash alternately
3. The Crazyflie is now in DFU mode

Flash using cfloader:

```bash
# Install cfloader
pip install cfloader

# Flash firmware
cfloader flash cf2.bin stm32-fw
```

Or use cfclient: Connect → Bootloader → Flash new firmware

---

## 3.5 Porting Your Code to C

### 3.5.1 Code Mapping

Your Python simulation code maps directly to C firmware code:

| Python (user_code.py) | C (UserCode.c) |
|-----------------------|----------------|
| `state.gyro[0]` | `sensorData.gyro.x` |
| `state.accel[0]` | `sensorData.acc.x` |
| `state.set_motors(m1,m2,m3,m4)` | `motorsSetRatio(...)` |
| `state.button_blue` | `buttonGetPressed(BUTTON_BLUE)` |

### 3.5.2 Create Your User Code

In the firmware directory, create your control module. Start with the template in `firmware/UserCode.c`:

```c
/**
 * UserCode.c - Student Control Implementation
 *
 * This is the C equivalent of user_code.py from simulation.
 */

#include "UserCode.h"
#include "motors.h"
#include "sensors.h"
#include "param.h"
#include "log.h"

// Team ID - set your team number
static uint8_t teamId = 0;

// Motor command limit for safety
#define MOTOR_TEST_LIMIT 10000  // PWM range is 0-65535

// Powertrain constants (from SimLab2)
static float PWM_TO_SPEED_A = 426.0f;    // rad/s at PWM=0
static float PWM_TO_SPEED_B = 6.8f;      // rad/s per PWM unit
static float THRUST_CONSTANT = 1.8e-8f;  // N/(rad/s)^2

/**
 * Convert desired motor speed (rad/s) to PWM command.
 */
static uint16_t pwmFromSpeed(float desiredSpeed) {
    float pwm = (desiredSpeed - PWM_TO_SPEED_A) / PWM_TO_SPEED_B;
    if (pwm < 0) return 0;
    if (pwm > 255) return 255;
    // Scale to 16-bit PWM (0-65535)
    return (uint16_t)(pwm * 257.0f);
}

/**
 * Convert desired thrust (N) to motor speed (rad/s).
 */
static float speedFromForce(float desiredForce) {
    if (desiredForce <= 0) return 0.0f;
    return sqrtf(desiredForce / THRUST_CONSTANT);
}

/**
 * Main control loop - called at 500 Hz
 */
void userCodeMainLoop(sensorData_t* sensors, uint16_t* motorPwm) {
    // Read gyroscope (rad/s)
    float gyroX = sensors->gyro.x;
    float gyroY = sensors->gyro.y;
    float gyroZ = sensors->gyro.z;

    // Read accelerometer (g units)
    float accelX = sensors->acc.x;
    float accelY = sensors->acc.y;
    float accelZ = sensors->acc.z;

    // =========================================
    // YOUR CODE HERE
    // =========================================

    // Example: Turn on motors when blue button pressed
    // if (buttonGetPressed(BUTTON_BLUE)) {
    //     uint16_t cmd = MOTOR_TEST_LIMIT;
    //     motorPwm[0] = cmd;
    //     motorPwm[1] = cmd;
    //     motorPwm[2] = cmd;
    //     motorPwm[3] = cmd;
    // } else {
    //     motorPwm[0] = 0;
    //     motorPwm[1] = 0;
    //     motorPwm[2] = 0;
    //     motorPwm[3] = 0;
    // }

    // Default: motors off
    motorPwm[0] = 0;
    motorPwm[1] = 0;
    motorPwm[2] = 0;
    motorPwm[3] = 0;
}

/**
 * Print status - called when requested
 */
void userCodePrintStatus(void) {
    DEBUG_PRINT("Team ID: %d\n", teamId);
    DEBUG_PRINT("PWM_TO_SPEED_A: %.2f\n", PWM_TO_SPEED_A);
    DEBUG_PRINT("PWM_TO_SPEED_B: %.4f\n", PWM_TO_SPEED_B);
    DEBUG_PRINT("THRUST_CONSTANT: %.2e\n", THRUST_CONSTANT);
}

// Parameter declarations for tuning via cfclient
PARAM_GROUP_START(userCode)
PARAM_ADD(PARAM_UINT8, teamId, &teamId)
PARAM_ADD(PARAM_FLOAT, pwmToSpeedA, &PWM_TO_SPEED_A)
PARAM_ADD(PARAM_FLOAT, pwmToSpeedB, &PWM_TO_SPEED_B)
PARAM_ADD(PARAM_FLOAT, thrustConst, &THRUST_CONSTANT)
PARAM_GROUP_STOP(userCode)
```

### 3.5.3 Integrate with Firmware

1. Copy your `UserCode.c` and `UserCode.h` to `src/modules/src/`
2. Add to `Makefile` or `Kbuild`
3. Hook into the main loop (modify `stabilizer.c` or create an app)

For a simpler approach, use the **App Layer**:

```c
// app_usercode.c
#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

void appMain() {
    while (1) {
        // Your code here
        vTaskDelay(M2T(2));  // 500 Hz
    }
}
```

---

## 3.6 Testing Without Propellers

### 3.6.1 Motor Test

1. **Remove all propellers**
2. Connect via cfclient
3. In the Parameters tab, find your `userCode` parameters
4. Verify your Team ID is set

5. Enable motor test mode (without arming):
   - Your code should command motors when button pressed
   - Observe motor spinning with fingers safely away

### 3.6.2 Sensor Verification

Use cfclient's Flight Data tab to observe:
- Gyroscope readings (should be near zero when stationary)
- Accelerometer readings (should show ~1g downward)

Compare with your simulation values.

---

## 3.7 Deliverables

### 1. Contributions
For each team member, describe their contribution.

### 2. Hardware Setup [30%]

a. Photo of your Crazyflie connected and powered on

b. Screenshot of cfclient showing connection

c. Describe any issues encountered during setup

### 3. Firmware Development [40%]

a. Provide the complete listing of your `UserCode.c` file

b. Describe the process of porting from Python to C

c. What changes were required beyond syntax?

### 4. Motor Test [30%]

a. Video or photo evidence of motors responding to commands

b. Compare sensor readings between simulation and hardware

| Sensor | Simulation | Hardware |
|--------|------------|----------|
| Gyro X (stationary) | | |
| Gyro Y (stationary) | | |
| Gyro Z (stationary) | | |
| Accel Z (at rest) | | |

c. Comment on any differences observed

---

## 3.8 Safety Checklist

Before proceeding to HWLab2 (with propellers):

- [ ] Motors respond correctly to commands
- [ ] Motors stop immediately when button released
- [ ] Battery voltage is adequate
- [ ] Propellers are undamaged
- [ ] You have a clear, safe flying area
- [ ] You know how to emergency stop (power off)

---

## 3.9 Next Steps

In **HWLab2**, you will:
- Validate your powertrain model with real measurements
- Attach propellers and test thrust
- Achieve stable hover using your identified parameters

**Warning**: HWLab2 involves spinning propellers. Take all safety precautions seriously.

