# Hardware Lab 1: Programming the Crazyflie

In this lab you will set up the Crazyflie hardware, flash firmware, and run your simulation code on the real vehicle.

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
- Computer with the development environment set up

---

## 3.2 Software Setup

### 3.2.1 Install cflib

For communication with the Crazyflie:

```bash
uv pip install cflib
```

### 3.2.2 USB Permissions (Linux)

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

## 3.3 Crazyflie Firmware

### 3.3.1 Clone the Firmware

```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git ~/crazyflie-firmware
cd ~/crazyflie-firmware
```

### 3.3.2 Build Environment

Install the ARM toolchain:

```bash
# Ubuntu/Debian
sudo apt install gcc-arm-none-eabi

# macOS
brew install --cask gcc-arm-embedded
```

### 3.3.3 Understanding the Code Sharing

**Key insight**: Your `user_code.c` and `user_code.h` work in both simulation (SITL) and hardware (Crazyflie firmware). The API is the same:

| API | Simulation (SITL) | Hardware (Crazyflie) |
|-----|-------------------|----------------------|
| `user_main_loop(state)` | Called by SITL binary | Called by `app_main.c` |
| `state->sensors.gyro` | From Elodin physics | From BMI088 IMU |
| `user_set_all_motors()` | Sent via UDP to sim | Calls `motorsSetRatio()` |

No code changes required when moving from simulation to hardware!

---

## 3.4 Deploying Your Code

### 3.4.1 Use the Deploy Script

The `firmware/deploy.sh` script copies your code to the Crazyflie firmware (from repo root):

```bash
# Deploy to ~/crazyflie-firmware (default location)
./examples/crazyflie-edu/firmware/deploy.sh

# Or specify a custom firmware path
./examples/crazyflie-edu/firmware/deploy.sh /path/to/crazyflie-firmware
```

This copies:
- `user_code.c` → firmware app directory
- `user_code.h` → firmware app directory
- `firmware/app_main.c` → firmware app directory

### 3.4.2 Build the Firmware

```bash
cd ~/crazyflie-firmware

# Configure for the user control app
make app_user_control_defconfig

# Build
make -j
```

This produces `build/cf2.bin`.

### 3.4.3 Flash the Firmware

Put the Crazyflie in bootloader mode:
1. Power off the Crazyflie
2. Hold down the power button for 3+ seconds until blue LEDs flash alternately
3. The Crazyflie is now in DFU mode

Flash using cfloader:

```bash
# Install cfloader if needed
uv pip install cfloader

# Flash firmware via radio
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```

---

## 3.5 Testing with HITL Mode

Before flying, verify your code works with real sensors using HITL (Hardware-In-The-Loop) mode:

### 3.5.1 Connect to Crazyflie

1. Plug in the Crazyradio PA USB dongle
2. Power on the Crazyflie
3. Run the simulation in HITL mode (from repo root):

```bash
elodin editor examples/crazyflie-edu/main.py --hitl
```

### 3.5.2 What HITL Does

In HITL mode:
- **Real sensors**: Gyro and accel data come from the actual Crazyflie hardware
- **Keyboard controls**: Your Q/Shift/E/R/T keys send commands to the real drone
- **Visualization**: Elodin shows sensor data from the real world
- **Physics disabled**: No simulated physics - you're seeing real sensor noise!

### 3.5.3 Compare Sensors

With the Crazyflie stationary, compare sensor readings:

| Sensor | Simulation | HITL (Real) |
|--------|------------|-------------|
| Gyro X | ~0 | |
| Gyro Y | ~0 | |
| Gyro Z | ~0 | |
| Accel X | ~0 | |
| Accel Y | ~0 | |
| Accel Z | ~1g | |

Note any differences - real sensors have different noise characteristics!

---

## 3.6 Testing Without Propellers

### 3.6.1 Motor Test

1. **Remove all propellers**
2. Run HITL mode: `elodin editor examples/crazyflie-edu/main.py --hitl`
3. Press Q to arm
4. Hold Shift (blue button) - motors should spin
5. Release Shift - motors should stop immediately

### 3.6.2 Verify Safety

- [ ] Motors respond correctly to commands
- [ ] Motors stop immediately when button released
- [ ] Arming/disarming works correctly

---

## 3.7 Deliverables

### 1. Contributions
For each team member, describe their contribution.

### 2. Hardware Setup [30%]

a. Photo of your Crazyflie powered on

b. Screenshot showing successful firmware flash

c. Describe any issues encountered during setup

### 3. HITL Testing [40%]

a. Screenshot of Elodin in HITL mode showing sensor data

b. Compare sensor readings between simulation and HITL:

| Sensor | Simulation | HITL (Real) | Difference |
|--------|------------|-------------|------------|
| Gyro X (stationary) | | | |
| Gyro Y (stationary) | | | |
| Gyro Z (stationary) | | | |
| Accel Z (at rest) | | | |

c. Comment on any differences observed (noise, bias, etc.)

### 4. Motor Test [30%]

a. Video or photo evidence of motors responding to keyboard commands

b. Verify all safety checks pass

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
