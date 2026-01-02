# Crazyflie Firmware Integration Research

## Overview

This document explores how students in drone control courses (like ME136 at UC Berkeley) interface with the Crazyflie firmware, and how one might build SITL (Software-In-The-Loop) simulation hooks similar to those provided for Betaflight.

---

## Table of Contents

1. [ME136 Course Architecture](#me136-course-architecture)
2. [Understanding the Crazyflie Ecosystem](#understanding-the-crazyflie-ecosystem)
3. [Firmware Integration Approaches](#firmware-integration-approaches)
4. [SITL Simulation Options](#sitl-simulation-options)
5. [Comparison with Betaflight SITL](#comparison-with-betaflight-sitl)
6. [Potential Elodin Integration Architecture](#potential-elodin-integration-architecture)
7. [References](#references)

---

## ME136 Course Architecture

Based on analysis of the ME136 course materials, the course uses a **dual-environment approach**:

### Simulation Labs (SimLab1, SimLab2)

The simulation environment consists of:

1. **Custom VirtualBox VM** containing:
   - Pre-compiled simulator binary (Green "S" icon)
   - Eclipse IDE for code editing
   - 3D Visualizer for viewing the simulated drone
   - Flight GUI for interaction (arming, button presses, vehicle manipulation)

2. **User Code Architecture**:
   ```
   Sim_UserCode/
   ├── UserCode.cpp      # Main control loop - students edit this
   ├── UtilityFunctions.cpp  # Powertrain constants (Lab 2)
   └── [Other supporting files]
   ```

3. **Key Interface**:
   ```cpp
   // MainLoop() called at 500 Hz
   void MainLoop() {
       // Read sensors (gyro, accel)
       // Check button states
       // Set motor commands
   }
   ```

### Hardware Labs (HWLab1, HWLab2)

The hardware environment uses:

1. **Custom quadcopter platform** (appears to use RP2040-based MCU):
   - UF2 bootloader (appears as "RPI-RP2" USB drive)
   - Flashing via USB or radio

2. **Same Code Structure**: The critical insight is that **the same `UserCode.cpp` runs on both the simulator and the hardware**, making the transition seamless.

3. **Communication Tools**:
   - Serial Terminal for debugging/shell access
   - Radio & Joystick program for RC control and telemetry

### Key ME136 Design Principles

| Principle | Implementation |
|-----------|----------------|
| Code Portability | Same C++ code runs on simulator and hardware |
| Abstracted Sensors | Sensor API identical in both environments |
| Abstracted Actuators | Motor commands work identically |
| Safety | Arm/disarm, dead man switch, motor limits |

---

## Understanding the Crazyflie Ecosystem

The Bitcraze Crazyflie has several key components relevant to educational use:

### Hardware

| Component | Details |
|-----------|---------|
| MCU | STM32F405 (168 MHz ARM Cortex-M4) |
| IMU | BMI088 (accelerometer + gyroscope) |
| Size | 27g, 92mm diagonal |
| Radio | nRF51822 for Crazyradio communication |
| Battery | 250mAh LiPo, ~7 minutes flight |

### Firmware Architecture

The [crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware) is modular:

```
crazyflie-firmware/
├── src/
│   ├── modules/
│   │   ├── src/
│   │   │   ├── controller/      # PID controllers
│   │   │   │   ├── controller_pid.c
│   │   │   │   ├── controller_mellinger.c
│   │   │   │   └── controller_indi.c
│   │   │   ├── commander.c      # Setpoint handling
│   │   │   ├── stabilizer.c     # Main control loop
│   │   │   └── estimator_*.c    # State estimation
│   │   └── interface/
│   ├── drivers/                 # Hardware drivers
│   └── hal/                     # Hardware abstraction
├── app_api/                     # App layer for custom code
└── Makefile
```

### Controller Interface

The firmware supports pluggable controllers:

```c
// Controller function signature
typedef void (*controller_t)(
    control_t *control,           // Output: motor commands
    setpoint_t *setpoint,         // Input: desired state
    const sensorData_t *sensors,  // Input: sensor readings
    const state_t *state,         // Input: estimated state
    const uint32_t tick           // Input: current tick
);
```

### Building Custom Controllers

Two approaches exist:

1. **In-tree modification**: Modify files directly in the firmware source
   ```bash
   git clone https://github.com/bitcraze/crazyflie-firmware.git
   cd crazyflie-firmware
   # Edit src/modules/src/controller/controller_pid.c
   make
   ```

2. **Out-of-tree app layer**: Build separate application code
   ```bash
   # In your app directory
   make CONTROLLER="Student"
   ```

---

## Firmware Integration Approaches

### Approach 1: Direct Firmware Modification

**How it works**: Students modify the Crazyflie firmware directly and flash it.

```c
// Example: Custom rate controller in controller_custom.c
void controllerCustom(
    control_t *control,
    setpoint_t *setpoint,
    const sensorData_t *sensors,
    const state_t *state,
    const uint32_t tick)
{
    // Custom PID logic here
    float error_roll = setpoint->attitude.roll - state->attitude.roll;
    // ... compute motor commands
    control->thrust = calculateThrust(...);
}
```

**Pros**:
- Full access to all firmware features
- Maximum flexibility

**Cons**:
- Steeper learning curve
- Risk of breaking critical code
- Harder to isolate student code

### Approach 2: App Layer (Recommended for Education)

**How it works**: Use the Crazyflie's app layer API to build isolated applications.

```c
// app_main.c
#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

void appMain() {
    while (1) {
        // Your control code here
        // Access sensors, set commands
        vTaskDelay(M2T(2)); // 500 Hz
    }
}
```

**Pros**:
- Clean separation from core firmware
- Easier build/flash cycle
- Safer for students

**Cons**:
- Some limitations on low-level access

### Approach 3: Python-Based Control (cflib)

**How it works**: Run control code on a PC, send commands over radio.

```python
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.commander import Commander

cf = Crazyflie()
cf.open_link('radio://0/80/2M')

# In control loop
cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)
```

**Pros**:
- No firmware modification required
- Python is more accessible for students
- Easy debugging

**Cons**:
- Radio latency (~10-20ms)
- Limited to ~100Hz effective control rate
- Not suitable for aggressive maneuvers

---

## SITL Simulation Options

Unlike Betaflight, the Crazyflie ecosystem doesn't have native SITL. Here are the available options:

### Option 1: CrazySim (Georgia Tech)

[CrazySim](https://github.com/gtfactslab/CrazySim) provides:

- Gazebo integration
- ROS 2 support via Crazyswarm2
- Actual firmware can run in simulation
- Multi-agent support

**Architecture**:
```
┌─────────────────┐     ROS 2      ┌─────────────────┐
│  Gazebo         │◄──────────────►│  Crazyflie      │
│  (Physics)      │                │  Firmware       │
└─────────────────┘                │  (Modified)     │
         │                         └─────────────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐
│  Sensor         │                │  Motor          │
│  Simulation     │                │  Commands       │
└─────────────────┘                └─────────────────┘
```

### Option 2: CrazyS (ROS/RotorS)

[CrazyS](https://github.com/gsilano/CrazyS) extends RotorS:

- More mature simulation
- Detailed aerodynamic models
- ROS 1 based (older)

### Option 3: Minimal Bitcraze Simulation

[crazyflie-simulation](https://github.com/bitcraze/crazyflie-simulation):

- Lightweight
- Not actively maintained
- Good for basic visualization

### Option 4: Custom SITL Implementation

This would involve:

1. Modifying firmware to accept simulated sensor data via UART/USB
2. Creating a physics simulation
3. Establishing communication protocol

---

## Comparison with Betaflight SITL

Betaflight's SITL provides a model for what Crazyflie SITL could look like:

### Betaflight SITL Architecture

```
┌──────────────────────┐        UDP        ┌──────────────────────┐
│   Physics Simulator  │◄─────────────────►│   Betaflight SITL    │
│   (Elodin/Gazebo)    │                   │   (Native Binary)    │
│                      │   Port 9003: FDM  │                      │
│  • Rigid Body Sim    │   Port 9004: RC   │  • Flight Control    │
│  • Motor Thrust      │   Port 9002: PWM  │  • PID Loops         │
│  • Sensor Output     │   Port 9001: RAW  │  • Attitude Est.     │
└──────────────────────┘                   └──────────────────────┘
```

### Key Betaflight SITL Features

| Feature | Description |
|---------|-------------|
| `SIMULATOR_GYROPID_SYNC` | Lockstep sync - blocks until FDM packet arrives |
| FDM Packets | Contain gyro, accel, position, velocity, baro |
| RC Packets | 16-channel RC input (throttle, roll, pitch, yaw, aux) |
| Motor Output | Normalized [0,1] or raw PWM |
| Native Build | Firmware compiles for Linux/Mac/Windows |

### What Crazyflie Lacks

1. **No native SITL target**: Firmware only compiles for STM32
2. **No UDP interface**: Would need to be added
3. **No sync mechanism**: No equivalent to GYROPID_SYNC
4. **No sensor injection**: Would need hooks added

---

## Potential Elodin Integration Architecture

Based on the analysis, here are possible architectures for Crazyflie + Elodin integration:

### Architecture A: Python Control Layer (Simplest)

**Description**: Python control code runs in Elodin, mirrors hardware API.

```
┌─────────────────────────────────────────────────────┐
│                    Elodin Editor                     │
├─────────────────────────────────────────────────────┤
│  user_code.py                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │  def main_loop(state: CrazyflieState):      │   │
│  │      if state.is_armed and state.button:    │   │
│  │          state.set_all_motors(50)           │   │
│  │                                              │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │  sim.py (Physics Simulation)                │   │
│  │  • 6-DOF rigid body dynamics               │   │
│  │  • Motor thrust model                       │   │
│  │  • Sensor simulation                        │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
                         │ (Code port for hardware)
                         ▼
┌─────────────────────────────────────────────────────┐
│  firmware/UserCode.c  (C equivalent)               │
│  • Same logic, C syntax                            │
│  • Flash to Crazyflie via Crazyradio               │
└─────────────────────────────────────────────────────┘
```

**This is what the existing `crazyflie-edu` example implements.**

### Architecture B: True SITL (Complex, Full Fidelity)

**Description**: Actual Crazyflie firmware runs with simulated sensors.

```
┌────────────────────┐     Shared Memory    ┌────────────────────┐
│    Elodin          │     or UDP           │  Crazyflie         │
│    Physics         │◄────────────────────►│  Firmware          │
│                    │                       │  (x86 build)       │
│  • World Sim       │   Sensors→           │                    │
│  • Collision       │   ←Motors            │  • Stabilizer      │
│  • Visualization   │                       │  • Controller      │
└────────────────────┘                       │  • Estimator       │
                                             └────────────────────┘
```

**Requirements**:
1. Modify Crazyflie firmware to build for x86/ARM host
2. Add sensor injection interfaces
3. Add motor output capture
4. Create synchronization mechanism

### Architecture C: Hardware-in-the-Loop (HIL)

**Description**: Physical Crazyflie MCU, simulated sensors.

```
┌────────────────────┐      USB/UART      ┌────────────────────┐
│    Elodin          │◄─────────────────►│  Crazyflie 2.1     │
│    Physics         │                    │  (Physical MCU)    │
│                    │   Sensors→         │                    │
│  • Sensor sim      │   ←Motors          │  • Real firmware   │
│  • Motor model     │                    │  • Real timing     │
└────────────────────┘                    └────────────────────┘
```

**Pros**: Tests actual firmware timing and behavior
**Cons**: Requires hardware, complex wiring

---

## Wireless Flashing via Crazyradio

The Crazyradio provides convenient wireless flashing:

### Prerequisites
- Crazyradio PA or Crazyradio 2.0
- `crazyflie-clients-python` installed
- Built firmware binary

### Manual Bootloader Method

```bash
# 1. Enter bootloader mode:
#    - Turn off Crazyflie
#    - Hold power button 3 seconds until blue LEDs blink

# 2. Flash firmware
make cload
```

### Automatic Bootloader Method

```bash
# Flash specific Crazyflie by URI
CLOAD_CMDS="-w radio://0/80/2M/E7E7E7E7E7" make cload
```

### Using cfloader Directly

```bash
# Flash with cfloader
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```

---

## Summary: How ME136 Students Interface with Firmware

Based on all research, here's the likely workflow:

### Phase 1: Simulation Development
1. Open Eclipse IDE in VM
2. Edit `UserCode.cpp` with control logic
3. Build with "All" target
4. Run simulator + 3D visualizer
5. Test with Flight GUI buttons
6. Analyze logs with Python/MATLAB

### Phase 2: Hardware Deployment
1. Connect Crazyradio
2. Build firmware for hardware target
3. Flash via `make cload` or USB
4. Run RadioAndJoystick program
5. Test with Xbox controller
6. Compare behavior to simulation

### Code Architecture (ME136 Style)

```cpp
// UserCode.cpp
#include "UserCode.h"

void MainLoop(sensorData_t* sensors, uint16_t* motorPwm) {
    // Read sensors
    float gyro_x = sensors->gyro.x;  // rad/s
    float accel_z = sensors->acc.z;  // g
    
    // Check buttons (set externally)
    if (isArmed && buttonBlue) {
        // Your control logic
        float thrust = calculateThrust(...);
        setAllMotors(motorPwm, thrust);
    } else {
        motorsOff(motorPwm);
    }
}
```

This code runs identically in:
- **Simulator**: Physics engine calls `MainLoop()` with simulated sensors
- **Hardware**: MCU firmware calls `MainLoop()` with real sensors

---

## References

### Bitcraze Documentation
- [Building and Flashing Firmware](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/)
- [Crazyflie Firmware GitHub](https://github.com/bitcraze/crazyflie-firmware)
- [Crazyradio Documentation](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyradio-2-0/)

### Simulation Projects
- [CrazySim (Georgia Tech)](https://github.com/gtfactslab/CrazySim)
- [CrazyS (ROS/RotorS)](https://github.com/gsilano/CrazyS)
- [Bitcraze crazyflie-simulation](https://github.com/bitcraze/crazyflie-simulation)

### Betaflight SITL
- [Betaflight SITL Wiki](https://betaflight.com/docs/wiki/guides/SITL)
- [Elodin Betaflight SITL Example](../../examples/betaflight-sitl/)

### Course Materials
- [ME136/236U UC Berkeley](https://undergraduate.catalog.berkeley.edu/courses/1155121)
- [Iowa State CPRE488 MP-4](https://class.ece.iastate.edu/cpre488/labs/MP-4.pdf)

---

## Appendix: Key Differences from ME136 VM

The ME136 course appears to use a **custom quadcopter platform** (not the standard Crazyflie 2.1):

| Feature | ME136 Custom | Crazyflie 2.1 |
|---------|--------------|---------------|
| MCU | RP2040 (based on "RPI-RP2" bootloader) | STM32F405 |
| Flashing | UF2 file to USB drive | `make cload` via Crazyradio |
| Mass | 32g (from lab docs) | 27g |
| Simulator | Custom binary | Third-party (CrazySim, etc.) |
| IDE | Eclipse (C++) | Any (VS Code typical) |

The **Elodin crazyflie-edu example** bridges this by:
1. Providing Python simulation with Elodin physics
2. Providing equivalent C firmware files for hardware
3. Matching the ME136 API (sensors, buttons, motor commands)
4. Targeting actual Crazyflie 2.1 hardware

This enables courses to use the modern Elodin editor instead of the VM-based approach.

