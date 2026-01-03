# Crazyflie Educational Labs

An Elodin-based learning experience for quadcopter dynamics and control, inspired by UC Berkeley's ME136/236U course taught by Professor Mark Mueller.

## Key Feature: Write C Once, Run Everywhere

The same C control code runs in:
1. **SITL Simulation** - Test with full physics before touching hardware
2. **Crazyflie Hardware** - Flash and fly with no code changes

```c
// user_code.c - Works in SITL and on real hardware!
void user_main_loop(user_state_t* state) {
    if (state->is_armed && state->button_blue) {
        user_set_all_motors(state, 10000);  // PWM 0-65535
    } else {
        user_motors_off(state);
    }
}
```

## Quick Start

### Prerequisites

```bash
# Enter Nix development environment (from repository root)
nix develop

# Set up Python environment with Elodin
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml

# Install keyboard input support
uv pip install pynput
```

### Build and Run SITL

```bash
# Build the C SITL binary (first time only, from repo root)
./examples/crazyflie-edu/sitl/build.sh

# Run simulation with 3D visualization
elodin editor examples/crazyflie-edu/main.py

# Or run headless (faster)
elodin run examples/crazyflie-edu/main.py
```

### Controls

| Key | Action |
|-----|--------|
| Q | Toggle armed state |
| Left Shift | Blue button (hold for motors) |
| E / R / T | Yellow / Green / Red buttons |
| Space | Play/Pause (editor mode) |

## Project Structure

```
examples/crazyflie-edu/
├── user_code.c          # 👈 YOUR CODE GOES HERE (C)
├── user_code.h          # Student API header
├── sitl/
│   ├── sitl_main.c      # SITL wrapper (UDP comms)
│   ├── build.sh         # Build script
│   └── Makefile         # For make users
├── firmware/
│   ├── app_main.c       # Crazyflie app layer entry
│   └── deploy.sh        # Deploy to crazyflie-firmware
├── main.py              # Elodin physics simulation + HITL
├── keyboard_controller.py  # Keyboard input handling
├── config.py            # Crazyflie physical parameters
├── sim.py               # 6-DOF physics
├── sensors.py           # IMU simulation
└── labs/                # Lab instructions
```

## Writing Control Code

Edit `user_code.c` to implement your control logic:

```c
#include "user_code.h"
#include <math.h>

// Your control code runs at 500 Hz
void user_main_loop(user_state_t* state) {
    // Read sensors
    float gyro_x = state->sensors.gyro.x;  // rad/s
    float accel_z = state->sensors.accel.z;  // g units
    
    // Check armed state and blue button (dead man switch)
    if (state->is_armed && state->button_blue) {
        // Set motor PWM (0-65535)
        user_set_all_motors(state, 10000);
    } else {
        user_motors_off(state);
    }
}
```

Available in `user_state_t`:
- `state->sensors.gyro.x/y/z` - Angular velocity (rad/s)
- `state->sensors.accel.x/y/z` - Acceleration (g units)
- `state->is_armed` - Vehicle armed state
- `state->button_blue/yellow/green/red` - Button states
- `state->time` - Current time (seconds)
- `state->dt` - Time step (0.002s at 500Hz)

Motor control functions:
- `user_set_all_motors(state, pwm)` - Set all motors to same PWM
- `user_set_motors(state, m1, m2, m3, m4)` - Set individual motors
- `user_motors_off(state)` - Turn all motors off

## Deploying to Hardware

### 1. Test in Simulation First

```bash
# Build and run SITL (from repo root)
./examples/crazyflie-edu/sitl/build.sh
elodin run examples/crazyflie-edu/main.py

# Verify your control code works:
# - Press Q to arm
# - Hold Shift (blue button) to spin motors
# - Check motor response in terminal output
```

### 2. Deploy to Crazyflie Firmware

```bash
# Clone crazyflie-firmware (if not already done)
git clone https://github.com/bitcraze/crazyflie-firmware.git ~/crazyflie-firmware
cd ~/crazyflie-firmware
git submodule update --init --recursive

# Deploy your code (from repo root)
./examples/crazyflie-edu/firmware/deploy.sh ~/crazyflie-firmware
```

### 3. Build and Flash

```bash
cd ~/crazyflie-firmware

# Build the firmware
make app_user_control_defconfig
make -j

# Flash via Crazyradio
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```

### 4. Test with HITL Mode

Once your code is flashed on the Crazyflie, use HITL mode to control it with the same keyboard interface:

```bash
# Install cflib for Crazyradio communication
uv pip install cflib

# Run in HITL mode - connects to real hardware!
elodin editor examples/crazyflie-edu/main.py --hitl
```

This provides the **exact same experience** as SITL simulation:
- Same keyboard controls (Q to arm, Shift for blue button)
- Real sensor data visualized in Elodin Editor
- Your user_code.c runs on the actual Crazyflie hardware

The `--hitl` flag:
1. Connects to the Crazyflie via Crazyradio
2. Sends keyboard inputs as Crazyflie parameters
3. Streams real IMU data for visualization
4. Disables physics simulation (real world is source of truth)

## Lab Sequence

### SimLab1: System Setup and Sensor Analysis
- Run the Elodin simulation
- Write code to turn motors on/off
- Analyze gyroscope and accelerometer data

### SimLab2: Powertrain Identification
- Determine PWM-to-speed relationship
- Determine speed-to-force relationship
- Implement utility functions

### HWLab1: Hardware Programming
- Set up Crazyflie hardware
- Deploy firmware from simulation code
- Test motors without propellers

### HWLab2: Hardware Validation
- Validate thrust model
- Achieve stable hover
- Compare simulation vs hardware

## Safety

⚠️ **Important Safety Guidelines:**

1. **Simulation first** - Fully test in SITL before hardware
2. **No propellers initially** - Test motor response without props
3. **Safety glasses** - Required during propeller tests
4. **Clear area** - 2m minimum radius for flight tests
5. **Tether** - Use a string tether for initial hover tests
6. **Low power** - Start with low motor commands
7. **Emergency stop** - Know how to disarm quickly (set armed=0)

## Target Hardware

**Bitcraze Crazyflie 2.1**

| Specification | Value |
|--------------|-------|
| Mass | 27 g |
| Diagonal | 92 mm |
| MCU | STM32F405 (168 MHz) |
| IMU | BMI088 (accel + gyro) |
| Motors | 7x16mm coreless DC |
| Props | 45mm |
| Battery | 250 mAh LiPo |

## Resources

- [Bitcraze Crazyflie 2.1](https://store.bitcraze.io/products/crazyflie-2-1)
- [Crazyflie Firmware](https://github.com/bitcraze/crazyflie-firmware)
- [Elodin Documentation](https://docs.elodin.systems)

## License

This educational example is part of the Elodin project. See the main repository LICENSE file.
