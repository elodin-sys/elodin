# Crazyflie Educational Labs

An Elodin-based learning experience for quadcopter dynamics and control, inspired by UC Berkeley's ME136/236U course taught by Professor Mark Mueller.

## Overview

This example provides a 4-lab educational sequence that takes students from simulation to real hardware flight:

| Lab | Name | Focus |
|-----|------|-------|
| SimLab1 | System Setup | Elodin simulation, motors, sensor analysis |
| SimLab2 | Powertrain Identification | PWM→speed→force mapping |
| HWLab1 | Hardware Programming | Crazyflie firmware, code porting |
| HWLab2 | Hardware Validation | Real-world powertrain testing, hover |

## Quick Start

### Prerequisites

From the repository root:

```bash
# Enter Nix development environment
nix develop

# Set up Python environment with Elodin
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
```

### Run Simulation

```bash
elodin editor examples/crazyflie-edu/main.py
```

This opens the Elodin editor with:
- 3D viewport showing the Crazyflie
- Sensor graphs (gyroscope, accelerometer)
- Motor telemetry graphs

### Controls

| Key | Action |
|-----|--------|
| Space | Play/Pause simulation |
| R | Reset simulation |
| Left-click + drag | Rotate camera |
| Scroll | Zoom |

## Project Structure

```
examples/crazyflie-edu/
├── main.py              # Entry point for Elodin
├── config.py            # Crazyflie 2.1 physical parameters
├── sim.py               # 6-DOF physics simulation
├── sensors.py           # IMU sensor simulation with noise
├── crazyflie_api.py     # Firmware-like API
├── user_code.py         # 👈 YOUR CODE GOES HERE
├── labs/
│   ├── simlab1.md       # Lab 1: System setup
│   ├── simlab2.md       # Lab 2: Powertrain ID
│   ├── hwlab1.md        # Lab 3: Hardware programming
│   └── hwlab2.md        # Lab 4: Hardware validation
├── firmware/
│   ├── UserCode.c       # C equivalent of user_code.py
│   ├── UserCode.h       # Header file
│   └── UtilityFunctions.c
└── analysis/
    └── (data analysis scripts)
```

## Writing Control Code

Edit `user_code.py` to implement your control logic:

```python
def main_loop(state: CrazyflieState) -> None:
    """Called every control cycle (500 Hz)."""
    
    # Read sensors
    gyro_x = state.gyro[0]  # rad/s
    accel_z = state.accel[2]  # g units
    
    # Check buttons
    if state.is_armed and state.button_blue:
        # Set motor commands (0-255)
        state.set_all_motors(50)
    else:
        state.motors_off()
```

## Lab Sequence

### SimLab1: System Setup and Sensor Analysis

**Goals:**
- Run the Elodin simulation
- Write code to turn motors on/off
- Analyze gyroscope and accelerometer data
- Compare sensor noise enabled vs disabled

[Full instructions: labs/simlab1.md](labs/simlab1.md)

### SimLab2: Powertrain Identification

**Goals:**
- Determine PWM-to-speed relationship (affine fit)
- Determine speed-to-force relationship (quadratic fit)
- Implement utility functions
- Validate with 90%/110% hover thrust tests

[Full instructions: labs/simlab2.md](labs/simlab2.md)

### HWLab1: Hardware Programming

**Goals:**
- Set up Crazyflie hardware and radio
- Install cfclient and cflib
- Port Python code to C firmware
- Test motors without propellers

[Full instructions: labs/hwlab1.md](labs/hwlab1.md)

### HWLab2: Hardware Validation

**Goals:**
- Validate PWM-to-speed with tachometer
- Validate thrust model with force measurements
- Achieve stable hover with propellers
- Compare simulation vs hardware

[Full instructions: labs/hwlab2.md](labs/hwlab2.md)

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

## Safety

⚠️ **Important Safety Guidelines:**

1. **Simulation first** - Fully test in simulation before hardware
2. **No propellers initially** - Test motor response without props
3. **Safety glasses** - Required during propeller tests
4. **Clear area** - 2m minimum radius for flight tests
5. **Tether** - Use a string tether for initial hover tests
6. **Low power** - Start with low motor commands (≤50)
7. **Emergency stop** - Know how to disarm quickly

## Comparison with ME136

This example is inspired by UC Berkeley's ME136 course, with key differences:

| Aspect | ME136 | Elodin Labs |
|--------|-------|-------------|
| Setup | VirtualBox VM | `elodin editor main.py` |
| Language | C++ (Eclipse) | Python → C |
| Simulator | Custom binary | Elodin physics engine |
| Hardware | Crazyflie 2.1 | Crazyflie 2.1 |
| 3D View | Custom visualizer | Elodin Editor |

## Resources

- [Bitcraze Crazyflie 2.1](https://store.bitcraze.io/products/crazyflie-2-1)
- [Crazyflie Firmware](https://github.com/bitcraze/crazyflie-firmware)
- [Elodin Documentation](https://docs.elodin.systems)
- [ME136 Course Info](https://undergraduate.catalog.berkeley.edu/courses/1155121)

## License

This educational example is part of the Elodin project. See the main repository LICENSE file.

