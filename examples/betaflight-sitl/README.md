# Betaflight SITL Drone Simulation

This example demonstrates how to run a Software-In-The-Loop (SITL) drone simulation
using Elodin's physics engine with Betaflight's flight controller software.

## Overview

The simulation provides:
- **6-DOF Physics**: Rigid body dynamics with motor thrust, drag, and gravity
- **Betaflight Integration**: Real Betaflight flight controller running as SITL
- **UDP Communication**: Bidirectional sensor/motor data exchange
- **1kHz+ Update Rate**: High-frequency control loop support

```
┌─────────────────────┐        UDP        ┌─────────────────────┐
│   Elodin Physics    │◄─────────────────►│   Betaflight SITL   │
│                     │                    │                     │
│  • Rigid Body Sim   │   Port 9003: FDM   │  • Flight Control   │
│  • Motor Thrust     │   Port 9004: RC    │  • PID Loops        │
│  • Sensor Output    │   Port 9002: PWM   │  • Attitude Est.    │
│  • 3D Visualization │   Port 9001: RAW   │  • Motor Mixing     │
└─────────────────────┘                    └─────────────────────┘
```

## Quick Start

### Prerequisites

1. Nix development environment:
   ```bash
   nix develop
   ```

2. Python virtual environment with Elodin:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
   ```

### Build Betaflight SITL

```bash
cd examples/betaflight-sitl
./build.sh
```

This compiles the Betaflight firmware for SITL mode. The binary will be at:
`betaflight/obj/main/betaflight_SITL.elf`

### First-Time Setup: Configure Arming

**IMPORTANT**: Before running the simulation, you must configure an ARM switch in Betaflight.
This only needs to be done once - the config is saved to `eeprom.bin`.

1. **Start SITL** (in terminal 1):
   ```bash
   ./betaflight/obj/main/betaflight_SITL.elf
   ```
   Wait for `bind port 5761 for UART1` to appear.

2. **Connect to CLI** (in terminal 2):
   ```bash
   # Create virtual serial port
   socat -d -d pty,raw,echo=0,link=/tmp/bf tcp:localhost:5761 &
   sleep 2
   
   # Connect with screen
   screen /tmp/bf
   ```

3. **Configure ARM switch** (in screen session):
   ```
   #
   status
   aux 0 0 0 1700 2100 0 0
   save
   ```
   
   This sets ARM mode on AUX1 channel (activated when AUX1 > 1700).

4. **Exit screen**: Press `Ctrl+A` then `K` then `Y`

### Run the Simulation

**With 3D Editor:**
```bash
elodin editor examples/betaflight-sitl/main.py
```

**Headless (for testing):**
```bash
python3 examples/betaflight-sitl/main.py headless
```

### Quick Arming Test

Test that arming works correctly:
```bash
# Start SITL in one terminal
./examples/betaflight-sitl/betaflight/obj/main/betaflight_SITL.elf

# In another terminal, run the test
source .venv/bin/activate
python3 examples/betaflight-sitl/test_comms.py
```

Expected output when working:
```
Phase 2: Setting AUX1=1800 to ARM (2 seconds)...
  t=5.5s motors=[0.055 0.055 0.055 0.055]  # Motors at idle!
Phase 3: Raising throttle...
  t=7.2s motors=[0.402 0.402 0.402 0.402]  # Motors responding!
```

## Project Structure

```
examples/betaflight-sitl/
├── build.sh           # Build script for Betaflight SITL
├── main.py            # Main simulation entry point
├── config.py          # Drone physical parameters
├── sim.py             # Physics simulation systems
├── sensors.py         # IMU sensor simulation
├── comms.py           # UDP communication bridge
├── test_comms.py      # Standalone communication test
├── eeprom.bin         # Betaflight saved config (created on first run)
├── betaflight/        # Betaflight submodule
│   └── obj/main/
│       └── betaflight_SITL.elf  # Built binary
└── README.md          # This file
```

## Communication Protocol

### Simulator → Betaflight

**FDM Packet (Port 9003)**: Flight Dynamics Model data
- Timestamp (seconds)
- IMU angular velocity (rad/s, body frame)
- IMU linear acceleration (m/s², NED body frame)
- Orientation quaternion (w, x, y, z)
- Velocity (m/s, ENU world frame)
- Position (m, ENU world frame)
- Barometric pressure (Pa)

**RC Packet (Port 9004)**: Remote Control channels
- Timestamp
- 16 RC channels (PWM microseconds, 1000-2000)
- Channel mapping: [0]=Roll, [1]=Pitch, [2]=Throttle, [3]=Yaw, [4-15]=AUX1-12

### Betaflight → Simulator

**Servo Packet (Port 9002)**: Normalized motor outputs
- 4 motor values [0.0, 1.0]

**Servo Raw Packet (Port 9001)**: Raw PWM outputs
- Motor count
- 16 PWM values (1000-2000 microseconds)

## Arming Requirements

For the drone to ARM, Betaflight requires:

1. **BOOTGRACE period expired** (~5 seconds after boot)
2. **Not in CLI mode** (don't have screen/nc connected)
3. **ARM switch configured** (AUX1 in range 1700-2100)
4. **Throttle low** (< 1050)
5. **Gyro calibrated** (automatic after stable sensor data)

Check arming status via CLI:
```
# status
...
Arming disable flags: BOOTGRACE CLI  # These must clear before arming
```

## Coordinate Systems

The simulation handles coordinate frame conversions:

- **Elodin**: ENU (East-North-Up), X=forward, Y=left, Z=up
- **Betaflight Sensors**: NED body frame
- **Betaflight GPS**: ENU for position/velocity

Motor mapping for Betaflight Quad-X:
```
    Front
  1(CCW)  2(CW)
     \\  /
      \\/
      /\\
     /  \\
  4(CW)  3(CCW)
    Back

Betaflight: [0]=FR, [1]=BR, [2]=BL, [3]=FL
Elodin:     [0]=FR, [1]=FL, [2]=BR, [3]=BL
```

## Configuration

Edit `config.py` to modify drone parameters:

```python
DroneConfig(
    mass=0.8,                    # kg
    arm_length=0.12,             # meters
    motor_max_thrust=15.0,       # Newtons per motor
    motor_time_constant=0.02,    # seconds
    sim_time_step=0.001,         # 1kHz physics
)
```

Pre-configured drone types:
- `create_5inch_racing_quad()` - 5" racing quadcopter
- `create_3inch_cinewhoop()` - 3" cinewhoop
- `create_7inch_long_range()` - 7" long range quad

## Troubleshooting

### Build Failures

**`-fuse-linker-plugin not supported`**: The build script automatically handles
this macOS compatibility issue.

**Missing symbols**: The build includes stub implementations for SITL-specific
functions not available on macOS.

### SITL Crashes

**`malloc: pointer being freed was not allocated`**: Delete `eeprom.bin` and restart:
```bash
rm -f examples/betaflight-sitl/eeprom.bin
pkill -f betaflight_SITL
./betaflight/obj/main/betaflight_SITL.elf
```

**`bind port 5761 failed`**: Another SITL instance is running:
```bash
pkill -f betaflight_SITL
lsof -i :5761  # Should show nothing
```

### Motors Always Zero (Not Arming)

1. **Check SITL is receiving data**: Look for `[SITL] new fdm` and `[SITL] new rc` in output
2. **Verify AUX1 value**: Should show `AUX1-4: 1800 ...` when arming
3. **Wait for BOOTGRACE**: First 5 seconds after boot, arming is blocked
4. **Check CLI status**: Connect with socat/screen and run `status` to see disable flags
5. **Re-configure ARM switch**:
   ```
   aux 0 0 0 1700 2100 0 0
   save
   ```

### Connection Issues

**Betaflight not receiving data**: Check that ports 9001-9004 are not in use:
```bash
lsof -i :9003
```

**CLI won't connect**: Make sure SITL is running and port 5761 is bound:
```bash
lsof -i :5761
```

### Performance

**Simulation too slow**: The simulation runs faster than real-time by default.
For 10kHz control loops, ensure:
- `sim_time_step = 0.0001` (10kHz)
- Sufficient CPU for both Elodin and Betaflight

## Development

### Adding New Sensors

1. Define component type in `sensors.py`
2. Create computation system
3. Add to `create_sensor_system()`
4. Update `comms.py` to send sensor data

### Modifying Physics

1. Edit systems in `sim.py`
2. Adjust parameters in `config.py`
3. Test with `sim.py` standalone

### Betaflight Configuration

For custom Betaflight setups:
1. Connect via CLI (socat + screen method above)
2. Use CLI commands for settings
3. Save configuration (persists to eeprom.bin)

**Note**: The Betaflight Configurator GUI (10.10.0+) doesn't support direct TCP
connections on macOS. Use the CLI method described above.

## References

- [Betaflight Documentation](https://www.betaflight.com/docs)
- [Elodin Documentation](../../docs)
- [Betaflight SITL Wiki](https://betaflight.com/docs/wiki/guides/SITL)
