# MSP OSD Service

MSP DisplayPort OSD service that connects to Elodin-DB for telemetry data. Supports Walksnail Avatar VTX and other MSP-compatible systems.

## Features

- Real-time telemetry subscription from Elodin-DB
- **Declarative input mappings** - configure how to extract telemetry from any Elodin-DB components
- MSP DisplayPort protocol implementation
- Debug terminal backend for local development
- OSD elements:
  - Compass heading with tick marks
  - Altitude ladder and climb rate
  - Speed indicator
  - Artificial horizon with pitch/roll

## Core Inputs

The OSD requires three inputs in world frame, all configured via `config.toml`:

1. **Position** (x, y, z) - Used for altitude display
2. **Orientation** (q0, q1, q2, q3) - Quaternion for horizon and compass
3. **Velocity** (x, y, z) - Used for speed and climb rate

## Configuration

The `config.toml` file uses declarative input mappings to specify how to extract these values from Elodin-DB components.

### BDX RC Jet Example Configuration

For the bdx simulation, `world_pos` contains `[q0, q1, q2, q3, x, y, z]` and `world_vel` contains `[ωx, ωy, ωz, vx, vy, vz]`:

```toml
[db]
host = "127.0.0.1"
port = 2240

[osd]
rows = 18
cols = 50
refresh_rate_hz = 20.0

[serial]
port = "/dev/ttyTHS7"
baud = 115200

# Position from world_pos indices 4,5,6
[inputs.position]
component = "bdx.world_pos"
x = 4
y = 5
z = 6

# Orientation from world_pos indices 0,1,2,3
[inputs.orientation]
component = "bdx.world_pos"
q0 = 0
q1 = 1
q2 = 2
q3 = 3

# Velocity from world_vel indices 3,4,5
[inputs.velocity]
component = "bdx.world_vel"
x = 3
y = 4
z = 5
```

### Satellite Example

For a satellite with separate position and attitude components:

```toml
[inputs.position]
component = "satellite.position"
x = 0
y = 1
z = 2

[inputs.orientation]
component = "satellite.attitude"
q0 = 0
q1 = 1
q2 = 2
q3 = 3

[inputs.velocity]
component = "satellite.velocity"
x = 0
y = 1
z = 2
```

## Usage

### Debug Mode (Terminal Display)

Run locally for development and testing:

```bash
# Using default configuration
cargo run -- --mode debug

# With custom DB address
cargo run -- --mode debug --db-addr 127.0.0.1:2240

# Verbose logging
cargo run -- --mode debug --verbose
```

### Serial Mode (MSP DisplayPort)

For actual hardware connection:

```bash
# Using configuration file
cargo run -- --mode serial

# Override serial port
cargo run -- --mode serial --serial-port /dev/ttyUSB0
```

## Testing with Drone Simulation

1. Start the bdx example simulation:
   ```bash
   elodin editor examples/rc-jet/main.py
   ```

2. In another terminal, run the OSD in debug mode:
   ```bash
   cd fsw/msp-osd
   cargo run -- --mode debug
   ```

You should see the OSD layout updating in your terminal with simulated telemetry.

## Hardware Setup

For Walksnail Avatar connection on Aleph:

1. Wire UART7 on the Orin NX to the Walksnail Avatar VTX:
   - Orin TX7 → Walksnail RX
   - Orin RX7 → Walksnail TX
   - GND ↔ GND

2. Configure Walksnail VTX for "MSP OSD" mode

3. Run the service with serial mode:
   ```bash
   cargo run -- --mode serial --serial-port /dev/ttyTHS7
   ```

## Architecture

- `config.rs` - Declarative input mapping configuration
- `db_client.rs` - Generic Elodin-DB client that extracts values based on config
- `telemetry.rs` - Core state: position, orientation, velocity
- `osd_grid.rs` - Text grid representation for OSD
- `layout.rs` - Renders telemetry into OSD elements
- `backends/terminal.rs` - Terminal display for debugging
- `backends/displayport.rs` - MSP DisplayPort protocol implementation

## MSP DisplayPort Protocol

Implements MSP v1 with DisplayPort extensions:
- `MSP_DISPLAYPORT` (ID 182) with subcommands:
  - `MSP_DP_HEARTBEAT` - Keep connection alive
  - `MSP_DP_CLEAR_SCREEN` - Clear OSD canvas
  - `MSP_DP_WRITE_STRING` - Write text at position
  - `MSP_DP_DRAW_SCREEN` - Present frame
