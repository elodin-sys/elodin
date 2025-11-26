# Avatar OSD Service

MSP DisplayPort OSD service for Walksnail Avatar VTX that connects to Elodin-DB for telemetry data.

## Features

- Real-time telemetry subscription from Elodin-DB
- MSP DisplayPort protocol implementation for Walksnail Avatar
- Debug terminal backend for local development
- Configurable OSD layout with:
  - Compass heading with tick marks
  - Altitude ladder and climb rate
  - Speed indicator
  - Artificial horizon with pitch/roll
  - System status and warnings

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

For actual hardware connection to Walksnail Avatar:

```bash
# Using configuration file
cargo run -- --mode serial

# Override serial port
cargo run -- --mode serial --serial-port /dev/ttyUSB0

# Custom configuration file
cargo run -- --mode serial --config custom-config.toml
```

## Configuration

Edit `config.toml` to configure:

- Database connection (host, port, components to subscribe)
- OSD grid dimensions and refresh rate
- Serial port settings for MSP DisplayPort

Default configuration subscribes to drone telemetry components:
- `drone.gyro` - Angular velocity
- `drone.accel` - Linear acceleration
- `drone.magnetometer` - Magnetic field
- `drone.world_pos` - Position (quaternion + xyz)
- `drone.world_vel` - Velocity (angular + linear)
- `drone.body_ang_vel` - Body-frame angular velocity
- `drone.attitude_target` - Target attitude

**Note:** Component names must match exactly as registered in Elodin-DB (including entity prefix).

## Testing with Drone Simulation

1. Start the drone example simulation:
   ```bash
   cd examples/drone
   python main.py run
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

- `db_client.rs` - Connects to Elodin-DB and subscribes to telemetry
- `telemetry.rs` - Processes and aggregates telemetry state
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

## Development Notes

- The service currently processes basic telemetry components
- Full decomponentization of table data from Elodin-DB is planned
- Additional OSD elements can be added to `layout.rs`
- The artificial horizon uses ASCII art for terminal display
