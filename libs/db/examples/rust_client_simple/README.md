# Simplified Rust Client for Elodin-DB

A streamlined Rust client example demonstrating how to connect to Elodin-DB and subscribe to rocket telemetry data.

## Quick Start

### Build
```bash
cargo build --release
```

### Run
```bash
# Make sure elodin-db and rocket.py are running first
cargo run --release
```

## What It Does

1. **Connects** to Elodin-DB using TCP
2. **Registers** rocket telemetry components
3. **Subscribes** to real-time data streams
4. **Demonstrates** the basic protocol structure

## Implementation Notes

This simplified version:
- Uses standard TCP sockets
- Implements basic Impeller2 packet structure
- Shows component registration process
- Demonstrates subscription flow

For a full implementation with complete packet processing, see the main `rust_client` example.

## Components Registered

- `rocket.mach` - Mach number
- `rocket.thrust` - Engine thrust
- `rocket.fin_deflect` - Fin deflection angle
- `rocket.angle_of_attack` - Angle of attack
- `rocket.dynamic_pressure` - Dynamic pressure
- `rocket.world_pos` - Position (quaternion + translation)
- `rocket.world_vel` - Velocity (spatial motion)
- `rocket.aero_force` - Aerodynamic forces  
- `rocket.center_of_gravity` - Center of gravity

## Dependencies

This example uses minimal dependencies:
- `tokio` - Async runtime
- `postcard` - Message serialization
- `colored` - Terminal colors
- `clap` - CLI parsing
