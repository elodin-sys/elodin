# Rust Client Example for Elodin-DB

## Overview

This document outlines the design and implementation plan for a Rust client example that subscribes to and consumes telemetry data from the Nox-Py rocket.py simulation through the Elodin-DB.

## Goals

1. **Demonstrate Rust API Usage**: Create a clear example of how to use the Rust ecosystem to interact with Elodin-DB
2. **Subscribe to Rocket Telemetry**: Connect to a running rocket.py simulation and consume its telemetry data
3. **Educational Value**: Provide well-documented code that serves as a reference for users building their own Rust clients
4. **Type Safety**: Leverage Rust's type system to create a robust client with compile-time guarantees

## Architecture Analysis

### Existing Examples Reference

#### C Client (`examples/client.c`)
- Direct socket connection using TCP
- Manual packet header construction
- Raw buffer management
- Demonstrates basic streaming pattern

#### C++ Client (`examples/client.cpp`)
- Uses helper classes for Socket management
- VTable builder pattern for schema definition
- Component metadata registration
- Separate reader thread for subscriptions
- Template-based field definitions for type safety

### Key Libraries and Dependencies

#### From Elodin-DB Ecosystem
- **`impeller2`**: Core protocol types (`LenPacket`, `VTable`, `ComponentId`, etc.)
- **`impeller2-stellar`**: TCP client implementation with async support
- **`impeller2-wkt`**: Well-known types (messages like `Stream`, `GetTimeSeries`)
- **`stellarator`**: Async runtime and networking utilities
- **`postcard`**: Serialization for message payloads

#### Rocket.py Component Analysis

The rocket simulation creates the following components under the "rocket" namespace:

```
rocket.angle_of_attack       → f64[1]     # Angle of attack
rocket.aero_coefs           → f64[6]     # Aerodynamic coefficients
rocket.center_of_gravity    → f64        # Center of gravity
rocket.mach                 → f64        # Mach number
rocket.dynamic_pressure     → f64        # Dynamic pressure
rocket.aero_force           → f64[12]    # Spatial force (6 linear + 6 angular)
rocket.wind                 → f64[3]     # Wind vector
rocket.motor                → f64        # Motor force
rocket.fin_deflect          → f64        # Fin deflection angle
rocket.fin_control          → f64        # Fin control input
rocket.v_rel_accel          → f64[3]     # Relative velocity acceleration
rocket.v_rel_accel_filtered → f64[3]     # Filtered acceleration
rocket.pitch_pid            → f64[3]     # PID controller gains
rocket.pitch_pid_state      → f64[3]     # PID controller state
rocket.accel_setpoint       → f64[2]     # Acceleration setpoint
rocket.accel_setpoint_smooth→ f64[2]     # Smoothed setpoint
rocket.thrust               → f64        # Thrust force
rocket.world_pos            → f64[7]     # World position (quaternion + translation)
rocket.world_vel            → f64[12]    # World velocity (6 linear + 6 angular)
rocket.inertia              → f64[10]    # Inertia tensor
```

## Design Decisions

### 1. Architecture Pattern
**Decision**: Use a **subscription-based consumer pattern** with async/await
- **Rationale**: Aligns with the streaming nature of telemetry data and leverages Rust's async ecosystem
- **Alternative Considered**: Polling pattern - rejected due to inefficiency for real-time data

### 2. Component Representation
**Decision**: Create strongly-typed Rust structs for rocket components
- **Rationale**: Provides compile-time safety and better documentation
- **Implementation**: Use a trait-based approach for component registration

```rust
trait TelemetryComponent {
    const NAME: &'static str;
    fn schema() -> ComponentSchema;
    fn from_bytes(data: &[u8]) -> Result<Self, Error>;
}
```

### 3. Connection Management
**Decision**: Use the existing `impeller2-stellar::Client` 
- **Rationale**: Reuse battle-tested networking code, avoid reimplementing protocol details
- **Features**: Automatic reconnection, request/response correlation, streaming support

### 4. Data Processing Pipeline
**Decision**: Implement a modular pipeline with stages:
1. **Connection Stage**: Establish TCP connection to DB
2. **Registration Stage**: Register interest in rocket components
3. **Subscription Stage**: Subscribe to real-time streams
4. **Processing Stage**: Parse and display telemetry data
5. **Analysis Stage**: Compute derived metrics (optional)

### 5. Error Handling
**Decision**: Use `thiserror` for error types with `anyhow` for application-level handling
- **Rationale**: Standard Rust error handling pattern, good ergonomics
- **Recovery Strategy**: Log errors and attempt reconnection for network failures

## Implementation Plan

### Phase 1: Core Infrastructure

#### File Structure
```
libs/db/examples/
├── rust_client/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs           # Entry point and CLI
│   │   ├── client.rs         # Client connection logic
│   │   ├── components.rs     # Rocket component definitions
│   │   ├── subscriber.rs     # Subscription handling
│   │   └── display.rs        # Terminal output formatting
```

#### Dependencies (`Cargo.toml`)
```toml
[package]
name = "elodin-db-rust-client"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core protocol
impeller2 = { path = "../../../impeller2" }
impeller2-stellar = { path = "../../../impeller2/stellar" }
impeller2-wkt = { path = "../../../impeller2/wkt" }

# Async runtime
stellarator = { path = "../../../stellarator", features = ["tokio"] }
tokio = { version = "1", features = ["full"] }

# Serialization
postcard = "1.1"
serde = { version = "1.0", features = ["derive"] }

# Error handling
thiserror = "2"
anyhow = "1"

# CLI and display
clap = { version = "4", features = ["derive"] }
colored = "2"
indicatif = "0.17"

# Utilities
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### Phase 2: Component Definitions

```rust
// components.rs
use impeller2::types::{ComponentId, ComponentSchema};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocketTelemetry {
    pub mach: f64,
    pub thrust: f64,
    pub fin_deflect: f64,
    pub angle_of_attack: f64,
    pub dynamic_pressure: f64,
    pub world_pos: [f64; 7],  // Quaternion + position
    pub world_vel: [f64; 12], // Spatial velocity
}

impl RocketTelemetry {
    pub fn component_ids() -> Vec<(&'static str, ComponentId)> {
        vec![
            ("rocket.mach", ComponentId::new("rocket.mach")),
            ("rocket.thrust", ComponentId::new("rocket.thrust")),
            // ... etc
        ]
    }
}
```

### Phase 3: Subscription Logic

```rust
// subscriber.rs
use impeller2_stellar::Client;
use impeller2_wkt::{Stream, StreamBehavior};

pub struct TelemetrySubscriber {
    client: Client,
    stream_id: u64,
}

impl TelemetrySubscriber {
    pub async fn subscribe_to_rocket(&mut self) -> Result<()> {
        let stream = Stream {
            behavior: StreamBehavior::RealTime,
            id: self.stream_id,
        };
        
        self.client.send(stream).await?;
        Ok(())
    }
    
    pub async fn process_telemetry(&mut self) -> Result<()> {
        loop {
            let packet = self.receive_packet().await?;
            self.handle_packet(packet)?;
        }
    }
}
```

### Phase 4: Display and Analysis

```rust
// display.rs
pub struct TelemetryDisplay {
    update_rate: Duration,
    last_update: Instant,
}

impl TelemetryDisplay {
    pub fn render(&mut self, telemetry: &RocketTelemetry) {
        // Clear screen and display formatted telemetry
        println!("🚀 Rocket Telemetry Dashboard");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Mach:     {:.2}", telemetry.mach);
        println!("Thrust:   {:.0} N", telemetry.thrust);
        println!("Altitude: {:.0} m", telemetry.world_pos[6]);
        // ... etc
    }
}
```

## Testing Strategy

### Integration Testing
1. **Setup**: Launch `elodin-db` server
2. **Run Simulation**: Execute `elodin editor libs/nox-py/examples/rocket.py`
3. **Connect Client**: Run Rust client example
4. **Verify**: Check that telemetry data is received and displayed correctly

### Test Cases
- ✓ Successful connection to DB
- ✓ Component metadata registration
- ✓ Real-time data subscription
- ✓ Graceful disconnection handling
- ✓ Reconnection after network failure
- ✓ Data validation and parsing

## Usage Documentation

### Running the Example

```bash
# Terminal 1: Start the database
elodin-db run [::]:2240 ~/.elodin/db

# Terminal 2: Run the rocket simulation
elodin editor libs/nox-py/examples/rocket.py

# Terminal 3: Run the Rust client
cd libs/db/examples/rust_client
cargo run -- --host 127.0.0.1 --port 2240
```

### Expected Output
```
🚀 Connected to Elodin-DB at 127.0.0.1:2240
📡 Subscribing to rocket telemetry...
✅ Receiving data from rocket simulation

╭─────────── Rocket Telemetry ───────────╮
│ Mach:          0.85                    │
│ Altitude:      1250.3 m                │
│ Velocity:      287.5 m/s               │
│ Thrust:        4500.0 N                │
│ Fin Deflect:   2.3°                    │
│ Angle of Attack: 5.1°                  │
╰─────────────────────────────────────────╯
```

## Future Enhancements

1. **Data Recording**: Add option to save telemetry to file (CSV, Parquet)
2. **Filtering**: Allow selective subscription to specific components
3. **Visualization**: Integration with plotting libraries for real-time graphs
4. **Replay Mode**: Support reading historical data from the database
5. **Performance Metrics**: Add latency and throughput measurements

## References

- [Impeller2 Protocol Documentation](../impeller2/README.md)
- [Elodin-DB Architecture](./README.md)
- [Nox-Py Examples](../nox-py/examples/)
- [C++ Client Implementation](./examples/client.cpp)
