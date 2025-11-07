# Roci

A reactive flight software framework for building composable control systems with real-time telemetry streaming.

<video autoplay loop muted playsinline style="width: 100%; height: auto;">
  <source src="roci-baselisk-demo.h264.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Overview

Roci (named after the ship from The Expanse) is a framework designed to simplify the development of flight software by providing:
- **Composable Systems** - Build complex control systems from simple, reusable components
- **Zero-Copy Communication** - Efficient data exchange with the Impeller2 protocol
- **Real-Time Drivers** - Support for different execution modes (fixed frequency, interrupt-driven)
- **Telemetry Integration** - Seamless connection to Elodin's simulation and visualization tools

## Architecture

### Core Concepts

#### Systems

The fundamental building block is the `System` trait:

```rust
pub trait System {
    type World: Default + Decomponentize + Componentize;
    type Driver: DriverMode;
    
    fn update(&mut self, world: &mut Self::World);
}
```

Systems operate on a `World` (state container) and can be composed using combinators:

```rust
// Pipe systems together - output of first becomes input to second
let combined = sensor_system
    .pipe(filter_system)
    .pipe(controller_system)
    .pipe(actuator_system);
```

#### Drivers

Drivers control how systems execute:

- **`Hz<N>`** - Fixed frequency execution (e.g., `Hz<120>` for 120Hz)
- **`Interrupt`** - Event-driven execution
- **`OsSleepDriver`** - Uses OS sleep for timing
- **`LoopDriver`** - Continuous execution without delays

```rust
// Run a system at 120Hz
os_sleep_driver::<120, _, _>(my_system).run();
```

#### World and Components

The `World` represents the system state. Using derive macros, you can automatically generate serialization/deserialization:

```rust
#[derive(Default, Debug, Componentize, Decomponentize, AsVTable, Metadatatize)]
struct DroneWorld {
    #[roci(component_id = "drone.imu.accel")]
    accel: [f64; 3],
    
    #[roci(component_id = "drone.imu.gyro")]
    gyro: [f64; 3],
    
    #[roci(component_id = "drone.control.torque")]
    control_torque: [f64; 3],
}
```

### System Functions

Regular Rust functions can become systems:

```rust
fn attitude_controller(
    imu: &mut ImuData,
    target: &mut TargetAttitude,
    control: &mut ControlOutput,
) {
    // Control logic here
}

// Convert to system and run at 100Hz
os_sleep_driver::<100, _, _>(attitude_controller).run();
```

## ADCS Sub-crate

The `roci-adcs` sub-crate provides attitude determination and control algorithms commonly used in spacecraft:

### Algorithms

#### Attitude Determination
- **TRIAD** - Two-vector attitude determination using two reference vectors
- **MEKF** (Multiplicative Extended Kalman Filter) - Quaternion-based attitude estimation with gyro bias estimation
- **UKF** (Unscented Kalman Filter) - Nonlinear state estimation with configurable sigma points

#### Magnetometer Calibration
- **MAG.I.CAL** - Iterative calibration algorithm for hard/soft iron compensation
- **MagKal** - UKF-based magnetometer calibration for real-time bias and scale factor estimation

#### Control
- **Yang LQR** - Linear Quadratic Regulator for quaternion-based attitude control

### Example: MEKF Attitude Estimation

```rust
use roci_adcs::mekf::State;
use nalgebra::vector;

// Initialize MEKF with noise parameters
let mut state = State::new(
    vector![0.01, 0.01, 0.01],  // Gyro noise
    vector![0.01, 0.01, 0.01],  // Gyro bias noise
    1.0 / 120.0,                 // Time step
);

// Update with measurements
state = state.estimate_attitude(
    [magnetometer_body, sun_sensor_body],  // Body measurements
    [mag_reference, sun_reference],        // Reference vectors
    [0.03, 0.03],                          // Measurement noise
);

let attitude = state.q_hat;  // Get estimated quaternion
let gyro_bias = state.b_hat; // Get estimated gyro bias
```

## Communication

### TCP/Network Integration

Roci integrates with Impeller2 for network communication:

```rust
use roci::tcp::{tcp_connect, tcp_listen};

// Connect to Elodin simulation
let (tx, rx) = tcp_connect::<Hz<120>>(
    "127.0.0.1:4488".parse()?,
    &[Query::all()],
    metadata,
);

// Compose with your control system
let system = rx
    .pipe(sensor_processing)
    .pipe(controller)
    .pipe(tx);
```

### CSV Logging

Log telemetry to CSV files for analysis:

```rust
use roci::csv::CSVLogger;

let logger = CSVLogger::try_from_path("telemetry.csv")?;
let system = my_system.pipe(logger);
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
roci = { path = "../path/to/elodin/libs/roci" }
roci-adcs = { path = "../path/to/elodin/libs/roci/adcs" }  # Optional ADCS algorithms
```

## Example: Complete Attitude Control System

```rust
use roci::{os_sleep_driver, IntoSystem};
use roci_adcs::{mekf, yang_lqr::YangLQR};
use nalgebra::{vector, Vector3, UnitQuaternion};

#[derive(Default, Componentize, Decomponentize)]
struct SatelliteWorld {
    #[roci(component_id = "sat.sensors.mag")]
    magnetometer: [f64; 3],
    
    #[roci(component_id = "sat.sensors.gyro")]
    gyroscope: [f64; 3],
    
    #[roci(component_id = "sat.state.quaternion")]
    attitude: [f64; 4],
    
    #[roci(component_id = "sat.control.torque")]
    control_torque: [f64; 3],
}

fn attitude_control_system(world: &mut SatelliteWorld) {
    // 1. Estimate attitude using MEKF
    // 2. Compute control torque using LQR
    // 3. Update world with results
}

fn main() {
    // Run at 100Hz with TCP communication
    let (tx, rx) = tcp_connect("sim.local:4488", &[Query::all()], metadata);
    
    let system = rx
        .pipe(attitude_control_system)
        .pipe(tx);
    
    os_sleep_driver::<100, _, _>(system).run();
}
```

## History

Roci has evolved significantly since its initial implementation. Here are some major milestones:

### Evolution Timeline

1. **Initial Implementation** ([#469](https://github.com/elodin-sys/paracosm/pull/469)) - May 2024
   - Core framework with World/Handler pattern
   - Basic Conduit (now Impeller2) integration
   - Entity-Component mapping with derive macros

2. **Basilisk Integration** ([#471](https://github.com/elodin-sys/paracosm/pull/471)) - May 2024
   - First demonstration of external algorithm integration
   - CubeSat attitude control example using Basilisk's MRP-PD controller

3. **Combinator Refactor** ([#508](https://github.com/elodin-sys/paracosm/pull/508))
   - Introduced pipe combinators for system composition
   - Simplified system chaining and data flow

4. **System Functions** ([#634](https://github.com/elodin-sys/paracosm/pull/634))
   - Added support for regular functions as systems
   - Reduced boilerplate for simple control logic

5. **UKF Implementation** ([#663](https://github.com/elodin-sys/paracosm/pull/663))
   - Dynamic dimension support for Unscented Kalman Filter
   - Foundation for advanced estimation algorithms

6. **Python UKF Wrapper** ([#668](https://github.com/elodin-sys/paracosm/pull/668))
   - Cross-language support for filter algorithms
   - Enabled prototyping in Python

7. **Impeller2 Protocol** ([#763](https://github.com/elodin-sys/paracosm/pull/763))
   - Migration from Conduit to Impeller2
   - Improved performance and flexibility

8. **Nalgebra Migration** (2025)
   - Migrated ADCS algorithms from internal tensor library to Nalgebra
   - Simplified dependencies and improved maintainability

## Design Philosophy

Roci embraces several key principles:

1. **Composition over Inheritance** - Build complex systems from simple, focused components
2. **Type Safety** - Leverage Rust's type system for compile-time correctness
3. **Zero-Cost Abstractions** - Framework overhead should be minimal
4. **Hardware Agnostic** - Same code runs in simulation and on real hardware
5. **Telemetry First** - Built-in support for debugging and analysis

## Use Cases

- **Spacecraft ADCS** - Attitude determination and control systems
- **Drone Flight Controllers** - Multi-rotor stabilization and navigation
- **Robotics** - Real-time control with sensor fusion
- **Hardware-in-the-Loop** - Testing real hardware with simulated environments
- **Simulation** - Pure software testing before hardware deployment

## Development

### Building

```bash
cargo build --release
```

### Testing

```bash
cargo test
```

### Examples

See the `examples/` directory for complete examples:
- `msg-test.rs` - Basic message passing
- Integration examples with Elodin simulation

## Related Projects

- **Impeller2** - High-performance telemetry protocol
- **Nalgebra** - Linear algebra library for Rust

## Contributing

When adding new ADCS algorithms:
1. Implement in `roci/adcs/src/`
2. Add reference implementation in Python in `roci/adcs/reference-impls/`
3. Include comprehensive tests comparing against reference
4. Document algorithm source papers and assumptions

## License

See the repository's LICENSE file for details.
