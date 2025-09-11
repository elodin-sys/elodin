# Stellarator

A minimalist async runtime designed for deterministic, high-performance flight software.

## Overview

Stellarator is a custom async executor built specifically for aerospace and embedded systems where:
- **Determinism** is critical for real-time control loops
- **Minimal allocations** are required during steady-state operation
- **Simple, auditable code** is essential for safety-critical systems
- **Zero-copy I/O** maximizes performance on constrained hardware

Unlike general-purpose runtimes, Stellarator prioritizes simplicity and predictability over features, making it ideal for flight computers, embedded controllers, and hardware-in-the-loop testing.

## Why Not Use Existing Runtimes?

Popular async runtimes like Tokio, async-std, or specialized ones like compio, monoio, and glommio are excellent for general applications but present challenges for flight software:

### The Problem with Existing Solutions

| Runtime | Issues for Flight Software |
|---------|----------------------------|
| **Tokio** | Complex scheduler, work-stealing, large dependency tree, allocations during runtime |
| **async-std** | Dynamic task spawning, complex abstractions, standard library dependencies |
| **monoio/glommio** | io_uring-only (no cross-platform), complex scheduling, not embedded-friendly |
| **compio** | Windows-focused, large API surface, not optimized for embedded |

### Stellarator's Approach

- **No work-stealing** - Tasks run where they're spawned, predictable CPU usage
- **No dynamic allocation** - All tasks allocated upfront or on stack
- **Minimal unsafe code** - Easier to audit for safety-critical applications
- **Small dependency tree** - Reduces supply chain risk and binary size
- **Platform-appropriate I/O**:
  - **Linux**: io_uring for maximum performance
  - **macOS/Windows**: Polling-based fallback
  - **Embedded**: Compatible with no_std environments (via maitake)

## Architecture

```
┌─────────────────────────────────────────────┐
│             User Application                │
├─────────────────────────────────────────────┤
│          Async Tasks & Futures              │
├─────────────────────────────────────────────┤
│            Stellarator API                  │
│  ┌─────────┬────────┬──────────┬────────┐   │
│  │   I/O   │  Net   │  Serial  │  Sync  │   │
│  └─────────┴────────┴──────────┴────────┘   │
├─────────────────────────────────────────────┤
│           Executor & Reactor                │
│  ┌──────────────────┬──────────────────┐    │
│  │  Maitake         │  Timer Wheel     │    │
│  │  Scheduler       │  (Maitake)       │    │
│  └──────────────────┴──────────────────┘    │
├─────────────────────────────────────────────┤
│           Platform Backend                  │
│  ┌──────────────────┬──────────────────┐    │
│  │   io_uring       │    Polling       │    │
│  │   (Linux)        │  (Mac/Windows)   │    │
│  └──────────────────┴──────────────────┘    │
└─────────────────────────────────────────────┘
```

## Features

### Core Runtime
- **Single-threaded executor** - Predictable, no synchronization overhead
- **Thread-local execution** - Each thread has its own executor
- **Structured concurrency** - Cancellation tokens and controlled task lifecycles
- **Timer integration** - Built-in timer wheel from maitake

### I/O Operations
All I/O operations use zero-copy buffers to minimize allocations:

- **File I/O** - Async read/write with configurable offsets
- **Network** - TCP/UDP with async connect/accept/send/receive
- **Serial Ports** - Hardware communication with configurable baud rates
- **Memory-mapped I/O** - Direct hardware register access (platform-specific)

### Platform Backends

#### Linux (io_uring)
- True async I/O without thread pools
- Kernel-level batching of operations
- Zero syscall overhead in hot path
- Supports all modern I/O operations

#### macOS/Windows (Polling)
- epoll/kqueue/IOCP based
- Fallback for platforms without io_uring
- Compatible with standard async patterns

## Usage

### Basic Example

```rust
use stellarator::{run, sleep, spawn};
use std::time::Duration;

fn main() {
    // Run an async task on the thread-local executor
    let result = run(async {
        println!("Starting task");
        
        // Spawn a background task
        let handle = spawn(async {
            for i in 0..5 {
                println!("Background task: {}", i);
                sleep(Duration::from_millis(100)).await;
            }
            42
        });
        
        // Do other work
        sleep(Duration::from_millis(250)).await;
        
        // Wait for background task
        handle.await.unwrap()
    });
    
    println!("Result: {}", result);
}
```

### File I/O

```rust
use stellarator::fs::File;

async fn read_sensor_data() -> Result<Vec<u8>, stellarator::Error> {
    let file = File::open("/dev/sensor0").await?;
    let mut buffer = vec![0u8; 1024];
    
    // Zero-copy read into buffer
    let (result, buffer) = file.read(buffer).await;
    let bytes_read = result?;
    
    Ok(buffer[..bytes_read].to_vec())
}
```

### Serial Communication

```rust
use stellarator::serial::{SerialPort, Baud};

async fn communicate_with_hardware() -> Result<(), stellarator::Error> {
    let mut port = SerialPort::open("/dev/ttyUSB0").await?;
    port.set_baud(Baud::B115200)?;
    
    // Send command
    let command = b"READ_SENSORS\n";
    let (result, _) = port.write(command.to_vec()).await;
    result?;
    
    // Read response
    let mut buffer = vec![0u8; 256];
    let (result, buffer) = port.read(buffer).await;
    let bytes_read = result?;
    
    println!("Received: {:?}", &buffer[..bytes_read]);
    Ok(())
}
```

### Structured Concurrency

```rust
use stellarator::{struc_con, util::CancelToken};

async fn cancellable_operation() {
    let cancel = CancelToken::new();
    
    // Spawn a thread that can be cancelled
    let thread = struc_con::stellar(|| async {
        loop {
            // Simulate work
            stellarator::sleep(Duration::from_millis(100)).await;
            println!("Working...");
        }
    });
    
    // Cancel after 1 second
    stellarator::sleep(Duration::from_secs(1)).await;
    thread.cancel().await.unwrap();
}
```

## Integration with Flight Software

Stellarator is designed to integrate seamlessly with:

1. **Roci** - Reactive flight software framework
2. **Impeller2** - High-performance telemetry protocol
3. **Elodin Simulation** - Hardware-in-the-loop testing
4. **Aleph** - Flight computer platform

Example integration with Roci:

```rust
use roci::{System, SystemFn};
use stellarator::serial::SerialPort;

#[derive(SystemFn)]
async fn sensor_reader(port: &SerialPort) -> SensorData {
    let mut buffer = vec![0u8; 64];
    let (result, buffer) = port.read(buffer).await;
    
    // Parse sensor data from buffer
    parse_sensor_data(&buffer[..result.unwrap()])
}
```

## Performance Characteristics

### Memory Usage
- **Zero allocations** during steady-state I/O operations
- **Stack-based futures** where possible
- **Pre-allocated task storage** via maitake

### Latency
- **Single-digit microsecond** wake latency on Linux with io_uring
- **No thread pool overhead** - operations complete in-place
- **Batch submission** of I/O operations on Linux

### Throughput
- **Saturates NVMe** on modern hardware
- **Line-rate networking** with zero-copy buffers
- **Minimal CPU overhead** - kernel does the heavy lifting

## Development

### Building

```bash
# Build with default features (includes io_uring on Linux)
cargo build --package stellarator

# Build for embedded (no_std via maitake)
cargo build --package stellarator --no-default-features
```

### Testing

```bash
# Run unit tests
cargo test --package stellarator

# Run with miri for safety verification
cargo +nightly miri test --package stellarator
```

### Architecture Decision Records

1. **io_uring on Linux only** - Other platforms lack equivalent APIs
2. **Single-threaded by default** - Multi-threading adds complexity without clear benefits for control systems
3. **Maitake for scheduling** - Battle-tested in embedded systems
4. **Zero-copy buffers everywhere** - Essential for high-frequency sensor data

## Contributing

When contributing to Stellarator:

1. **Keep it simple** - Reject complexity unless absolutely necessary
2. **Minimize unsafe** - Prefer safe abstractions even at minor performance cost
3. **Document safety invariants** - Every unsafe block needs justification
4. **Test on real hardware** - Ensure changes work on flight computers
5. **Profile before optimizing** - Measure impact on real workloads

## History

Stellarator was created to solve specific challenges in flight software development where existing async runtimes fell short. The initial implementation ([PR #775](https://github.com/elodin-sys/paracosm/pull/775)) focused on:

- Creating a minimal async executor for deterministic behavior
- Using io_uring on Linux for maximum I/O performance
- Maintaining compatibility with no_std environments
- Providing simple, auditable code for safety-critical systems

The name "Stellarator" comes from the fusion reactor design - like its namesake, this runtime aims to create a controlled, stable environment for high-energy operations.

## License

See the repository's LICENSE file for details.
