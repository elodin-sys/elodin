# Postcard-C

A C/C++ code generator and runtime library for the [Postcard](https://github.com/jamesmunns/postcard) serialization format, enabling seamless data exchange between Rust and C/C++ systems.

## Overview

Postcard-C bridges the gap between Rust and C/C++ code in the Elodin ecosystem, particularly for:
- **Flight Software** - Embedded systems written in C/C++ that need to send telemetry
- **Ground Software** - C/C++ applications that need to communicate with Elodin components
- **Hardware-in-the-Loop** - Real hardware running C/C++ firmware interfacing with Elodin simulations
- **Legacy Systems** - Existing C/C++ codebases that need to integrate with Elodin

## Why Postcard?

[Postcard](https://github.com/jamesmunns/postcard) is a compact, efficient, `#![no_std]` wire format designed for embedded systems:
- **Compact** - Uses variable-length encoding (varint) to minimize message size
- **Fast** - Zero-copy deserialization where possible
- **Simple** - No schema negotiation or versioning complexity
- **Deterministic** - No dynamic allocation, suitable for real-time systems

## Architecture

Postcard-C consists of three main components:

### 1. `postcard.h` - Header-Only C Library
A lightweight, dependency-free C library that provides low-level encoding/decoding primitives:
```c
// Encode a value
postcard_encode_u32(&slice, 12345);
postcard_encode_string(&slice, "telemetry", 9);

// Decode a value
uint32_t id;
postcard_decode_u32(&slice, &id);
```

### 2. Code Generator (`postcard-c-codegen`)
A Rust tool that generates C++ classes from type definitions:
```rust
// Input: Type definition in RON format
(name:"Telemetry", ty:Struct([
  (name:"timestamp", ty:(name:"u64", ty:U64)),
  (name:"temperature", ty:(name:"f32", ty:F32)),
  (name:"position", ty:(name:"Vec<f64>", ty:Seq((name:"f64", ty:F64))))
]))
```

Generates:
```cpp
struct Telemetry {
  uint64_t timestamp;
  float temperature;
  std::vector<double> position;
  
  // Auto-generated serialization
  std::vector<uint8_t> encode_vec() const;
  postcard_error_t decode(std::span<const uint8_t>& input);
};
```

### 3. Generated C++ Classes
Type-safe wrappers with automatic serialization/deserialization:
- Structs with fields
- Enums with variants (using `std::variant`)
- Collections (`std::vector`, `std::unordered_map`)
- Optional values (`std::optional`)
- Nested types

## Usage

### Generating Bindings

1. Define your types in RON format:
```rust
// telemetry.ron
(name:"ImuData", ty:Struct([
  (name:"accel", ty:(name:"Vec<f32>", ty:Seq((name:"f32", ty:F32)))),
  (name:"gyro", ty:(name:"Vec<f32>", ty:Seq((name:"f32", ty:F32)))),
  (name:"timestamp", ty:(name:"u64", ty:U64))
]))
```

2. Generate C++ header:
```bash
cargo run --package postcard-c-codegen telemetry.ron > telemetry.hpp
# Or with formatting:
cargo run --package postcard-c-codegen telemetry.ron | clang-format > telemetry.hpp
```

### Using in C++ Flight Software

```cpp
#include "telemetry.hpp"
#include <vector>

// Create telemetry packet
ImuData imu {
    .accel = {0.0f, 0.0f, 9.81f},
    .gyro = {0.1f, 0.2f, 0.0f},
    .timestamp = 1234567890
};

// Serialize to bytes
auto encoded = imu.encode_vec();

// Send over UART, TCP, or any transport
send_to_elodin(encoded.data(), encoded.size());
```

### Receiving in Elodin (Rust)

```rust
use postcard::from_bytes;
use serde::Deserialize;

#[derive(Deserialize)]
struct ImuData {
    accel: Vec<f32>,
    gyro: Vec<f32>,
    timestamp: u64,
}

// Receive and deserialize
let bytes = receive_from_flight_software();
let imu: ImuData = from_bytes(&bytes)?;
```

## Integration with Impeller2

Postcard-C is designed to work seamlessly with Impeller2 for telemetry streaming:

```cpp
// Flight software sending component data
struct DroneState {
    WorldPos position;
    Quaternion attitude;
    Vec3 velocity;
};

// Serialize and send via Impeller2 message
DroneState state = get_current_state();
auto packet = state.encode_vec();

// Wrap in Impeller2 message format
send_impeller_msg(ComponentId("drone.state"), packet);
```

## Supported Types

### Primitive Types
- Integers: `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`
- Floats: `f32`, `f64`
- Boolean: `bool`
- Strings: `std::string`

### Collections
- Sequences: `std::vector<T>`
- Maps: `std::unordered_map<K, V>`
- Byte arrays: `std::vector<uint8_t>` (optimized encoding)

### Complex Types
- Structs with named fields
- Enums with unit, newtype, and struct variants
- Optional values: `std::optional<T>`
- Tuples: `std::tuple<T1, T2, ...>`
- Nested types of arbitrary depth

## Performance Characteristics

- **Zero-copy deserialization** for primitive types
- **Variable-length encoding** for integers (1-10 bytes based on value)
- **Compact boolean encoding** (1 byte)
- **Length-prefixed collections** for safe parsing
- **No heap allocation** in C library (caller manages buffers)
- **C++23 features** for modern, efficient code generation

## Examples

### Complete Example: Sensor Telemetry

```cpp
// sensor_telemetry.hpp (generated)
struct SensorReading {
    uint64_t timestamp;
    std::string sensor_id;
    std::vector<float> values;
    std::optional<float> temperature;
    bool is_valid;
    
    size_t encoded_size() const;
    std::vector<uint8_t> encode_vec() const;
    postcard_error_t decode(std::span<const uint8_t>& input);
};

// main.cpp
#include "sensor_telemetry.hpp"

int main() {
    SensorReading reading {
        .timestamp = get_timestamp_ns(),
        .sensor_id = "IMU_001",
        .values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        .temperature = 25.5f,  // Optional with value
        .is_valid = true
    };
    
    // Encode
    auto bytes = reading.encode_vec();
    printf("Encoded %zu bytes\n", bytes.size());
    
    // Decode
    SensorReading decoded;
    auto input = std::span<const uint8_t>(bytes);
    if (decoded.decode(input) == POSTCARD_SUCCESS) {
        printf("Decoded sensor: %s\n", decoded.sensor_id.c_str());
    }
    
    return 0;
}
```

## Building

### Requirements
- C++23 compiler (for generated code)
- Rust toolchain (for code generator)
- No external C/C++ dependencies

### For Code Generator
```bash
cd libs/postcard-c/codegen
cargo build --release
```

### For C++ Projects
Include `postcard.h` and generated headers in your project:
```cmake
# CMakeLists.txt
target_include_directories(your_target PRIVATE 
    ${ELODIN_PATH}/libs/postcard-c
)
```

## Design Principles

1. **Safety First** - All decoding operations check bounds and return errors
2. **Zero Dependencies** - Header-only C library with no external dependencies
3. **Embedded-Friendly** - No dynamic allocation in the C library
4. **Type Safety** - Generated C++ code provides strong typing
5. **Compatibility** - Wire format 100% compatible with Rust Postcard

## Common Use Cases

### 1. Flight Computer Telemetry
Embedded systems sending IMU, GPS, and sensor data to Elodin:
```cpp
// Send high-frequency IMU updates
ImuPacket imu = read_imu_sensor();
auto bytes = imu.encode_vec();
uart_send(bytes.data(), bytes.size());
```

### 2. Ground Station Commands
Sending commands from Elodin to flight software:
```rust
// Rust side
let command = SetThrottle { value: 0.75 };
let bytes = postcard::to_vec(&command)?;
serial_port.write(&bytes)?;
```

```cpp
// C++ side
SetThrottle command;
if (command.decode(received_bytes) == POSTCARD_SUCCESS) {
    apply_throttle(command.value);
}
```

### 3. Hardware-in-the-Loop Testing
Real hardware exchanging state with Elodin simulation:
```cpp
// Hardware sends actual sensor readings
// Simulation sends simulated environment data
// Both use the same Postcard-encoded messages
```

## Troubleshooting

### Common Issues

**Encoding Buffer Too Small**
```cpp
// Error: POSTCARD_ERROR_BUFFER_TOO_SMALL
// Solution: Use encoded_size() to pre-calculate required buffer size
size_t required = data.encoded_size();
std::vector<uint8_t> buffer(required);
```

**Type Mismatch Between Rust and C++**
- Ensure RON definitions match Rust struct definitions exactly
- Field order matters for structs
- Use the same integer sizes (u32 in Rust = uint32_t in C++)

**Decoding Failures**
- Check that sender and receiver use the same type definitions
- Verify byte order (Postcard uses little-endian)
- Ensure complete message is received (no truncation)

## Contributing

When contributing to Postcard-C:

1. **Update postcard.h** for new primitive types
2. **Extend code generator** templates for new language features
3. **Add tests** for both encoding and decoding paths
4. **Ensure compatibility** with Rust Postcard crate
5. **Document** wire format changes

## Related Projects

- [Postcard](https://github.com/jamesmunns/postcard) - The original Rust implementation
- [Impeller2](../impeller2/README.md) - Protocol using Postcard for messages
- [Elodin Database](../db/README.md) - Stores Postcard-encoded telemetry

## License

See the repository's LICENSE file for details.
