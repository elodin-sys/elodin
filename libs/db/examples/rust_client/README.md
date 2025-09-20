# Rust Client Example for Elodin-DB

This example demonstrates how to build a Rust client that connects to Elodin-DB, discovers available components dynamically, and sets up telemetry subscriptions.

## Features

- 🔌 **TCP Connection**: Connect to Elodin-DB server
- 🔍 **Dynamic Discovery**: Automatically discovers components registered in the database
- 📝 **Schema Detection**: Retrieves component schemas and metadata from the database
- 📡 **Stream Setup**: Configure real-time telemetry streaming
- 🎯 **Type-safe API**: Uses the impeller2 protocol for communication
- 🚀 **Rocket-Aware**: Specifically detects and categorizes rocket simulation components

## Prerequisites

- Rust toolchain (1.70 or later)
- Running instance of `elodin-db`

## Building

From the repository root:

```bash
cargo build -p elodin-db-rust-client --release
```

## Running

### 1. Start the Database

First, ensure the Elodin-DB server is running:

```bash
elodin-db run [::]:2240 ~/.elodin/db --config examples/db-config.lua
```

### 2. Run the Rust Client

```bash
./target/release/rust_client
```

Or with custom host/port:

```bash
./target/release/rust_client --host 192.168.1.100 --port 2240
```

### 3. Generate Test Data (Optional)

To see the client in action with real telemetry, run the rocket simulation:

```bash
# In another terminal
cd libs/nox-py/examples
python rocket.py
```

## Architecture

The client demonstrates:
1. **Connection**: Establishing a TCP connection to the database
2. **Registration**: Registering component metadata using `SetComponentMetadata`
3. **Streaming**: Setting up real-time streams and VTable subscriptions
4. **Protocol**: Using the impeller2 wire protocol for communication

## Implementation Details

### Key Components

- **Client Connection**: Uses `impeller2_stellar::Client` for TCP communication
- **Message Types**: Leverages `impeller2_wkt` well-known types
- **Async Runtime**: Built on `stellarator` (tokio-based) async runtime

### Protocol Flow

1. Connect to database via TCP
2. Register component metadata (name, type, shape)
3. Subscribe to real-time stream
4. Request VTable stream for structured data
5. Process incoming packets (extend for your use case)

## Extending the Example

This is a foundational example showing connectivity. To build a full client:

1. **Add Packet Processing**: Implement handlers for incoming `OwnedPacket` types
2. **Parse Telemetry**: Deserialize component data from table packets
3. **Add Visualization**: Display or graph incoming telemetry
4. **Implement Commands**: Send control messages back to the simulation
5. **Add Persistence**: Store telemetry to files or secondary databases

## Comparison with Other Examples

- **C Client** (`client.c`): Manual packet construction, raw socket handling
- **C++ Client** (`client.cpp`): Object-oriented wrapper with type safety
- **Rust Client**: Full type safety, async/await, integrated with Rust ecosystem

## Troubleshooting

If connection fails:
- Verify elodin-db is running: `ps aux | grep elodin-db`
- Check the port is open: `nc -zv localhost 2240`
- Enable verbose logging: `./rust_client --verbose`

## Dynamic Component Discovery

This client now includes automatic component discovery! When connecting to a database with an active simulation (like rocket.py), the client will:

### Discovery Features

1. **Query Available Components** - Uses `DumpMetadata` and `DumpSchema` messages to get all registered components
2. **Display Component Info** - Shows each component's:
   - Name (e.g., `rocket.mach`)
   - Data type and shape (e.g., `f64[3]` for 3D vectors)
   - Associated metadata
3. **Categorize Rocket Components** - Groups rocket-specific components by:
   - Aerodynamics (mach, dynamic pressure, etc.)
   - Propulsion (thrust, motor)
   - Control (fin deflection, PID states)
   - Position/Motion (world position, velocity)

### Example Discovery Output

```
🔍 Discovering registered components:
  Found 20 components registered
  ✓ rocket.mach → f64
  ✓ rocket.thrust → f64
  ✓ rocket.world_pos → f64[7]
  ✓ rocket.aero_force → f64[6]
  ...

🚀 Rocket Components Summary:
  20 rocket-specific components available
```

### No Manual Registration Required

Unlike the C/C++ examples which manually register components, this Rust client:
- Discovers what's already in the database
- Retrieves schemas automatically
- Adapts to whatever simulation is running

This makes the client much more flexible - it can work with any Elodin simulation without code changes!

## Next Steps

See the [design document](../../RUST_CLIENT_DESIGN.md) for the full architecture plan and future enhancements.