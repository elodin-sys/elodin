# Rust Client Example for Elodin-DB

This example demonstrates how to build a Rust client that connects to Elodin-DB and sets up telemetry subscriptions.

## Features

- 🔌 **TCP Connection**: Connect to Elodin-DB server
- 📝 **Component Registration**: Register telemetry components with metadata
- 📡 **Stream Setup**: Configure real-time telemetry streaming
- 🎯 **Type-safe API**: Uses the impeller2 protocol for communication

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

## Next Steps

See the [design document](../../RUST_CLIENT_DESIGN.md) for the full architecture plan and future enhancements.