---
name: elodin-db
description: Work with Elodin-DB, the time-series telemetry database. Use when running elodin-db, writing client integrations (C, C++, Rust, Python), configuring replication/follow mode, querying data via the Lua REPL, or connecting the Elodin Editor to a database.
---

# Elodin-DB

Elodin-DB is a high-performance time-series database for telemetry data. It stores components, messages, and metadata using the Impeller2 protocol, and serves as the central data bus between simulations, flight software, and the Elodin Editor.

## Quick Start

```bash
# Install (from source)
just install

# Run the database
elodin-db run [::]:2240 $HOME/.local/share/elodin/db --config libs/db/examples/db-config.lua --log-level warn

# Connect the Elodin Editor
elodin editor 127.0.0.1:2240

# Launch the Lua REPL
elodin-db lua
```

## Running the Database

```bash
elodin-db run <bind_addr> <data_dir> [--config <lua_file>] [--log-level <level>]
```

| Parameter | Example | Purpose |
|-----------|---------|---------|
| `bind_addr` | `[::]:2240` | Listen address (IPv4/IPv6 + port) |
| `data_dir` | `$HOME/.local/share/elodin/db` | Storage directory |
| `--config` | `libs/db/examples/db-config.lua` | Lua configuration script |
| `--log-level` | `warn` | Log verbosity: error, warn, info, debug, trace |

## Lua REPL

Interactive database shell for debugging and exploration:

```bash
elodin-db lua
```

```lua
db> client = connect("127.0.0.1:2240")
db> client:dump_metadata()
db> :help
```

## Client Integration

### C Client

```bash
cc examples/client.c -lm -o /tmp/client && /tmp/client
```

See `libs/db/examples/client.c` for streaming fake sensor data.

### C++ Client

```bash
c++ -std=c++23 examples/client.cpp -o /tmp/client-cpp && /tmp/client-cpp
```

The C++ library is C++20 compatible. See `libs/db/examples/client.cpp` for subscription example.

### Rust Client

See `libs/db/examples/rust_client/` for a complete Rust client using Impeller2.

### C++ Header Generation

Generate the single-header C++20 library with message definitions:

```bash
cargo run --bin elodin-db gen-cpp > libs/db/examples/db.hpp
```

### Python (via Simulation)

Simulations automatically create an embedded database, or connect to an external one:

```python
# Embedded (temporary)
w.run(system)

# Explicit path
w.run(system, db_path="./my_data")

# Connect to external DB
w.run(system, db_addr="127.0.0.1:2240")
```

## Follow Mode (Replication)

Replicate data from one database to another over TCP:

```bash
# Source database
elodin-db run [::]:2240 $HOME/.local/share/elodin/source-db

# Follower database (replicates from source)
elodin-db run [::]:2241 $HOME/.local/share/elodin/follower-db --follows 127.0.0.1:2240
```

The follower:
1. Synchronizes all existing metadata and schemas
2. Backfills historical component data and message logs
3. Streams real-time updates as they arrive

### Packet Size Tuning

Default batches outgoing data into ~1500-byte TCP writes (standard Ethernet MTU):

```bash
elodin-db run [::]:2241 ./follower-db --follows 127.0.0.1:2240 --follow-packet-size 9000
```

### Dual-Source Example

Run a simulation on source, follow on target, and add local streams:

```bash
# Source machine
elodin editor examples/video-stream/main.py

# Target — follow source
elodin-db run [::]:2241 ./follower-db --follows SOURCE_IP:2240

# Target — connect editor
elodin editor 127.0.0.1:2241

# Target — add local video stream
examples/video-stream/stream-video.sh
```

## Trimming a Database

Remove data from the beginning or end of a recording. Values are in microseconds. Without `--output`, modifies in place.

```bash
# Remove the first 3 minutes from a recording
elodin-db trim --from-start 180000000 ./my-db

# Remove the last 2 minutes from a recording
elodin-db trim --from-end 120000000 --output ./trimmed ./my-db

# Trim 1 minute from the start and 2 minutes from the end
elodin-db trim --from-start 60000000 --from-end 120000000 --output ./window ./my-db

# Preview without modifying
elodin-db trim --from-start 180000000 --dry-run ./my-db
```

## Editor Connection

The Elodin Editor connects to any running database:

```bash
elodin editor 127.0.0.1:2240
```

From a simulation, the editor connects automatically when launched via `elodin editor sim.py`.

## Architecture

Elodin-DB uses the Impeller2 protocol internally:
- **Components**: Time-series data indexed by entity + component name + timestamp
- **Messages**: Ordered log entries (commands, events)
- **Metadata**: Schema information, entity names, component types

Storage is append-only with configurable retention. The database supports concurrent readers and writers with lock-free data structures.

## Key References

- Full documentation: [libs/db/README.md](../../../libs/db/README.md)
- C client example: [libs/db/examples/client.c](../../../libs/db/examples/client.c)
- C++ client example: [libs/db/examples/client.cpp](../../../libs/db/examples/client.cpp)
- Rust client example: [libs/db/examples/rust_client/](../../../libs/db/examples/rust_client/)
- Lua config example: [libs/db/examples/db-config.lua](../../../libs/db/examples/db-config.lua)
- DB architecture docs: [docs/public/content/home/db/](../../../docs/public/content/home/db/)
