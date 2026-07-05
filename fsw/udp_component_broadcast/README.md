# UDP Component Broadcast

A Python-based component broadcast system for Elodin-DB that enables real-time data sharing between simulation instances over UDP.

## Overview

This example demonstrates how to:
- Subscribe to components from a local Elodin-DB instance
- UDP broadcast component data at a controlled rate
- Receive broadcasts and write to another Elodin-DB instance

This enables distributed simulation scenarios like having a target drone chase a BDX jet plane across two separate simulation instances running on different machines.

## Architecture

```
┌─────────────────────┐         UDP Broadcast         ┌──────────────────────┐
│  BDX Simulation     │  ────────────────────────────▶│  Target Simulation   │
│  (Elodin Editor)    │                               │  (Elodin Editor)     │
│                     │                               │                      │
│  ┌───────────────┐  │         ┌───────────┐         │  ┌────────────────┐  │
│  │  Elodin-DB    │──┼────────▶│ Broadcast │────────▶│  │  Elodin-DB     │  │
│  │  bdx.world_pos│  │         │  Script   │         │  │target.world_pos│  │
│  └───────────────┘  │         └───────────┘         │  └────────────────┘  │
└─────────────────────┘                               └──────────────────────┘
        Machine A                                            Machine B
```

## Installation

The scripts use the first-class `elodin.db` client from the `elodin` Python
wheel (see `libs/nox-py`).

```bash
cd fsw/udp_component_broadcast

# Create virtual environment
uv venv --python 3.13 && source .venv/bin/activate

# Install dependencies (includes the elodin wheel)
uv pip install -r requirements.txt

# Generate protobuf code (if not already generated)
protoc --python_out=. component_broadcast.proto
```

From the repo's nix devshell, `just install` builds and installs the local
elodin wheel instead.

## Quick Start

### 1. Start the Source Simulation

On Machine A, run the RC-jet simulation:
```bash
elodin editor examples/rc-jet/main.py
```

### 2. Start the Broadcaster

On Machine A, in a separate terminal:
```bash
cd fsw/udp_component_broadcast
source .venv/bin/activate
python3 broadcast_component.py \
    --component bdx.world_pos \
    --rename target.world_pos \
    --source-id bdx-jet
```

### 3. Start the Receiver

On Machine B:
```bash
cd fsw/udp_component_broadcast
source .venv/bin/activate
python3 receive_broadcast.py
```

### 4. Start the Target Simulation

On Machine B:
```bash
elodin editor examples/rc-jet/main.py
```

The target drone will receive position updates from the BDX jet on Machine A!

## Usage

### Broadcast Script

```bash
python3 broadcast_component.py --help
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--component` | (required) | Component to subscribe to (e.g., `bdx.world_pos`) |
| `--rename` | same as component | Rename component for broadcast |
| `--broadcast-rate` | 10 | Broadcast rate in Hz |
| `--broadcast-port` | 41235 | UDP broadcast port |
| `--source-id` | `source` | Source identifier |
| `--db-addr` | `127.0.0.1:2240` | Elodin-DB address |
| `-v, --verbose` | | Enable verbose logging |

**Examples:**

```bash
# Basic: broadcast world_pos unchanged
python3 broadcast_component.py --component bdx.world_pos

# Rename for target tracking
python3 broadcast_component.py \
    --component bdx.world_pos \
    --rename target.world_pos

# High-rate broadcast
python3 broadcast_component.py \
    --component bdx.world_pos \
    --broadcast-rate 50
```

### Receive Script

```bash
python3 receive_broadcast.py --help
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--listen-port` | 41235 | UDP listen port |
| `--db-addr` | `127.0.0.1:2240` | Elodin-DB address for writing |
| `--filter` | (none) | Only accept specific components (repeatable) |
| `--timestamp-mode` | `sender` | Timestamp mode: `sender`, `local`, or `monotonic` |
| `-v, --verbose` | | Enable verbose logging |

**Timestamp Modes:**
| Mode | Description |
|------|-------------|
| `sender` | Use timestamp from broadcaster (default) |
| `local` | Use local wall-clock time |
| `monotonic` | Use Linux monotonic clock (relative to system boot) |

**Examples:**

```bash
# Receive all broadcasts
python3 receive_broadcast.py

# Filter to specific components
python3 receive_broadcast.py --filter target.world_pos

# Custom port
python3 receive_broadcast.py --listen-port 41300

# Use local wall-clock time instead of sender timestamp
python3 receive_broadcast.py --timestamp-mode local

# Use monotonic clock (useful for consistent timing)
python3 receive_broadcast.py --timestamp-mode monotonic
```

## Protocol

The broadcast uses Protocol Buffers for serialization with two message types:

**ComponentBroadcast**: Component data
- `source_id`, `component_name`, `renamed_component`
- `timestamp_us`, `sequence`
- `data_type` (F32, F64, etc.), `shape`, `data`

**BroadcastHeartbeat**: Connection monitoring
- `source_id`, `components`, `broadcast_rate_hz`, `timestamp_us`

## Troubleshooting

### No data being received

1. Verify both machines are on the same subnet
2. Check firewall allows UDP on port 41235
3. Use `-v` flag to enable verbose logging

### Connection to Elodin-DB fails

1. Verify Elodin-DB is running
2. The receiver can run in print-only mode without a DB connection

## File Structure

```
udp_component_broadcast/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── component_broadcast.proto    # Protobuf message definitions
├── component_broadcast_pb2.py   # Generated protobuf code
├── broadcast_component.py       # Sender script
└── receive_broadcast.py         # Receiver script
```

## License

See repository LICENSE file.
