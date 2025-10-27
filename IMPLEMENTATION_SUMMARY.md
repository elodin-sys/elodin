# External Database Mirroring - Final Implementation

## What We Built

A feature that allows Elodin simulations to optionally mirror data to an external `elodin-db` instance, leveraging the database's existing mirroring capability.

## Design Evolution

**Initial Approach (Discarded)**: Created a new `impeller2_client` module with dual-mode behavior
- ❌ 230+ lines of new code
- ❌ Complex VTable streaming implementation needed
- ❌ Duplicate behavior for embedded vs external modes

**Second Approach (Discarded)**: Modified `impeller2_server` with `DbConnection` enum
- ❌ ~140 lines of new code
- ❌ Still needed custom streaming implementation
- ❌ Changed core server behavior

**Final Approach (Implemented)**: Use existing database mirroring ✅
- ✅ **Only ~90 lines total**
- ✅ **Zero custom streaming code** - uses built-in `UdpUnicast`
- ✅ **Zero behavior changes** - embedded DB works exactly as before
- ✅ **Proven infrastructure** - same as `downlink.lua` example

## Implementation

### Changes Made

#### 1. `libs/nox-ecs/src/impeller2_server.rs` (~55 lines)

```rust
// Added optional mirror_addr field
pub struct Server {
    db: elodin_db::Server,
    world: WorldExec<Compiled>,
    mirror_addr: Option<SocketAddr>,  // NEW
}

// New constructor for mirroring
pub fn with_mirror(db: elodin_db::Server, world: WorldExec<Compiled>, mirror_addr: SocketAddr) -> Self

// Modified run_with_cancellation to call setup_mirror if mirror_addr is set

// New helper function
async fn setup_mirror(source_addr, mirror_addr, db) -> Result<(), Error>
```

#### 2. `libs/nox-py/src/world_builder.rs` (~10 lines)

```rust
// Added CLI argument
#[arg(long, default_value = None)]
db_addr: Option<SocketAddr>,

// Choose constructor based on db_addr
let server = if let Some(mirror_addr) = db_addr_clone {
    Server::with_mirror(embedded_db, exec, mirror_addr)
} else {
    Server::new(embedded_db, exec)
};
```

#### 3. `libs/nox-py/python/elodin/__init__.py` (~30 lines)

```python
def run(self, system, ..., db_addr: Optional[str] = None):
    # Injects --db-addr into sys.argv if provided
```

### How Mirroring Works

```
┌─────────────────────────────────────────────────────────────┐
│  Simulation Process                                         │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │ Embedded elodin-db (temp directory)        │             │
│  │  - Stores all simulation data              │             │
│  │  - Accepts local connections               │             │
│  │  - Streams via UDP when configured         │             │
│  └──────────────┬─────────────────────────────┘             │
│                 │                                            │
│                 │ ① DumpMetadata                            │
│                 │ ② SetComponentMetadata (to mirror)        │
│                 │ ③ UdpUnicast (configure streaming)        │
│                 │                                            │
└─────────────────┼────────────────────────────────────────────┘
                  │
                  │ Real-time UDP streaming (automatic)
                  │
                  ▼
    ┌──────────────────────────────────┐
    │  External elodin-db              │
    │   - Persistent storage           │
    │   - Multiple editors can connect │
    │   - Historical data preserved    │
    └──────────────────────────────────┘
```

## Usage

### Python API

```python
import elodin as el

w = el.World()
# ... build simulation ...

# Option 1: Embedded only (original behavior)
w.run(system)

# Option 2: Embedded + Mirror to external database
w.run(system, db_addr="127.0.0.1:2240")
```

### Command Line

```bash
# Start external database first
elodin-db run "[::]:2240" /tmp/my-database

# Run simulation with mirroring
python examples/rocket/main.py run --db-addr 127.0.0.1:2240

# View in editor (connected to external DB)
elodin editor 127.0.0.1:2240
```

## Testing

### Quick Test

Terminal 1:
```bash
elodin-db run "[::]:2240" /tmp/test-db
```

Terminal 2:
```bash
cd libs/nox-py
source .venv/bin/activate
python ../../examples/test_external_db.py
```

**Look for these log messages:**
- `configuring mirror: ... -> ...`
- `dumped N components from source database`
- `metadata sent to mirror database`
- `mirror streaming configured successfully`

Terminal 3 (optional):
```bash
elodin editor 127.0.0.1:2240
```

Should show live visualization of the simulation data!

## Advantages Over Other Approaches

| Aspect | This Approach | Client-Only Mode | Dual Mode |
|--------|--------------|------------------|-----------|
| **Code Added** | ~90 lines | ~230 lines | ~200 lines |
| **Embedded DB Behavior** | Unchanged | Replaced | Conditionally changed |
| **Streaming Implementation** | Reused (UdpUnicast) | Custom VTable code | Custom VTable code |
| **Risk** | Minimal | Medium | Medium |
| **Complexity** | Low | High | High |
| **Testing** | Embedded mode unaffected | All simulations affected | Both modes need testing |

## Known Limitations

1. **UDP Only**: Mirroring uses UDP (may have packet loss on congested networks)
2. **One-Way**: Data flows embedded → external, not bidirectional
3. **External Control**: For external control of simulations, still use the existing `external_control` metadata pattern

## Future Enhancements (Optional)

If needed, could add:
1. **TCP mirroring**: Use `UdpVTableStream` equivalent for TCP (if it exists)
2. **Bidirectional sync**: Mirror data back from external DB to simulation
3. **Multiple mirrors**: Support mirroring to multiple external databases
4. **Reconnection**: Handle external database disconnections gracefully

But these aren't needed for the core use case!

## Migration Guide

Existing simulations work unchanged:
```python
w.run(system)  # ✅ Still works exactly as before
```

To add mirroring, just add one parameter:
```python
w.run(system, db_addr="127.0.0.1:2240")  # ✅ New capability
```

That's it! No other code changes needed.

---

**Status: COMPLETE ✅**

This implementation solves your original problem ("connect to an already existing database") while requiring minimal code and zero behavior changes to the existing system.

