# Testing Bidirectional Database Mirroring

## Overview

This guide demonstrates how to test the bidirectional mirroring feature using the rocket simulation's `fin_control_trim` external control component.

## The Test Scenario

The rocket simulation has a `fin_control_trim` component marked with `external_control: "true"`. We'll:
1. Run the simulation with bidirectional mirroring
2. Write control values to the **external database**
3. Verify they flow back through the mirror to affect the simulation

## Data Flow

```
Rust Client
    ↓ (writes fin_control_trim)
External DB (port 2242)
    ↓ (UDP stream #2 - reverse)
Embedded DB (temp directory)
    ↓ (copy_db_to_world)
Simulation (GPU buffers)
    ↓ (physics applies trim to fins)
Results visible in visualization
```

## Step-by-Step Test

### Terminal 1: Start External Database

```bash
elodin-db run "[::]:2242" /tmp/test-mirror-db
```

This starts a persistent external database on port 2242.

### Terminal 2: Run Rocket Simulation with Mirroring

```bash
cd libs/nox-py
source .venv/bin/activate
python ../../examples/rocket/main.py run --db-addr="127.0.0.1:2242"
```

**Look for these log messages:**
```
INFO nox_ecs::impeller2_server: setting up mirror to external database at 127.0.0.1:2242
INFO nox_ecs::impeller2_server: configuring mirror: [::]:2240 -> 127.0.0.1:2242
INFO nox_ecs::impeller2_server: dumped 28 components from source database
INFO nox_ecs::impeller2_server: metadata sent to mirror database
INFO nox_ecs::impeller2_server: forward streaming configured: embedded → external
INFO nox_ecs::impeller2_server: reverse streaming configured: external → embedded
INFO nox_ecs::impeller2_server: bidirectional mirroring active: [::]:2240 ↔ 127.0.0.1:2242
INFO elodin_db: UDP unicasting to 127.0.0.1:2242  # Forward stream
INFO elodin_db: UDP unicasting to [::]:2240       # Reverse stream
```

### Terminal 3: Send Control Commands via Rust Client

```bash
# Build the Rust client if you haven't already
cargo build --release -p elodin-db-rust-client

# Run it connected to the EXTERNAL database
./target/release/rust_client -H 127.0.0.1:2242
```

The Rust client will:
1. Connect to the **external database** (port 2242)
2. Send sinusoidal `fin_control_trim` values
3. These values flow back through the mirror to the simulation

### Terminal 4: Visualize (Optional)

```bash
# Connect to EITHER database - both have the same data!
elodin editor 127.0.0.1:2242   # External DB
# OR
elodin editor 127.0.0.1:2240   # Embedded DB (temporary)
```

## What You Should See

### In the Logs

**External DB (Terminal 1):**
- Receives metadata from simulation
- Receives real-time simulation data (forward stream)
- Receives control commands from Rust client
- Sends control values back (reverse stream)

**Simulation (Terminal 2):**
- Normal simulation operation
- `fin_control_trim` values updating from external source
- Rocket behavior affected by external control

**Rust Client (Terminal 3):**
- Discovers components including `fin_control_trim`
- Sends sinusoidal trim commands
- Shows telemetry dashboard

### In the Editor (Terminal 4)

- **3D View**: Rocket should rock back and forth due to fin trim
- **fin_control_trim graph**: Shows sinusoidal control input (~±1° at 0.25 Hz)
- **fin_deflect graph**: Shows combined PID + external trim
- **Flight path**: Affected by the external control input

## Verification

The key proof of bidirectionality:

1. **Without external control**: Rocket flies straight (only PID control)
2. **With external control to external DB**: Rocket rocks side-to-side (external trim applied)
3. **Data in both databases**: Same telemetry visible whether you connect editor to port 2240 or 2242

## How It Works Internally

```rust
// In tick loop (libs/nox-ecs/src/impeller2_server.rs:274)
db.with_state(|state| copy_db_to_world(state, &mut world));
```

This function (line 163):
1. Reads latest values from embedded database
2. Compares with simulation state
3. If values changed (e.g., from reverse stream), copies them into simulation
4. Marks components as dirty for GPU update
5. Next simulation tick uses the updated values

The `external_control: "true"` metadata prevents the simulation from writing these components back, so external control has authority.

## Troubleshooting

### "No effect from external control"

Check:
- Component has `external_control: "true"` metadata
- `copy_db_to_world()` is being called (it is, line 274 in tick loop)
- External DB is receiving the control values
- Reverse stream is configured (look for "reverse streaming configured" log)

### "Logs show no reverse stream"

The reverse stream setup happens in `setup_mirror()`. Look for:
```
INFO nox_ecs::impeller2_server: reverse streaming configured: external → embedded
```

If missing, check that both databases are reachable and the mirror setup didn't error.

## Next Steps

Try modifying the Rust client (`libs/db/examples/rust_client/src/control.rs`) to:
- Send different control patterns (step functions, ramps, etc.)
- Control different components
- Demonstrate multi-client control

The bidirectional mirror makes this all possible without any simulation code changes!

