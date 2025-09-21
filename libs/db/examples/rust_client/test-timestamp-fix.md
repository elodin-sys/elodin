# Testing the Timestamp Fix

## Test Procedure

### 1. Start Fresh Rocket Simulation
```bash
cd libs/nox-py
python examples/rocket.py run 0.0.0.0:2240
```

### 2. Run the Updated Rust Client
In another terminal:
```bash
cd /Users/danieldriscoll/dev/elodin
RUST_LOG=info ./target/release/rust_client --host 127.0.0.1 --port 2240
```

### 3. Verify No Time Travel Errors
Watch the rocket simulation terminal. You should see:
- ✅ Normal telemetry logging
- ✅ Component registration messages
- ❌ NO "time travel" warnings
- ❌ NO "error committing head" messages

### 4. Connect Elodin Editor (Optional)
```bash
elodin editor 127.0.0.1:2240
```
The editor should:
- Display telemetry correctly
- Show continuous data updates
- Not freeze or show gaps in data

## Expected Output

### Before Fix (BAD) ❌
```
WARN elodin_db::time_series: time travel last_timestamp=Timestamp(1758413363095391) timestamp=Timestamp(1758413362764224)
WARN nox_ecs::impeller2_server: error committing head err=DB(TimeTravel)
```

### After Fix (GOOD) ✅
```
INFO elodin_db: inserting component.id=7786622236758618069
INFO nox_ecs::impeller2_server: running server with cancellation
# Normal operation, no warnings
```

## What Changed
The control client now sends packets with explicit timestamps in the VTable definition, preventing timestamp conflicts between the simulation and control data streams.

## Troubleshooting
If you still see time travel errors:
1. Ensure you rebuilt the Rust client after the fix
2. Check that both processes are using port 2240
3. Restart both the simulation and client
4. Clear any cached database files if necessary
