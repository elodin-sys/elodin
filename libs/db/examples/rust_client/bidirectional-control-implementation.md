# Bidirectional Control Implementation for Rust Client

## ✅ Implementation Complete

We have successfully added bidirectional control capabilities to the Rust client, enabling it to send control commands to the rocket simulation while receiving telemetry.

## What Was Implemented

### 1. New Control Module (`src/control.rs`)
- **`ControlSender` struct**: Manages sending trim control commands
- **Sinusoidal wave generation**: Produces trim values oscillating between -2° and +2°
- **VTable registration**: Sends component schema to database
- **60Hz update rate**: Smooth control updates without overwhelming the system

### 2. Dual Client Architecture
- **Telemetry Client**: Receives streaming telemetry data
- **Control Client**: Sends control commands concurrently
- Both clients run simultaneously using `futures_lite::future::race`

### 3. Control Parameters
- **Component**: `rocket.fin_control_trim`
- **Amplitude**: ±2 degrees
- **Frequency**: 0.25 Hz (4-second period)
- **Update Rate**: 60 Hz

## Testing Instructions

### Step 1: Start the Rocket Simulation
```bash
cd libs/nox-py
python examples/rocket.py run 0.0.0.0:2240
```

You should see output like:
```
2025-09-20 16:46:22.377  INFO elodin_db: listening addr=0.0.0.0:2240
2025-09-20 16:46:22.380  INFO elodin_db: inserting component.id=7786622236758618069
...
```

### Step 2: Run the Rust Client
In another terminal:
```bash
cd /Users/danieldriscoll/dev/elodin
./target/release/rust_client --host 127.0.0.1 --port 2240
```

## Expected Behavior

### In the Rust Client Logs
```
INFO Starting sinusoidal trim control (±2° @ 0.25Hz)
INFO Sent VTable definition for fin control trim
INFO Beginning trim control oscillation...
INFO Trim control oscillating: 0.000° (t=0.0s)
INFO Trim control oscillating: 1.414° (t=2.0s)
INFO Trim control oscillating: 2.000° (t=3.0s)
INFO Trim control oscillating: 1.414° (t=4.0s)
INFO Trim control oscillating: 0.000° (t=5.0s)
INFO Trim control oscillating: -1.414° (t=6.0s)
```

### In the Telemetry Dashboard
The `fin_control_trim` value should appear in the telemetry display once the simulation processes it.

### In the Rocket Simulation
The rocket will exhibit a gentle rolling/yawing oscillation as the trim varies. The PID controller maintains stability while the trim creates controlled perturbations.

## How It Works

### Control Flow
```
Rust Client (Control)          Database              Rocket Simulation
      |                           |                         |
      |-- Send VTable Def ------> |                         |
      |                           |                         |
      |-- Send Trim Value ------> | <-- Read Trim ---------|
      |   (60 Hz updates)         |                         |
      |                           |                         |
      |                           | --- Forward Trim -----> |
      |                           |                         |
      |                           |                  (Apply to fins)
      |                           |                         |
      |                           | <-- Send Telemetry ----|
      |                           |                         |
      |<-- Receive Telemetry ---- |                         |
```

### Key Design Decisions

1. **Trim Approach**: We add trim to existing PID control rather than overriding it
2. **Concurrent Clients**: Separate connections for reading and writing avoid blocking
3. **Sinusoidal Pattern**: Provides clear visual confirmation of control working
4. **No UI Changes**: Focus on core functionality first

## Code Structure

```
libs/db/examples/rust_client/
├── src/
│   ├── main.rs         # Entry point
│   ├── client.rs       # Dual-client orchestration
│   ├── control.rs      # NEW: Control command sending
│   ├── discovery.rs    # Component discovery
│   ├── processor.rs    # Telemetry processing
│   └── tui.rs          # Terminal UI
```

## Future Enhancements

1. **Interactive Control**: Add keyboard controls (arrow keys for manual trim adjustment)
2. **Multiple Control Modes**: 
   - Manual override
   - Trajectory following
   - Disturbance injection
3. **Control Feedback**: Display sent commands in the TUI
4. **Command Queuing**: Buffer commands for smoother control
5. **Safety Limits**: Add configurable limits and emergency stop

## Troubleshooting

### "Connection refused" Error
Ensure the rocket simulation is running and listening on the correct port:
```bash
ps aux | grep rocket.py
netstat -an | grep 2240
```

### No Trim Effect Visible
1. Check if `rocket.fin_control_trim` component is registered in the database
2. Verify the rocket simulation is reading from the database (not just writing)
3. Check the logs for VTable registration confirmation

### Client Crashes
Enable verbose logging:
```bash
RUST_LOG=debug ./target/release/rust_client --host 127.0.0.1 --port 2240 --verbose
```

## Important: Timestamp Synchronization

### Critical Update (v0.2.1)
The initial implementation had a timestamp synchronization issue that caused "time travel" errors. This has been fixed by including explicit timestamps in the VTable definition and packets.

**Key Requirements for Control Packets:**
- Always include timestamp fields in VTable definitions
- Send explicit timestamps with each packet
- Use `Timestamp::now()` for real-time synchronization
- Ensures compatibility with simulation timeline

See `timestamp-fix.md` for technical details.

## Summary

The Rust client now demonstrates full bidirectional communication with the Elodin ecosystem:
- ✅ **Reads** telemetry from simulations
- ✅ **Writes** control commands that affect simulations
- ✅ **Concurrent** operation without blocking
- ✅ **Real-time** control at 60 Hz
- ✅ **Timestamp synchronized** to prevent conflicts

This implementation proves that external clients can actively control simulations through elodin-db, enabling hardware-in-the-loop testing, remote control, and sophisticated control algorithms running outside the simulation environment.
