# Real-Time Streaming Investigation

## Executive Summary

The Elodin-DB system already supports real-time streaming, and the Rust client is configured correctly to receive data in real-time. However, the simulation needs configuration adjustments to ensure it generates data at the proper real-time rate rather than as fast as possible.

## Current State

### Client Configuration ✅ 
The Rust client is already correctly configured for real-time streaming:

```rust
// libs/db/examples/rust_client/src/client.rs
let stream = Stream {
    behavior: StreamBehavior::RealTime,  // Real-time mode
    id: 1,
};
```

### Stream Behaviors Available

Elodin-DB supports two streaming modes:

1. **`StreamBehavior::RealTime`** (Currently Used)
   - Streams data immediately as it arrives at the database
   - Perfect for live telemetry and control feedback
   - No buffering or delay
   - The `handle_real_time_stream` function waits for new data and sends it immediately

2. **`StreamBehavior::FixedRate`**
   - Plays back data at a specific frequency
   - Can start from:
     - `InitialTimestamp::Earliest` - Beginning of recorded data
     - `InitialTimestamp::Latest` - Most recent data point
     - `InitialTimestamp::Manual(timestamp)` - Specific timestamp
   - Used for replay/analysis scenarios

## The Problem: Simulation Data Rate

The issue is not with the client or streaming mode, but with the **simulation's data generation rate**. Currently:

1. The rocket simulation (`rocket.py`) runs with:
   ```python
   w.run(sys, sim_time_step=SIM_TIME_STEP)  # SIM_TIME_STEP = 1/120 = 8.33ms
   ```

2. The simulation may be running **faster than real-time** if no rate limiting is applied, generating data as quickly as the CPU can compute it.

## Solution: Configure Real-Time Simulation Rate

### Method 1: Add `run_time_step` Parameter (Recommended)

The `w.run()` function accepts a `run_time_step` parameter to control the real-time execution rate:

```python
# libs/nox-py/examples/rocket.py
w.run(
    sys, 
    sim_time_step=SIM_TIME_STEP,        # Physics timestep (1/120 sec)
    run_time_step=1/120.0,               # Real-time rate limiting
    default_playback_speed=1.0           # 1.0 = real-time speed
)
```

**Parameters explained:**
- `sim_time_step`: Physics simulation timestep (how often physics is calculated)
- `run_time_step`: Real-world time between simulation executions
- `default_playback_speed`: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)

### Method 2: Launch with Elodin-DB Address

When the simulation connects to an external Elodin-DB instance, it can be configured to respect real-time constraints:

```bash
# Start the database first
elodin-db run [::]:2240 ~/.elodin/db

# Run the simulation with database address
python rocket.py run [::]:2240
```

The simulation will then stream data to the database at the configured real-time rate.

## Testing Real-Time Streaming

### Verification Steps

1. **Configure the simulation for real-time:**
   ```python
   # In rocket.py, modify the last line:
   w.run(
       sys, 
       sim_time_step=1/120.0,
       run_time_step=1/120.0,    # Add this for real-time
       default_playback_speed=1.0
   )
   ```

2. **Start the components in order:**
   ```bash
   # Terminal 1: Start the Rust client
   ./target/debug/rust_client
   
   # Terminal 2: Start the rocket simulation
   python libs/nox-py/examples/rocket.py run 127.0.0.1:2240
   ```

3. **Observe the behavior:**
   - Data should arrive at ~120Hz (every 8.33ms)
   - The client should display telemetry immediately upon generation
   - No playback from historical data

## Control Feedback Implementation

Once real-time streaming is verified, you can implement the control feedback loop:

1. **Read current telemetry** from the real-time stream
2. **Calculate control inputs** based on telemetry
3. **Send control commands** to the database
4. **Simulation reads control inputs** and adjusts behavior
5. **New telemetry reflects changes** immediately

### Example Control Flow

```rust
// In the Rust client
match telemetry {
    TelemetryRow { component_name: "rocket.angle_of_attack", values, .. } => {
        // Calculate fin control based on angle of attack
        let control_value = calculate_pid_control(values[0]);
        
        // Send control command back to database
        client.send_component_value(
            "rocket.fin_control",
            control_value
        ).await?;
    }
    _ => {}
}
```

## Additional Considerations

### Latency Optimization

For minimal latency in the control loop:

1. **Network**: Use localhost or low-latency network connections
2. **Buffer Sizes**: The client uses a bounded channel of 1000 messages - this could be reduced for lower latency
3. **Processing**: Process control decisions in the telemetry callback for immediate response

### Data Synchronization

When implementing control feedback:
- The simulation timestep (1/120s) defines the control loop frequency
- Control inputs should be synchronized with simulation ticks
- Consider implementing interpolation for control inputs arriving between ticks

## Conclusion

The infrastructure for real-time streaming is already in place and working correctly. The key requirement is to ensure the simulation generates data at real-time rates by adding the `run_time_step` parameter. This will enable the closed-loop control system you need for fin control feedback.

## Next Steps

1. ✅ Client already configured for `StreamBehavior::RealTime`
2. ⏳ Add `run_time_step=1/120.0` to rocket.py simulation
3. ⏳ Implement control command sending in the Rust client
4. ⏳ Add control input reading in the Python simulation
5. ⏳ Test closed-loop control with fin adjustments
