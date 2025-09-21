# Future Timestamp Workaround

## The Core Problem

The simulation and control client have incompatible timestamp strategies:

### Simulation Timeline
1. Starts with `timestamp = Timestamp::now()`
2. Increments by `sim_time_step` each tick (e.g., 1/120 second)
3. Reads ALL components from database
4. Runs simulation tick
5. **Writes ALL components back** with its calculated timestamp

### Control Client Timeline
1. Uses `Timestamp::now()` for each packet
2. Runs independently from simulation
3. Timestamps may be slightly behind or ahead

### The Conflict
When both write to the same component (`rocket.fin_control_trim`):
- Simulation writes with timestamp T
- Control client writes with timestamp T' 
- If T' < T, database rejects with "time travel" error
- Components can only have monotonically increasing timestamps

## The Future Timestamp Solution

Write control values with timestamps **far in the future** (e.g., 10 seconds ahead):

```rust
// Instead of current time
let timestamp = Timestamp::now();

// Use future time
let timestamp = Timestamp(Timestamp::now().0 + 10_000_000_000); // 10 seconds ahead
```

### Why This Works

1. **Database accepts future writes**: No time travel error since future > current
2. **Simulation reads latest value**: `copy_db_to_world` gets our future value
3. **Simulation writes don't overwrite**: Its current timestamp < our future timestamp
4. **Control values persist**: Our future values remain the "latest" in the time series

### Timeline Visualization

```
Time ->
[Sim@T=100] [Sim@T=200] [Sim@T=300] [Sim@T=400] ...
                  ^
                  |
                  Reads control value from T=10000+
                  Writes back at T=200 (doesn't overwrite)
                  
[Control writes @ T=10000+] -----------------------> Always latest
```

## Implementation

```rust
pub async fn send_trim_update(&mut self, client: &mut Client) -> Result<()> {
    // ... calculate trim_value ...
    
    // Write 10 seconds into the future
    let timestamp = Timestamp(Timestamp::now().0 + 10_000_000_000);
    
    // Build packet with future timestamp
    let mut packet = LenPacket::table(self.trim_vtable_id, 16);
    packet.extend_aligned(&timestamp.0.to_le_bytes());
    packet.extend_aligned(&trim_value.to_le_bytes());
    
    // Send to database
    let (result, _) = client.send(packet).await;
    result?;
}
```

## Advantages

✅ No time travel errors
✅ Simulation reads control values correctly
✅ No modifications needed to simulation code
✅ Works with existing infrastructure

## Limitations

⚠️ Control values appear "in the future" in database queries
⚠️ Must ensure future offset is larger than simulation duration
⚠️ Timestamps in logs/debug may be confusing

## Alternative Solutions Considered

1. **Separate component name** - Simulation wouldn't read it
2. **Remove from simulation** - Would break `aero_coefs` function
3. **Synchronize timestamps** - Complex, requires coordination
4. **Read-only metadata** - Not supported by current system

The future timestamp approach is the simplest workaround that requires no changes to the simulation or database infrastructure.
