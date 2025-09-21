# Timestamp Synchronization Fix

## Problem
The "time travel" errors occurred when both the simulation and control client wrote to the database with conflicting timestamps:
```
WARN elodin_db::time_series: time travel last_timestamp=Timestamp(1758413363095391) timestamp=Timestamp(1758413362764224)
```

## Root Cause
When table packets don't include timestamp fields in their VTable definition:
1. The database assigns `Timestamp::now()` to incoming data
2. The simulation has its own timeline with different timestamps
3. Timestamps can appear to go "backwards" causing rejection

## Solution
Include explicit timestamps in the VTable definition and packets:

### Before (No Timestamp)
```rust
// VTable without timestamp field
let vtable = vtable(vec![
    raw_field(0, 8, schema(PrimType::F64, &[], component(self.trim_component_id))),
]);

// Packet with just the value
let mut packet = LenPacket::table(self.trim_vtable_id, 8); // 8 bytes f64
packet.extend_aligned(&trim_value.to_le_bytes());
```

### After (With Timestamp)
```rust
// VTable with timestamp field
let time_field = raw_table(0, 8);  // First 8 bytes for timestamp
let vtable = vtable(vec![
    raw_field(8, 8, schema(PrimType::F64, &[], timestamp(time_field, component(self.trim_component_id)))),
]);

// Packet with timestamp and value
let mut packet = LenPacket::table(self.trim_vtable_id, 16); // 8 bytes timestamp + 8 bytes f64
packet.extend_aligned(&timestamp.0.to_le_bytes());
packet.extend_aligned(&trim_value.to_le_bytes());
```

## How It Works
1. **Timestamp Field**: The VTable now declares that the first 8 bytes contain a timestamp
2. **Explicit Timestamps**: Each packet includes the current timestamp
3. **Database Respects Timestamps**: The database uses our timestamps instead of generating its own
4. **No Time Conflicts**: Both simulation and control client have valid, monotonically increasing timestamps

## Testing
After this fix:
- ✅ No more "time travel" warnings
- ✅ Elodin Editor displays data correctly
- ✅ Both simulation and control data coexist peacefully

## Key Learning
Always include timestamps in VTable definitions when sending time-series data to elodin-db. This ensures proper temporal ordering and prevents conflicts between multiple data sources.
