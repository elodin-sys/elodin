# Tegrastats Bridge

Reads system telemetry from the NVIDIA Orin SoC and writes it to `elodin-db`.

## Overview

The tegrastats bridge:
1. Connects to `elodin-db` at `127.0.0.1:2240`
2. Reads CPU usage, frequency, temperature, and GPU load from `/sys` filesystem
3. Publishes system telemetry at ~200Hz for real-time monitoring

## Telemetry Format

The bridge publishes an `Output` struct with system metrics:

```rust
#[repr(C)]
pub struct Output {
    pub cpu_usage: [f32; 8],      // CPU usage per core (0-100%)
    pub cpu_freq: [f32; 8],       // CPU frequency per core in kHz
    pub thermal_zones: [f32; 10], // Temperature sensors in °C
    pub gpu_usage: f32,           // GPU usage (0-1000, divide by 1000 for percentage)
}
```

Total size: 108 bytes (27 x 4-byte floats)

### Data Sources

The bridge reads from these `/sys` filesystem locations:

- **CPU Usage**: Calculated via `sysinfo` crate
- **CPU Frequency**: `/sys/devices/system/cpu/cpu{0-7}/cpufreq/scaling_cur_freq`
- **Thermal Zones**: `/sys/devices/virtual/thermal/thermal_zone{0-9}/temp`
- **GPU Load**: `/sys/devices/platform/gpu.0/load`

## Database Setup

To properly decode the telemetry data, configure `elodin-db` with a VTable definition matching the `Output` structure.

### Using Lua Configuration

Create a configuration file `tegrastats-config.lua`:

```lua
-- Define VTable for Orin system telemetry
-- The VTable describes the binary layout of the Output struct
tegrastats_vtable = vtable_msg(2, {
    field(0,  32, schema("f32", {8}, component("aleph.cpu_usage"))),
    field(32, 32, schema("f32", {8}, component("aleph.cpu_freq"))),
    field(64, 40, schema("f32", {10}, component("aleph.thermal_zones"))),
    field(104, 4, schema("f32", {}, component("aleph.gpu_usage"))),
})

-- Set up metadata for better visualization
msgs = {
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.cpu_usage"), 
        name = "CPU Usage",
        metadata = { 
            unit = "%",
            element_names = "core0,core1,core2,core3,core4,core5,core6,core7"
        }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.cpu_freq"), 
        name = "CPU Frequency",
        metadata = { 
            unit = "kHz",
            element_names = "core0,core1,core2,core3,core4,core5,core6,core7"
        }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.thermal_zones"), 
        name = "Thermal Zones",
        metadata = { 
            unit = "°C",
            element_names = "zone0,zone1,zone2,zone3,zone4,zone5,zone6,zone7,zone8,zone9"
        }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.gpu_usage"), 
        name = "GPU Usage",
        metadata = { unit = "‰" }  -- Note: value is in parts per thousand
    }),
    tegrastats_vtable,
}

-- Connect to local elodin-db and send configuration
client = connect("127.0.0.1:2240")
client:send_msgs(msgs)
```

Start the database with this configuration:

```bash
elodin-db run [::]:2240 /path/to/db --config tegrastats-config.lua
```

### Using C++ Client

Alternatively, bootstrap the database using a C++ client:

```cpp
#include "db.hpp"

using namespace vtable::builder;

struct Output {
    float cpu_usage[8];
    float cpu_freq[8];
    float thermal_zones[10];
    float gpu_usage;
};

int main() {
    auto sock = Socket("127.0.0.1", 2240);
    
    // Define the VTable matching the Output struct
    auto vtable = vtable({
        field<Output, &Output::cpu_usage>(
            schema(PrimType::F32(), {8}, component("aleph.cpu_usage"))),
        field<Output, &Output::cpu_freq>(
            schema(PrimType::F32(), {8}, component("aleph.cpu_freq"))),
        field<Output, &Output::thermal_zones>(
            schema(PrimType::F32(), {10}, component("aleph.thermal_zones"))),
        field<Output, &Output::gpu_usage>(
            schema(PrimType::F32(), {}, component("aleph.gpu_usage"))),
    });
    
    sock.send(VTableMsg { .id = {2, 0}, .vtable = vtable });
    
    // Send metadata for better display
    sock.send(set_component_metadata("aleph.cpu_usage", 
        {"core0", "core1", "core2", "core3", "core4", "core5", "core6", "core7"}));
    sock.send(set_component_metadata("aleph.cpu_freq",
        {"core0", "core1", "core2", "core3", "core4", "core5", "core6", "core7"}));
    sock.send(set_component_metadata("aleph.thermal_zones",
        {"zone0", "zone1", "zone2", "zone3", "zone4", "zone5", "zone6", "zone7", "zone8", "zone9"}));
}
```

### Using Rust Client

You can also use the Rust client from `libs/db/examples/rust_client`:

```rust
use impeller2::types::{ComponentId, PrimType};
use impeller2::vtable::builder::{vtable, raw_field, schema, component};
use impeller2_wkt::{VTableMsg, SetComponentMetadata};

async fn setup_tegrastats(client: &mut Client) -> Result<()> {
    // VTable ID must match what tegrastats-bridge uses
    let vtable_id = [2, 0];
    
    let vtable = vtable(vec![
        raw_field(0, 32, schema(PrimType::F32, &[8], component("aleph.cpu_usage"))),
        raw_field(32, 32, schema(PrimType::F32, &[8], component("aleph.cpu_freq"))),
        raw_field(64, 40, schema(PrimType::F32, &[10], component("aleph.thermal_zones"))),
        raw_field(104, 4, schema(PrimType::F32, &[], component("aleph.gpu_usage"))),
    ]);
    
    client.send(&VTableMsg { id: vtable_id, vtable }).await?;
    client.send(&SetComponentMetadata::new("aleph.cpu_usage", "CPU Usage")).await?;
    // ... send metadata for other components
    
    Ok(())
}
```

## Deployment

On Aleph, tegrastats-bridge runs as a systemd service that automatically starts on boot. The service is defined in `aleph/modules/tegrastats-bridge.nix`.

### Manual Testing

To test the tegrastats bridge manually:

```bash
# On Aleph
RUST_LOG=debug tegrastats-bridge
```

### Service Management

```bash
# Check service status
systemctl status tegrastats-bridge

# View logs
journalctl -u tegrastats-bridge -f

# Restart service
systemctl restart tegrastats-bridge
```

## Visualizing Data

Once the database is configured, visualize system telemetry using the Elodin editor:

```bash
elodin editor 127.0.0.1:2240
```

Useful graphs to create:
- **CPU Usage**: Line graph of `aleph.cpu_usage` - shows per-core utilization
- **CPU Frequency**: Line graph of `aleph.cpu_freq` - shows frequency scaling
- **Thermal Zones**: Line graph of `aleph.thermal_zones` - monitor temperatures
- **GPU Usage**: Line graph of `aleph.gpu_usage` - GPU load (remember to divide by 1000)

## Performance Impact

The tegrastats bridge has minimal performance impact:
- ~200Hz update rate (5ms sleep between samples)
- Reads are from in-memory `/sys` files (fast)
- Single TCP connection to local database
- No significant CPU or memory overhead

## Troubleshooting

### No telemetry appearing in database

1. **Check service is running**:
   ```bash
   systemctl status tegrastats-bridge
   ```

2. **Verify database is accessible**:
   ```bash
   nc -zv 127.0.0.1 2240
   ```

3. **Check logs for errors**:
   ```bash
   journalctl -u tegrastats-bridge -n 50
   ```

### Missing `/sys` files

If thermal zone or GPU load files are missing:
- Some files may not exist on all Orin variants
- The bridge handles missing files gracefully (returns `NaN`)
- Check which thermal zones exist:
  ```bash
  ls -l /sys/devices/virtual/thermal/
  ```

### VTable ID mismatch

The tegrastats bridge uses VTable ID `[2, 0]`. Ensure your database configuration uses the same ID, or telemetry won't be decoded.

### High CPU usage

If the bridge is using excessive CPU:
- Check the sleep duration (should be 5ms)
- Verify no other processes are reading the same `/sys` files rapidly
- Consider increasing the sleep duration if real-time monitoring isn't critical

## Data Format Details

### CPU Usage
- Range: 0-100 (percentage)
- Updated via `sysinfo` crate with `CpuRefreshKind::everything()`
- One value per core (8 cores on Orin NX)

### CPU Frequency  
- Units: kHz (kilohertz)
- Read from kernel's cpufreq subsystem
- Shows current operating frequency (affected by governor and thermal throttling)

### Thermal Zones
- Units: °C (degrees Celsius)
- Raw values from sysfs are in millidegrees, converted by dividing by 1000
- Typically includes: CPU cores, GPU, board sensors

### GPU Usage
- Units: Parts per thousand (‰)
- Raw value from `/sys/devices/platform/gpu.0/load` (0-1000)
- Divide by 1000 to get percentage (0-100%)
- Divide by 10 to get 0-1 range

