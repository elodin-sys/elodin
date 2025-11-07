# Aleph Serial Bridge

Reads sensor data being streamed over the serial port from the Aleph expansion board and writes it to `elodin-db`.

## Overview

The serial bridge:
1. Connects to `elodin-db` at `127.0.0.1:2240`
2. Reads sensor data from `/dev/ttyTHS0` (UART connected to expansion board)
3. Decodes COBS-framed sensor records
4. Forwards the data to the database as binary packets

## Sensor Data Format

The expansion board streams `Record` structs over UART:

```rust
#[repr(C)]
pub struct Record {
    pub ts: u32,           // Timestamp in milliseconds
    pub mag: [f32; 3],     // Magnetometer (x, y, z) in μT
    pub gyro: [f32; 3],    // Gyroscope (x, y, z) in rad/s
    pub accel: [f32; 3],   // Accelerometer (x, y, z) in m/s²
    pub mag_temp: f32,     // Magnetometer temperature in °C
    pub mag_sample: u32,   // Magnetometer sample counter
    pub baro: f32,         // Barometric pressure in Pa
    pub baro_temp: f32,    // Barometer temperature in °C
    pub vin: f32,          // Input voltage in V
    pub vbat: f32,         // Battery voltage in V
    pub aux_current: f32,  // Auxiliary current in A
    pub rtc_vbat: f32,     // RTC battery voltage in V
    pub cpu_temp: f32,     // CPU temperature in °C
}
```

Total size: 68 bytes (17 x 4-byte fields)

## Database Setup

The serial bridge sends raw binary data to the database. To properly decode this data, you need to configure `elodin-db` with a VTable definition that matches the `Record` structure.

### Using Lua Configuration

Create a configuration file `serial-bridge-config.lua`:

```lua
-- Define VTable for sensor data
-- The VTable describes the binary layout of the Record struct
sensor_vtable = vtable_msg(1, {
    field(0,  4, schema("u32", {}, component("aleph.ts"))),
    field(4,  12, schema("f32", {3}, component("aleph.mag"))),
    field(16, 12, schema("f32", {3}, component("aleph.gyro"))),
    field(28, 12, schema("f32", {3}, component("aleph.accel"))),
    field(40, 4, schema("f32", {}, component("aleph.mag_temp"))),
    field(44, 4, schema("u32", {}, component("aleph.mag_sample"))),
    field(48, 4, schema("f32", {}, component("aleph.baro"))),
    field(52, 4, schema("f32", {}, component("aleph.baro_temp"))),
    field(56, 4, schema("f32", {}, component("aleph.vin"))),
    field(60, 4, schema("f32", {}, component("aleph.vbat"))),
    field(64, 4, schema("f32", {}, component("aleph.aux_current"))),
    field(68, 4, schema("f32", {}, component("aleph.rtc_vbat"))),
    field(72, 4, schema("f32", {}, component("aleph.cpu_temp"))),
})

-- Set up metadata for better visualization
msgs = {
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.mag"), 
        name = "Magnetometer",
        metadata = { unit = "μT", element_names = "x,y,z" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.gyro"), 
        name = "Gyroscope",
        metadata = { unit = "rad/s", element_names = "x,y,z" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.accel"), 
        name = "Accelerometer",
        metadata = { unit = "m/s²", element_names = "x,y,z" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.mag_temp"), 
        name = "Mag Temperature",
        metadata = { unit = "°C" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.baro"), 
        name = "Barometric Pressure",
        metadata = { unit = "Pa" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.baro_temp"), 
        name = "Baro Temperature",
        metadata = { unit = "°C" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.vin"), 
        name = "Input Voltage",
        metadata = { unit = "V" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.vbat"), 
        name = "Battery Voltage",
        metadata = { unit = "V" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.aux_current"), 
        name = "Auxiliary Current",
        metadata = { unit = "A" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.rtc_vbat"), 
        name = "RTC Battery Voltage",
        metadata = { unit = "V" }
    }),
    SetComponentMetadata({ 
        component_id = ComponentId("aleph.cpu_temp"), 
        name = "CPU Temperature",
        metadata = { unit = "°C" }
    }),
    sensor_vtable,
}

-- Connect to local elodin-db and send configuration
client = connect("127.0.0.1:2240")
client:send_msgs(msgs)
```

Then start the database with this configuration:

```bash
elodin-db run [::]:2240 /path/to/db --config serial-bridge-config.lua
```

### Using C++ Client

Alternatively, bootstrap the database using a C++ client:

```cpp
#include "db.hpp"

using namespace vtable::builder;

struct Record {
    uint32_t ts;
    float mag[3];
    float gyro[3];
    float accel[3];
    float mag_temp;
    uint32_t mag_sample;
    float baro;
    float baro_temp;
    float vin;
    float vbat;
    float aux_current;
    float rtc_vbat;
    float cpu_temp;
};

int main() {
    auto sock = Socket("127.0.0.1", 2240);
    
    // Define the VTable matching the Record struct
    auto vtable = vtable({
        field<Record, &Record::ts>(schema(PrimType::U32(), {}, component("aleph.ts"))),
        field<Record, &Record::mag>(schema(PrimType::F32(), {3}, component("aleph.mag"))),
        field<Record, &Record::gyro>(schema(PrimType::F32(), {3}, component("aleph.gyro"))),
        field<Record, &Record::accel>(schema(PrimType::F32(), {3}, component("aleph.accel"))),
        field<Record, &Record::mag_temp>(schema(PrimType::F32(), {}, component("aleph.mag_temp"))),
        field<Record, &Record::mag_sample>(schema(PrimType::U32(), {}, component("aleph.mag_sample"))),
        field<Record, &Record::baro>(schema(PrimType::F32(), {}, component("aleph.baro"))),
        field<Record, &Record::baro_temp>(schema(PrimType::F32(), {}, component("aleph.baro_temp"))),
        field<Record, &Record::vin>(schema(PrimType::F32(), {}, component("aleph.vin"))),
        field<Record, &Record::vbat>(schema(PrimType::F32(), {}, component("aleph.vbat"))),
        field<Record, &Record::aux_current>(schema(PrimType::F32(), {}, component("aleph.aux_current"))),
        field<Record, &Record::rtc_vbat>(schema(PrimType::F32(), {}, component("aleph.rtc_vbat"))),
        field<Record, &Record::cpu_temp>(schema(PrimType::F32(), {}, component("aleph.cpu_temp"))),
    });
    
    sock.send(VTableMsg { .id = {1, 0}, .vtable = vtable });
    sock.send(set_component_metadata("aleph.mag", {"x", "y", "z"}));
    sock.send(set_component_metadata("aleph.gyro", {"x", "y", "z"}));
    sock.send(set_component_metadata("aleph.accel", {"x", "y", "z"}));
    // ... send metadata for other components
}
```

## Deployment

On Aleph, the serial bridge runs as a systemd service that automatically starts on boot. The service is defined in `aleph/modules/aleph-serial-bridge.nix`.

### Manual Testing

To test the serial bridge manually:

```bash
# On Aleph
RUST_LOG=debug aleph-serial-bridge
```

### Service Management

```bash
# Check service status
systemctl status serial-bridge

# View logs
journalctl -u serial-bridge -f

# Restart service
systemctl restart serial-bridge
```

## Visualizing Data

Once the database is configured, you can visualize the sensor data using the Elodin editor:

```bash
elodin editor 127.0.0.1:2240
```

Create graphs for:
- `aleph.mag` - 3D magnetometer readings
- `aleph.gyro` - Angular velocity
- `aleph.accel` - Linear acceleration
- `aleph.baro` - Barometric pressure
- Temperature sensors (`mag_temp`, `baro_temp`, `cpu_temp`)
- Power metrics (`vin`, `vbat`, `aux_current`)

## Bidirectional Control

The serial bridge also supports sending commands back to the expansion board. Commands are sent as `Command` structs:

```rust
struct Command {
    gpios: [bool; 8],  // GPIO pin states
}
```

To send GPIO commands from an external client, subscribe to the command stream:

```lua
client = connect("127.0.0.1:2240")
client:stream_msgs("Command")
```

## Troubleshooting

### No data appearing in the database

1. **Check serial connection**:
   ```bash
   ls -l /dev/ttyTHS0  # Should exist and be accessible
   ```

2. **Verify expansion board is powered and running**:
   ```bash
   # Monitor serial output
   tio /dev/ttyTHS0
   ```

3. **Check service logs**:
   ```bash
   journalctl -u serial-bridge -n 100
   ```

### COBS framing errors

The expansion board uses COBS (Consistent Overhead Byte Stuffing) framing. If you see framing errors:
- Check baud rate is correct (115200)
- Verify firmware version matches expected Record format
- Check for electrical noise or connection issues

### VTable ID mismatch

The serial bridge uses VTable ID `[1, 0]`. Ensure your database configuration uses the same ID, or the data won't be properly decoded.

