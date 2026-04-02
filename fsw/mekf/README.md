# Aleph Sensor Pipeline & MEKF Engineering Specification

## Overview

The Aleph flight computer runs a sensor pipeline that reads dual BMI270 IMUs, a BMM350 magnetometer, a BMP581 barometer, an optional QMC5883L external compass, and an optional u-blox GPS (M10Q or M9N) on an STM32H747 microcontroller. The raw sensor data is pre-integrated using a coning & sculling filter, then streamed over UART to the Orin NX where it is written to Elodin-DB and consumed by a Multiplicative Extended Kalman Filter (MEKF) for attitude estimation.

---

## Hardware

### STM32H747 (Cortex-M7 @ 400 MHz)

| Sensor | Part | Bus | ODR | Notes |
|---|---|---|---|---|
| IMU (primary) | BMI270 (U16) | I2C2 @ 1 MHz | 1600 Hz | Back of PCB |
| IMU (secondary) | BMI270 (U18) | SPI5 | 1600 Hz | Front of PCB, optional |
| Magnetometer | BMM350 | I2C3 | ~400 Hz | On-board |
| Barometer | BMP581 | I2C3 | ~50 Hz | On-board |
| GPS | u-blox M10Q or M9N | USART2 | 5 Hz | External on J7, optional |
| External compass | QMC5883L | I2C4 | 50 Hz | External on J7, optional |

### Orin NX (Linux, aarch64)

| Service | Purpose |
|---|---|
| `serial-bridge` | Reads UART1 COBS frames, writes to Elodin-DB |
| `elodin-db` | Time-series telemetry database on port 2240 |
| `mekf` | Attitude estimation from IMU/mag data |

---

## Coordinate Frame: FRD Body Frame

All sensor data output by the STM32 is in **FRD (Forward-Right-Down)** body frame:

- **+X** = forward (board front edge, toward connectors)
- **+Y** = right
- **+Z** = down (toward Earth when level)

When the board is level and stationary:
- `accel = [0, 0, +1.0]` (gravity in +Z = down)
- `gyro = [0, 0, 0]` (no rotation)

### Axis Corrections Applied on STM32

The two BMI270 IMUs are mounted on opposite sides of the PCB and require different axis transformations to produce FRD output:

| Sensor | Location | Raw-to-FRD Transform |
|---|---|---|
| BMI270 I2C (U16) | Back of PCB | `[x, y, -z]` |
| BMI270 SPI (U18) | Front of PCB | `[-x, y, z]` |

These corrections are applied in `fsw/sensor-fw/src/main.rs` before the coning/sculling integrator. Both sensors produce identical FRD output after correction.

---

## Coning & Sculling Pre-Integration

The STM32 performs 2-sample Bortz coning and sculling correction on the raw IMU data before transmission. This removes high-frequency conical motion and rotation-acceleration coupling errors that would be lost during decimation.

### Algorithm

Implemented in `fsw/sensor-fw/src/coning_sculling.rs`:

```
FOR EACH raw sample (gyro_frd, accel_frd, dt):
  delta_angle = gyro * dt
  delta_vel   = accel * dt

  coning_correction += (2/3) * cross(prev_delta_angle, delta_angle)
  sculling_correction = (1/2) * cross(accum_delta_angle, delta_vel)
  corrected_delta_vel = delta_vel + sculling_correction

  accum_delta_angle += delta_angle
  accum_delta_vel   += corrected_delta_vel

  EVERY N samples:
    output_gyro  = (accum_delta_angle + coning_correction) / accum_dt
    output_accel = accum_delta_vel / accum_dt
    EMIT and RESET
```

### Decimation

| IMU Config | Input Rate | N | Output Rate |
|---|---|---|---|
| 2 IMUs (I2C + SPI) | ~3,080 Hz combined | 4 | ~770 Hz |
| 1 IMU (I2C only) | ~1,530 Hz | 2 | ~765 Hz |

The output rate is consistent at ~765-770 Hz regardless of IMU count.

### Output Units

| Field | Unit | Description |
|---|---|---|
| `accel[3]` | g (9.81 m/s^2 = 1.0) | Coning/sculling-corrected specific force, FRD |
| `gyro[3]` | degrees/second | Coning/sculling-corrected angular rate, FRD |
| `mag[3]` | microtesla (uT) | Latest BMM350 magnetometer reading (repeated at IMU rate) |

---

## UART Wire Protocol

### Physical Layer

- **UART**: USART1 (STM32 PA9 TX / PA10 RX) to Tegra ttyTHS1
- **Baud**: 1,000,000 (1 Mbaud, 8N1)
- **Direction**: Primarily STM32 -> Orin; Orin -> STM32 for GPIO commands only

### Framing: COBS with Leading Delimiter

Each frame on the wire:

```
0x00 | COBS_encoded_payload
```

The `0x00` byte serves as the frame delimiter. COBS encoding guarantees no `0x00` bytes appear in the payload. Consecutive frames share the delimiter (the terminating `0x00` of a decoded frame is also the start delimiter of the next).

### EL Frame Format

Most sensor data uses the "EL" frame format inside the COBS payload:

```
Byte 0-1: 'E' 'L'     (magic)
Byte 2:   version (1)
Byte 3:   kind
Byte 4:   reserved (0)
Byte 5+:  payload
```

| Kind | Payload | Rate |
|---|---|---|
| 1 | Log message (UTF-8 string) | ~5/s (diagnostics) |
| 2 | GpsRecord (64 bytes) | 5 Hz |
| 3 | CompassRecord (8 bytes) | 50 Hz |
| 4 | ImuRecord (36 bytes) | ~770 Hz |

Legacy `Record` (baro/voltage, 28 bytes) uses plain COBS without the EL header, at 10 Hz.

### ImuRecord Wire Layout (36 bytes, kind=4)

```c
struct ImuRecord {
    float accel[3];  // g, FRD body frame
    float gyro[3];   // dps, FRD body frame
    float mag[3];    // uT, sensor frame (BMM350)
};
```

### GpsRecord Wire Layout (64 bytes, kind=2)

```c
struct GpsRecord {
    int64_t  unix_epoch_ms;   // UTC milliseconds since epoch
    uint32_t itow;            // GPS time of week (ms)
    int32_t  lat;             // 1e-7 degrees
    int32_t  lon;             // 1e-7 degrees
    int32_t  alt_msl;         // mm above mean sea level
    int32_t  alt_wgs84;       // mm above WGS84 ellipsoid
    int32_t  vel_ned[3];      // mm/s [north, east, down]
    uint32_t ground_speed;    // mm/s
    int32_t  heading_motion;  // 1e-5 degrees
    uint32_t h_acc;           // mm horizontal accuracy
    uint32_t v_acc;           // mm vertical accuracy
    uint32_t s_acc;           // mm/s speed accuracy
    uint8_t  fix_type;        // 0=none, 2=2D, 3=3D
    uint8_t  satellites;
    uint8_t  valid_flags;
    uint8_t  _pad;
};
```

---

## Elodin-DB Component Schema

The serial-bridge writes sensor data to Elodin-DB as time-series components. Each component has a GPS-disciplined microsecond timestamp (when GPS is enabled) or wall-clock timestamp.

### IMU Component (`imu`)

| Field | Type | Unit |
|---|---|---|
| `time` | i64 | microseconds since epoch |
| `accel` | [f32; 3] | g, FRD |
| `gyro` | [f32; 3] | dps, FRD |
| `mag` | [f32; 3] | uT |

Rate: ~770 Hz. Queryable as `imu.accel`, `imu.gyro`, `imu.mag`.

### GPS Component (`ublox`)

| Field | Type | Unit |
|---|---|---|
| `time` | i64 | microseconds since epoch |
| `unix_epoch_ms` | i64 | UTC ms |
| `itow` | u32 | GPS time of week, ms |
| `lat` | i32 | 1e-7 degrees |
| `lon` | i32 | 1e-7 degrees |
| `alt_msl` | i32 | mm |
| `alt_wgs84` | i32 | mm |
| `vel_ned` | [i32; 3] | mm/s [N, E, D] |
| `ground_speed` | u32 | mm/s |
| `heading_motion` | i32 | 1e-5 degrees |
| `h_acc` | u32 | mm |
| `v_acc` | u32 | mm |
| `s_acc` | u32 | mm/s |
| `fix_type` | u8 | 0=none, 2=2D, 3=3D |
| `satellites` | u8 | |
| `valid_flags` | u8 | |

Rate: 5 Hz. Supports M10Q (9600 baud) and M9N (38400 baud) via NixOS config.

### Barometer / Power Component (`aleph`)

| Field | Type | Unit |
|---|---|---|
| `baro` | f32 | Pascals |
| `baro_temp` | f32 | Celsius |
| `vin` | f32 | Volts |
| `vbat` | f32 | Volts |
| `aux_current` | f32 | Amps |

Rate: 10 Hz.

### External Compass Component (`qmc5883l`)

| Field | Type | Unit |
|---|---|---|
| `mag` | [i16; 3] | raw LSB |
| `status` | u8 | |

Rate: 50 Hz.

---

## MEKF Attitude Estimation

The MEKF runs on the Orin NX as the `mekf` systemd service. It subscribes to the `imu` component in Elodin-DB and produces attitude estimates.

### Input

Subscribes to the `imu` vtable (accel, gyro, mag) at ~770 Hz.

- **gyro**: converted from dps to rad/s, used directly as angular velocity
- **accel**: normalized, used as gravity reference vector `[0, 0, 1]` (FRD: down = +Z)
- **mag**: soft-iron calibrated via configurable `mag_cal.a` (3x3 matrix) and `mag_cal.b` (3-vector offset), then transformed with `[-1, 1, -1]` axis flip, then normalized

### Output Component (`aleph`)

| Field | Type | Description |
|---|---|---|
| `q_hat` | [f64; 4] | Attitude quaternion (scalar-last: `[qx, qy, qz, qw]`) |
| `b_hat` | [f64; 3] | Gyro bias estimate (rad/s) |
| `gyro_est` | [f64; 3] | Bias-corrected angular velocity (rad/s) |
| `world_pos` | SpatialTransform | Rotation as a spatial transform for visualization |
| `mag_cal` | [f64; 3] | Calibrated+normalized magnetometer vector |

Rate: ~770 Hz (one output per IMU input).

### Default Configuration

```lua
-- /root/mekf.lua (configurable per-Aleph)
config.mekf.accel_sigma = 160e-6
config.mekf.mag_sigma = 3e-4
config.mekf.gyro_sigma = math.rad(0.008)
config.mekf.gyro_bias_sigma = 0.001
config.mekf.dt = 1.0 / 1400.0
config.mekf.mag_ref = wmm(latitude, longitude, altitude)  -- WMM reference field
```

The `mag_ref` should be set to the local magnetic field vector using the World Magnetic Model. The `wmm()` Lua helper is available in the config script.

---

## Timestamping

### GPS-Disciplined Mode (default when GPS configured)

When `services.sensor-fw.gps.model` is set:
1. The serial-bridge waits for the first valid GPS NAV-PVT frame with `unix_epoch_ms > 0`
2. It anchors a local clock to that GPS time
3. All subsequent sensor timestamps are derived from `GPS_anchor + elapsed_monotonic`
4. GPS timestamps are refreshed on each NAV-PVT frame (5 Hz)
5. Timestamps are guaranteed monotonically increasing (max of candidate vs last emitted)

### Wall-Clock Mode (no GPS)

When no GPS is configured, the serial-bridge uses `Timestamp::now()` (system wall-clock in microseconds since epoch).

---

## Deployment

### NixOS Configuration

```nix
# Minimal: IMU/mag/baro only (no GPS)
services.sensor-fw.enable = true;

# With M10Q GPS:
services.sensor-fw.gps.model = "m10q";

# With M9N-5883 GPS:
services.sensor-fw.gps.model = "m9n";
```

Setting `gps.model` automatically enables GPS-disciplined timestamping.

### Deploy Commands

```bash
# Default (no GPS):
./deploy.sh -h <aleph-ip> -u root

# With M10Q GPS:
./deploy.sh -h <aleph-ip> -u root -c m10q

# With M9N GPS:
./deploy.sh -h <aleph-ip> -u root -c m9n
```

### Querying Data

```bash
# IMU accel (last 10 samples)
elodin-db query --eql "imu.accel" --offset -10 --flatten /db/default-*

# GPS position
elodin-db query --eql "ublox.lat" --offset -5 /db/default-*

# MEKF attitude quaternion
elodin-db query --eql "aleph.q_hat" --offset -5 --flatten /db/default-*

# Connect live from development machine
elodin editor <aleph-ip>:2240
```

---

## Performance Summary

| Metric | Value |
|---|---|
| IMU output rate | ~770 Hz (coning/sculling corrected) |
| IMU latency (sensor to DB) | < 2 ms |
| GPS rate | 5 Hz (NAV-PVT) |
| MEKF rate | ~770 Hz (1:1 with IMU) |
| UART bandwidth | 33 KB/s at 1 Mbaud (33% utilization) |
| STM32 CPU utilization | < 15% |
| Accel range | ±8g |
| Gyro range | ±2000 dps |
| Accel noise (typical) | ±0.03g per sample |
| Gyro noise (typical) | ±0.5 dps per sample |

---

## Source Files
Relative to the public repo path:
https://github.com/elodin-sys/elodin

| File | Description |
|---|---|
| `fsw/sensor-fw/src/main.rs` | STM32 main loop, sensor polling, axis corrections, integrator |
| `fsw/sensor-fw/src/coning_sculling.rs` | Coning & sculling pre-integration filter |
| `fsw/sensor-fw/src/bmi270.rs` | BMI270 I2C driver |
| `fsw/sensor-fw/src/bmi270_spi.rs` | BMI270 SPI driver |
| `fsw/sensor-fw/src/ubx.rs` | u-blox UBX protocol driver (M10Q + M9N) |
| `fsw/sensor-fw/src/command.rs` | UART COBS framing and EL protocol |
| `fsw/serial-bridge/src/main.rs` | Orin-side UART receiver and Elodin-DB writer |
| `fsw/mekf/src/main.rs` | MEKF attitude estimator |
| `fsw/blackbox/src/lib.rs` | Shared record definitions (ImuRecord, GpsRecord, etc.) |
| `libs/impeller2/frame/src/lib.rs` | COBS FrameDecoder |
| `aleph/modules/sensor-fw.nix` | NixOS module (GPS config, flash service) |
