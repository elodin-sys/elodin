#![no_std]

#[derive(
    zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout, Clone,
)]
#[repr(C)]
pub struct Record {
    pub baro: f32,
    pub baro_temp: f32,
    pub vin: f32,
    pub vbat: f32,
    pub aux_current: f32,
    pub rtc_vbat: f32,
    pub cpu_temp: f32,
}

/// GPS data from UBX-NAV-PVT, sent as EL frame kind=2
#[derive(
    zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout, Clone,
)]
#[repr(C)]
pub struct GpsRecord {
    pub unix_epoch_ms: i64,  // unix epoch milliseconds derived from UTC fields
    pub itow: u32,           // GPS time of week, ms
    pub lat: i32,            // 1e-7 degrees
    pub lon: i32,            // 1e-7 degrees
    pub alt_msl: i32,        // mm above mean sea level
    pub alt_wgs84: i32,      // mm above WGS84 ellipsoid
    pub vel_ned: [i32; 3],   // mm/s [north, east, down]
    pub ground_speed: u32,   // mm/s
    pub heading_motion: i32, // 1e-5 degrees
    pub h_acc: u32,          // mm horizontal accuracy
    pub v_acc: u32,          // mm vertical accuracy
    pub s_acc: u32,          // mm/s speed accuracy
    pub fix_type: u8,        // 0=none, 2=2D, 3=3D
    pub satellites: u8,
    pub valid_flags: u8,
    pub _pad: u8,
}

/// High-rate IMU data from BMI270, sent as EL frame kind=4
#[derive(
    zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout, Clone,
)]
#[repr(C)]
pub struct ImuRecord {
    pub accel: [f32; 3], // g
    pub gyro: [f32; 3],  // dps
    pub mag: [f32; 3],   // uT (latest BMM350 value, repeated between mag updates)
}

/// External compass data from QMC5883L, sent as EL frame kind=3
#[derive(
    zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout, Clone,
)]
#[repr(C)]
pub struct CompassRecord {
    pub mag: [i16; 3], // raw LSB [x, y, z]
    pub status: u8,
    pub _pad: u8,
}
