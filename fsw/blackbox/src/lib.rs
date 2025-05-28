#![cfg_attr(not(any(test, feature = "std")), no_std)]

#[derive(
    zerocopy::FromBytes, zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout, Clone,
)]
#[cfg_attr(feature = "roci", derive(roci::AsVTable, roci::Metadatatize))]
#[cfg_attr(feature = "roci", roci(entity_id = 1))]
#[repr(C)]
pub struct Record {
    pub ts: u32, // in milliseconds
    pub mag: [f32; 3],
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
    pub mag_temp: f32,
    pub mag_sample: u32,
    pub baro: f32,
    pub baro_temp: f32,
    pub vin: f32,
    pub vbat: f32,
    pub aux_current: f32,
    pub rtc_vbat: f32,
    pub cpu_temp: f32,
}
