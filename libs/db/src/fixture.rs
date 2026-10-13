//! A small deterministic recording for tests of export, the catalog, and the analysis SDK.
//!
//! - `imu.accel` (f64[3], x/y/z) and `imu.temp` (f64) share every timestamp at 100 Hz,
//!   including one repeated timestamp, so `--join` zips them and keeps both rows.
//! - `gps.alt` (f64, 10 Hz) and `gps.count` (u32 counter, 5 Hz) differ in rate, so `--join`
//!   outer-joins them; `gps.alt` repeats one timestamp, which the outer join collapses.
//! - `baro.pressure` (f64, 50 Hz) is alone in its group.
//! - `empty.signal` has no samples, and `secret.key` is private and never exported.
//!
//! Values are exact binary fractions, so every platform writes identical bytes.

use std::collections::HashMap;
use std::path::Path;

use impeller::types::{ComponentId, PrimType, Timestamp};
use impeller_wkt::ComponentMetadata;
use zerocopy::IntoBytes;

use crate::{ComponentSchema, DB, Error};

/// Recorder clock start, microseconds.
pub const START_US: i64 = 1_000_000;
const IMU_SAMPLES: i64 = 6_000;
const IMU_STEP_US: i64 = 10_000;
/// The IMU step whose timestamp is recorded twice.
pub const REPEAT_STEP: i64 = 1_000;

struct Spec {
    name: &'static str,
    prim: PrimType,
    dim: &'static [usize],
    metadata: &'static [(&'static str, &'static str)],
}

const SPECS: &[Spec] = &[
    Spec {
        name: "imu.accel",
        prim: PrimType::F64,
        dim: &[3],
        metadata: &[("element_names", "x,y,z")],
    },
    Spec {
        name: "imu.temp",
        prim: PrimType::F64,
        dim: &[],
        metadata: &[("unit", "degC")],
    },
    Spec {
        name: "gps.alt",
        prim: PrimType::F64,
        dim: &[],
        metadata: &[("unit", "m")],
    },
    Spec {
        name: "gps.count",
        prim: PrimType::U32,
        dim: &[],
        metadata: &[],
    },
    Spec {
        name: "baro.pressure",
        prim: PrimType::F64,
        dim: &[],
        metadata: &[("unit", "kPa")],
    },
    Spec {
        name: "empty.signal",
        prim: PrimType::F64,
        dim: &[],
        metadata: &[],
    },
    Spec {
        name: "secret.key",
        prim: PrimType::F64,
        dim: &[],
        metadata: &[("private", "true")],
    },
];

/// Write the fixture recording to `path`, which must not exist yet.
pub fn write(path: &Path) -> Result<(), Error> {
    if path.exists() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} already exists", path.display()),
        )));
    }
    let db = DB::create(path.to_path_buf())?;
    db.with_state_mut(|s| -> Result<(), Error> {
        for spec in SPECS {
            let id = ComponentId::new(spec.name);
            let metadata: HashMap<String, String> = spec
                .metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: id,
                    name: spec.name.to_string(),
                    metadata,
                },
                path,
            )?;
            s.insert_component(id, ComponentSchema::new(spec.prim, spec.dim), path)?;
        }
        Ok(())
    })?;

    let push = |name: &str, t: i64, buf: &[u8]| -> Result<(), Error> {
        db.with_state(|s| {
            s.get_component(ComponentId::new(name))
                .expect("fixture component")
                .time_series
                .push_buf(Timestamp(t), buf)
        })
    };
    for i in 0..IMU_SAMPLES {
        let t = START_US + i * IMU_STEP_US;
        let repeats = if i == REPEAT_STEP { 2 } else { 1 };
        for r in 0..repeats {
            let k = (i + r) as f64;
            let accel = [(i % 100) as f64 * 0.125, -9.8125 + r as f64, k * 0.0625];
            push("imu.accel", t, accel.as_bytes())?;
            push("imu.temp", t, (20.0 + (i % 64) as f64 * 0.25).as_bytes())?;
        }
        if i % 2 == 0 {
            push(
                "baro.pressure",
                t,
                (101.25 - i as f64 * 0.001953125).as_bytes(),
            )?;
        }
        if i % 10 == 0 {
            let alt = i as f64 * 0.5;
            push("gps.alt", t, alt.as_bytes())?;
            if i == REPEAT_STEP {
                push("gps.alt", t, (alt + 1.0).as_bytes())?;
            }
            push("secret.key", t, (i as f64).as_bytes())?;
        }
        if i % 20 == 5 {
            push("gps.count", t, ((i / 20) as u32).as_bytes())?;
        }
    }
    db.flush_all()?;
    Ok(())
}
