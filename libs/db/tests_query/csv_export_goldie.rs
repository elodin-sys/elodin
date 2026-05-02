//! Golden snapshot of `elodin_db::export::run` for CSV output.
//!
//! Locks the externally-visible CSV bytes for a small synthetic database that exercises:
//! - Every primitive type (F64/F32/I*/U*/Bool) at scalar, vector, and matrix shapes.
//! - Components with `element_names` metadata (drives the flatten path's column suffixes).
//! - Edge floats: NaN, +/- infinity, very small/large magnitudes, signed zero.
//!
//! Each later optimization phase that promises to keep the default CSV format byte-stable
//! must keep this test passing without regenerating the goldens.
//!
//! Update goldens: `GOLDIE_UPDATE=1 cargo test -p elodin-db --test csv_export_goldie`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;
use zerocopy::IntoBytes;

const TS_BASE: i64 = 1_700_000_000_000_000;
const TS_STEP: i64 = 1_000_000;
const NUM_ROWS: usize = 5;

#[derive(Clone)]
struct CompSpec {
    name: &'static str,
    prim: PrimType,
    dim: &'static [usize],
    element_names: Option<&'static str>,
}

fn fixture_specs() -> Vec<CompSpec> {
    vec![
        CompSpec {
            name: "scalar_f64",
            prim: PrimType::F64,
            dim: &[],
            element_names: None,
        },
        CompSpec {
            name: "vec3_f64",
            prim: PrimType::F64,
            dim: &[3],
            element_names: Some("x,y,z"),
        },
        CompSpec {
            name: "vec3_f64_unnamed",
            prim: PrimType::F64,
            dim: &[3],
            element_names: None,
        },
        CompSpec {
            name: "mat3_f64",
            prim: PrimType::F64,
            dim: &[3, 3],
            element_names: None,
        },
        CompSpec {
            name: "vec3_f32",
            prim: PrimType::F32,
            dim: &[3],
            element_names: Some("x,y,z"),
        },
        CompSpec {
            name: "scalar_i64",
            prim: PrimType::I64,
            dim: &[],
            element_names: None,
        },
        CompSpec {
            name: "vec2_i32",
            prim: PrimType::I32,
            dim: &[2],
            element_names: Some("a,b"),
        },
        CompSpec {
            name: "scalar_u8",
            prim: PrimType::U8,
            dim: &[],
            element_names: None,
        },
        CompSpec {
            name: "vec3_u16",
            prim: PrimType::U16,
            dim: &[3],
            element_names: None,
        },
        CompSpec {
            name: "scalar_bool",
            prim: PrimType::Bool,
            dim: &[],
            element_names: None,
        },
        CompSpec {
            name: "vec3_bool",
            prim: PrimType::Bool,
            dim: &[3],
            element_names: Some("p,q,r"),
        },
        CompSpec {
            name: "edge_floats",
            prim: PrimType::F64,
            dim: &[4],
            element_names: Some("nan,pinf,ninf,tiny"),
        },
    ]
}

fn elem_count(dim: &[usize]) -> usize {
    if dim.is_empty() {
        1
    } else {
        dim.iter().product()
    }
}

fn build_sample(spec: &CompSpec, step: usize) -> Vec<u8> {
    let n = elem_count(spec.dim);
    match spec.prim {
        PrimType::F64 => {
            let v: Vec<f64> = if spec.name == "edge_floats" {
                vec![
                    f64::NAN,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    1.0e-7 * (step as f64 + 1.0),
                ]
            } else {
                (0..n)
                    .map(|i| (step as f64) * 1.5 + (i as f64) * 0.25)
                    .collect()
            };
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::F32 => {
            let v: Vec<f32> = (0..n)
                .map(|i| (step as f32) * 0.5 + (i as f32) * 0.125)
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I64 => {
            let v: Vec<i64> = (0..n)
                .map(|i| -1_000i64 + (step as i64) * 13 + (i as i64))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I32 => {
            let v: Vec<i32> = (0..n)
                .map(|i| 100i32 + (step as i32) * 7 + (i as i32))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I16 => {
            let v: Vec<i16> = (0..n).map(|i| (step as i16) * 3 + (i as i16)).collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I8 => (0..n)
            .map(|i| ((step as i8).wrapping_mul(2).wrapping_add(i as i8)) as u8)
            .collect(),
        PrimType::U64 => {
            let v: Vec<u64> = (0..n)
                .map(|i| 1u64 + (step as u64) * 11 + (i as u64))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::U32 => {
            let v: Vec<u32> = (0..n)
                .map(|i| 1u32 + (step as u32) * 5 + (i as u32))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::U16 => {
            let v: Vec<u16> = (0..n)
                .map(|i| 1u16 + (step as u16) * 5 + (i as u16))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::U8 => (0..n).map(|i| 1u8 + (step as u8) * 3 + (i as u8)).collect(),
        PrimType::Bool => (0..n).map(|i| ((step + i) % 2) as u8).collect(),
    }
}

fn build_fixture(path: PathBuf) -> DB {
    let db = DB::create(path.clone()).expect("DB::create");
    let specs = fixture_specs();

    db.with_state_mut(|s| {
        for spec in &specs {
            let cid = ComponentId::new(spec.name);
            let mut metadata = HashMap::new();
            if let Some(en) = spec.element_names {
                metadata.insert("element_names".to_string(), en.to_string());
            }
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: cid,
                    name: spec.name.to_string(),
                    metadata,
                },
                &path,
            )
            .expect("set_component_metadata");
            s.insert_component(cid, ComponentSchema::new(spec.prim, spec.dim), &path)
                .expect("insert_component");
        }
    });

    for step in 0..NUM_ROWS {
        let ts = Timestamp(TS_BASE + TS_STEP * step as i64);
        db.with_state(|s| {
            for spec in &specs {
                let cid = ComponentId::new(spec.name);
                let buf = build_sample(spec, step);
                let c = s.get_component(cid).expect("component");
                c.time_series.push_buf(ts, &buf).expect("push_buf");
            }
        });
    }

    db.flush_all().expect("flush_all");
    db
}

fn snapshot_csv_dir(dir: &Path, label: &str) -> String {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("csv"))
        .collect();
    entries.sort();

    let mut s = String::new();
    s.push_str(&format!("# csv export goldens: {}\n\n", label));
    for p in entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let content =
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        s.push_str(&format!("=== {} ===\n{}\n", name, content));
    }
    s
}

fn run_export(db_path: PathBuf, flatten: bool, csv_fast_floats: bool, label: &str) -> String {
    run_export_full(
        db_path,
        flatten,
        csv_fast_floats,
        false,
        elodin_db::export::TimeFormat::Iso8601,
        false,
        label,
    )
}

fn run_export_full(
    db_path: PathBuf,
    flatten: bool,
    csv_fast_floats: bool,
    join: bool,
    time_format: elodin_db::export::TimeFormat,
    include_private: bool,
    label: &str,
) -> String {
    let out = std::env::temp_dir().join(format!(
        "elodin_csv_goldie_{}_{}",
        label.replace(' ', "_"),
        fastrand::u64(..)
    ));
    let _ = std::fs::remove_dir_all(&out);

    elodin_db::export::run(
        db_path,
        out.clone(),
        elodin_db::export::ExportFormat::Csv,
        elodin_db::export::ExportOptions {
            flatten,
            pattern: None,
            csv_fast_floats,
            join,
            time_format,
            include_private,
        },
    )
    .expect("export::run");

    let snap = snapshot_csv_dir(&out, label);
    let _ = std::fs::remove_dir_all(&out);
    snap
}

/// Build a small DB whose components form two groups:
/// - `MSG_A.{POS,VEL}` (3-vec each, identical timestamps -> fast-path zip).
/// - `MSG_B.LABEL` (single member).
///
/// Plus one component without a `.` (`tick`) which becomes its own one-member group.
fn build_join_fixture(path: PathBuf) -> DB {
    use std::collections::HashMap;
    let db = DB::create(path.clone()).expect("DB::create");

    let make = |s: &mut elodin_db::State,
                name: &str,
                prim: PrimType,
                dim: &[usize],
                element_names: Option<&str>| {
        let cid = ComponentId::new(name);
        let mut metadata = HashMap::new();
        if let Some(en) = element_names {
            metadata.insert("element_names".to_string(), en.to_string());
        }
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid,
                name: name.to_string(),
                metadata,
            },
            &path,
        )
        .expect("set_component_metadata");
        s.insert_component(cid, ComponentSchema::new(prim, dim), &path)
            .expect("insert_component");
        cid
    };

    let (a_pos, a_vel, b_label, tick) = db.with_state_mut(|s| {
        let a_pos = make(s, "MSG_A.POS", PrimType::F64, &[3], Some("x,y,z"));
        let a_vel = make(s, "MSG_A.VEL", PrimType::F64, &[3], Some("x,y,z"));
        let b_label = make(s, "MSG_B.LABEL", PrimType::I32, &[], None);
        let tick = make(s, "tick", PrimType::U64, &[], None);
        (a_pos, a_vel, b_label, tick)
    });

    db.with_state(|s| {
        let pos = s.get_component(a_pos).unwrap();
        let vel = s.get_component(a_vel).unwrap();
        let lbl = s.get_component(b_label).unwrap();
        let tk = s.get_component(tick).unwrap();
        for step in 0..NUM_ROWS {
            let ts = Timestamp(TS_BASE + TS_STEP * step as i64);
            let pos_buf: Vec<f64> = vec![step as f64, step as f64 + 0.5, step as f64 + 1.0];
            pos.time_series
                .push_buf(ts, pos_buf.as_slice().as_bytes())
                .unwrap();
            let vel_buf: Vec<f64> = vec![10.0 + step as f64, 11.0, 12.0];
            vel.time_series
                .push_buf(ts, vel_buf.as_slice().as_bytes())
                .unwrap();
            let lbl_buf: Vec<i32> = vec![100 + step as i32];
            lbl.time_series
                .push_buf(ts, lbl_buf.as_slice().as_bytes())
                .unwrap();
            let tk_buf: Vec<u64> = vec![step as u64];
            tk.time_series
                .push_buf(ts, tk_buf.as_slice().as_bytes())
                .unwrap();
        }
    });

    db.flush_all().expect("flush_all");
    db
}

/// Build a DB with two members of a group whose timestamps differ — exercises the
/// outer-join slow path (sorted union + take with NULL fill).
fn build_join_outer_fixture(path: PathBuf) -> DB {
    use std::collections::HashMap;
    let db = DB::create(path.clone()).expect("DB::create");

    let cid_a = ComponentId::new("PAIR.A");
    let cid_b = ComponentId::new("PAIR.B");

    db.with_state_mut(|s| {
        for (cid, name) in [(cid_a, "PAIR.A"), (cid_b, "PAIR.B")] {
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: cid,
                    name: name.to_string(),
                    metadata: HashMap::new(),
                },
                &path,
            )
            .expect("set_component_metadata");
            s.insert_component(cid, ComponentSchema::new(PrimType::I64, &[]), &path)
                .expect("insert_component");
        }
    });

    db.with_state(|s| {
        let a = s.get_component(cid_a).unwrap();
        let b = s.get_component(cid_b).unwrap();
        // A samples at even ticks; B samples at odd ticks. Union axis interleaves both.
        for step in 0..6 {
            let ts_a = Timestamp(TS_BASE + TS_STEP * (step * 2) as i64);
            let ts_b = Timestamp(TS_BASE + TS_STEP * (step * 2 + 1) as i64);
            let a_val: i64 = step as i64 * 100;
            let b_val: i64 = step as i64 * 1000 + 1;
            a.time_series
                .push_buf(ts_a, std::slice::from_ref(&a_val).as_bytes())
                .unwrap();
            b.time_series
                .push_buf(ts_b, std::slice::from_ref(&b_val).as_bytes())
                .unwrap();
        }
    });

    db.flush_all().expect("flush_all");
    db
}

/// Build a small DB containing one regular component and one private component
/// (`metadata = {"private": "true"}`). Exercises the `--include-private` filter.
fn build_private_fixture(path: PathBuf) -> DB {
    use std::collections::HashMap;
    let db = DB::create(path.clone()).expect("DB::create");

    let cid_pub = ComponentId::new("public_scalar");
    let cid_secret = ComponentId::new("secret_scalar");

    db.with_state_mut(|s| {
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid_pub,
                name: "public_scalar".to_string(),
                metadata: HashMap::new(),
            },
            &path,
        )
        .expect("set_component_metadata public");
        s.insert_component(cid_pub, ComponentSchema::new(PrimType::F64, &[]), &path)
            .expect("insert_component public");

        let mut secret_meta = HashMap::new();
        secret_meta.insert("private".to_string(), "true".to_string());
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid_secret,
                name: "secret_scalar".to_string(),
                metadata: secret_meta,
            },
            &path,
        )
        .expect("set_component_metadata secret");
        s.insert_component(cid_secret, ComponentSchema::new(PrimType::F64, &[]), &path)
            .expect("insert_component secret");
    });

    db.with_state(|s| {
        let public = s.get_component(cid_pub).unwrap();
        let secret = s.get_component(cid_secret).unwrap();
        for step in 0..NUM_ROWS {
            let ts = Timestamp(TS_BASE + TS_STEP * step as i64);
            let pub_val: f64 = step as f64;
            let sec_val: f64 = step as f64 + 100.0;
            public
                .time_series
                .push_buf(ts, std::slice::from_ref(&pub_val).as_bytes())
                .unwrap();
            secret
                .time_series
                .push_buf(ts, std::slice::from_ref(&sec_val).as_bytes())
                .unwrap();
        }
    });

    db.flush_all().expect("flush_all");
    db
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_export_default_no_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_fixture(dir.path().to_path_buf());
        let snapshot = run_export(dir.path().to_path_buf(), false, false, "no_flatten");
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_default_no_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    #[test]
    fn csv_export_default_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_fixture(dir.path().to_path_buf());
        let snapshot = run_export(dir.path().to_path_buf(), true, false, "flatten");
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_default_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// Locks the `--csv-fast-floats` opt-in path. Floats are formatted via `ryu`, which is
    /// faster than `Display` but produces a slightly different (still parseable) text format.
    /// Integers/bools go through `Display` so they stay byte-identical to the default path.
    #[test]
    fn csv_export_fast_floats_no_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_fixture(dir.path().to_path_buf());
        let snapshot = run_export(dir.path().to_path_buf(), false, true, "fast_no_flatten");
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_fast_floats_no_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// `--join` fast path: members of a group share identical timestamps so we zero-copy
    /// zip them onto a shared time axis. Covers both the multi-member group (MSG_A) and
    /// single-member groups (MSG_B, tick).
    #[test]
    fn csv_export_join_identical_ts_no_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_join_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            false,
            false,
            true,
            elodin_db::export::TimeFormat::Iso8601,
            false,
            "join_identical_no_flatten",
        );
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_join_identical_ts_no_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// `--join --flatten`: same fixture as above, with the vector members expanded into
    /// per-element columns. Confirms column naming `<short>.<element>`.
    #[test]
    fn csv_export_join_identical_ts_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_join_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            true,
            false,
            true,
            elodin_db::export::TimeFormat::Iso8601,
            false,
            "join_identical_flatten",
        );
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_join_identical_ts_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// `--join` slow path: members have disjoint timestamp axes, so the union is
    /// interleaved and each member contributes NULLs at the other's timestamps.
    #[test]
    fn csv_export_join_outer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_join_outer_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            false,
            false,
            true,
            elodin_db::export::TimeFormat::Iso8601,
            false,
            "join_outer",
        );
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_join_outer"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// `--mono-ns`: the time column is renamed `time_ns` and carries integer nanoseconds
    /// since unix epoch (= microseconds * 1000). Locks against the customer's
    /// `TIME_MONOTONIC` matching expectation.
    #[test]
    fn csv_export_mono_ns_no_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            false,
            false,
            false,
            elodin_db::export::TimeFormat::MonoNanoseconds,
            false,
            "mono_ns_no_flatten",
        );
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_mono_ns_no_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// `--mono-us`: the time column is renamed `time_us` and carries integer microseconds
    /// since unix epoch (zero-copy retype of the underlying TimestampMicrosecond buffer).
    /// Combined with `--flatten` here to verify time-column rewrite composes with flatten.
    #[test]
    fn csv_export_mono_us_flatten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            true,
            false,
            false,
            elodin_db::export::TimeFormat::MonoMicroseconds,
            false,
            "mono_us_flatten",
        );
        drop(db);
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_mono_us_flatten"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// Default behaviour: a component flagged `metadata = {"private": "true"}` does not
    /// appear in the export output, and the export log mentions the skip.
    #[test]
    fn csv_export_private_default_skips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_private_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            false,
            false,
            false,
            elodin_db::export::TimeFormat::Iso8601,
            false, // include_private = false
            "private_default_skips",
        );
        drop(db);
        // Snapshot should contain only public_scalar.csv, not secret_scalar.csv.
        assert!(
            snapshot.contains("=== public_scalar.csv ==="),
            "public_scalar.csv missing from default export"
        );
        assert!(
            !snapshot.contains("=== secret_scalar.csv ==="),
            "secret_scalar.csv must NOT appear in default export (it is `private: true`)"
        );
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_private_default_skips"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }

    /// `--include-private` opt-out: both the regular and private components export.
    #[test]
    fn csv_export_private_include_overrides() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = build_private_fixture(dir.path().to_path_buf());
        let snapshot = run_export_full(
            dir.path().to_path_buf(),
            false,
            false,
            false,
            elodin_db::export::TimeFormat::Iso8601,
            true, // include_private = true
            "private_include_overrides",
        );
        drop(db);
        assert!(
            snapshot.contains("=== public_scalar.csv ==="),
            "public_scalar.csv missing from --include-private export"
        );
        assert!(
            snapshot.contains("=== secret_scalar.csv ==="),
            "secret_scalar.csv must appear under --include-private"
        );
        goldie::Builder::new(
            env!("CARGO_MANIFEST_DIR"),
            "tests_query/csv_export_goldie.rs",
            concat!(module_path!(), "::csv_export_private_include_overrides"),
        )
        .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
        .build()
        .assert(&snapshot);
    }
}
