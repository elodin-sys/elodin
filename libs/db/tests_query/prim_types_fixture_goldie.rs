//! Golden snapshot of a small DB that exercises every [`PrimType`] variant (11 types) with shapes `[]`, `[2]`, and `[3, 4]`.
//!
//! - On-disk: hex dump of `schema`, `metadata`, and the **committed prefix** of each `index` / `data`
//!   append log (sparse files are ~8GiB; we only snapshot bytes `0..committed_len`).
//! - `db_state` is omitted (it records `CARGO_PKG_VERSION` and would churn every release).
//!
//! Update goldens: `GOLDIE_UPDATE=1 cargo test -p elodin-db --test prim_types_fixture_goldie`

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;
use std::collections::HashMap;
use zerocopy::IntoBytes;

const NUM_SAMPLES: usize = 16;
const TS_BASE: i64 = 5_000_000;
const TS_STEP: i64 = 1_000;

const ALL_PRIMS: [PrimType; 11] = [
    PrimType::U8,
    PrimType::U16,
    PrimType::U32,
    PrimType::U64,
    PrimType::I8,
    PrimType::I16,
    PrimType::I32,
    PrimType::I64,
    PrimType::Bool,
    PrimType::F32,
    PrimType::F64,
];

const SHAPES: [&[usize]; 3] = [&[], &[2], &[3, 4]];

fn shape_suffix(shape: &[usize]) -> &'static str {
    match shape {
        [] => "s",
        [2] => "v2",
        [3, 4] => "m34",
        _ => unreachable!(),
    }
}

fn elem_count(shape: &[usize]) -> usize {
    shape.iter().product::<usize>()
}

fn component_name(prim: PrimType, shape: &[usize]) -> String {
    format!("{}.{}", prim.as_str(), shape_suffix(shape))
}

/// Distinct, mostly incrementing payloads per (prim, shape, timestep).
fn build_sample(prim: PrimType, shape: &[usize], step: usize, tag: u8) -> Vec<u8> {
    let n = elem_count(shape);
    let st = step as i64;
    match prim {
        PrimType::U8 => (0..n)
            .map(|i| {
                (step as u8)
                    .wrapping_add(tag)
                    .wrapping_add(i as u8)
                    .wrapping_mul(3)
            })
            .collect(),
        PrimType::U16 => {
            let v: Vec<u16> = (0..n)
                .map(|i| {
                    10u16
                        .wrapping_add((step as u16).wrapping_mul(7))
                        .wrapping_add(tag as u16)
                        .wrapping_add(i as u16)
                })
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::U32 => {
            let v: Vec<u32> = (0..n)
                .map(|i| {
                    100u32
                        .wrapping_add((step as u32).wrapping_mul(101))
                        .wrapping_add((tag as u32).wrapping_mul(17))
                        .wrapping_add(i as u32)
                })
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::U64 => {
            let v: Vec<u64> = (0..n)
                .map(|i| {
                    1_000u64
                        .wrapping_add((step as u64).wrapping_mul(1_003))
                        .wrapping_add((tag as u64).wrapping_mul(97))
                        .wrapping_add(i as u64)
                })
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I8 => {
            let v: Vec<i8> = (0..n)
                .map(|i| -12i8 + (step as i8).wrapping_add(tag as i8) + (i as i8))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I16 => {
            let v: Vec<i16> = (0..n)
                .map(|i| -300i16 + (st as i16).wrapping_mul(11) + (tag as i16) + (i as i16))
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I32 => {
            let v: Vec<i32> = (0..n)
                .map(|i| {
                    -50_000i32
                        + (st as i32).wrapping_mul(1_009)
                        + (tag as i32).wrapping_mul(13)
                        + (i as i32)
                })
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I64 => {
            let v: Vec<i64> = (0..n)
                .map(|i| {
                    st.wrapping_mul(10_007)
                        .wrapping_add(tag as i64)
                        .wrapping_add(i as i64)
                })
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::Bool => (0..n)
            .map(|i| !(step + i + tag as usize).is_multiple_of(5) as u8)
            .collect(),
        PrimType::F32 => {
            let v: Vec<f32> = (0..n)
                .map(|i| (step as f32) * 1.25 + (tag as f32) * 0.0625 + (i as f32) * 0.00390625)
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::F64 => {
            let v: Vec<f64> = (0..n)
                .map(|i| (step as f64) * 1.125 + (tag as f64) * 0.03125 + (i as f64) * 0.001953125)
                .collect();
            v.as_slice().as_bytes().to_vec()
        }
    }
}

fn read_small_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Read bytes `[0, committed_len)` from an append log (avoids reading the sparse tail).
fn read_append_log_committed(path: &Path) -> Vec<u8> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut first = [0u8; 8];
    f.read_exact(&mut first)
        .unwrap_or_else(|e| panic!("read header {}: {e}", path.display()));
    let committed = u64::from_le_bytes(first) as usize;
    assert!(
        committed <= 512 * 1024,
        "unexpectedly large append log in fixture ({}): {}",
        committed,
        path.display()
    );
    let mut buf = vec![0u8; committed];
    f.seek(SeekFrom::Start(0))
        .unwrap_or_else(|e| panic!("seek {}: {e}", path.display()));
    f.read_exact(&mut buf)
        .unwrap_or_else(|e| panic!("read committed {}: {e}", path.display()));
    buf
}

fn hex_lines(bytes: &[u8]) -> String {
    const CHUNK: usize = 32;
    let mut s = String::new();
    for chunk in bytes.chunks(CHUNK) {
        for b in chunk {
            use std::fmt::Write;
            write!(&mut s, "{b:02x}").unwrap();
        }
        s.push('\n');
    }
    s
}

fn rel_path(root: &Path, full: &Path) -> String {
    full.strip_prefix(root)
        .unwrap()
        .to_string_lossy()
        .replace('\\', "/")
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.is_file() {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

fn snapshot_fixture_bytes(db_root: &Path) -> String {
    let mut paths = collect_files(db_root);
    paths.retain(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n != "db_state")
            .unwrap_or(true)
    });
    paths.sort_by_key(|p| rel_path(db_root, p));

    let mut doc = String::new();
    doc.push_str("# prim_types fixture: on-disk bytes (no db_state).\n");
    doc.push_str(
        "# index/data = full committed prefix [0..committed_len); schema/metadata = full file.\n\n",
    );

    for p in paths {
        let rel = rel_path(db_root, &p);
        let base = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let bytes = match base {
            "index" | "data" => read_append_log_committed(&p),
            _ => read_small_file(&p),
        };
        doc.push_str(&rel);
        doc.push('\n');
        doc.push_str(&hex_lines(&bytes));
        doc.push('\n');
    }
    doc
}

fn fixture_specs() -> Vec<(ComponentId, PrimType, &'static [usize], u8)> {
    let mut v = Vec::new();
    let mut tag = 0u8;
    for &prim in &ALL_PRIMS {
        for shape in &SHAPES {
            let name = component_name(prim, shape);
            let cid = ComponentId::from_pair("fixture", &name);
            v.push((cid, prim, *shape, tag));
            tag = tag.wrapping_add(1);
        }
    }
    v
}

fn build_fixture(path: PathBuf) -> DB {
    let db = DB::create(path.clone()).expect("DB::create");
    let specs = fixture_specs();

    db.with_state_mut(|s| {
        for (cid, prim, shape, _) in &specs {
            s.set_component_metadata(
                ComponentMetadata {
                    component_id: *cid,
                    name: component_name(*prim, shape),
                    metadata: HashMap::new(),
                },
                &path,
            )
            .expect("set_component_metadata");
            s.insert_component(*cid, ComponentSchema::new(*prim, shape), &path)
                .expect("insert_component");
        }
    });

    for step in 0..NUM_SAMPLES {
        let ts = Timestamp(TS_BASE + TS_STEP * step as i64);
        db.with_state(|s| {
            for (cid, prim, shape, tag) in &specs {
                let buf = build_sample(*prim, shape, step, *tag);
                let c = s.get_component(*cid).expect("component");
                c.time_series.push_buf(ts, &buf).expect("push_buf");
            }
        });
    }

    db.flush_all().expect("flush_all");
    db
}

#[test]
fn prim_types_fixture_on_disk_and_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();

    let db = build_fixture(path.clone());
    let snapshot = snapshot_fixture_bytes(&path);

    goldie::Builder::new(
        env!("CARGO_MANIFEST_DIR"),
        "tests_query/prim_types_fixture_goldie.rs",
        concat!(module_path!(), "::prim_types_fixture_on_disk_and_roundtrip"),
    )
    .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
    .build()
    .assert(&snapshot);

    drop(db);

    let db2 = DB::open(path.clone()).expect("DB::open");
    let specs = fixture_specs();
    for step in 0..NUM_SAMPLES {
        let ts = Timestamp(TS_BASE + TS_STEP * step as i64);
        db2.with_state(|s| {
            for (cid, prim, shape, tag) in &specs {
                let c = s.get_component(*cid).expect("component after reopen");
                let got = c
                    .time_series
                    .get(ts)
                    .unwrap_or_else(|| panic!("missing ts {ts:?} for {:?}", cid));
                let want = build_sample(*prim, shape, step, *tag);
                assert_eq!(
                    got,
                    want.as_slice(),
                    "round-trip {:?} shape={shape:?} step={step}",
                    prim
                );
            }
        });
    }
}
