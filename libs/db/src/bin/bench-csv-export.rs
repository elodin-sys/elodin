//! Microbenchmark and inspection tool for the CSV export hot path.
//!
//! Subcommands:
//!   - `bench` (default): synthesize a `Component` with a given shape and primitive type, time
//!     `fixed_size_list_to_string` (via the non-flatten CSV path), `as_flattened_columns` (via
//!     the flatten CSV path), and the full `export::run` to a tempdir. Prints a CSV table.
//!   - `decode-schema <db-path>`: decode the postcard-encoded `Schema` and `ComponentMetadata`
//!     for every component dir, printing `(name, prim_type, dim, num_rows, bytes/row)`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use elodin_db::export::{self, ExportFormat, ExportOptions};
use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;
use zerocopy::IntoBytes;

#[derive(Parser)]
#[command(about = "elodin-db CSV export microbenchmark")]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the CSV export microbench across a matrix of shapes.
    Bench {
        /// Comma-separated list of (size,prim) pairs, e.g. "3:f64,9:f64,3000:f64,3:f32"
        #[arg(long, default_value = "3:f64,9:f64,100:f64,3000:f64,3:f32,6000:f32")]
        shapes: String,
        /// Comma-separated list of row counts.
        #[arg(long, default_value = "1000,56000,152000")]
        rows: String,
        /// Where to materialize the synthetic DB / output. Default: a fresh tempdir each run.
        #[arg(long)]
        scratch: Option<PathBuf>,
    },
    /// Decode all component schemas+metadata under a DB directory.
    DecodeSchema {
        /// Path to an `elodin-db` database directory.
        path: PathBuf,
    },
}

fn parse_prim(s: &str) -> PrimType {
    match s.to_ascii_lowercase().as_str() {
        "f64" => PrimType::F64,
        "f32" => PrimType::F32,
        "i64" => PrimType::I64,
        "i32" => PrimType::I32,
        "i16" => PrimType::I16,
        "i8" => PrimType::I8,
        "u64" => PrimType::U64,
        "u32" => PrimType::U32,
        "u16" => PrimType::U16,
        "u8" => PrimType::U8,
        "bool" => PrimType::Bool,
        other => panic!("unknown prim type: {other}"),
    }
}

fn prim_size(prim: PrimType) -> usize {
    match prim {
        PrimType::F64 | PrimType::I64 | PrimType::U64 => 8,
        PrimType::F32 | PrimType::I32 | PrimType::U32 => 4,
        PrimType::I16 | PrimType::U16 => 2,
        PrimType::I8 | PrimType::U8 | PrimType::Bool => 1,
    }
}

fn synth_payload(prim: PrimType, n: usize, step: usize) -> Vec<u8> {
    match prim {
        PrimType::F64 => {
            let v: Vec<f64> = (0..n)
                .map(|i| (step as f64) * 1.5 + (i as f64) * 0.25)
                .collect();
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
        PrimType::I16 => {
            let v: Vec<i16> = (0..n).map(|i| (step as i16) * 3 + (i as i16)).collect();
            v.as_slice().as_bytes().to_vec()
        }
        PrimType::I8 => (0..n)
            .map(|i| ((step as i8).wrapping_mul(2).wrapping_add(i as i8)) as u8)
            .collect(),
        PrimType::Bool => (0..n).map(|i| ((step + i) % 2) as u8).collect(),
    }
}

fn build_component_db(
    db_path: PathBuf,
    name: &str,
    prim: PrimType,
    size: usize,
    rows: usize,
) -> DB {
    let db = DB::create(db_path.clone()).expect("DB::create");
    let cid = ComponentId::new(name);
    let dim_buf = if size == 1 { vec![] } else { vec![size] };
    let dim_slice: &[usize] = &dim_buf;

    db.with_state_mut(|s| {
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid,
                name: name.to_string(),
                metadata: HashMap::new(),
            },
            &db_path,
        )
        .expect("set_component_metadata");
        s.insert_component(cid, ComponentSchema::new(prim, dim_slice), &db_path)
            .expect("insert_component");
    });

    let payload_step = synth_payload(prim, size, 0);
    let elem_bytes = size * prim_size(prim);
    assert_eq!(payload_step.len(), elem_bytes);

    db.with_state(|s| {
        let c = s.get_component(cid).expect("component");
        for step in 0..rows {
            let ts = Timestamp(1_700_000_000_000_000 + step as i64);
            let buf = synth_payload(prim, size, step);
            c.time_series.push_buf(ts, &buf).expect("push_buf");
        }
    });

    db.flush_all().expect("flush_all");
    db
}

fn time_export(db_path: PathBuf, flatten: bool, csv_fast_floats: bool) -> std::time::Duration {
    let out = std::env::temp_dir().join(format!("bench_csv_out_{}", fastrand::u64(..)));
    let _ = std::fs::remove_dir_all(&out);
    let opts = ExportOptions {
        flatten,
        pattern: None,
        csv_fast_floats,
        join: false,
        ..Default::default()
    };
    let t0 = Instant::now();
    export::run(db_path, out.clone(), ExportFormat::Csv, opts).expect("export::run");
    let elapsed = t0.elapsed();
    let _ = std::fs::remove_dir_all(&out);
    elapsed
}

fn fmt_rate(bytes: usize, dur: std::time::Duration) -> String {
    let secs = dur.as_secs_f64();
    if secs <= 0.0 {
        return "-".to_string();
    }
    let mbps = (bytes as f64) / 1_048_576.0 / secs;
    format!("{:>7.1} MB/s", mbps)
}

fn run_bench(shapes: String, rows: String, scratch: Option<PathBuf>) {
    let shape_specs: Vec<(usize, PrimType)> = shapes
        .split(',')
        .map(|chunk| {
            let (sz, p) = chunk.split_once(':').expect("expected size:prim");
            (sz.parse().expect("size"), parse_prim(p))
        })
        .collect();
    let row_specs: Vec<usize> = rows.split(',').map(|r| r.parse().expect("rows")).collect();

    println!(
        "{:>8} {:>5} {:>9} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "rows", "size", "prim", "no_flatten", "flatten", "nf+fastfp", "fl+fastfp", "data"
    );
    for &(size, prim) in &shape_specs {
        for &rows in &row_specs {
            let scratch_root = scratch.clone().unwrap_or_else(std::env::temp_dir);
            let db_path = scratch_root.join(format!(
                "bench_csv_db_{}_{}_{}_{}",
                size,
                prim.as_str(),
                rows,
                fastrand::u64(..)
            ));
            let _ = std::fs::remove_dir_all(&db_path);
            let _db = build_component_db(db_path.clone(), "bench_comp", prim, size, rows);
            let elem_bytes = size * prim_size(prim);
            let total_bytes = elem_bytes * rows;
            let nf = time_export(db_path.clone(), false, false);
            let fl = time_export(db_path.clone(), true, false);
            let nf_fast = time_export(db_path.clone(), false, true);
            let fl_fast = time_export(db_path.clone(), true, true);
            println!(
                "{:>8} {:>5} {:>9} {:>14} {:>14} {:>14} {:>14} {:>10} MB",
                rows,
                size,
                prim.as_str(),
                fmt_rate(total_bytes, nf),
                fmt_rate(total_bytes, fl),
                fmt_rate(total_bytes, nf_fast),
                fmt_rate(total_bytes, fl_fast),
                total_bytes / 1_048_576
            );
            drop(_db);
            let _ = std::fs::remove_dir_all(&db_path);
        }
    }
}

fn run_decode_schema(db_path: PathBuf) {
    println!(
        "{:>5} {:>14} {:>20} {:>10} {:>10}  name",
        "rows", "bytes/row", "shape", "prim", "name_short"
    );
    let entries: Vec<_> = std::fs::read_dir(&db_path)
        .expect("read db dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    let mut by_size: Vec<(u64, String)> = Vec::new();
    for e in entries {
        let p = e.path();
        let cid_str = match p.file_name().and_then(|s| s.to_str()) {
            Some(s) if s.chars().all(|c| c.is_ascii_digit()) => s.to_string(),
            _ => continue,
        };
        let schema_bytes = std::fs::read(p.join("schema"));
        let metadata_bytes = std::fs::read(p.join("metadata"));
        let (schema_bytes, metadata_bytes) = match (schema_bytes, metadata_bytes) {
            (Ok(s), Ok(m)) => (s, m),
            _ => continue,
        };
        let schema: impeller2::schema::Schema<Vec<u64>> =
            postcard::from_bytes(&schema_bytes).expect("decode schema");
        let metadata: ComponentMetadata =
            postcard::from_bytes(&metadata_bytes).expect("decode metadata");
        let prim = schema.prim_type();
        let dim: Vec<u64> = schema.dim().to_vec();
        let elem_count: u64 = if dim.is_empty() {
            1
        } else {
            dim.iter().copied().product()
        };
        let bytes_per_row = elem_count as usize * prim_size(prim);
        let index_blocks = std::fs::metadata(p.join("index"))
            .map(|m| {
                use std::os::unix::fs::MetadataExt;
                m.blocks()
            })
            .unwrap_or(0);
        let index_bytes = index_blocks * 512;
        let approx_rows = if index_bytes >= 24 {
            (index_bytes - 24) / 16
        } else {
            0
        };
        let _ = cid_str;
        let short = metadata
            .name
            .rsplit_once('.')
            .map(|(_, s)| s.to_string())
            .unwrap_or_else(|| metadata.name.clone());
        by_size.push((
            approx_rows * bytes_per_row as u64,
            format!(
                "{:>5} {:>14} {:>20} {:>10} {:>10}  {}",
                approx_rows,
                bytes_per_row,
                format!("{:?}", dim),
                format!("{:?}", prim),
                short,
                metadata.name
            ),
        ));
    }
    by_size.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, line) in by_size {
        println!("{line}");
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd.unwrap_or(Cmd::Bench {
        shapes: "3:f64,9:f64,100:f64,3000:f64,3:f32,6000:f32".to_string(),
        rows: "1000,56000,152000".to_string(),
        scratch: None,
    }) {
        Cmd::Bench {
            shapes,
            rows,
            scratch,
        } => run_bench(shapes, rows, scratch),
        Cmd::DecodeSchema { path } => run_decode_schema(path),
    }
}
