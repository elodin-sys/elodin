use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;

fn create_fixture(name: &str, prim_type: PrimType, rows: &[Vec<u8>]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    let cid = ComponentId::new(name);

    let db = DB::create(path.clone()).expect("DB::create");
    db.with_state_mut(|s| {
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid,
                name: name.to_string(),
                metadata: HashMap::new(),
            },
            &path,
        )
        .expect("set_component_metadata");
        s.insert_component(cid, ComponentSchema::new(prim_type, &[]), &path)
            .expect("insert_component");
    });

    for (i, value) in rows.iter().enumerate() {
        db.with_state(|s| {
            let c = s.get_component(cid).expect("component");
            c.time_series
                .push_buf(Timestamp(1_000_000 + i as i64), value)
                .expect("push_buf");
        });
    }

    db.flush_all().expect("flush_all");
    drop(db);
    dir
}

fn run_eql_query(db: &Path, eql: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_elodin-db"))
        .args([
            "query",
            "--eql",
            eql,
            "--format",
            "csv",
            "--time-format",
            "omit",
            "--precision",
            "full",
            db.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("spawn elodin-db query")
}

fn csv_values(stdout: &str) -> Vec<&str> {
    stdout.lines().skip(1).collect()
}

fn assert_query_success(output: std::process::Output) -> String {
    assert!(
        output.status.success(),
        "elodin-db query failed (status {:?})\nstderr:\n{}\nstdout:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );

    String::from_utf8(output.stdout).expect("stdout utf8")
}

#[test]
fn eql_cast_identifier_syntax_allows_u16_float_arithmetic() {
    let dir = create_fixture(
        "sensor.count",
        PrimType::U16,
        &[
            10u16.to_le_bytes().to_vec(),
            11u16.to_le_bytes().to_vec(),
            12u16.to_le_bytes().to_vec(),
        ],
    );
    let output = run_eql_query(dir.path(), "sensor.count.cast(f64) + 1.0");
    let stdout = assert_query_success(output);
    assert_eq!(csv_values(&stdout), vec!["11", "12", "13"]);
}

#[test]
fn eql_cast_string_syntax_is_also_supported() {
    let dir = create_fixture(
        "sensor.count",
        PrimType::U16,
        &[
            10u16.to_le_bytes().to_vec(),
            11u16.to_le_bytes().to_vec(),
            12u16.to_le_bytes().to_vec(),
        ],
    );
    let output = run_eql_query(dir.path(), "sensor.count.cast(\"f32\") / 2.0");
    let stdout = assert_query_success(output);
    assert_eq!(csv_values(&stdout), vec!["5", "5.5", "6"]);
}

#[test]
fn eql_cast_f32_to_i32_is_supported() {
    let dir = create_fixture(
        "sensor.value",
        PrimType::F32,
        &[
            10.0f32.to_le_bytes().to_vec(),
            11.0f32.to_le_bytes().to_vec(),
            12.0f32.to_le_bytes().to_vec(),
        ],
    );
    let output = run_eql_query(dir.path(), "sensor.value.cast(i32)");
    let stdout = assert_query_success(output);
    assert_eq!(csv_values(&stdout), vec!["10", "11", "12"]);
}

#[test]
fn eql_cast_f64_to_u16_is_supported() {
    let dir = create_fixture(
        "sensor.value",
        PrimType::F64,
        &[
            20.0f64.to_le_bytes().to_vec(),
            21.0f64.to_le_bytes().to_vec(),
            22.0f64.to_le_bytes().to_vec(),
        ],
    );
    let output = run_eql_query(dir.path(), "sensor.value.cast(u16)");
    let stdout = assert_query_success(output);
    assert_eq!(csv_values(&stdout), vec!["20", "21", "22"]);
}

#[test]
fn eql_cast_i32_to_f32_is_supported() {
    let dir = create_fixture(
        "sensor.value",
        PrimType::I32,
        &[
            30i32.to_le_bytes().to_vec(),
            31i32.to_le_bytes().to_vec(),
            32i32.to_le_bytes().to_vec(),
        ],
    );
    let output = run_eql_query(dir.path(), "sensor.value.cast(f32) / 2.0");
    let stdout = assert_query_success(output);
    assert_eq!(csv_values(&stdout), vec!["15", "15.5", "16"]);
}
