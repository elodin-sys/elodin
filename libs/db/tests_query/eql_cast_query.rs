use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;

fn create_u16_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    let cid = ComponentId::new("sensor.count");

    let db = DB::create(path.clone()).expect("DB::create");
    db.with_state_mut(|s| {
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid,
                name: "sensor.count".to_string(),
                metadata: HashMap::new(),
            },
            &path,
        )
        .expect("set_component_metadata");
        s.insert_component(cid, ComponentSchema::new(PrimType::U16, &[]), &path)
            .expect("insert_component");
    });

    for (i, value) in [10u16, 11u16, 12u16].iter().enumerate() {
        db.with_state(|s| {
            let c = s.get_component(cid).expect("component");
            c.time_series
                .push_buf(Timestamp(1_000_000 + i as i64), &value.to_le_bytes())
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

#[test]
fn eql_cast_identifier_syntax_allows_u16_float_arithmetic() {
    let dir = create_u16_fixture();
    let output = run_eql_query(dir.path(), "sensor.count.cast(f64) + 1.0");

    assert!(
        output.status.success(),
        "elodin-db query failed (status {:?})\nstderr:\n{}\nstdout:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(csv_values(&stdout), vec!["11", "12", "13"]);
}

#[test]
fn eql_cast_string_syntax_is_also_supported() {
    let dir = create_u16_fixture();
    let output = run_eql_query(dir.path(), "sensor.count.cast(\"f32\") / 2.0");

    assert!(
        output.status.success(),
        "elodin-db query failed (status {:?})\nstderr:\n{}\nstdout:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(csv_values(&stdout), vec!["5", "5.5", "6"]);
}
