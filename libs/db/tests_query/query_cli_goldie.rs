//! Integration test: build an on-disk DB, run `elodin-db query`, compare stdout with goldie.
//!
//! Update golden: `GOLDIE_UPDATE=1 cargo test -p elodin-db --test query_cli_goldie`

use std::collections::HashMap;
use std::process::Command;

use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;

#[test]
fn query_cli_bool_alternating_pattern() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    let cid = ComponentId::new("goldie.bool_alternating");

    let db = DB::create(path.clone()).expect("DB::create");
    db.with_state_mut(|s| {
        s.set_component_metadata(
            ComponentMetadata {
                component_id: cid,
                name: "bool_alternating".to_string(),
                metadata: HashMap::new(),
            },
            &path,
        )
        .expect("set_component_metadata");
        s.insert_component(cid, ComponentSchema::new(PrimType::Bool, &[]), &path)
            .expect("insert_component");
    });
    for (i, byte) in [1u8, 0u8, 1u8, 0u8].iter().enumerate() {
        db.with_state(|s| {
            let c = s.get_component(cid).expect("component");
            c.time_series
                .push_buf(Timestamp(1000 * (i as i64 + 1)), &[*byte])
                .expect("push_buf");
        });
    }
    db.flush_all().expect("flush_all");
    drop(db);

    let exe = env!("CARGO_BIN_EXE_elodin-db");
    let output = Command::new(exe)
        .args([
            "query",
            "--sql",
            "SELECT * FROM bool_alternating",
            "--precision",
            "full",
            path.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("spawn elodin-db query");

    assert!(
        output.status.success(),
        "elodin-db query failed (status {:?})\nstderr:\n{}\nstdout:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    // `file!()` can be workspace-prefixed (`libs/db/...`), which breaks goldie's default
    // manifest-relative layout; pin paths under `tests_query/testdata/`.
    goldie::Builder::new(
        env!("CARGO_MANIFEST_DIR"),
        "tests_query/query_cli_goldie.rs",
        concat!(module_path!(), "::query_cli_bool_alternating_pattern"),
    )
    .golden_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
    .build()
    .assert(&stdout);
}
