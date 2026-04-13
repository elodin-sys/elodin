//! Integration test: run `elodin-db query` and compare stdout with goldie.
//!
//! Update goldens: `GOLDIE_UPDATE=1 cargo test -p elodin-db --test query_cli_goldie`
//!
//! Archive fixtures (add under `tests_query/testdata/`):
//! - `bools-alternating-db-linux.tar.zst` — used on Linux only
//! - `bools-alternating-db-mac.tar.zst` — used on macOS only
//!
//! Extract uses `gtar` on macOS and `tar` on Linux (`-x --zstd --sparse`), matching `nix develop`.
//!
//! Query matches shell `elodin-db query --eql vehicle.a <db-dir> --precision full` (the `elodin` app
//! does not expose this; use the `elodin-db` binary).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use elodin_db::{ComponentSchema, DB};
use impeller2::types::{ComponentId, PrimType, Timestamp};
use impeller2_wkt::ComponentMetadata;

fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata")
}

fn extract_tar_zstd_sparse(archive: &Path, dest: &Path) {
    let tar_bin = "gtar";
    let status = Command::new(tar_bin)
        .arg("-x")
        .arg("--zstd")
        .arg("--sparse")
        .arg("-f")
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .status()
        .unwrap_or_else(|e| panic!("spawn {tar_bin}: {e}; on macOS `nix develop` provides gtar."));
    assert!(
        status.success(),
        "{} extract failed (status {:?}) for {}",
        tar_bin,
        status,
        archive.display()
    );
}

/// Walk `root` for a directory that contains `db_state` (Elodin DB root).
fn find_db_root(root: &Path) -> Option<PathBuf> {
    if root.join("db_state").is_file() {
        return Some(root.to_path_buf());
    }
    let entries = fs::read_dir(root).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir()
            && let Some(found) = find_db_root(&p)
        {
            return Some(found);
        }
    }
    None
}

fn run_elodin_db_query_eql_vehicle_a(db: &Path) -> String {
    let exe = env!("CARGO_BIN_EXE_elodin-db");
    let output = Command::new(exe)
        .args([
            "query",
            "--eql",
            "vehicle.a",
            "--precision",
            "full",
            db.to_str().expect("utf-8 db path"),
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

    String::from_utf8(output.stdout).expect("stdout utf8")
}

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

    let stdout = run_elodin_db_query_sql_bool_alternating(&path);
    goldie::Builder::new(
        env!("CARGO_MANIFEST_DIR"),
        "tests_query/query_cli_goldie.rs",
        concat!(module_path!(), "::query_cli_bool_alternating_pattern"),
    )
    .golden_dir(testdata_dir())
    .build()
    .assert(&stdout);
}

fn run_elodin_db_query_sql_bool_alternating(path: &Path) -> String {
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

    String::from_utf8(output.stdout).expect("stdout utf8")
}

/// `elodin-db query --eql vehicle.a` on the Linux-packaged DB (`tar.zstd` fixture).
#[test]
fn query_cli_vehicle_a_from_linux_bools_archive() {
    let archive = testdata_dir().join("bools-alternating-db-linux.tar.zst");
    assert!(
        archive.is_file(),
        "missing {}; add the Linux DB archive under tests_query/testdata/",
        archive.display()
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    extract_tar_zstd_sparse(&archive, tmp.path());
    let db = find_db_root(tmp.path()).expect("db_state under extracted archive");

    let stdout = run_elodin_db_query_eql_vehicle_a(&db);
    goldie::Builder::new(
        env!("CARGO_MANIFEST_DIR"),
        "tests_query/query_cli_goldie.rs",
        concat!(
            module_path!(),
            "::query_cli_vehicle_a_from_linux_bools_archive"
        ),
    )
    .golden_dir(testdata_dir())
    .build()
    .assert(&stdout);
}

/// `elodin-db query --eql vehicle.a` on the macOS-packaged DB (`tar.zstd` fixture).
#[test]
fn query_cli_vehicle_a_from_macos_bools_archive() {
    let archive = testdata_dir().join("bools-alternating-db-mac.tar.zst");
    assert!(
        archive.is_file(),
        "missing {}; add the macOS DB archive under tests_query/testdata/",
        archive.display()
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    extract_tar_zstd_sparse(&archive, tmp.path());
    let db = find_db_root(tmp.path()).expect("db_state under extracted archive");

    let stdout = run_elodin_db_query_eql_vehicle_a(&db);
    goldie::Builder::new(
        env!("CARGO_MANIFEST_DIR"),
        "tests_query/query_cli_goldie.rs",
        concat!(
            module_path!(),
            "::query_cli_vehicle_a_from_macos_bools_archive"
        ),
    )
    .golden_dir(testdata_dir())
    .build()
    .assert(&stdout);
}
