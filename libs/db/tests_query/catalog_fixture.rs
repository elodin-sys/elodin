//! Pins the export catalog and today's `--join` rows on the synthetic fixture, so any change
//! to join behavior (repeated timestamps, outer-join collapsing) is a deliberate golden update.
//!
//! Update goldens: `GOLDIE_UPDATE=1 cargo test -p elodin-db --test catalog_fixture`

use std::path::Path;

use arrow::array::RecordBatch;
use elodin_db::catalog::{self, Read};
use elodin_db::{DB, fixture};

fn csv(batch: &RecordBatch) -> String {
    use arrow::util::display::{ArrayFormatter, FormatOptions};
    let opts = FormatOptions::default().with_null("null");
    let fmts: Vec<_> = batch
        .columns()
        .iter()
        .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).unwrap())
        .collect();
    let names: Vec<_> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let mut out = names.join(",") + "\n";
    for row in 0..batch.num_rows() {
        let cells: Vec<_> = fmts.iter().map(|f| f.value(row).to_string()).collect();
        out += &(cells.join(",") + "\n");
    }
    out
}

fn window(db: &DB, table: &str, step: i64, rows: i64) -> RecordBatch {
    let t = fixture::START_US + step * 10_000;
    catalog::read(
        db,
        true,
        &Read {
            table: table.into(),
            start_us: Some(t - 20_000),
            end_us: Some(t + rows * 10_000),
            ..Default::default()
        },
    )
    .unwrap()
    .unwrap()
}

#[test]
fn catalog_and_joined_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("fixture");
    fixture::write(&path).unwrap();
    let db = DB::open(path.clone()).unwrap();

    let joined = catalog::catalog(&db, true, None).unwrap();
    let split = catalog::catalog(&db, false, None).unwrap();
    let mut snapshot = format!("{}\n{}\n", joined.to_json(), split.to_json());
    for (table, step) in [("imu", fixture::REPEAT_STEP), ("gps", fixture::REPEAT_STEP)] {
        snapshot.push_str(&format!(
            "=== {table} ===\n{}",
            csv(&window(&db, table, step, 3))
        ));
    }
    assert!(
        !snapshot.contains("secret"),
        "private components never appear"
    );

    // An unfiltered read returns every exported row; a nonnull read only the member's rows.
    let gps = joined.table("gps").unwrap();
    let all = catalog::read(
        &db,
        true,
        &Read {
            table: "gps".into(),
            ..Default::default()
        },
    );
    assert_eq!(all.unwrap().unwrap().num_rows() as u64, gps.rows);
    let count = catalog::read(
        &db,
        true,
        &Read {
            table: "gps".into(),
            columns: vec!["count".into()],
            nonnull: Some("count".into()),
            ..Default::default()
        },
    );
    assert_eq!(
        count.unwrap().unwrap().num_rows() as u64,
        joined.signal("gps.count").unwrap().rows
    );

    // `export --manifest` writes the same catalog, and a file for each non-empty table.
    let out = tmp.path().join("out");
    let manifest = tmp.path().join("manifest.json");
    elodin_db::export::run(
        path,
        out.clone(),
        elodin_db::export::ExportFormat::ArrowIpc,
        elodin_db::export::ExportOptions {
            join: true,
            time_format: elodin_db::export::TimeFormat::MonoMicroseconds,
            manifest: Some(manifest.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let written: catalog::Catalog =
        serde_json::from_str(&std::fs::read_to_string(manifest).unwrap()).unwrap();
    assert_eq!(written, joined);
    let mut files: Vec<_> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    let expected: Vec<_> = joined
        .tables
        .iter()
        .filter(|t| t.rows > 0)
        .map(|t| format!("{}.arrow", t.name))
        .collect();
    assert_eq!(files, expected);

    goldie::Builder::new(
        env!("CARGO_MANIFEST_DIR"),
        "tests_query/catalog_fixture.rs",
        "catalog_fixture::catalog_and_joined_rows",
    )
    .golden_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests_query/testdata"))
    .build()
    .assert(&snapshot);
}
