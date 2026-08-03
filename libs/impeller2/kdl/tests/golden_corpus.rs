//! Phase 0 golden corpus: every checked-in schematic must parse, emit
//! deterministically, and round-trip to an equal model. Canonical models are
//! snapshotted as sorted JSON under `tests/corpus/goldens/`.
//!
//! Refresh goldens with `BLESS_GOLDENS=1 cargo test -p impeller2-kdl --test golden_corpus`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use impeller2_kdl::{parse_schematic, serialize_schematic};
use impeller2_wkt::Schematic;
use serde_json::Value;

fn corpus_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

fn source_kdl_files() -> Vec<PathBuf> {
    let sources = corpus_root().join("sources");
    let mut files = Vec::new();
    for group in ["examples", "fsw"] {
        let dir = sources.join(group);
        for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display())) {
            let path = entry.expect("dir entry").path();
            if path.extension().is_some_and(|e| e == "kdl") {
                files.push(path);
            }
        }
    }
    files.sort();
    assert!(
        files.len() >= 20,
        "expected ≥20 corpus schematics under tests/corpus/sources, found {}",
        files.len()
    );
    files
}

fn corpus_stem(path: &Path) -> String {
    let group = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("schematic");
    format!("{group}__{stem}")
}

fn golden_path(path: &Path) -> PathBuf {
    corpus_root()
        .join("goldens")
        .join(format!("{}.json", corpus_stem(path)))
}

fn sort_value(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k, sort_value(v));
            }
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_value).collect()),
        other => other,
    }
}

fn canonical_model(src: &str) -> Schematic {
    let parsed = parse_schematic(src).expect("parse source");
    let emitted = serialize_schematic(&parsed);
    parse_schematic(&emitted).expect("parse emitted kdl")
}

fn stable_json(schematic: &Schematic) -> String {
    let value = serde_json::to_value(schematic).expect("schematic → json");
    let sorted = sort_value(value);
    let mut text = serde_json::to_string_pretty(&sorted).expect("pretty json");
    text.push('\n');
    text
}

#[test]
fn all_corpus_schematics_parse() {
    for path in source_kdl_files() {
        let text =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if let Err(err) = parse_schematic(&text) {
            panic!("{} failed to parse:\n{err:?}", path.display());
        }
    }
}

#[test]
fn emission_is_byte_deterministic() {
    for path in source_kdl_files() {
        let text = fs::read_to_string(&path).expect("read");
        let model = parse_schematic(&text).expect("parse");
        let once = serialize_schematic(&model);
        let twice = serialize_schematic(&model);
        assert_eq!(
            once,
            twice,
            "{}: serialize_schematic must be byte-identical across calls",
            path.display()
        );
    }
}

#[test]
fn parse_emit_parse_model_equality() {
    for path in source_kdl_files() {
        let text = fs::read_to_string(&path).expect("read");
        let first = parse_schematic(&text).expect("parse");
        let emitted = serialize_schematic(&first);
        let second = parse_schematic(&emitted).expect("re-parse");
        let reemitted = serialize_schematic(&second);
        let third = parse_schematic(&reemitted).expect("third parse");

        assert_eq!(
            second,
            third,
            "{}: parse→emit→parse must reach a fixed-point model",
            path.display()
        );
        assert_eq!(
            reemitted,
            serialize_schematic(&third),
            "{}: emitted KDL must be a fixed point",
            path.display()
        );
        assert_eq!(
            emitted,
            reemitted,
            "{}: first emit of source should already be canonical",
            path.display()
        );
    }
}

#[test]
fn golden_json_snapshots() {
    let bless = std::env::var_os("BLESS_GOLDENS").is_some();
    let goldens_dir = corpus_root().join("goldens");
    fs::create_dir_all(&goldens_dir).expect("create goldens dir");

    for path in source_kdl_files() {
        let text = fs::read_to_string(&path).expect("read");
        let canonical = canonical_model(&text);
        let actual = stable_json(&canonical);
        let golden = golden_path(&path);

        if bless {
            fs::write(&golden, &actual).unwrap_or_else(|e| {
                panic!("write {}: {e}", golden.display());
            });
            continue;
        }

        assert!(
            golden.exists(),
            "missing golden {} — run with BLESS_GOLDENS=1 to create it",
            golden.display()
        );
        let expected = fs::read_to_string(&golden).unwrap_or_else(|e| {
            panic!("read {}: {e}", golden.display());
        });
        assert_eq!(
            actual,
            expected,
            "{} diverged from {}; re-run with BLESS_GOLDENS=1 if intentional",
            path.display(),
            golden.display()
        );
    }
}
