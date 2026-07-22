//! Every KDL schematic shipped with the examples must parse: the Python SDK
//! embeds schematic text without validating it, so a syntax error would
//! otherwise surface only when someone opens the editor.

use std::path::PathBuf;

#[test]
fn all_example_schematics_parse() {
    let examples = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../examples");
    let mut checked = 0;
    for entry in std::fs::read_dir(&examples).expect("examples dir") {
        let dir = entry.expect("dir entry").path();
        if !dir.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&dir).expect("example dir") {
            let path = file.expect("file entry").path();
            if path.extension().is_some_and(|e| e == "kdl") {
                let text = std::fs::read_to_string(&path).expect("read kdl");
                if let Err(err) = impeller2_kdl::parse_schematic(&text) {
                    panic!("{} failed to parse: {err:?}", path.display());
                }
                checked += 1;
            }
        }
    }
    assert!(
        checked >= 2,
        "expected to find example schematics, got {checked}"
    );
}
