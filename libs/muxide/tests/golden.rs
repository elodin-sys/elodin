mod support;

use std::{fs, path::Path};
use support::parse_boxes;

#[test]
fn golden_minimal_contains_expected_boxes() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/minimal.mp4");
    let data = fs::read(fixture)?;
    let boxes = parse_boxes(&data);

    assert!(boxes.len() >= 2, "expected at least two top-level boxes");
    assert_eq!(boxes[0].typ, *b"ftyp");
    assert_eq!(boxes[1].typ, *b"moov");

    let moov = boxes.iter().find(|b| b.typ == *b"moov").unwrap();
    let moov_payload = moov.payload(&data);
    let child_boxes = parse_boxes(moov_payload);
    assert!(child_boxes.iter().any(|b| b.typ == *b"mvhd"));

    Ok(())
}
