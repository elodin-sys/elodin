use impeller_kdl::{parse_schematic, schematic_to_python, serialize_schematic};

#[test]
fn geographic_names_round_trip_without_losing_precision() {
    let source = r#"place_names visible=#true {
        place "LC-39A" lat=28.60839 lon=-80.60433 alt=3.0
        place "Landing Zone 1" lat=28.48580 lon=-80.54440 alt=5.0
    }"#;
    let parsed = parse_schematic(source).unwrap();
    let names = parsed.place_names.as_ref().unwrap();
    assert!(names.visible);
    assert_eq!(names.places[0].latitude, 28.60839);
    assert_eq!(names.places[1].longitude, -80.54440);
    assert_eq!(
        parse_schematic(&serialize_schematic(&parsed)).unwrap(),
        parsed
    );
    let python = schematic_to_python(source, None).unwrap();
    assert!(python.contains("ui.place_names("));
    assert!(python.contains("Landing Zone 1"));
}

#[test]
fn invalid_geographic_names_are_rejected() {
    for source in [
        r#"place_names { place "bad" lat=91 lon=0 alt=0; }"#,
        r#"place_names { place "bad" lat=0 lon=181 alt=0; }"#,
        r#"place_names { place "" lat=0 lon=0 alt=0; }"#,
        r#"place_names { place "bad" lat=0 lon=0; }"#,
        r#"place_names visible="yes" {}"#,
    ] {
        assert!(
            parse_schematic(source).is_err(),
            "accepted invalid place names"
        );
    }
}
