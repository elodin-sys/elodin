use impeller_kdl::{FromKdl, ToKdl};
use impeller_wkt::Schematic;

#[test]
fn map_overlays_roundtrip_and_follow_db_asset_rewriting() {
    let mut scene = Schematic::from_kdl("map_overlay path=\"overlays/local.json\"\nmap_overlay path=\"overlays/world.json\" visible=#false\nviewport").unwrap();
    assert!(scene.map_overlays[0].visible);
    assert!(!scene.map_overlays[1].visible);
    assert_eq!(Schematic::from_kdl(&scene.to_kdl()).unwrap(), scene);
    assert_eq!(
        impeller_kdl::collect_local_asset_paths(&scene),
        ["overlays/local.json", "overlays/world.json"]
    );
    impeller_kdl::rewrite_asset_paths(&mut scene, |p| Some(format!("db:{p}")));
    assert!(impeller_kdl::collect_local_asset_paths(&scene).is_empty());
    assert_eq!(
        impeller_kdl::collect_db_asset_names(&scene),
        ["overlays/local.json", "overlays/world.json"]
    );
    assert_eq!(Schematic::from_kdl(&scene.to_kdl()).unwrap(), scene);
}

#[test]
fn map_overlay_rejects_invalid_declarations() {
    for input in [
        "map_overlay",
        "map_overlay path=\"../x.json\"",
        "map_overlay path=\"https://host/x.json\"",
        "map_overlay path=\"db:overlays/%2e%2e/x.json\"",
        "map_overlay path=\"x.glb\"",
        "map_overlay path=\"x.json\" visible=1",
    ] {
        assert!(Schematic::from_kdl(input).is_err(), "{input}");
    }
    assert!(
        Schematic::from_kdl("viewport")
            .unwrap()
            .map_overlays
            .is_empty()
    );
}
