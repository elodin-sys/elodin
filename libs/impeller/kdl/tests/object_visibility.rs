use impeller_kdl::{FromKdl, ToKdl};
use impeller_wkt::{Schematic, SchematicElem};

fn visible(scene: &Schematic) -> Option<&str> {
    let SchematicElem::Object3d(object) = &scene.elems[0] else {
        panic!("expected object")
    };
    object.visible.as_deref()
}

#[test]
fn recorded_object_visibility_survives_roundtrip_and_asset_rewriting() {
    let mut scene = Schematic::from_kdl(
        "object_3d \"booster.visual_pos\" visible=\"booster.upper_mass[0]\" { glb path=\"falcon9-upper-stage.glb\" }\nviewport",
    ).unwrap();
    assert_eq!(visible(&scene), Some("booster.upper_mass[0]"));
    assert_eq!(Schematic::from_kdl(&scene.to_kdl()).unwrap(), scene);
    impeller_kdl::rewrite_asset_paths(&mut scene, |p| Some(format!("db:{p}")));
    assert_eq!(Schematic::from_kdl(&scene.to_kdl()).unwrap(), scene);
    assert_eq!(visible(&scene), Some("booster.upper_mass[0]"));
    let python = impeller_kdl::schematic_to_python(&scene.to_kdl(), None).unwrap();
    assert!(python.contains("visible=\"booster.upper_mass[0]\""));
}

#[test]
fn existing_objects_remain_unconditionally_visible() {
    let scene = Schematic::from_kdl("object_3d \"ball.world_pos\" { sphere radius=0.2 }\nviewport")
        .unwrap();
    assert!(visible(&scene).is_none());
    assert!(!scene.to_kdl().contains("visible="));
}

#[test]
fn visibility_requires_a_nonempty_expression() {
    for value in ["#true", "1", "\"\""] {
        let input = format!("object_3d \"ball.world_pos\" visible={value} {{ sphere radius=0.2 }}");
        assert!(Schematic::from_kdl(&input).is_err());
    }
}
