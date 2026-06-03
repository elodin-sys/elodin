use impeller2_wkt::{Object3DMesh, Panel, Schematic, SchematicElem};
use std::path::Path;

pub fn is_local_asset_path(path: &str) -> bool {
    !(path.starts_with("db:") || path.starts_with("http://") || path.starts_with("https://"))
}

pub fn local_glb_asset_name(path: &str) -> Option<String> {
    if !is_local_asset_path(path) {
        return None;
    }
    let name = Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())?;
    Some(name.to_string())
}

pub fn rewrite_glb_paths<F>(schematic: &mut Schematic, mut map: F)
where
    F: FnMut(&str) -> Option<String>,
{
    for elem in &mut schematic.elems {
        rewrite_elem(elem, &mut map);
    }
}

fn rewrite_elem<F>(elem: &mut SchematicElem, map: &mut F)
where
    F: FnMut(&str) -> Option<String>,
{
    match elem {
        SchematicElem::Object3d(obj) => rewrite_glb_mesh(&mut obj.mesh, map),
        SchematicElem::Panel(panel) => rewrite_panel(panel, map),
        _ => {}
    }
}

#[allow(clippy::only_used_in_recursion)]
fn rewrite_panel<F>(panel: &mut Panel, map: &mut F)
where
    F: FnMut(&str) -> Option<String>,
{
    match panel {
        Panel::VSplit(split) | Panel::HSplit(split) => {
            for child in &mut split.panels {
                rewrite_panel(child, map);
            }
        }
        Panel::Tabs(panels) => {
            for child in panels {
                rewrite_panel(child, map);
            }
        }
        _ => {}
    }
}

fn rewrite_glb_mesh<F>(mesh: &mut Object3DMesh, map: &mut F)
where
    F: FnMut(&str) -> Option<String>,
{
    if let Object3DMesh::Glb { path, .. } = mesh
        && let Some(new_path) = map(path)
    {
        *path = new_path;
    }
}

pub fn collect_local_glb_paths(schematic: &Schematic) -> Vec<String> {
    let mut paths = Vec::new();
    for elem in &schematic.elems {
        collect_elem_paths(elem, &mut paths);
    }
    paths.sort();
    paths.dedup();
    paths
}

fn collect_elem_paths(elem: &SchematicElem, paths: &mut Vec<String>) {
    match elem {
        SchematicElem::Object3d(obj) => {
            if let Object3DMesh::Glb { path, .. } = &obj.mesh
                && is_local_asset_path(path)
            {
                paths.push(path.clone());
            }
        }
        SchematicElem::Panel(panel) => collect_panel_paths(panel, paths),
        _ => {}
    }
}

#[allow(clippy::only_used_in_recursion)]
fn collect_panel_paths(panel: &Panel, paths: &mut Vec<String>) {
    match panel {
        Panel::VSplit(split) | Panel::HSplit(split) => {
            for child in &split.panels {
                collect_panel_paths(child, paths);
            }
        }
        Panel::Tabs(panels) => {
            for child in panels {
                collect_panel_paths(child, paths);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2_wkt::{Object3D, Object3DMesh, SchematicElem};

    #[test]
    fn rewrite_local_glb_to_db_scheme() {
        let mut schematic = Schematic {
            elems: vec![SchematicElem::Object3d(Object3D {
                eql: "e".into(),
                mesh: Object3DMesh::glb("path/to/rocket.glb"),
                frame: None,
                icon: None,
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };

        rewrite_glb_paths(&mut schematic, |path| {
            local_glb_asset_name(path).map(|name| format!("db:{name}"))
        });

        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("expected object_3d");
        };
        let Object3DMesh::Glb { path, .. } = &obj.mesh else {
            panic!("expected glb");
        };
        assert_eq!(path, "db:rocket.glb");
    }

    #[test]
    fn collect_local_glb_paths_skips_db_and_http() {
        let schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("models/a.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("db:b.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "c".into(),
                    mesh: Object3DMesh::glb("http://127.0.0.1:2241/c.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        assert_eq!(
            collect_local_glb_paths(&schematic),
            vec!["models/a.glb".to_string()]
        );
    }

    #[test]
    fn rewrite_multiple_root_object_3d_glbs() {
        let mut schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("models/a.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("models/b.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        rewrite_glb_paths(&mut schematic, |path| {
            local_glb_asset_name(path).map(|name| format!("db:{name}"))
        });

        let paths: Vec<_> = schematic
            .elems
            .iter()
            .filter_map(|elem| match elem {
                SchematicElem::Object3d(obj) => match &obj.mesh {
                    Object3DMesh::Glb { path, .. } => Some(path.as_str()),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(paths, vec!["db:a.glb", "db:b.glb"]);
    }

    #[test]
    fn serialize_round_trip_preserves_db_scheme() {
        use crate::serialize_schematic;

        let mut schematic = Schematic {
            elems: vec![SchematicElem::Object3d(Object3D {
                eql: "rocket.world_pos".into(),
                mesh: Object3DMesh::glb("path/to/rocket.glb"),
                frame: None,
                icon: None,
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };

        rewrite_glb_paths(&mut schematic, |path| {
            local_glb_asset_name(path).map(|name| format!("db:{name}"))
        });

        let kdl = serialize_schematic(&schematic);
        assert!(
            kdl.contains("path=db:rocket.glb"),
            "serialized KDL should contain db: reference, got:\n{kdl}"
        );

        let reparsed = crate::parse_schematic(&kdl).expect("reparsed KDL");
        let SchematicElem::Object3d(obj) = &reparsed.elems[0] else {
            panic!("expected object_3d");
        };
        let Object3DMesh::Glb { path, .. } = &obj.mesh else {
            panic!("expected glb");
        };
        assert_eq!(path, "db:rocket.glb");
    }

    #[test]
    fn basename_collision_rewrites_to_same_db_reference() {
        let mut schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("models/rocket.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("other/rocket.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        rewrite_glb_paths(&mut schematic, |path| {
            local_glb_asset_name(path).map(|name| format!("db:{name}"))
        });

        for elem in &schematic.elems {
            let SchematicElem::Object3d(obj) = elem else {
                continue;
            };
            let Object3DMesh::Glb { path, .. } = &obj.mesh else {
                panic!("expected glb");
            };
            assert_eq!(path, "db:rocket.glb");
        }
    }
}
