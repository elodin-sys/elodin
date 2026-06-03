use impeller2_wkt::{Object3DMesh, Panel, Schematic, SchematicElem};
use std::path::{Component, Path};

pub fn is_local_asset_path(path: &str) -> bool {
    !(path.starts_with("db:") || path.starts_with("http://") || path.starts_with("https://"))
}

pub fn local_glb_asset_name(path: &str) -> Option<String> {
    if !is_local_asset_path(path) {
        return None;
    }
    path_components_to_asset_name(Path::new(path))
}

/// Strips `db:` and returns the relative asset key stored under `{db}/assets/`.
pub fn db_glb_asset_name(path: &str) -> Option<String> {
    let name = path.strip_prefix("db:")?.trim_start_matches('/');
    if name.is_empty() || name.contains('\0') {
        return None;
    }
    Some(name.to_owned())
}

fn path_components_to_asset_name(path: &Path) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::Prefix(prefix) => {
                let drive = prefix.as_os_str().to_string_lossy();
                let drive = drive.trim_end_matches(':').trim_end_matches('\\');
                if !drive.is_empty() {
                    parts.push(drive.to_owned());
                }
            }
            Component::RootDir => {}
            Component::ParentDir => return None,
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("/"))
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

pub fn collect_db_glb_asset_names(schematic: &Schematic) -> Vec<String> {
    let mut names = Vec::new();
    for elem in &schematic.elems {
        collect_elem_db_asset_names(elem, &mut names);
    }
    names.sort();
    names.dedup();
    names
}

fn collect_elem_db_asset_names(elem: &SchematicElem, names: &mut Vec<String>) {
    match elem {
        SchematicElem::Object3d(obj) => {
            if let Object3DMesh::Glb { path, .. } = &obj.mesh
                && let Some(name) = db_glb_asset_name(path)
            {
                names.push(name);
            }
        }
        SchematicElem::Panel(panel) => collect_panel_db_asset_names(panel, names),
        _ => {}
    }
}

#[allow(clippy::only_used_in_recursion)]
fn collect_panel_db_asset_names(panel: &Panel, names: &mut Vec<String>) {
    match panel {
        Panel::VSplit(split) | Panel::HSplit(split) => {
            for child in &split.panels {
                collect_panel_db_asset_names(child, names);
            }
        }
        Panel::Tabs(panels) => {
            for child in panels {
                collect_panel_db_asset_names(child, names);
            }
        }
        _ => {}
    }
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
        assert_eq!(path, "db:path/to/rocket.glb");
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
        assert_eq!(paths, vec!["db:models/a.glb", "db:models/b.glb"]);
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
            kdl.contains("path=\"db:path/to/rocket.glb\""),
            "serialized KDL should contain db: reference, got:\n{kdl}"
        );

        let reparsed = crate::parse_schematic(&kdl).expect("reparsed KDL");
        let SchematicElem::Object3d(obj) = &reparsed.elems[0] else {
            panic!("expected object_3d");
        };
        let Object3DMesh::Glb { path, .. } = &obj.mesh else {
            panic!("expected glb");
        };
        assert_eq!(path, "db:path/to/rocket.glb");
    }

    #[test]
    fn basename_collision_preserves_relative_paths() {
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
        assert_eq!(paths, vec!["db:models/rocket.glb", "db:other/rocket.glb"]);
    }

    #[test]
    fn absolute_glb_paths_use_full_component_path() {
        assert_eq!(
            local_glb_asset_name("/projets/a/rocket.glb").as_deref(),
            Some("projets/a/rocket.glb")
        );
        assert_eq!(
            local_glb_asset_name("/autre/rocket.glb").as_deref(),
            Some("autre/rocket.glb")
        );
        assert_ne!(
            local_glb_asset_name("/projets/a/rocket.glb"),
            local_glb_asset_name("/autre/rocket.glb")
        );
    }

    #[test]
    fn absolute_paths_rewrite_to_distinct_db_scheme() {
        let mut schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("/projets/a/rocket.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("/autre/rocket.glb"),
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
        assert_eq!(
            paths,
            vec!["db:projets/a/rocket.glb", "db:autre/rocket.glb"]
        );
    }

    #[test]
    fn collect_db_glb_asset_names_from_schematic() {
        let schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("db:models/a.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("models/local.glb"),
                    frame: None,
                    icon: None,
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };
        assert_eq!(
            collect_db_glb_asset_names(&schematic),
            vec!["models/a.glb".to_string()]
        );
    }
}
