use impeller2_wkt::{
    Object3DIcon, Object3DIconSource, Object3DMesh, Panel, Schematic, SchematicElem,
};
use serde::Deserialize;
use std::path::{Component, Path};

pub const SKYBOX_MANIFEST_ASSET_NAME: &str = "skyboxes/manifest.ron";

#[derive(Debug, thiserror::Error)]
pub enum SkyboxManifestError {
    #[error("failed to parse skybox manifest: {0}")]
    Ron(String),
    #[error("invalid skybox cubemap file path `{0}`")]
    InvalidCubemapPath(String),
}

impl From<ron::error::SpannedError> for SkyboxManifestError {
    fn from(value: ron::error::SpannedError) -> Self {
        Self::Ron(value.to_string())
    }
}

#[derive(Deserialize)]
struct SkyboxManifest {
    entries: Vec<SkyboxManifestEntry>,
}

#[derive(Deserialize)]
struct SkyboxManifestEntry {
    name: String,
    cubemap_file: String,
}

pub fn is_local_asset_path(path: &str) -> bool {
    !(path.starts_with("db:") || path.starts_with("http://") || path.starts_with("https://"))
}

pub fn local_asset_name(path: &str) -> Option<String> {
    if !is_local_asset_path(path) {
        return None;
    }
    path_components_to_asset_name(Path::new(path))
}

pub fn local_glb_asset_name(path: &str) -> Option<String> {
    local_asset_name(path)
}

/// Strips `db:` and returns the relative asset key stored under `{db}/assets/`.
pub fn db_asset_name(path: &str) -> Option<String> {
    let name = path.strip_prefix("db:")?.trim_start_matches('/');
    if name.is_empty() || name.contains('\0') {
        return None;
    }
    Some(name.to_owned())
}

pub fn db_glb_asset_name(path: &str) -> Option<String> {
    db_asset_name(path)
}

pub fn skybox_cubemap_asset_name(cubemap_file: &str) -> Option<String> {
    let mut parts = vec!["skyboxes".to_string()];
    for component in Path::new(cubemap_file).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    if parts.len() == 1 {
        return None;
    }
    Some(parts.join("/"))
}

pub fn skybox_manifest_cubemap_asset_name(
    manifest_ron: &str,
    skybox_name: &str,
) -> Result<Option<String>, SkyboxManifestError> {
    let manifest: SkyboxManifest = ron::from_str(manifest_ron)?;
    let Some(entry) = manifest
        .entries
        .iter()
        .find(|entry| entry.name == skybox_name)
    else {
        return Ok(None);
    };
    skybox_cubemap_asset_name(&entry.cubemap_file)
        .map(Some)
        .ok_or_else(|| SkyboxManifestError::InvalidCubemapPath(entry.cubemap_file.clone()))
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

pub fn rewrite_asset_paths<F>(schematic: &mut Schematic, mut map: F)
where
    F: FnMut(&str) -> Option<String>,
{
    for elem in &mut schematic.elems {
        rewrite_elem(elem, &mut map);
    }
}

pub fn rewrite_glb_paths<F>(schematic: &mut Schematic, map: F)
where
    F: FnMut(&str) -> Option<String>,
{
    rewrite_asset_paths(schematic, map);
}

fn rewrite_elem<F>(elem: &mut SchematicElem, map: &mut F)
where
    F: FnMut(&str) -> Option<String>,
{
    match elem {
        SchematicElem::Object3d(obj) => {
            rewrite_glb_mesh(&mut obj.mesh, map);
            rewrite_icon_path(&mut obj.icon, map);
        }
        SchematicElem::Panel(panel) => rewrite_panel(panel, map),
        SchematicElem::Window(window) => {
            if let Some(path) = &mut window.path
                && let Some(new_path) = map(path)
            {
                *path = new_path;
            }
        }
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

fn rewrite_icon_path<F>(icon: &mut Option<Object3DIcon>, map: &mut F)
where
    F: FnMut(&str) -> Option<String>,
{
    let Some(icon) = icon else {
        return;
    };
    if let Object3DIconSource::Path(path) = &mut icon.source
        && let Some(new_path) = map(path)
    {
        *path = new_path;
    }
}

pub fn collect_local_asset_paths(schematic: &Schematic) -> Vec<String> {
    let mut paths = Vec::new();
    if schematic.skybox.is_some() {
        paths.push(SKYBOX_MANIFEST_ASSET_NAME.to_string());
    }
    for elem in &schematic.elems {
        collect_elem_paths(elem, &mut paths);
    }
    paths.sort();
    paths.dedup();
    paths
}

pub fn collect_local_glb_paths(schematic: &Schematic) -> Vec<String> {
    collect_local_asset_paths(schematic)
}

pub fn collect_db_asset_names(schematic: &Schematic) -> Vec<String> {
    let mut names = Vec::new();
    if schematic.skybox.is_some() {
        names.push(SKYBOX_MANIFEST_ASSET_NAME.to_string());
    }
    for elem in &schematic.elems {
        collect_elem_db_asset_names(elem, &mut names);
    }
    names.sort();
    names.dedup();
    names
}

pub fn collect_db_glb_asset_names(schematic: &Schematic) -> Vec<String> {
    collect_db_asset_names(schematic)
}

fn collect_elem_db_asset_names(elem: &SchematicElem, names: &mut Vec<String>) {
    match elem {
        SchematicElem::Object3d(obj) => {
            if let Object3DMesh::Glb { path, .. } = &obj.mesh
                && let Some(name) = db_asset_name(path)
            {
                names.push(name);
            }
            if let Some(icon) = &obj.icon
                && let Object3DIconSource::Path(path) = &icon.source
                && let Some(name) = db_asset_name(path)
            {
                names.push(name);
            }
        }
        SchematicElem::Panel(panel) => collect_panel_db_asset_names(panel, names),
        SchematicElem::Window(window) => {
            if let Some(path) = &window.path
                && let Some(name) = db_asset_name(path)
            {
                names.push(name);
            }
        }
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
            if let Some(icon) = &obj.icon
                && let Object3DIconSource::Path(path) = &icon.source
                && is_local_asset_path(path)
            {
                paths.push(path.clone());
            }
        }
        SchematicElem::Panel(panel) => collect_panel_paths(panel, paths),
        SchematicElem::Window(window) => {
            if let Some(path) = &window.path
                && is_local_asset_path(path)
            {
                paths.push(path.clone());
            }
        }
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
    use impeller2_wkt::{
        Object3D, Object3DIcon, Object3DIconSource, Object3DMesh, SchematicElem, SkyboxConfig,
        default_icon_color, default_icon_size,
    };

    #[test]
    fn rewrite_local_glb_to_db_scheme() {
        let mut schematic = Schematic {
            elems: vec![SchematicElem::Object3d(Object3D {
                eql: "e".into(),
                mesh: Object3DMesh::glb("path/to/rocket.glb"),
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                icon: None,
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
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
    fn collect_local_asset_paths_skips_db_and_http() {
        let schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("models/a.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("db:b.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "c".into(),
                    mesh: Object3DMesh::glb("http://127.0.0.1:2241/c.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        assert_eq!(
            collect_local_asset_paths(&schematic),
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
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("models/b.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
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
                frame_orientation: None,
                orientation: Default::default(),
                icon: None,
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
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
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("other/rocket.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
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
            local_asset_name("/projets/a/rocket.glb").as_deref(),
            Some("projets/a/rocket.glb")
        );
        assert_eq!(
            local_asset_name("/autre/rocket.glb").as_deref(),
            Some("autre/rocket.glb")
        );
        assert_ne!(
            local_asset_name("/projets/a/rocket.glb"),
            local_asset_name("/autre/rocket.glb")
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
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("/autre/rocket.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
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
    fn collect_db_asset_names_from_schematic() {
        let schematic = Schematic {
            elems: vec![
                SchematicElem::Object3d(Object3D {
                    eql: "a".into(),
                    mesh: Object3DMesh::glb("db:models/a.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
                SchematicElem::Object3d(Object3D {
                    eql: "b".into(),
                    mesh: Object3DMesh::glb("models/local.glb"),
                    frame: None,
                    frame_orientation: None,
                    orientation: Default::default(),
                    icon: None,
                    thrusters: Vec::new(),
                    mesh_visibility_range: None,
                    node_id: Default::default(),
                }),
            ],
            ..Default::default()
        };
        assert_eq!(
            collect_db_asset_names(&schematic),
            vec!["models/a.glb".to_string()]
        );
    }

    #[test]
    fn rewrite_local_icon_png_to_db_scheme() {
        let mut schematic = Schematic {
            elems: vec![SchematicElem::Object3d(Object3D {
                eql: "e".into(),
                mesh: Object3DMesh::glb("model.glb"),
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                icon: Some(Object3DIcon {
                    source: Object3DIconSource::Path("icons/marker.png".into()),
                    color: default_icon_color(),
                    size: default_icon_size(),
                    visibility_range: None,
                }),
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
        });

        let SchematicElem::Object3d(obj) = &schematic.elems[0] else {
            panic!("expected object_3d");
        };
        let Object3DIconSource::Path(path) = &obj.icon.as_ref().unwrap().source else {
            panic!("expected icon path");
        };
        assert_eq!(path, "db:icons/marker.png");
        let Object3DMesh::Glb { path, .. } = &obj.mesh else {
            panic!("expected glb");
        };
        assert_eq!(path, "db:model.glb");
    }

    #[test]
    fn collect_local_paths_includes_icon_png() {
        let schematic = Schematic {
            elems: vec![SchematicElem::Object3d(Object3D {
                eql: "a".into(),
                mesh: Object3DMesh::glb("a.glb"),
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                icon: Some(Object3DIcon {
                    source: Object3DIconSource::Path("icons/a.png".into()),
                    color: default_icon_color(),
                    size: default_icon_size(),
                    visibility_range: None,
                }),
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };

        assert_eq!(
            collect_local_asset_paths(&schematic),
            vec!["a.glb".to_string(), "icons/a.png".to_string()]
        );
    }

    #[test]
    fn collect_db_asset_names_includes_icon_png() {
        let schematic = Schematic {
            elems: vec![SchematicElem::Object3d(Object3D {
                eql: "a".into(),
                mesh: Object3DMesh::glb("db:mesh.glb"),
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                icon: Some(Object3DIcon {
                    source: Object3DIconSource::Path("db:icons/a.png".into()),
                    color: default_icon_color(),
                    size: default_icon_size(),
                    visibility_range: None,
                }),
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                node_id: Default::default(),
            })],
            ..Default::default()
        };
        assert_eq!(
            collect_db_asset_names(&schematic),
            vec!["icons/a.png".to_string(), "mesh.glb".to_string()]
        );
    }

    #[test]
    fn collect_local_asset_paths_includes_skybox_manifest() {
        let schematic = Schematic {
            skybox: Some(SkyboxConfig {
                name: "desert_night".into(),
            }),
            ..Default::default()
        };

        assert_eq!(
            collect_local_asset_paths(&schematic),
            vec![SKYBOX_MANIFEST_ASSET_NAME.to_string()]
        );
    }

    #[test]
    fn collect_db_asset_names_includes_skybox_manifest() {
        let schematic = Schematic {
            skybox: Some(SkyboxConfig {
                name: "desert_night".into(),
            }),
            ..Default::default()
        };

        assert_eq!(
            collect_db_asset_names(&schematic),
            vec![SKYBOX_MANIFEST_ASSET_NAME.to_string()]
        );
    }

    #[test]
    fn skybox_manifest_cubemap_asset_name_extracts_active_ktx2() {
        let manifest = r#"
(
    version: 2,
    entries: [
        (
            name: "desert_night",
            prompt: "mojave",
            style: M3Photoreal,
            blockade: None,
            cubemap_file: "desert_night.cubemap.ktx2",
            face_size: 2048,
            created_at: "2026-05-11T05:34:26Z",
        ),
    ],
    default: Some("desert_night"),
)
"#;

        assert_eq!(
            skybox_manifest_cubemap_asset_name(manifest, "desert_night").unwrap(),
            Some("skyboxes/desert_night.cubemap.ktx2".to_string())
        );
    }

    #[test]
    fn rewrite_and_collect_window_schematic_path() {
        use impeller2_wkt::WindowSchematic;

        let mut schematic = Schematic {
            elems: vec![SchematicElem::Window(WindowSchematic {
                title: Some("detail".into()),
                path: Some("schematics/window-detail.kdl".into()),
                screen: None,
                screen_rect: None,
            })],
            ..Default::default()
        };

        assert_eq!(
            collect_local_asset_paths(&schematic),
            vec!["schematics/window-detail.kdl".to_string()]
        );

        rewrite_asset_paths(&mut schematic, |path| {
            local_asset_name(path).map(|name| format!("db:{name}"))
        });

        let SchematicElem::Window(window) = &schematic.elems[0] else {
            panic!("expected window");
        };
        assert_eq!(
            window.path.as_deref(),
            Some("db:schematics/window-detail.kdl")
        );
        assert_eq!(
            collect_db_asset_names(&schematic),
            vec!["schematics/window-detail.kdl".to_string()]
        );
    }

    #[test]
    fn collect_window_without_path_is_empty() {
        use impeller2_wkt::WindowSchematic;

        let schematic = Schematic {
            elems: vec![SchematicElem::Window(WindowSchematic::default())],
            ..Default::default()
        };
        assert!(collect_local_asset_paths(&schematic).is_empty());
        assert!(collect_db_asset_names(&schematic).is_empty());
    }

    #[test]
    fn skybox_manifest_cubemap_asset_name_rejects_traversal() {
        let manifest = r#"
(
    entries: [
        (
            name: "bad",
            cubemap_file: "../bad.ktx2",
        ),
    ],
)
"#;

        assert!(matches!(
            skybox_manifest_cubemap_asset_name(manifest, "bad"),
            Err(SkyboxManifestError::InvalidCubemapPath(_))
        ));
    }
}
