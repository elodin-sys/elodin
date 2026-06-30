use crate::ui::schematic::CurrentSchematic;
use bevy::asset::{AssetPath, AssetServer};
use bevy::prelude::*;
use impeller2_bevy::DbMessage;
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::env::schematic_file;
use impeller2_kdl::{FromKdl, ToKdl};
use impeller2_wkt::{DbConfig, Schematic, SkyboxConfig};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::commands::*;
use super::messages::*;
use super::types::*;

use crate::plugins::kdl_asset_source::canonicalize_or_original;

/// Updates the in-memory schematic skybox (document asset + `CurrentSchematic`) and returns
/// the root KDL text for DB metadata sync.
pub fn sync_document_skybox(
    skybox: Option<SkyboxConfig>,
    current_document: &mut CurrentDocument,
    document_assets: &mut Assets<SchematicDocumentAsset>,
    schematic: &mut CurrentSchematic,
) -> String {
    schematic.skybox = skybox.clone();
    if let Some(handle) = current_document.handle.as_ref() {
        // Keep the in-memory document in sync for save, but suppress the asset
        // Modified event so skybox-only edits do not reload the full schematic.
        current_document.suppress_ids.insert(handle.id());
        if let Some(document) = document_assets.get_mut(handle) {
            document.root.skybox = skybox;
        }
    }
    schematic.to_kdl()
}

pub(crate) fn filesystem_to_asset_path(path: &Path) -> AssetPath<'static> {
    let resolved = canonicalize_or_original(path);
    let source = super::super::kdl_asset_source::KDL_ASSET_SOURCE;
    if let Ok(root) = impeller2_kdl::env::schematic_dir_or_cwd() {
        let canonical_root = canonicalize_or_original(&root);
        if let Ok(relative) = resolved.strip_prefix(&canonical_root) {
            return AssetPath::from_path_buf(relative.to_path_buf()).with_source(source);
        }
    }
    AssetPath::from_path_buf(resolved).with_source(source)
}

pub fn open_document_path(
    path: &Path,
    asset_server: &AssetServer,
    current_document: &mut CurrentDocument,
) -> Result<SchematicDocumentAsset, KdlSchematicError> {
    let resolved_path = schematic_file(path);
    if !resolved_path.try_exists().unwrap_or(false) {
        return Err(KdlSchematicError::NoSuchFile {
            path: resolved_path,
        });
    }

    let asset_path = filesystem_to_asset_path(&resolved_path);
    let handle: Handle<SchematicDocumentAsset> = asset_server.load(asset_path.clone());
    let document = read_document_from_disk(&resolved_path, &asset_path, asset_server)?;
    current_document.set_file(handle.clone(), asset_path, resolved_path);
    // Suppress the AssetServer LoadedWithDependencies/Modified event for this open;
    // DocumentLoaded already applied the synchronously read document.
    current_document.suppress_ids.insert(handle.id());
    Ok(document)
}

fn read_document_from_disk(
    resolved_path: &Path,
    asset_path: &AssetPath<'static>,
    asset_server: &AssetServer,
) -> Result<SchematicDocumentAsset, KdlSchematicError> {
    let contents =
        std::fs::read_to_string(resolved_path).map_err(|_| KdlSchematicError::NoSuchFile {
            path: resolved_path.to_path_buf(),
        })?;
    let root = Schematic::from_kdl(&contents)?;
    let base_dir = asset_path
        .path()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let source = asset_path.source().clone_owned();
    let windows = root
        .elems
        .iter()
        .filter_map(|elem| match elem {
            impeller2_wkt::SchematicElem::Window(window) => window.path.as_deref(),
            _ => None,
        })
        .map(|path| {
            let asset_path =
                AssetPath::from_path_buf(base_dir.join(path)).with_source(source.clone());
            let handle = asset_server.load(asset_path.clone());
            SchematicWindow {
                handle,
                asset_path: asset_path.clone_owned(),
            }
        })
        .collect();
    Ok(SchematicDocumentAsset { root, windows })
}

pub fn open_document_from_content(
    content: &str,
    save_path: Option<PathBuf>,
    current_document: &mut CurrentDocument,
) -> Result<SchematicDocumentAsset, KdlSchematicError> {
    let root = Schematic::from_kdl(content)?;
    current_document.set_unsaved_content(save_path);
    Ok(SchematicDocumentAsset {
        root,
        windows: Vec::new(),
    })
}

/// Fetch the active schematic's KDL from the DB Asset Server over HTTP. The
/// request is bounded so an unreachable DB cannot hang the load.
pub(crate) fn fetch_active_schematic_kdl(
    key: &str,
    connection_addr: Option<SocketAddr>,
) -> Result<String, String> {
    let url = crate::object_3d::resolve_db_asset_url(&format!("db:{key}"), connection_addr);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|err| format!("{url}: {err}"))?;
    let response = client
        .get(&url)
        .send()
        .map_err(|err| format!("{url}: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    response.text().map_err(|err| format!("{url}: {err}"))
}

/// What to `PUT` to the DB Asset Server before pointing `schematic.active` at a
/// freshly authored schematic (RFD #724 Phase 2).
pub(crate) struct DbSavePlan {
    /// `(asset key, bytes)` for window sub-schematics and, last, the active
    /// schematic itself.
    schematic_uploads: Vec<(String, Vec<u8>)>,
    /// `(asset key, local source path)` for meshes/icons still referenced by a
    /// local path; their bytes are read from disk at upload time.
    local_assets: Vec<(String, String)>,
}

/// Normalizes a root schematic for DB-native storage: rewrites local mesh, icon
/// and window references to `db:` keys and records what must be uploaded. Pure
/// (no I/O) so the rewrite and keying stay unit-testable.
pub(crate) fn plan_db_save(root: &Schematic, windows: &[WindowDocumentSave]) -> DbSavePlan {
    use std::collections::HashMap;
    let window_kdl: HashMap<&str, &str> = windows
        .iter()
        .map(|w| (w.file_name.as_str(), w.kdl.as_str()))
        .collect();

    let mut local_assets = Vec::new();
    let mut window_keys = Vec::new();
    let mut root = root.clone();
    impeller2_kdl::rewrite_asset_paths(&mut root, |path| {
        if !impeller2_kdl::is_local_asset_path(path) {
            return None;
        }
        // A detached-window sub-schematic: store it under `schematics/<file>`
        // and reference it there. Its bytes come from the in-memory window list,
        // not from disk.
        if window_kdl.contains_key(path) {
            let key = format!("schematics/{path}");
            window_keys.push(key.clone());
            return Some(format!("db:{key}"));
        }
        // A local mesh/icon: key by its component path and upload its bytes from
        // disk at PUT time.
        let name = impeller2_kdl::local_asset_name(path)?;
        local_assets.push((name.clone(), path.to_string()));
        Some(format!("db:{name}"))
    });

    let mut schematic_uploads = Vec::new();
    for key in &window_keys {
        if let Some(file_name) = key.strip_prefix("schematics/")
            && let Some(kdl) = window_kdl.get(file_name)
        {
            schematic_uploads.push((key.clone(), kdl.as_bytes().to_vec()));
        }
    }
    // The active schematic is uploaded last so its dependencies are present
    // before `schematic.active` is repointed at it.
    schematic_uploads.push((ACTIVE_SCHEMATIC_KEY.to_string(), root.to_kdl().into_bytes()));

    DbSavePlan {
        schematic_uploads,
        local_assets,
    }
}

/// Uploads a [`DbSavePlan`] to the DB Asset Server over HTTP `PUT`. Returns `Ok`
/// only when every upload was accepted, so the caller may then point
/// `schematic.active` at the uploaded schematic with no torn-write window.
pub(crate) fn upload_db_save_plan(
    plan: &DbSavePlan,
    connection_addr: Option<SocketAddr>,
) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|err| err.to_string())?;

    for (key, src) in &plan.local_assets {
        let path = schematic_file(Path::new(src));
        let bytes = std::fs::read(&path).map_err(|err| format!("{}: {err}", path.display()))?;
        put_db_asset(&client, key, bytes, connection_addr)?;
    }
    for (key, bytes) in &plan.schematic_uploads {
        put_db_asset(&client, key, bytes.clone(), connection_addr)?;
    }
    Ok(())
}

fn put_db_asset(
    client: &reqwest::blocking::Client,
    key: &str,
    bytes: Vec<u8>,
    connection_addr: Option<SocketAddr>,
) -> Result<(), String> {
    let url = crate::object_3d::resolve_db_asset_url(&format!("db:{key}"), connection_addr);
    let response = client
        .put(&url)
        .body(bytes)
        .send()
        .map_err(|err| format!("{url}: {err}"))?;
    if !response.status().is_success() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    Ok(())
}

pub fn save_current_document(
    path: Option<PathBuf>,
    root_kdl: &str,
    windows: &[WindowDocumentSave],
    asset_server: &AssetServer,
    current_document: &mut CurrentDocument,
) -> Result<PathBuf, SaveCurrentDocumentError> {
    let path = path
        .or_else(|| current_document.save_path.clone())
        .ok_or(SaveCurrentDocumentError::MissingSavePath)?;
    let path = Path::new(&path).with_extension("kdl");
    let dest = schematic_file(&path);

    std::fs::write(&dest, root_kdl).map_err(|source| SaveCurrentDocumentError::WriteRoot {
        path: dest.clone(),
        source,
    })?;
    write_window_schematics(dest.parent().unwrap_or_else(|| Path::new(".")), windows)?;

    let asset_path = filesystem_to_asset_path(&dest);
    let handle: Handle<SchematicDocumentAsset> = asset_server.load(asset_path.clone());
    let root_id = handle.id();
    current_document.set_file(handle, asset_path, dest.clone());
    current_document.suppress_ids.insert(root_id);
    Ok(dest)
}

fn write_window_schematics(
    base_dir: &Path,
    windows: &[WindowDocumentSave],
) -> Result<(), SaveCurrentDocumentError> {
    for entry in windows {
        let dest = base_dir.join(&entry.file_name);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                SaveCurrentDocumentError::CreateWindowDir {
                    path: dest.clone(),
                    source,
                }
            })?;
        }

        std::fs::write(&dest, &entry.kdl)
            .map_err(|source| SaveCurrentDocumentError::WriteWindow { path: dest, source })?;
    }

    Ok(())
}

/// Applies `InitialKdlPath` to `DbConfig` so that document sync can load that file.
/// Runs before `sync_document_from_config`. Re-applies when the path is missing or different (e.g.
/// after the connection overwrote DbConfig with metadata) so the schematic loads.
pub fn apply_initial_kdl_path(
    mut reader: MessageReader<DbMessage>,
    mut initial: ResMut<InitialKdlPath>,
    current_document: Res<CurrentDocument>,
) -> Option<PathBuf> {
    // If the user passed `--kdl`, we want to open that file exactly once.
    //
    // Historically this was only triggered by a DB config update message, but
    // that prevents using `elodin editor --kdl <file>` in offline / no-DB
    // scenarios.
    let path = initial.0.take()?;

    // Only apply when either:
    // - a DB config update arrived (normal flow), OR
    // - there is no current document loaded yet (offline flow).
    let db_updated = reader.read().any(|m| matches!(m, DbMessage::UpdateConfig));
    if db_updated || current_document.handle.is_none() {
        Some(path)
    } else {
        // Put it back; we'll try again on the next config update.
        initial.0 = Some(path);
        None
    }
}

fn current_document_matches_path(current_document: &CurrentDocument, path: &Path) -> bool {
    current_document.handle.is_some()
        && current_document
            .save_path
            .as_deref()
            .is_some_and(|save_path| {
                canonicalize_or_original(save_path) == canonicalize_or_original(path)
            })
}

/// Returns true when two KDL strings describe the same schematic (exact or semantically).
pub(crate) fn schematic_content_equivalent(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    match (Schematic::from_kdl(left), Schematic::from_kdl(right)) {
        (Ok(left), Ok(right)) => left.to_kdl() == right.to_kdl(),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sync_document_from_config(
    In(given_path): In<Option<PathBuf>>,
    config: Res<DbConfig>,
    last_synced_content: Res<LastSyncedSchematicContent>,
    mut current_document: ResMut<CurrentDocument>,
    mut open_document: MessageWriter<OpenDocumentRequest>,
    mut open_document_from_active: MessageWriter<OpenDocumentFromActiveRequest>,
    mut open_document_from_content: MessageWriter<OpenDocumentFromContentRequest>,
    mut cleared: MessageWriter<DocumentCleared>,
) {
    if given_path.is_none() && !config.is_changed() {
        return;
    }

    // An explicit path override (user opened a specific file) wins outright.
    if let Some(path) = given_path {
        let resolved_path = schematic_file(&path);
        if resolved_path.try_exists().unwrap_or(false) {
            if current_document_matches_path(&current_document, &resolved_path) {
                return;
            }
            open_document.write(OpenDocumentRequest(path));
            return;
        }
        // Overridden path is missing: fall through to the DB-backed sources.
    }

    // DB-centric load (RFD #724): when the DB advertises an active schematic it
    // is authoritative (its KDL is `db:`-rewritten), so it must take precedence
    // over any stale local on-disk file that still carries local asset paths.
    // The mirrored `schematic.content` doubles as the change guard (the shim
    // keeps it equal to the active asset) and as the offline fallback.
    if let Some(active_key) = config.schematic_active() {
        if let Some(mirror) = config.schematic_content()
            && last_synced_content
                .0
                .as_deref()
                .is_some_and(|last| schematic_content_equivalent(last, mirror))
        {
            return;
        }
        open_document_from_active.write(OpenDocumentFromActiveRequest {
            key: active_key.to_string(),
            content_fallback: config.schematic_content().map(str::to_string),
            save_path: config.schematic_path().map(Path::new).map(schematic_file),
        });
        return;
    }

    // No active schematic: fall back to the configured local path if present.
    if let Some(path) = config.schematic_path().map(PathBuf::from) {
        let resolved_path = schematic_file(&path);
        if resolved_path.try_exists().unwrap_or(false) {
            if current_document_matches_path(&current_document, &resolved_path) {
                return;
            }
            open_document.write(OpenDocumentRequest(path));
            return;
        }
        if config.schematic_content().is_some() {
            bevy::log::info!(
                "Schematic file {:?} not found; using embedded schematic content fallback",
                resolved_path.display()
            );
        }
    }

    if let Some(content) = config.schematic_content() {
        if last_synced_content
            .0
            .as_deref()
            .is_some_and(|last| schematic_content_equivalent(last, content))
        {
            return;
        }
        open_document_from_content.write(OpenDocumentFromContentRequest {
            content: content.to_string(),
            save_path: config.schematic_path().map(Path::new).map(schematic_file),
        });
        return;
    }

    current_document.clear();
    cleared.write(DocumentCleared);
}

#[cfg(test)]
mod db_save_tests {
    use super::*;
    use crate::ui::tiles::WindowId;
    use impeller2_wkt::{Object3D, Object3DMesh, SchematicElem, WindowSchematic};

    fn glb_object(eql: &str, mesh: &str) -> SchematicElem {
        SchematicElem::Object3d(Object3D {
            eql: eql.into(),
            mesh: Object3DMesh::glb(mesh),
            frame: None,
            frame_orientation: None,
            orientation: Default::default(),
            icon: None,
            thrusters: Vec::new(),
            mesh_visibility_range: None,
            node_id: Default::default(),
        })
    }

    #[test]
    fn plan_db_save_rewrites_local_assets_and_windows() {
        let root = Schematic {
            elems: vec![
                glb_object("a", "models/local.glb"),
                glb_object("b", "db:models/remote.glb"),
                SchematicElem::Window(WindowSchematic {
                    title: Some("detail".into()),
                    path: Some("detail.kdl".into()),
                    screen: None,
                    screen_rect: None,
                }),
            ],
            ..Default::default()
        };
        let windows = vec![WindowDocumentSave {
            window_id: WindowId(1),
            file_name: "detail.kdl".into(),
            kdl: "viewport {\n}\n".into(),
        }];

        let plan = plan_db_save(&root, &windows);

        // The local mesh is queued for upload; the `db:` mesh is left untouched.
        assert_eq!(
            plan.local_assets,
            vec![(
                "models/local.glb".to_string(),
                "models/local.glb".to_string()
            )]
        );

        // Window content is stored under `schematics/<file>`, and the active
        // schematic is uploaded last so its deps land first.
        let keys: Vec<&str> = plan
            .schematic_uploads
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, vec!["schematics/detail.kdl", ACTIVE_SCHEMATIC_KEY]);

        // Every reference in the active schematic is now a `db:` key.
        let active = String::from_utf8(plan.schematic_uploads.last().unwrap().1.clone()).unwrap();
        assert!(active.contains("db:models/local.glb"), "{active}");
        assert!(active.contains("db:models/remote.glb"), "{active}");
        assert!(active.contains("db:schematics/detail.kdl"), "{active}");
    }

    #[test]
    fn plan_db_save_without_deps_uploads_only_active() {
        let plan = plan_db_save(&Schematic::default(), &[]);
        assert!(plan.local_assets.is_empty());
        assert_eq!(plan.schematic_uploads.len(), 1);
        assert_eq!(plan.schematic_uploads[0].0, ACTIVE_SCHEMATIC_KEY);
    }
}
