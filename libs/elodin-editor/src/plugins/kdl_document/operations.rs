use crate::ui::schematic::CurrentSchematic;
use bevy::asset::{AssetPath, AssetServer};
use bevy::prelude::*;
use impeller2_bevy::DbMessage;
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::env::schematic_file;
use impeller2_kdl::{FromKdl, ToKdl};
use impeller2_wkt::{DbConfig, Schematic, SkyboxConfig};
use std::collections::{HashMap, HashSet};
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
pub fn fetch_active_schematic_kdl(
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

/// List the schematic asset keys (`schematics/*.kdl`) the DB Asset Server holds,
/// via its `GET /__index__/<prefix>` listing (RFD #724). Sorted for a stable
/// picker; the request is bounded so an unreachable DB cannot hang the UI.
pub(crate) fn fetch_schematic_index(
    connection_addr: Option<SocketAddr>,
) -> Result<Vec<String>, String> {
    #[derive(serde::Deserialize)]
    struct IndexEntry {
        key: String,
    }

    let url = crate::object_3d::resolve_db_asset_url("db:__index__/schematics/", connection_addr);
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
    let entries: Vec<IndexEntry> = response.json().map_err(|err| format!("{url}: {err}"))?;
    let mut keys: Vec<String> = entries
        .into_iter()
        .map(|entry| entry.key)
        .filter(|key| key.ends_with(".kdl"))
        .collect();
    keys.sort();
    Ok(keys)
}

/// Builds the DB asset key (`schematics/<name>.kdl`) for a user-entered "Save
/// As" name. Validates rather than silently mangling: a trailing `.kdl` is
/// tolerated, but the stem must be non-empty and limited to `[A-Za-z0-9_-]` so
/// the key can never escape the `schematics/` prefix or hit a reserved/odd key.
pub(crate) fn schematic_save_key_from_name(name: &str) -> Result<String, String> {
    let stem = name.trim().strip_suffix(".kdl").unwrap_or(name.trim());
    if stem.is_empty() {
        return Err("Schematic name cannot be empty".to_string());
    }
    if !stem
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("Use only letters, digits, '-' or '_' in a schematic name".to_string());
    }
    Ok(format!("schematics/{stem}.kdl"))
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

/// Rewrites a schematic's local mesh, icon and window references to `db:` keys,
/// recording mesh/icon uploads in `local_assets` and the file names of any
/// referenced window sub-schematics in `referenced_windows`. Shared by the root
/// and every detached window so 3D content referenced only inside a window is
/// stored DB-natively too.
fn rewrite_schematic_for_db(
    schematic: &mut Schematic,
    window_kdl: &HashMap<&str, &str>,
    local_assets: &mut Vec<(String, String)>,
    referenced_windows: &mut Vec<String>,
) {
    impeller2_kdl::rewrite_asset_paths(schematic, |path| {
        if !impeller2_kdl::is_local_asset_path(path) {
            return None;
        }
        // A detached-window sub-schematic: store it under `schematics/<file>`
        // and reference it there. Its rewritten bytes are uploaded separately.
        if window_kdl.contains_key(path) {
            referenced_windows.push(path.to_string());
            return Some(format!("db:schematics/{path}"));
        }
        // A local mesh/icon: key by its component path and upload its bytes from
        // disk at PUT time.
        let name = impeller2_kdl::local_asset_name(path)?;
        local_assets.push((name.clone(), path.to_string()));
        Some(format!("db:{name}"))
    });
}

/// Normalizes a root schematic for DB-native storage: rewrites local mesh, icon
/// and window references to `db:` keys (including assets referenced only inside
/// detached window sub-schematics) and records what must be uploaded. The active
/// schematic is stored under `active_key` (e.g. `schematics/main.kdl`, or a
/// user-named `schematics/<name>.kdl` for "Save As"). Pure (no I/O) so the
/// rewrite and keying stay unit-testable.
pub(crate) fn plan_db_save(
    root: &Schematic,
    windows: &[WindowDocumentSave],
    active_key: &str,
) -> DbSavePlan {
    let window_kdl: HashMap<&str, &str> = windows
        .iter()
        .map(|w| (w.file_name.as_str(), w.kdl.as_str()))
        .collect();

    let mut local_assets = Vec::new();
    let mut schematic_uploads = Vec::new();

    // Rewrite the root, discovering the window sub-schematics it references.
    let mut root = root.clone();
    let mut referenced_windows = Vec::new();
    rewrite_schematic_for_db(
        &mut root,
        &window_kdl,
        &mut local_assets,
        &mut referenced_windows,
    );

    // Rewrite each referenced window the same way and upload the rewritten KDL,
    // so meshes/icons referenced only inside a window are also `db:`-keyed and
    // queued for upload. Windows may reference further windows, so follow them
    // transitively, keying each by `schematics/<file>` and uploading once.
    let mut uploaded = HashSet::new();
    let mut next = 0;
    while next < referenced_windows.len() {
        let file_name = referenced_windows[next].clone();
        next += 1;
        let key = format!("schematics/{file_name}");
        if !uploaded.insert(key.clone()) {
            continue;
        }
        let Some(&kdl) = window_kdl.get(file_name.as_str()) else {
            continue;
        };
        let bytes = match Schematic::from_kdl(kdl) {
            Ok(mut window_root) => {
                rewrite_schematic_for_db(
                    &mut window_root,
                    &window_kdl,
                    &mut local_assets,
                    &mut referenced_windows,
                );
                window_root.to_kdl().into_bytes()
            }
            // If a window's KDL cannot be parsed, fall back to its raw bytes so
            // the save still includes it rather than silently dropping it.
            Err(_) => kdl.as_bytes().to_vec(),
        };
        schematic_uploads.push((key, bytes));
    }

    // A window and the root (or another window) may reference the same local
    // asset; keep only the first upload of each, preserving order.
    let mut seen_assets = HashSet::new();
    local_assets.retain(|(key, _)| seen_assets.insert(key.clone()));

    // The active schematic is uploaded last so its dependencies are present
    // before `schematic.active` is repointed at it.
    schematic_uploads.push((active_key.to_string(), root.to_kdl().into_bytes()));

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
        let path = local_asset_file(src);
        let bytes = std::fs::read(&path).map_err(|err| format!("{}: {err}", path.display()))?;
        put_db_asset(&client, key, bytes, connection_addr)?;
    }
    for (key, bytes) in &plan.schematic_uploads {
        put_db_asset(&client, key, bytes.clone(), connection_addr)?;
    }
    Ok(())
}

/// Resolves a local mesh/icon path referenced by a schematic to a filesystem
/// path for upload. These paths are loaded by the editor's `AssetServer`, whose
/// default source is `$ELODIN_ASSETS` or `assets/` under cwd — not the schematic
/// root. Resolving them the same way here keeps a DB save reading the exact
/// bytes the editor displays.
fn local_asset_file(src: &str) -> PathBuf {
    let path = Path::new(src);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match crate::plugins::env_asset_source::resolve_assets_dir() {
        Some(root) => root.join(path),
        None => path.to_path_buf(),
    }
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

/// Display/save path derived from the DB asset key (`schematics/foo.kdl`).
pub(crate) fn save_path_for_active_key(key: &str) -> PathBuf {
    PathBuf::from(key)
}

#[allow(clippy::too_many_arguments)]
pub fn sync_document_from_config(
    In(given_path): In<Option<PathBuf>>,
    config: Res<DbConfig>,
    last_synced_key: Res<LastSyncedActiveKey>,
    mut pending_active: ResMut<PendingActiveSchematic>,
    mut current_document: ResMut<CurrentDocument>,
    mut open_document: MessageWriter<OpenDocumentRequest>,
    mut open_document_from_active: MessageWriter<OpenDocumentFromActiveRequest>,
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

    // DB-centric load (RFD #724): `schematic.active` is authoritative; fetch its
    // bytes over HTTP. Reload on a change of the active *key* alone — opening
    // another stored schematic with byte-identical KDL still switches the active
    // key, and the editor must follow it.
    if let Some(active_key) = config.schematic_active() {
        if let Some(pending) = pending_active.0.clone() {
            if pending == active_key {
                pending_active.0 = None;
            } else {
                return;
            }
        }

        if last_synced_key.0.as_deref() == Some(active_key) {
            return;
        }
        open_document_from_active.write(OpenDocumentFromActiveRequest {
            key: active_key.to_string(),
            save_path: Some(save_path_for_active_key(active_key)),
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

        let plan = plan_db_save(&root, &windows, ACTIVE_SCHEMATIC_KEY);

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
    fn plan_db_save_rewrites_window_internal_assets() {
        let root = Schematic {
            elems: vec![SchematicElem::Window(WindowSchematic {
                title: Some("detail".into()),
                path: Some("detail.kdl".into()),
                screen: None,
                screen_rect: None,
            })],
            ..Default::default()
        };
        // A mesh referenced only inside the window, nowhere in the root.
        let window = Schematic {
            elems: vec![glb_object("c", "models/window_only.glb")],
            ..Default::default()
        };
        let windows = vec![WindowDocumentSave {
            window_id: WindowId(2),
            file_name: "detail.kdl".into(),
            kdl: window.to_kdl(),
        }];

        let plan = plan_db_save(&root, &windows, ACTIVE_SCHEMATIC_KEY);

        // The window's local mesh must be queued for upload...
        assert!(
            plan.local_assets
                .iter()
                .any(|(key, _)| key == "models/window_only.glb"),
            "{:?}",
            plan.local_assets
        );

        // ...and the stored window KDL must reference it over `db:`.
        let window_upload = plan
            .schematic_uploads
            .iter()
            .find(|(key, _)| key == "schematics/detail.kdl")
            .map(|(_, bytes)| String::from_utf8(bytes.clone()).unwrap())
            .expect("window sub-schematic should be uploaded");
        assert!(
            window_upload.contains("db:models/window_only.glb"),
            "{window_upload}"
        );
    }

    #[test]
    fn plan_db_save_without_deps_uploads_only_active() {
        let plan = plan_db_save(&Schematic::default(), &[], ACTIVE_SCHEMATIC_KEY);
        assert!(plan.local_assets.is_empty());
        assert_eq!(plan.schematic_uploads.len(), 1);
        assert_eq!(plan.schematic_uploads[0].0, ACTIVE_SCHEMATIC_KEY);
    }

    #[test]
    fn plan_db_save_stores_active_under_named_key() {
        let plan = plan_db_save(&Schematic::default(), &[], "schematics/orbit.kdl");
        assert_eq!(
            plan.schematic_uploads.last().unwrap().0,
            "schematics/orbit.kdl"
        );
    }

    #[test]
    fn schematic_save_key_from_name_validates_and_keys() {
        assert_eq!(
            schematic_save_key_from_name("orbit"),
            Ok("schematics/orbit.kdl".to_string())
        );
        // A typed `.kdl` suffix is tolerated, not doubled.
        assert_eq!(
            schematic_save_key_from_name(" my-run_2.kdl "),
            Ok("schematics/my-run_2.kdl".to_string())
        );
        assert!(schematic_save_key_from_name("   ").is_err());
        // No traversal or nested keys.
        assert!(schematic_save_key_from_name("../escape").is_err());
        assert!(schematic_save_key_from_name("a/b").is_err());
    }
}
