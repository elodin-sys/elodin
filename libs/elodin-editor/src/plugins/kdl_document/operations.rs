use crate::ui::schematic::CurrentSchematic;
use bevy::asset::{AssetPath, AssetServer};
use bevy::prelude::*;
use impeller2_bevy::DbMessage;
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::env::schematic_file;
use impeller2_kdl::{FromKdl, ToKdl};
use impeller2_wkt::{DbConfig, Schematic, SkyboxConfig};
use std::path::{Path, PathBuf};

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
    initial: Res<InitialKdlPath>,
) -> Option<PathBuf> {
    if !reader.read().any(|m| matches!(m, DbMessage::UpdateConfig)) {
        None
    } else {
        initial.0.clone()
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

pub fn sync_document_from_config(
    In(given_path): In<Option<PathBuf>>,
    config: Res<DbConfig>,
    last_synced_content: Res<LastSyncedSchematicContent>,
    mut current_document: ResMut<CurrentDocument>,
    mut open_document: MessageWriter<OpenDocumentRequest>,
    mut open_document_from_content: MessageWriter<OpenDocumentFromContentRequest>,
    mut cleared: MessageWriter<DocumentCleared>,
) {
    if given_path.is_none() && !config.is_changed() {
        return;
    }

    let has_content_fallback = config.schematic_content().is_some();
    let path_was_overridden = given_path.is_some();

    if let Some(path) = given_path.or(config.schematic_path().map(PathBuf::from)) {
        let resolved_path = schematic_file(&path);
        if resolved_path.try_exists().unwrap_or(false) {
            if current_document_matches_path(&current_document, &resolved_path) {
                return;
            }
            open_document.write(OpenDocumentRequest(path));
            return;
        }

        if has_content_fallback && !path_was_overridden {
            bevy::log::info!(
                "Schematic file {:?} not found; using embedded schematic content fallback",
                resolved_path.display()
            );
        }
    }

    if let Some(content) = config.schematic_content() {
        if last_synced_content.0.as_deref() == Some(content) {
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
