use bevy::asset::{AssetPath, AssetServer, Assets};
use bevy::prelude::*;
use impeller2_bevy::DbMessage;
use impeller2_kdl::FromKdl;
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::env::schematic_file;
use impeller2_wkt::{DbConfig, Schematic};
use std::path::{Path, PathBuf};

use super::types::*;

fn canonicalize_or_original(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
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
    document_assets: &Assets<SchematicDocumentAsset>,
) -> Result<Option<SchematicDocumentAsset>, KdlSchematicError> {
    let resolved_path = schematic_file(path);
    if !resolved_path.try_exists().unwrap_or(false) {
        return Err(KdlSchematicError::NoSuchFile {
            path: resolved_path,
        });
    }

    let asset_path = filesystem_to_asset_path(&resolved_path);
    let handle: Handle<SchematicDocumentAsset> = asset_server.load(asset_path.clone());
    current_document.set_file(handle.clone(), asset_path, resolved_path);
    Ok(document_assets.get(&handle).cloned())
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
        secondary: Vec::new(),
    })
}

pub fn save_current_document(
    path: Option<PathBuf>,
    root_kdl: &str,
    secondary: &[SecondaryDocumentSave],
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
    write_secondary_schematics(dest.parent().unwrap_or_else(|| Path::new(".")), secondary)?;

    let asset_path = filesystem_to_asset_path(&dest);
    let handle: Handle<SchematicDocumentAsset> = asset_server.load(asset_path.clone());
    current_document.set_file(handle, asset_path, dest.clone());
    Ok(dest)
}

fn write_secondary_schematics(
    base_dir: &Path,
    secondary: &[SecondaryDocumentSave],
) -> Result<(), SaveCurrentDocumentError> {
    for entry in secondary {
        let dest = base_dir.join(&entry.file_name);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                SaveCurrentDocumentError::CreateSecondaryDir {
                    path: dest.clone(),
                    source,
                }
            })?;
        }

        std::fs::write(&dest, &entry.kdl)
            .map_err(|source| SaveCurrentDocumentError::WriteSecondary { path: dest, source })?;
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

pub fn sync_document_from_config(
    In(given_path): In<Option<PathBuf>>,
    config: Res<DbConfig>,
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
        open_document_from_content.write(OpenDocumentFromContentRequest {
            content: content.to_string(),
            save_path: config.schematic_path().map(Path::new).map(schematic_file),
        });
        return;
    }

    current_document.clear();
    cleared.write(DocumentCleared);
}
