use bevy::asset::{AssetEvent, AssetLoadFailedEvent};
use bevy::prelude::*;
use bevy_ai_skybox::prelude::{SetActiveSkybox, SkyboxCache};
use impeller2_wkt::SkyboxConfig;
use std::path::PathBuf;

use super::commands::*;
use super::messages::*;
use super::operations::{open_document_from_content, open_document_path, save_current_document};
use super::types::*;

fn cloned_current_document_asset(
    current_document: &CurrentDocument,
    document_assets: &Assets<SchematicDocumentAsset>,
) -> Option<(Option<PathBuf>, SchematicDocumentAsset)> {
    let handle = current_document.handle.clone()?;
    let save_path = current_document.save_path.clone();
    let document = document_assets.get(&handle).cloned()?;
    Some((save_path, document))
}

pub(super) fn handle_open_document_requests(
    mut requests: MessageReader<OpenDocumentRequest>,
    asset_server: Res<AssetServer>,
    mut current_document: ResMut<CurrentDocument>,
    mut loaded: MessageWriter<DocumentLoaded>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    for request in requests.read() {
        match open_document_path(&request.0, &asset_server, &mut current_document) {
            Ok(document) => {
                loaded.write(DocumentLoaded {
                    save_path: current_document.save_path.clone(),
                    document,
                });
            }
            Err(error) => {
                failed.write(DocumentCommandFailed {
                    title: format!("Invalid Schematic in {}", request.0.display()),
                    message: error.to_string(),
                });
            }
        }
    }
}

pub(super) fn handle_open_document_from_content_requests(
    mut requests: MessageReader<OpenDocumentFromContentRequest>,
    mut current_document: ResMut<CurrentDocument>,
    mut loaded: MessageWriter<DocumentLoaded>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    for request in requests.read() {
        match open_document_from_content(
            &request.content,
            request.save_path.clone(),
            &mut current_document,
        ) {
            Ok(document) => {
                loaded.write(DocumentLoaded {
                    save_path: current_document.save_path.clone(),
                    document,
                });
            }
            Err(error) => {
                failed.write(DocumentCommandFailed {
                    title: "Invalid Schematic".to_string(),
                    message: error.to_string(),
                });
            }
        }
    }
}

pub(super) fn handle_save_current_document_requests(
    mut requests: MessageReader<SaveCurrentDocumentRequest>,
    asset_server: Res<AssetServer>,
    mut current_document: ResMut<CurrentDocument>,
    mut saved: MessageWriter<DocumentSaved>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    for request in requests.read() {
        match save_current_document(
            request.path.clone(),
            &request.root_kdl,
            &request.windows,
            &asset_server,
            &mut current_document,
        ) {
            Ok(save_path) => {
                saved.write(DocumentSaved {
                    save_path,
                    windows: request
                        .windows
                        .iter()
                        .map(|w| SavedWindowInfo {
                            window_id: w.window_id,
                            file_name: w.file_name.clone(),
                        })
                        .collect(),
                });
            }
            Err(error) => {
                failed.write(DocumentCommandFailed {
                    title: "Failed to Save Schematic".to_string(),
                    message: error.to_string(),
                });
            }
        }
    }
}

pub(super) fn emit_document_reloads(
    mut events: MessageReader<AssetEvent<SchematicDocumentAsset>>,
    mut current_document: ResMut<CurrentDocument>,
    document_assets: Res<Assets<SchematicDocumentAsset>>,
    mut reloaded: MessageWriter<DocumentReloaded>,
) {
    let window_handles = current_document.window_handles(&document_assets);

    // Lazily expand the suppress set to include window sub-asset IDs once they
    // become available (they aren't known at save time).
    if !current_document.suppress_ids.is_empty() {
        current_document
            .suppress_ids
            .extend(window_handles.iter().copied());
    }

    let mut root_changed = false;
    let mut changed_window_indices = Vec::new();

    for event in events.read() {
        let id = match event {
            AssetEvent::LoadedWithDependencies { id } | AssetEvent::Modified { id } => *id,
            _ => continue,
        };

        let is_root = current_document.matches(id);
        let window_idx = window_handles.iter().position(|wid| *wid == id);

        if !is_root && window_idx.is_none() {
            continue;
        }

        if current_document.suppress_ids.remove(&id) {
            continue;
        }

        if is_root {
            root_changed = true;
        }
        if let Some(i) = window_idx {
            changed_window_indices.push(i);
        }
    }

    if !root_changed && changed_window_indices.is_empty() {
        return;
    }

    // Genuine external changes detected; discard any remaining suppress IDs.
    current_document.suppress_ids.clear();

    // When only window sub-assets changed, report the specific indices.
    // When the root changed, report an empty vec to signal a full reload.
    let changed = if root_changed {
        Vec::new()
    } else {
        changed_window_indices
    };

    if let Some((save_path, document)) =
        cloned_current_document_asset(&current_document, &document_assets)
    {
        reloaded.write(DocumentReloaded {
            save_path,
            document,
            changed_window_indices: changed,
        });
    }
}

pub(super) fn activate_document_skybox(
    mut loaded: MessageReader<DocumentLoaded>,
    mut reloaded: MessageReader<DocumentReloaded>,
    mut skyboxes: MessageWriter<SetActiveSkybox>,
    mut cache: Option<ResMut<SkyboxCache>>,
) {
    for event in loaded.read() {
        activate_skybox_config(
            event.document.root.skybox.as_ref(),
            &mut skyboxes,
            &mut cache,
        );
    }

    for event in reloaded.read() {
        if !event.changed_window_indices.is_empty() {
            continue;
        }
        activate_skybox_config(
            event.document.root.skybox.as_ref(),
            &mut skyboxes,
            &mut cache,
        );
    }
}

fn activate_skybox_config(
    skybox: Option<&SkyboxConfig>,
    skyboxes: &mut MessageWriter<SetActiveSkybox>,
    cache: &mut Option<ResMut<SkyboxCache>>,
) {
    if let Some(cache) = cache.as_mut() {
        // Drop stale cache state; bevy_ai_skybox sets `active` once the cubemap is ready.
        cache.active = None;
    }
    match skybox {
        Some(skybox) => {
            skyboxes.write(SetActiveSkybox::ByName(skybox.name.clone()));
        }
        None => {
            skyboxes.write(SetActiveSkybox::Clear);
        }
    }
}

pub(super) fn emit_document_load_failures(
    mut events: MessageReader<AssetLoadFailedEvent<SchematicDocumentAsset>>,
    current_document: Res<CurrentDocument>,
    document_assets: Res<Assets<SchematicDocumentAsset>>,
    mut failed: MessageWriter<DocumentLoadFailed>,
) {
    let window_handles = current_document.window_handles(&document_assets);
    for event in events.read() {
        if !current_document.matches(event.id) && !window_handles.contains(&event.id) {
            continue;
        }
        failed.write(DocumentLoadFailed {
            path: format!("{:?}", event.path),
            message: event.error.to_string(),
        });
    }
}
