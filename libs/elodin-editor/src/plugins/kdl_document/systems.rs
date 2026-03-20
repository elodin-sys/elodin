use bevy::asset::{AssetEvent, AssetLoadFailedEvent};
use bevy::prelude::*;
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
    document_assets: Res<Assets<SchematicDocumentAsset>>,
    mut current_document: ResMut<CurrentDocument>,
    mut loaded: MessageWriter<DocumentLoaded>,
    mut failed: MessageWriter<DocumentCommandFailed>,
) {
    for request in requests.read() {
        match open_document_path(
            &request.0,
            &asset_server,
            &mut current_document,
            &document_assets,
        ) {
            Ok(Some(document)) => {
                loaded.write(DocumentLoaded {
                    save_path: current_document.save_path.clone(),
                    document,
                });
            }
            Ok(None) => {}
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
    let mut root_changed = false;
    let mut changed_window_indices = Vec::new();

    for event in events.read() {
        let id = match event {
            AssetEvent::LoadedWithDependencies { id } | AssetEvent::Modified { id } => *id,
            _ => continue,
        };

        if current_document.matches(id) {
            root_changed = true;
        }
        for (i, window_id) in window_handles.iter().enumerate() {
            if *window_id == id {
                changed_window_indices.push(i);
            }
        }
    }

    if !root_changed && changed_window_indices.is_empty() {
        return;
    }

    if current_document.suppress_next_reload {
        current_document.suppress_next_reload = false;
        return;
    }

    // When only window sub-assets changed (not the root), report the specific indices.
    // When the root changed, report an empty vec to signal a full reload.
    let changed = if root_changed && changed_window_indices.is_empty() {
        Vec::new()
    } else if !root_changed {
        changed_window_indices
    } else {
        Vec::new()
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
