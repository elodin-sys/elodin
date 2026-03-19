use bevy::asset::{AssetEvent, AssetLoadFailedEvent};
use bevy::prelude::*;
use std::path::PathBuf;

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

fn matching_current_document_event_count(
    events: &mut MessageReader<AssetEvent<SchematicDocumentAsset>>,
    current_document: &CurrentDocument,
) -> usize {
    events
        .read()
        .filter_map(|event| match event {
            AssetEvent::LoadedWithDependencies { id } | AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .filter(|id| current_document.matches(*id))
        .count()
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
            &request.secondary,
            &asset_server,
            &mut current_document,
        ) {
            Ok(save_path) => {
                saved.write(DocumentSaved { save_path });
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
    current_document: Res<CurrentDocument>,
    document_assets: Res<Assets<SchematicDocumentAsset>>,
    mut reloaded: MessageWriter<DocumentReloaded>,
) {
    if matching_current_document_event_count(&mut events, &current_document) == 0 {
        return;
    }

    if let Some((save_path, document)) =
        cloned_current_document_asset(&current_document, &document_assets)
    {
        reloaded.write(DocumentReloaded {
            save_path,
            document,
        });
    }
}

pub(super) fn emit_document_load_failures(
    mut events: MessageReader<AssetLoadFailedEvent<SchematicDocumentAsset>>,
    current_document: Res<CurrentDocument>,
    mut failed: MessageWriter<DocumentLoadFailed>,
) {
    for event in events.read() {
        if !current_document.matches(event.id) {
            continue;
        }
        failed.write(DocumentLoadFailed {
            path: format!("{:?}", event.path),
            message: event.error.to_string(),
        });
    }
}
