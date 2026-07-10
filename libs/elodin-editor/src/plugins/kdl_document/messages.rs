use super::types::SchematicDocumentAsset;
use bevy::prelude::*;
use std::path::PathBuf;

#[derive(Message, Clone, Debug)]
pub struct DocumentLoaded {
    pub save_path: Option<PathBuf>,
    pub document: SchematicDocumentAsset,
    /// The load was user-initiated ("Open Schematic…", opening a file) rather
    /// than a background sync (connect-time load, external repoint, byte
    /// change). An explicit open re-applies the document's skybox even when
    /// the DB carries a sticky clear (`skybox.active=""`): the user asked for
    /// this schematic, skybox included.
    pub explicit: bool,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentCommandFailed {
    pub title: String,
    pub message: String,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentReloaded {
    pub save_path: Option<PathBuf>,
    pub document: SchematicDocumentAsset,
    /// Indices of windows whose assets changed. Empty means full reload.
    pub changed_window_indices: Vec<usize>,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentLoadFailed {
    pub path: String,
    pub message: String,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentCleared;
