use super::types::SchematicDocumentAsset;
use bevy::prelude::*;
use std::path::PathBuf;

#[derive(Message, Clone, Debug)]
pub struct DocumentLoaded {
    pub save_path: Option<PathBuf>,
    pub document: SchematicDocumentAsset,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentCommandFailed {
    pub title: String,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct SavedWindowInfo {
    pub window_id: crate::ui::tiles::WindowId,
    pub file_name: String,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentSaved {
    pub save_path: PathBuf,
    pub windows: Vec<SavedWindowInfo>,
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
