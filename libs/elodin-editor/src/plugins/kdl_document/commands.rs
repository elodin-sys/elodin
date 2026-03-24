use super::types::WindowDocumentSave;
use bevy::prelude::*;
use std::path::PathBuf;

#[derive(Message, Clone, Debug)]
pub struct OpenDocumentRequest(pub PathBuf);

#[derive(Message, Clone, Debug)]
pub struct OpenDocumentFromContentRequest {
    pub content: String,
    pub save_path: Option<PathBuf>,
}

#[derive(Message, Clone, Debug)]
pub struct SaveCurrentDocumentRequest {
    pub path: Option<PathBuf>,
    pub root_kdl: String,
    pub windows: Vec<WindowDocumentSave>,
}
