use bevy::prelude::*;
use std::path::PathBuf;

#[derive(Message, Clone, Debug)]
pub struct OpenDocumentRequest(pub PathBuf);

#[derive(Message, Clone, Debug)]
pub struct OpenDocumentFromContentRequest {
    pub content: String,
    pub save_path: Option<PathBuf>,
}

/// Load the main schematic from `schematic.active` by fetching its asset bytes
/// from the DB Asset Server over HTTP (RFD #724). `content_fallback` is the
/// legacy inline `schematic.content` mirror, used only if the fetch fails.
#[derive(Message, Clone, Debug)]
pub struct OpenDocumentFromActiveRequest {
    pub key: String,
    pub content_fallback: Option<String>,
    pub save_path: Option<PathBuf>,
}

/// Asset key under the DB Asset Server that holds the editor's active schematic.
/// Matches the key the Python SDK primes, so editor saves overwrite the same
/// active schematic.
pub const ACTIVE_SCHEMATIC_KEY: &str = "schematics/main.kdl";
