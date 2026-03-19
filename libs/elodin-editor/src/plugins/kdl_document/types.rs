use bevy::{asset::AssetPath, prelude::*, reflect::TypePath};
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::ToKdl;
use impeller2_wkt::Schematic;
use std::path::PathBuf;
use thiserror::Error;

/// When set (e.g. by CLI `--kdl`), the path is applied to `DbConfig` once so that the schematic
/// is loaded after connecting to the database.
#[derive(Resource, Default)]
pub struct InitialKdlPath(pub Option<PathBuf>);

#[derive(Asset, TypePath, Debug, Clone)]
pub struct SchematicDocumentAsset {
    pub root: Schematic,
    pub secondary: Vec<SecondarySchematicAsset>,
}

#[derive(Debug, Clone)]
pub struct SecondarySchematicAsset {
    pub asset_path: AssetPath<'static>,
    pub schematic: Schematic,
}

#[derive(Resource, Default, Debug, Clone)]
pub struct CurrentDocument {
    pub handle: Option<Handle<SchematicDocumentAsset>>,
    pub asset_path: Option<AssetPath<'static>>,
    pub save_path: Option<PathBuf>,
    applied_root_kdl: Option<String>,
    applied_secondary_kdls: Vec<String>,
}

impl CurrentDocument {
    pub fn clear(&mut self) {
        self.handle = None;
        self.asset_path = None;
        self.save_path = None;
        self.clear_applied();
    }

    pub fn set_file(
        &mut self,
        handle: Handle<SchematicDocumentAsset>,
        asset_path: AssetPath<'static>,
        save_path: PathBuf,
    ) {
        let changed = self.handle.as_ref().map(Handle::id) != Some(handle.id())
            || self.asset_path.as_ref() != Some(&asset_path)
            || self.save_path.as_ref() != Some(&save_path);
        self.handle = Some(handle);
        self.asset_path = Some(asset_path);
        self.save_path = Some(save_path);
        if changed {
            self.clear_applied();
        }
    }

    pub fn set_unsaved_content(&mut self, save_path: Option<PathBuf>) {
        self.handle = None;
        self.asset_path = None;
        self.save_path = save_path;
        self.clear_applied();
    }

    pub(crate) fn matches(&self, id: AssetId<SchematicDocumentAsset>) -> bool {
        self.handle.as_ref().map(Handle::id) == Some(id)
    }

    fn clear_applied(&mut self) {
        self.applied_root_kdl = None;
        self.applied_secondary_kdls.clear();
    }

    pub(crate) fn set_applied(&mut self, root: &Schematic, secondary_kdls: Vec<String>) {
        self.applied_root_kdl = Some(root.to_kdl());
        self.applied_secondary_kdls = secondary_kdls;
    }

    pub(crate) fn changed_secondary_indices(
        &self,
        document: &SchematicDocumentAsset,
    ) -> Option<Vec<usize>> {
        let root_kdl = self.applied_root_kdl.as_ref()?;
        if root_kdl != &document.root.to_kdl() {
            return None;
        }
        if self.applied_secondary_kdls.len() != document.secondary.len() {
            return None;
        }

        Some(
            document
                .secondary
                .iter()
                .enumerate()
                .filter_map(|(index, secondary)| {
                    (self.applied_secondary_kdls.get(index) != Some(&secondary.schematic.to_kdl()))
                        .then_some(index)
                })
                .collect(),
        )
    }

    pub(crate) fn matches_applied(&self, document: &SchematicDocumentAsset) -> bool {
        matches!(
            self.changed_secondary_indices(document),
            Some(changed_indices) if changed_indices.is_empty()
        )
    }
}

#[derive(Default)]
pub struct SchematicDocumentLoader;

#[derive(SystemSet, Debug, Clone, Hash, PartialEq, Eq)]
pub enum KdlDocumentSet {
    Commands,
    AssetEvents,
}

#[derive(Message, Clone, Debug)]
pub struct OpenDocumentRequest(pub PathBuf);

#[derive(Message, Clone, Debug)]
pub struct OpenDocumentFromContentRequest {
    pub content: String,
    pub save_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct SecondaryDocumentSave {
    pub file_name: String,
    pub kdl: String,
}

#[derive(Message, Clone, Debug)]
pub struct SaveCurrentDocumentRequest {
    pub path: Option<PathBuf>,
    pub root_kdl: String,
    pub secondary: Vec<SecondaryDocumentSave>,
}

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

#[derive(Message, Clone, Debug)]
pub struct DocumentSaved {
    pub save_path: PathBuf,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentReloaded {
    pub save_path: Option<PathBuf>,
    pub document: SchematicDocumentAsset,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentLoadFailed {
    pub path: String,
    pub message: String,
}

#[derive(Message, Clone, Debug)]
pub struct DocumentCleared;

#[derive(Debug, Error)]
pub enum SchematicDocumentLoaderError {
    #[error("Could not read schematic document: {0}")]
    Io(#[from] std::io::Error),
    #[error("Schematic document is not valid UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    RootKdl(#[from] KdlSchematicError),
    #[error("Could not read secondary schematic {path}: {reason}")]
    ReadSecondary {
        path: AssetPath<'static>,
        reason: String,
    },
    #[error("Could not parse secondary schematic {path}")]
    ParseSecondary {
        path: AssetPath<'static>,
        #[source]
        source: KdlSchematicError,
    },
}

#[derive(Debug, Error)]
pub enum SaveCurrentDocumentError {
    #[error("No save path is available for the current document")]
    MissingSavePath,
    #[error("Could not save schematic to {path}: {source}")]
    WriteRoot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not create directory for secondary schematic {path}: {source}")]
    CreateSecondaryDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not save secondary schematic to {path}: {source}")]
    WriteSecondary {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
