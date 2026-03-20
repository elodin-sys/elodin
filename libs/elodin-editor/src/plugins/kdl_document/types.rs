use bevy::{asset::AssetPath, prelude::*, reflect::TypePath};
use impeller2_kdl::KdlSchematicError;
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
    pub windows: Vec<SchematicWindow>,
}

#[derive(Debug, Clone)]
pub struct SchematicWindow {
    pub handle: Handle<SchematicDocumentAsset>,
    pub asset_path: AssetPath<'static>,
}

impl bevy::asset::VisitAssetDependencies for SchematicWindow {
    fn visit_dependencies(&self, visit: &mut impl FnMut(bevy::asset::UntypedAssetId)) {
        self.handle.visit_dependencies(visit);
    }
}

#[derive(Resource, Default, Debug, Clone)]
pub struct CurrentDocument {
    pub handle: Option<Handle<SchematicDocumentAsset>>,
    pub asset_path: Option<AssetPath<'static>>,
    pub save_path: Option<PathBuf>,
    pub(crate) suppress_next_reload: bool,
}

impl CurrentDocument {
    pub fn clear(&mut self) {
        self.handle = None;
        self.asset_path = None;
        self.save_path = None;
        self.suppress_next_reload = false;
    }

    pub fn set_file(
        &mut self,
        handle: Handle<SchematicDocumentAsset>,
        asset_path: AssetPath<'static>,
        save_path: PathBuf,
    ) {
        self.handle = Some(handle);
        self.asset_path = Some(asset_path);
        self.save_path = Some(save_path);
    }

    pub fn set_unsaved_content(&mut self, save_path: Option<PathBuf>) {
        self.handle = None;
        self.asset_path = None;
        self.save_path = save_path;
        self.suppress_next_reload = false;
    }

    pub(crate) fn matches(&self, id: AssetId<SchematicDocumentAsset>) -> bool {
        self.handle.as_ref().map(Handle::id) == Some(id)
    }

    pub(crate) fn window_handles(
        &self,
        assets: &Assets<SchematicDocumentAsset>,
    ) -> Vec<AssetId<SchematicDocumentAsset>> {
        self.handle
            .as_ref()
            .and_then(|h| assets.get(h))
            .map(|doc| doc.windows.iter().map(|w| w.handle.id()).collect())
            .unwrap_or_default()
    }
}

#[derive(Default)]
pub struct SchematicDocumentLoader;

#[derive(SystemSet, Debug, Clone, Hash, PartialEq, Eq)]
pub enum KdlDocumentSet {
    Commands,
    AssetEvents,
}

#[derive(Clone, Debug)]
pub struct WindowDocumentSave {
    pub window_id: u32,
    pub file_name: String,
    pub kdl: String,
}

#[derive(Debug, Error)]
pub enum SchematicDocumentLoaderError {
    #[error("Could not read schematic document: {0}")]
    Io(#[from] std::io::Error),
    #[error("Schematic document is not valid UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error(transparent)]
    RootKdl(#[from] KdlSchematicError),
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
    #[error("Could not create directory for window schematic {path}: {source}")]
    CreateWindowDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not save window schematic to {path}: {source}")]
    WriteWindow {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
