use bevy::{asset::AssetPath, prelude::*, reflect::TypePath};
use impeller2_kdl::KdlSchematicError;
use impeller2_wkt::Schematic;
use std::collections::HashSet;
use std::path::PathBuf;
use thiserror::Error;

/// When set (e.g. by CLI `--kdl`), the path is applied to `DbConfig` once so that the schematic
/// is loaded after connecting to the database.
#[derive(Resource, Default)]
pub struct InitialKdlPath(pub Option<PathBuf>);

/// Last `schematic.active` key synced from `DbConfig`, to skip redundant full
/// reloads when only other metadata (e.g. `skybox.active`) changes.
#[derive(Resource, Default)]
pub struct LastSyncedActiveKey(pub Option<String>);

/// Asset revision (`assets.revision`) observed at the last active-schematic
/// (re)load. Lets config sync reload when the bytes at an *unchanged*
/// `schematic.active` key were replaced by another client (RFD #724, Bug 1).
#[derive(Resource, Default)]
pub struct LastSyncedAssetsRevision {
    /// Revision current when the active schematic was last loaded.
    pub revision: Option<u64>,
    /// Adopt the next revision change as a new baseline without reloading —
    /// set after a *local* save so the editor doesn't reload bytes it just
    /// wrote (its own write bumps the revision too).
    pub suppress_next: bool,
}

/// Normalized KDL the editor last knows to be stored at the active schematic
/// key — recorded on every DB-active load and on every local write-back
/// (skybox edit, save). Lets a revision-triggered refetch tell "the schematic
/// bytes actually changed" apart from "an unrelated asset (e.g. a skybox
/// cubemap) bumped `assets.revision`", so the latter doesn't tear down and
/// respawn the whole document (Bug 1/2).
#[derive(Resource, Default)]
pub struct LastActiveSchematicContent {
    key: Option<String>,
    normalized: Option<String>,
}

/// Formatting-insensitive form of a schematic KDL document (hand-written
/// ingested files vs. `to_kdl` output differ only in layout). `None` when the
/// content does not parse.
pub(crate) fn normalized_schematic_kdl(content: &str) -> Option<String> {
    use impeller2_kdl::{FromKdl, ToKdl};
    Schematic::from_kdl(content).ok().map(|s| s.to_kdl())
}

impl LastActiveSchematicContent {
    pub fn record(&mut self, key: &str, content: &str) {
        self.key = Some(key.to_string());
        self.normalized = normalized_schematic_kdl(content);
    }

    /// Whether `content` matches the last known stored bytes for `key`.
    /// Unparseable or unknown content never matches, so the caller falls back
    /// to a full reload.
    pub fn matches(&self, key: &str, content: &str) -> bool {
        self.key.as_deref() == Some(key)
            && self.normalized.is_some()
            && self.normalized == normalized_schematic_kdl(content)
    }
}

/// Active-schematic key the editor has optimistically switched to (via "Save
/// As…" or "Open Schematic…") but the DB has not yet echoed. While set and not
/// yet matched by `DbConfig.schematic_active`, config sync ignores the stale
/// pointer so it can't briefly reload the schematic being replaced; the pin
/// clears as soon as the DB confirms the requested key.
#[derive(Resource, Default)]
pub struct PendingActiveSchematic {
    /// Key the editor optimistically switched to, awaiting the DB echo.
    pub target: Option<String>,
    /// `schematic.active` value in effect when the pin was set — the stale
    /// pointer we expect to keep seeing until the repoint echoes. Lets config
    /// sync tell "still waiting for our echo" apart from "the pointer moved
    /// elsewhere" (an external repoint or a failed local one), so a pin can
    /// never strand sync on a key the DB will never confirm.
    pub superseding: Option<String>,
}

impl PendingActiveSchematic {
    /// Pin `target`, recording the `superseded` active key it replaces.
    pub fn pin(&mut self, target: String, superseded: Option<String>) {
        self.target = Some(target);
        self.superseding = superseded;
    }

    pub fn clear(&mut self) {
        self.target = None;
        self.superseding = None;
    }

    pub fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }
}

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
    /// Asset IDs whose next reload event should be suppressed (e.g. after a save).
    /// Expanded lazily to include window sub-asset IDs in `emit_document_reloads`.
    pub(crate) suppress_ids: HashSet<AssetId<SchematicDocumentAsset>>,
}

impl CurrentDocument {
    pub fn clear(&mut self) {
        self.handle = None;
        self.asset_path = None;
        self.save_path = None;
        self.suppress_ids.clear();
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
        self.suppress_ids.clear();
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

#[derive(Default, TypePath)]
pub struct SchematicDocumentLoader;

#[derive(SystemSet, Debug, Clone, Hash, PartialEq, Eq)]
pub enum KdlDocumentSet {
    Commands,
    AssetEvents,
}

#[derive(Clone, Debug)]
pub struct WindowDocumentSave {
    pub window_id: crate::ui::tiles::WindowId,
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
