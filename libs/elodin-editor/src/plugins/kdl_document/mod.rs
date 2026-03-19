use bevy::{
    asset::{AssetEvent, AssetLoadFailedEvent, AssetLoader, AssetPath, io::Reader},
    prelude::*,
    reflect::TypePath,
};
use impeller2_bevy::DbMessage;
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::env::schematic_file;
use impeller2_kdl::{FromKdl, ToKdl};
use impeller2_wkt::{DbConfig, Schematic};
use std::path::{Path, PathBuf};
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

impl AssetLoader for SchematicDocumentLoader {
    type Asset = SchematicDocumentAsset;
    type Settings = ();
    type Error = SchematicDocumentLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        load_context: &mut bevy::asset::LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let root = Schematic::from_kdl(&String::from_utf8(bytes)?)?;
        let mut secondary = Vec::new();
        let base_dir = load_context
            .asset_path()
            .path()
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let source = load_context.asset_path().source().clone_owned();

        for window in root.elems.iter().filter_map(|elem| match elem {
            impeller2_wkt::SchematicElem::Window(window) => Some(window),
            _ => None,
        }) {
            let Some(path) = window.path.as_deref() else {
                continue;
            };
            let asset_path =
                AssetPath::from_path_buf(base_dir.join(path)).with_source(source.clone());
            let bytes = load_context
                .read_asset_bytes(asset_path.clone())
                .await
                .map_err(|err| SchematicDocumentLoaderError::ReadSecondary {
                    path: asset_path.clone_owned(),
                    reason: err.to_string(),
                })?;
            let text = String::from_utf8(bytes)?;
            let schematic = Schematic::from_kdl(&text).map_err(|source| {
                SchematicDocumentLoaderError::ParseSecondary {
                    path: asset_path.clone_owned(),
                    source,
                }
            })?;
            secondary.push(SecondarySchematicAsset {
                asset_path: asset_path.clone_owned(),
                schematic,
            });
        }

        Ok(SchematicDocumentAsset { root, secondary })
    }

    fn extensions(&self) -> &[&str] {
        &["kdl"]
    }
}

pub(crate) fn plugin(app: &mut App) {
    super::kdl_asset_source::plugin(app);
    app.configure_sets(
        PreUpdate,
        (KdlDocumentSet::Commands, KdlDocumentSet::AssetEvents).chain(),
    )
    .init_resource::<InitialKdlPath>()
    .init_resource::<CurrentDocument>()
    .init_asset::<SchematicDocumentAsset>()
    .init_asset_loader::<SchematicDocumentLoader>()
    .add_message::<OpenDocumentRequest>()
    .add_message::<OpenDocumentFromContentRequest>()
    .add_message::<SaveCurrentDocumentRequest>()
    .add_message::<DocumentLoaded>()
    .add_message::<DocumentCommandFailed>()
    .add_message::<DocumentSaved>()
    .add_message::<DocumentReloaded>()
    .add_message::<DocumentLoadFailed>()
    .add_message::<DocumentCleared>()
    .add_systems(
        PreUpdate,
        (
            handle_open_document_requests,
            handle_open_document_from_content_requests,
            handle_save_current_document_requests,
        )
            .chain()
            .in_set(KdlDocumentSet::Commands),
    )
    .add_systems(
        PreUpdate,
        (emit_document_reloads, emit_document_load_failures).in_set(KdlDocumentSet::AssetEvents),
    );
}

/// Applies `InitialKdlPath` to `DbConfig` so that document sync can load that file.
/// Runs before `sync_document_from_config`. Re-applies when the path is missing or different (e.g.
/// after the connection overwrote DbConfig with metadata) so the schematic loads.
pub fn apply_initial_kdl_path(
    mut reader: MessageReader<DbMessage>,
    initial: Res<InitialKdlPath>,
) -> Option<PathBuf> {
    if !reader.read().any(|m| matches!(m, DbMessage::UpdateConfig)) {
        None
    } else {
        initial.0.clone()
    }
}

pub fn sync_document_from_config(
    In(given_path): In<Option<PathBuf>>,
    config: Res<DbConfig>,
    mut current_document: ResMut<CurrentDocument>,
    mut open_document: MessageWriter<OpenDocumentRequest>,
    mut open_document_from_content: MessageWriter<OpenDocumentFromContentRequest>,
    mut cleared: MessageWriter<DocumentCleared>,
) {
    if given_path.is_none() && !config.is_changed() {
        return;
    }

    let has_content_fallback = config.schematic_content().is_some();
    let path_was_overridden = given_path.is_some();

    if let Some(path) = given_path.or(config.schematic_path().map(PathBuf::from)) {
        let resolved_path = schematic_file(&path);
        if resolved_path.try_exists().unwrap_or(false) {
            open_document.write(OpenDocumentRequest(path));
            return;
        }

        if has_content_fallback && !path_was_overridden {
            bevy::log::info!(
                "Schematic file {:?} not found; using embedded schematic content fallback",
                resolved_path.display()
            );
        }
    }

    if let Some(content) = config.schematic_content() {
        open_document_from_content.write(OpenDocumentFromContentRequest {
            content: content.to_string(),
            save_path: config.schematic_path().map(Path::new).map(schematic_file),
        });
        return;
    }

    current_document.clear();
    cleared.write(DocumentCleared);
}

fn canonicalize_or_original(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub(crate) fn filesystem_to_asset_path(path: &Path) -> AssetPath<'static> {
    let resolved = canonicalize_or_original(path);
    let source = super::kdl_asset_source::KDL_ASSET_SOURCE;
    if let Ok(root) = impeller2_kdl::env::schematic_dir_or_cwd() {
        let canonical_root = canonicalize_or_original(&root);
        if let Ok(relative) = resolved.strip_prefix(&canonical_root) {
            return AssetPath::from_path_buf(relative.to_path_buf()).with_source(source);
        }
    }
    AssetPath::from_path_buf(resolved).with_source(source)
}

pub fn open_document_path(
    path: &Path,
    asset_server: &AssetServer,
    current_document: &mut CurrentDocument,
    document_assets: &Assets<SchematicDocumentAsset>,
) -> Result<Option<SchematicDocumentAsset>, KdlSchematicError> {
    let resolved_path = schematic_file(path);
    if !resolved_path.try_exists().unwrap_or(false) {
        return Err(KdlSchematicError::NoSuchFile {
            path: resolved_path,
        });
    }

    let asset_path = filesystem_to_asset_path(&resolved_path);
    let handle: Handle<SchematicDocumentAsset> = asset_server.load(asset_path.clone());
    current_document.set_file(handle.clone(), asset_path, resolved_path);
    Ok(document_assets.get(&handle).cloned())
}

pub fn open_document_from_content(
    content: &str,
    save_path: Option<PathBuf>,
    current_document: &mut CurrentDocument,
) -> Result<SchematicDocumentAsset, KdlSchematicError> {
    let root = Schematic::from_kdl(content)?;
    current_document.set_unsaved_content(save_path);
    Ok(SchematicDocumentAsset {
        root,
        secondary: Vec::new(),
    })
}

pub fn save_current_document(
    path: Option<PathBuf>,
    root_kdl: &str,
    secondary: &[SecondaryDocumentSave],
    asset_server: &AssetServer,
    current_document: &mut CurrentDocument,
) -> Result<PathBuf, SaveCurrentDocumentError> {
    let path = path
        .or_else(|| current_document.save_path.clone())
        .ok_or(SaveCurrentDocumentError::MissingSavePath)?;
    let path = Path::new(&path).with_extension("kdl");
    let dest = schematic_file(&path);

    std::fs::write(&dest, root_kdl).map_err(|source| SaveCurrentDocumentError::WriteRoot {
        path: dest.clone(),
        source,
    })?;
    write_secondary_schematics(dest.parent().unwrap_or_else(|| Path::new(".")), secondary)?;

    let asset_path = filesystem_to_asset_path(&dest);
    let handle: Handle<SchematicDocumentAsset> = asset_server.load(asset_path.clone());
    current_document.set_file(handle, asset_path, dest.clone());
    Ok(dest)
}

fn write_secondary_schematics(
    base_dir: &Path,
    secondary: &[SecondaryDocumentSave],
) -> Result<(), SaveCurrentDocumentError> {
    for entry in secondary {
        let dest = base_dir.join(&entry.file_name);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                SaveCurrentDocumentError::CreateSecondaryDir {
                    path: dest.clone(),
                    source,
                }
            })?;
        }

        std::fs::write(&dest, &entry.kdl)
            .map_err(|source| SaveCurrentDocumentError::WriteSecondary { path: dest, source })?;
    }

    Ok(())
}

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

fn handle_open_document_requests(
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

fn handle_open_document_from_content_requests(
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

fn handle_save_current_document_requests(
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

fn emit_document_reloads(
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

fn emit_document_load_failures(
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

#[cfg(test)]
mod tests {
    use super::{
        CurrentDocument, SchematicDocumentAsset, SecondarySchematicAsset,
        open_document_from_content, plugin,
    };
    use bevy::{
        app::TaskPoolPlugin,
        asset::{AssetPath, AssetPlugin, UnapprovedPathMode},
        prelude::*,
    };
    use impeller2_kdl::{FromKdl, ToKdl};
    use impeller2_wkt::{Panel, Schematic, SchematicElem};
    use std::{
        ffi::OsString,
        fs,
        path::{Path, PathBuf},
        sync::{Mutex, OnceLock},
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    struct TempTestDir(PathBuf);

    impl TempTestDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time went backwards")
                .as_nanos();
            let path = std::env::temp_dir().join(format!("elodin-schematic-load-{name}-{unique}"));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct EnvVarGuard {
        previous: Option<OsString>,
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            // SAFETY: These tests serialize all `ELODIN_KDL_DIR` mutations behind `ENV_LOCK`,
            // so restoring the previous value here cannot race with another test in this module.
            unsafe {
                if let Some(value) = self.previous.take() {
                    std::env::set_var("ELODIN_KDL_DIR", value);
                } else {
                    std::env::remove_var("ELODIN_KDL_DIR");
                }
            }
        }
    }

    fn set_kdl_dir(path: &Path) -> EnvVarGuard {
        let previous = std::env::var_os("ELODIN_KDL_DIR");
        // SAFETY: This helper is only used in tests that hold `ENV_LOCK`, so mutating the
        // process environment here is serialized and scoped to the test's lifetime.
        unsafe {
            std::env::set_var("ELODIN_KDL_DIR", path);
        }
        EnvVarGuard { previous }
    }

    fn root_schematic(title: &str) -> Schematic {
        Schematic::from_kdl(&format!(
            "window path=\"rate-control-panel.kdl\" title=\"{title}\"\n"
        ))
        .expect("parse root kdl")
    }

    fn secondary_schematic(name: &str) -> Schematic {
        Schematic::from_kdl(&format!("graph \"drone.gyro\" name=\"{name}\"\n")).expect("parse kdl")
    }

    fn single_secondary_document(root_title: &str, secondary_name: &str) -> SchematicDocumentAsset {
        SchematicDocumentAsset {
            root: root_schematic(root_title),
            secondary: vec![SecondarySchematicAsset {
                asset_path: AssetPath::from_path_buf(PathBuf::from("rate-control-panel.kdl")),
                schematic: secondary_schematic(secondary_name),
            }],
        }
    }

    fn write_test_document(root: &Path, root_title: &str, secondary_name: &str) {
        fs::write(
            root.join("drone.kdl"),
            format!("window path=\"rate-control-panel.kdl\" title=\"{root_title}\"\n"),
        )
        .expect("write root kdl");
        fs::write(
            root.join("rate-control-panel.kdl"),
            format!("graph \"drone.gyro\" name=\"{secondary_name}\"\n"),
        )
        .expect("write secondary kdl");
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default());
        app.add_plugins(AssetPlugin {
            watch_for_changes_override: Some(true),
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..Default::default()
        });
        plugin(&mut app);
        app
    }

    fn load_document(app: &mut App) -> Handle<SchematicDocumentAsset> {
        app.world().resource::<AssetServer>().load(
            AssetPath::from_path_buf(PathBuf::from("drone.kdl"))
                .with_source(super::super::kdl_asset_source::KDL_ASSET_SOURCE),
        )
    }

    fn wait_for<T>(
        app: &mut App,
        timeout: Duration,
        mut predicate: impl FnMut(&mut App) -> Option<T>,
    ) -> T {
        let start = std::time::Instant::now();
        loop {
            app.update();
            if let Some(value) = predicate(app) {
                return value;
            }
            assert!(start.elapsed() < timeout, "timed out waiting for condition");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn window_title(doc: &SchematicDocumentAsset) -> Option<&str> {
        doc.root.elems.iter().find_map(|elem| match elem {
            SchematicElem::Window(window) => window.title.as_deref(),
            _ => None,
        })
    }

    fn first_secondary_graph_name(doc: &SchematicDocumentAsset) -> Option<&str> {
        doc.secondary.first().and_then(|secondary| {
            secondary
                .schematic
                .elems
                .iter()
                .find_map(|elem| match elem {
                    SchematicElem::Panel(Panel::Graph(graph)) => graph.name.as_deref(),
                    _ => None,
                })
        })
    }

    #[cfg(unix)]
    fn load_symlinked_document(
        name: &str,
        root_title: &str,
        secondary_name: &str,
    ) -> (
        TempTestDir,
        EnvVarGuard,
        App,
        Handle<SchematicDocumentAsset>,
    ) {
        use std::os::unix::fs::symlink;

        let temp = TempTestDir::new(name);
        let real_root = temp.path().join("real");
        fs::create_dir_all(&real_root).expect("create real root");
        let linked_root = temp.path().join("linked");
        symlink(&real_root, &linked_root).expect("create symlink root");

        write_test_document(&real_root, root_title, secondary_name);

        let var_guard = set_kdl_dir(&linked_root);
        let mut app = test_app();
        let handle = load_document(&mut app);
        (temp, var_guard, app, handle)
    }

    #[test]
    fn current_document_matches_equivalent_applied_document() {
        let root = root_schematic("Panel A");
        let secondary = secondary_schematic("Panel A");
        let document = single_secondary_document("Panel A", "Panel A");

        let mut current_document = CurrentDocument::default();
        current_document.set_applied(&root, vec![secondary.to_kdl()]);

        assert!(current_document.matches_applied(&document));
    }

    #[test]
    fn current_document_detects_secondary_changes() {
        let root = root_schematic("Panel A");
        let applied_secondary = secondary_schematic("Panel A");
        let document = single_secondary_document("Panel A", "Panel B");

        let mut current_document = CurrentDocument::default();
        current_document.set_applied(&root, vec![applied_secondary.to_kdl()]);

        assert!(!current_document.matches_applied(&document));
        assert_eq!(
            current_document.changed_secondary_indices(&document),
            Some(vec![0])
        );
    }

    #[test]
    fn current_document_uses_full_reload_when_root_changes() {
        let applied_root = root_schematic("Panel A");
        let secondary = secondary_schematic("Panel A");
        let document = single_secondary_document("Panel B", "Panel A");

        let mut current_document = CurrentDocument::default();
        current_document.set_applied(&applied_root, vec![secondary.to_kdl()]);

        assert_eq!(current_document.changed_secondary_indices(&document), None);
    }

    #[test]
    fn opening_unsaved_content_clears_applied_snapshot() {
        let root = root_schematic("Panel A");
        let secondary = secondary_schematic("Panel A");
        let mut current_document = CurrentDocument::default();
        current_document.set_applied(&root, vec![secondary.to_kdl()]);

        let document = open_document_from_content(
            "window path=\"rate-control-panel.kdl\" title=\"Panel A\"\n",
            Some(PathBuf::from("drone.kdl")),
            &mut current_document,
        )
        .expect("parse kdl");

        assert!(!current_document.matches_applied(&document));
    }

    #[cfg(unix)]
    #[test]
    fn root_document_reloads_when_root_kdl_changes_under_symlinked_dir() {
        let _env_guard = env_lock().lock().expect("env lock");
        let (temp, _var_guard, mut app, handle) =
            load_symlinked_document("root-reload", "Panel A", "Panel A");

        wait_for(&mut app, Duration::from_secs(10), |app| {
            app.world()
                .resource::<Assets<SchematicDocumentAsset>>()
                .get(&handle)
                .filter(|doc| window_title(doc) == Some("Panel A"))
                .cloned()
        });

        fs::write(
            temp.path().join("real").join("drone.kdl"),
            "window path=\"rate-control-panel.kdl\" title=\"Panel B\"\n",
        )
        .expect("update root kdl");

        let reloaded = wait_for(&mut app, Duration::from_secs(10), |app| {
            app.world()
                .resource::<Assets<SchematicDocumentAsset>>()
                .get(&handle)
                .filter(|doc| window_title(doc) == Some("Panel B"))
                .cloned()
        });

        assert_eq!(window_title(&reloaded), Some("Panel B"));
        drop(app);
    }

    #[cfg(unix)]
    #[test]
    fn root_document_reloads_when_secondary_kdl_changes_under_symlinked_dir() {
        let _env_guard = env_lock().lock().expect("env lock");
        let (temp, _var_guard, mut app, handle) =
            load_symlinked_document("secondary-reload", "Panel A", "Panel A");

        wait_for(&mut app, Duration::from_secs(10), |app| {
            app.world()
                .resource::<Assets<SchematicDocumentAsset>>()
                .get(&handle)
                .filter(|doc| first_secondary_graph_name(doc) == Some("Panel A"))
                .cloned()
        });

        fs::write(
            temp.path().join("real").join("rate-control-panel.kdl"),
            "graph \"drone.gyro\" name=\"Panel B\"\n",
        )
        .expect("update secondary kdl");

        let reloaded = wait_for(&mut app, Duration::from_secs(10), |app| {
            app.world()
                .resource::<Assets<SchematicDocumentAsset>>()
                .get(&handle)
                .filter(|doc| first_secondary_graph_name(doc) == Some("Panel B"))
                .cloned()
        });

        assert_eq!(first_secondary_graph_name(&reloaded), Some("Panel B"));
        drop(app);
    }
}
