use bevy::{
    asset::{AssetLoader, AssetPath, io::Reader},
    prelude::*,
    reflect::TypePath,
};
use impeller2_bevy::DbMessage;
use impeller2_kdl::KdlSchematicError;
use impeller2_kdl::{FromKdl, ToKdl};
use impeller2_wkt::Schematic;
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
        self.handle = Some(handle);
        self.asset_path = Some(asset_path);
        self.save_path = Some(save_path);
    }

    pub fn set_unsaved_content(&mut self, save_path: Option<PathBuf>) {
        self.handle = None;
        self.asset_path = None;
        self.save_path = save_path;
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
    app.init_resource::<InitialKdlPath>()
        .init_resource::<CurrentDocument>()
        .init_asset::<SchematicDocumentAsset>()
        .init_asset_loader::<SchematicDocumentLoader>();
}

/// Applies `InitialKdlPath` to `DbConfig` so that `sync_schematic` will load that file.
/// Runs before `sync_schematic`. Re-applies when the path is missing or different (e.g.
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

#[cfg(test)]
mod tests {
    use super::{
        CurrentDocument, SchematicDocumentAsset, SecondarySchematicAsset,
        plugin as kdl_document_plugin,
    };
    use bevy::{
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
        app.add_plugins(MinimalPlugins);
        kdl_document_plugin(&mut app);
        app.add_plugins(AssetPlugin {
            watch_for_changes_override: Some(true),
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..Default::default()
        });
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
