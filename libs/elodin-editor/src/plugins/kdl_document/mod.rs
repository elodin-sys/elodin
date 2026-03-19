mod loader;
mod operations;
mod systems;
mod types;

pub use operations::{apply_initial_kdl_path, sync_document_from_config};
pub use types::*;

use bevy::prelude::*;

pub(crate) fn plugin(app: &mut App) {
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
            systems::handle_open_document_requests,
            systems::handle_open_document_from_content_requests,
            systems::handle_save_current_document_requests,
        )
            .chain()
            .in_set(KdlDocumentSet::Commands),
    )
    .add_systems(
        PreUpdate,
        (
            systems::emit_document_reloads,
            systems::emit_document_load_failures,
        )
            .in_set(KdlDocumentSet::AssetEvents),
    );
}

#[cfg(test)]
mod tests {
    use super::{
        CurrentDocument, SchematicDocumentAsset, SecondarySchematicAsset,
        operations::open_document_from_content, plugin,
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
        super::super::kdl_asset_source::plugin(&mut app);
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
    #[ignore = "requires OS file watcher events; run with --ignored"]
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
    #[ignore = "requires OS file watcher events; run with --ignored"]
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
