mod commands;
mod loader;
mod messages;
mod operations;
mod systems;
mod types;

pub use commands::*;
pub use messages::*;
pub use operations::{apply_initial_kdl_path, sync_document_from_config, sync_document_skybox};
pub use types::*;

use bevy::prelude::*;

pub(crate) fn plugin(app: &mut App) {
    app.configure_sets(
        PreUpdate,
        (KdlDocumentSet::Commands, KdlDocumentSet::AssetEvents).chain(),
    )
    .init_resource::<InitialKdlPath>()
    .init_resource::<LastSyncedSchematicContent>()
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
            systems::activate_document_skybox,
        )
            .chain()
            .in_set(KdlDocumentSet::AssetEvents),
    );
}

#[cfg(test)]
mod tests {
    use super::{
        CurrentDocument, DocumentCleared, DocumentCommandFailed, DocumentLoaded, DocumentReloaded,
        LastSyncedSchematicContent, OpenDocumentFromContentRequest, OpenDocumentRequest,
        SchematicDocumentAsset,
        operations::{open_document_from_content, sync_document_skybox},
        plugin,
    };
    use crate::ui::schematic::CurrentSchematic;
    use bevy::{
        app::TaskPoolPlugin,
        asset::{AssetPath, AssetPlugin, UnapprovedPathMode},
        core_pipeline::Skybox,
        image::Image,
        prelude::*,
        render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    };
    use bevy_ai_skybox::{
        ManifestEntry, SkyboxStyle,
        prelude::{PrimarySkybox, SetActiveSkybox, SkyboxAssetPlugin, SkyboxCache},
    };
    use impeller2_wkt::{DbConfig, Schematic, SchematicElem, SkyboxConfig};
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

    fn write_test_document(root: &Path, root_title: &str, window_name: &str) {
        fs::write(
            root.join("drone.kdl"),
            format!("window path=\"rate-control-panel.kdl\" title=\"{root_title}\"\n"),
        )
        .expect("write root kdl");
        fs::write(
            root.join("rate-control-panel.kdl"),
            format!("graph \"drone.gyro\" name=\"{window_name}\"\n"),
        )
        .expect("write window kdl");
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

    fn first_window_graph_name<'a>(
        doc: &SchematicDocumentAsset,
        assets: &'a Assets<SchematicDocumentAsset>,
    ) -> Option<&'a str> {
        let window = doc.windows.first()?;
        let window_doc = assets.get(&window.handle)?;
        window_doc.root.elems.iter().find_map(|elem| match elem {
            SchematicElem::Panel(impeller2_wkt::Panel::Graph(graph)) => graph.name.as_deref(),
            _ => None,
        })
    }

    #[derive(Resource, Default)]
    struct SeenSkyboxMessages(Vec<SetActiveSkybox>);

    fn collect_skybox_messages(
        mut reader: MessageReader<SetActiveSkybox>,
        mut seen: ResMut<SeenSkyboxMessages>,
    ) {
        seen.0.extend(reader.read().cloned());
    }

    #[derive(Resource, Default)]
    struct SeenOpenDocumentRequests(Vec<PathBuf>);

    fn collect_open_document_requests(
        mut reader: MessageReader<OpenDocumentRequest>,
        mut seen: ResMut<SeenOpenDocumentRequests>,
    ) {
        seen.0
            .extend(reader.read().map(|request| request.0.clone()));
    }

    fn skybox_message_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_message::<DocumentLoaded>()
            .add_message::<DocumentReloaded>()
            .add_message::<SetActiveSkybox>()
            .insert_resource(SkyboxCache::empty(PathBuf::from("manifest.ron")))
            .init_resource::<SeenSkyboxMessages>()
            .add_systems(
                Update,
                (
                    super::systems::activate_document_skybox,
                    collect_skybox_messages,
                )
                    .chain(),
            );
        app
    }

    fn config_sync_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedSchematicContent>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromContentRequest>()
            .add_message::<DocumentCleared>()
            .init_resource::<SeenOpenDocumentRequests>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_open_document_requests,
                )
                    .chain(),
            );
        app
    }

    fn test_manifest_entry(name: &str) -> ManifestEntry {
        ManifestEntry {
            name: name.to_string(),
            prompt: name.to_string(),
            style: SkyboxStyle::default(),
            blockade: None,
            equirect_file: None,
            cubemap_file: format!("{name}.cubemap.ktx2"),
            face_size: 1,
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn test_cubemap_image() -> Image {
        Image::new(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            vec![255; 4 * 6],
            TextureFormat::Rgba8UnormSrgb,
            default(),
        )
    }

    fn skybox_runtime_test_app(cache_dir: &Path) -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin {
                unapproved_path_mode: UnapprovedPathMode::Allow,
                ..Default::default()
            })
            .init_asset::<Image>()
            .add_plugins(SkyboxAssetPlugin {
                cache_dir: cache_dir.to_path_buf(),
                asset_dir: PathBuf::from("skyboxes"),
                manifest_file: PathBuf::from("manifest.ron"),
                default_skybox: None,
                apply_to_all_cameras: false,
                env_lighting: false,
                watch_manifest: false,
                manifest_poll_secs: 1.0,
            });
        plugin(&mut app);
        app.world_mut().spawn((Camera3d::default(), PrimarySkybox));
        app
    }

    fn register_runtime_test_skybox(app: &mut App, name: &str) -> Handle<Image> {
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(test_cubemap_image());
        let mut cache = app.world_mut().resource_mut::<SkyboxCache>();
        cache.manifest.upsert(test_manifest_entry(name));
        cache.handles.insert(name.to_string(), handle.clone());
        handle
    }

    fn camera_skybox_handle(app: &mut App) -> Option<Handle<Image>> {
        let world = app.world_mut();
        let mut query = world.query::<&Skybox>();
        query.iter(world).next().map(|skybox| skybox.image.clone())
    }

    fn load_test_document(app: &mut App, skybox: Option<&str>) {
        app.world_mut().write_message(DocumentLoaded {
            save_path: None,
            document: SchematicDocumentAsset {
                root: impeller2_wkt::Schematic {
                    skybox: skybox.map(|name| SkyboxConfig {
                        name: name.to_string(),
                    }),
                    ..default()
                },
                windows: Vec::new(),
            },
        });
        app.update();
    }

    #[test]
    fn loaded_document_with_skybox_requests_activation() {
        let mut app = skybox_message_test_app();
        app.world_mut().resource_mut::<SkyboxCache>().active = Some("alien_swamp".to_string());
        app.world_mut().write_message(DocumentLoaded {
            save_path: None,
            document: SchematicDocumentAsset {
                root: impeller2_wkt::Schematic {
                    skybox: Some(SkyboxConfig {
                        name: "grand_canyon".to_string(),
                    }),
                    ..default()
                },
                windows: Vec::new(),
            },
        });
        app.update();

        let messages = &app.world().resource::<SeenSkyboxMessages>().0;
        assert!(matches!(
            messages.as_slice(),
            [SetActiveSkybox::ByName(name)] if name == "grand_canyon"
        ));
        assert!(app.world().resource::<SkyboxCache>().active.is_none());
    }

    #[test]
    fn loaded_document_without_skybox_requests_clear() {
        let mut app = skybox_message_test_app();
        app.world_mut().resource_mut::<SkyboxCache>().active = Some("grand_canyon".to_string());
        app.world_mut().write_message(DocumentLoaded {
            save_path: None,
            document: SchematicDocumentAsset {
                root: impeller2_wkt::Schematic::default(),
                windows: Vec::new(),
            },
        });
        app.update();

        let messages = &app.world().resource::<SeenSkyboxMessages>().0;
        assert!(matches!(messages.as_slice(), [SetActiveSkybox::Clear]));
        assert!(app.world().resource::<SkyboxCache>().active.is_none());
    }

    #[test]
    fn loaded_document_skybox_applies_to_camera_without_current_skybox() {
        let temp = TempTestDir::new("skybox-runtime-empty");
        let mut app = skybox_runtime_test_app(temp.path());
        let expected = register_runtime_test_skybox(&mut app, "grand_canyon");

        load_test_document(&mut app, Some("grand_canyon"));

        assert_eq!(camera_skybox_handle(&mut app).as_ref(), Some(&expected));
        assert_eq!(
            app.world().resource::<SkyboxCache>().active.as_deref(),
            Some("grand_canyon")
        );
    }

    #[test]
    fn loaded_document_skybox_replaces_existing_camera_skybox() {
        let temp = TempTestDir::new("skybox-runtime-replace");
        let mut app = skybox_runtime_test_app(temp.path());
        let first = register_runtime_test_skybox(&mut app, "alien_swamp");
        let second = register_runtime_test_skybox(&mut app, "grand_canyon");

        load_test_document(&mut app, Some("alien_swamp"));
        assert_eq!(camera_skybox_handle(&mut app).as_ref(), Some(&first));

        load_test_document(&mut app, Some("grand_canyon"));

        assert_eq!(camera_skybox_handle(&mut app).as_ref(), Some(&second));
        assert_eq!(
            app.world().resource::<SkyboxCache>().active.as_deref(),
            Some("grand_canyon")
        );
    }

    #[test]
    fn loaded_document_without_skybox_clears_existing_camera_skybox() {
        let temp = TempTestDir::new("skybox-runtime-clear");
        let mut app = skybox_runtime_test_app(temp.path());
        let active = register_runtime_test_skybox(&mut app, "grand_canyon");

        load_test_document(&mut app, Some("grand_canyon"));
        assert_eq!(camera_skybox_handle(&mut app).as_ref(), Some(&active));

        load_test_document(&mut app, None);

        assert!(camera_skybox_handle(&mut app).is_none());
        assert!(app.world().resource::<SkyboxCache>().active.is_none());
    }

    #[test]
    fn config_sync_opens_configured_file_path_when_not_loaded() {
        let temp = TempTestDir::new("config-sync-open");
        let path = temp.path().join("drone.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");

        let mut app = config_sync_test_app();
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_path(path.to_string_lossy().to_string());
        app.update();

        assert_eq!(
            app.world().resource::<SeenOpenDocumentRequests>().0,
            vec![path]
        );
    }

    #[derive(Resource, Default)]
    struct SeenDocumentCommandFailures(usize);

    fn count_document_command_failures(
        mut reader: MessageReader<DocumentCommandFailed>,
        mut seen: ResMut<SeenDocumentCommandFailures>,
    ) {
        seen.0 += reader.read().count();
    }

    #[test]
    fn config_sync_does_not_mark_invalid_embedded_content_as_synced() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin {
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..Default::default()
        });
        super::super::kdl_asset_source::plugin(&mut app);
        app.add_message::<SetActiveSkybox>();
        app.insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedSchematicContent>()
            .init_resource::<SeenDocumentCommandFailures>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    count_document_command_failures,
                )
                    .chain(),
            );
        plugin(&mut app);

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_content("not valid kdl".to_string());
        app.update();
        app.update();

        assert!(
            app.world()
                .resource::<LastSyncedSchematicContent>()
                .0
                .is_none(),
            "invalid embedded content must not be recorded before a successful load"
        );
        assert_eq!(
            app.world().resource::<SeenDocumentCommandFailures>().0,
            1,
            "invalid embedded content should surface a document command failure"
        );

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_content("not valid kdl".to_string());
        app.update();
        app.update();

        assert_eq!(
            app.world().resource::<SeenDocumentCommandFailures>().0,
            2,
            "the same invalid embedded content should be retried after a failed load"
        );
    }

    #[test]
    fn schematic_content_equivalent_treats_reserialized_kdl_as_equal() {
        use impeller2_kdl::{FromKdl, ToKdl};
        use impeller2_wkt::Schematic;

        let raw = "skybox name=\"seaport\"\n";
        let reserialized = Schematic::from_kdl(raw).unwrap().to_kdl();
        assert!(super::operations::schematic_content_equivalent(
            &reserialized,
            raw,
        ));
    }

    #[test]
    fn config_sync_skips_configured_file_path_when_already_loaded() {
        let temp = TempTestDir::new("config-sync-skip-current");
        let path = temp.path().join("drone.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");
        let resolved_path = impeller2_kdl::env::schematic_file(&path);

        let mut app = config_sync_test_app();
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_path(path.to_string_lossy().to_string());
        {
            let mut current_document = app.world_mut().resource_mut::<CurrentDocument>();
            current_document.handle = Some(Handle::<SchematicDocumentAsset>::default());
            current_document.save_path = Some(resolved_path);
        }
        app.update();

        assert!(
            app.world()
                .resource::<SeenOpenDocumentRequests>()
                .0
                .is_empty()
        );
    }

    #[test]
    fn suppress_ids_cleared_on_clear() {
        let mut current_document = CurrentDocument::default();
        assert!(current_document.suppress_ids.is_empty());

        let fake_handle = Handle::<SchematicDocumentAsset>::default();
        current_document.suppress_ids.insert(fake_handle.id());
        assert!(!current_document.suppress_ids.is_empty());

        current_document.clear();
        assert!(current_document.suppress_ids.is_empty());
    }

    #[test]
    fn opening_unsaved_content_clears_suppress_ids() {
        let mut current_document = CurrentDocument::default();
        let fake_handle = Handle::<SchematicDocumentAsset>::default();
        current_document.suppress_ids.insert(fake_handle.id());

        let _document = open_document_from_content(
            "window path=\"rate-control-panel.kdl\" title=\"Panel A\"\n",
            Some(PathBuf::from("drone.kdl")),
            &mut current_document,
        )
        .expect("parse kdl");

        assert!(current_document.suppress_ids.is_empty());
        assert!(current_document.handle.is_none());
    }

    #[derive(Resource, Default)]
    struct SeenDocumentReloads(usize);

    fn count_document_reloads(
        mut reader: MessageReader<DocumentReloaded>,
        mut seen: ResMut<SeenDocumentReloads>,
    ) {
        seen.0 += reader.read().count();
    }

    #[test]
    fn sync_document_skybox_suppresses_document_reload() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.add_plugins(AssetPlugin {
            unapproved_path_mode: UnapprovedPathMode::Allow,
            ..Default::default()
        });
        app.add_message::<SetActiveSkybox>();
        plugin(&mut app);
        app.init_resource::<SeenDocumentReloads>();
        app.add_systems(Update, count_document_reloads);

        let handle = app
            .world_mut()
            .resource_mut::<Assets<SchematicDocumentAsset>>()
            .add(SchematicDocumentAsset {
                root: Schematic::default(),
                windows: Vec::new(),
            });
        app.world_mut().resource_mut::<CurrentDocument>().set_file(
            handle.clone(),
            AssetPath::from_path_buf(PathBuf::from("drone.kdl")),
            PathBuf::from("drone.kdl"),
        );

        let mut schematic = CurrentSchematic(Schematic::default());
        app.world_mut()
            .resource_scope(|world, mut current_document: Mut<CurrentDocument>| {
                let mut document_assets = world.resource_mut::<Assets<SchematicDocumentAsset>>();
                sync_document_skybox(
                    Some(SkyboxConfig {
                        name: "seaport".to_string(),
                    }),
                    &mut current_document,
                    &mut document_assets,
                    &mut schematic,
                );
            });

        app.update();

        assert_eq!(app.world().resource::<SeenDocumentReloads>().0, 0);
        assert_eq!(
            app.world()
                .resource::<Assets<SchematicDocumentAsset>>()
                .get(&handle)
                .and_then(|doc| doc.root.skybox.as_ref())
                .map(|skybox| skybox.name.as_str()),
            Some("seaport")
        );
        assert_eq!(
            schematic.skybox.as_ref().map(|skybox| skybox.name.as_str()),
            Some("seaport")
        );
    }

    #[cfg(unix)]
    fn load_symlinked_document(
        name: &str,
        root_title: &str,
        window_name: &str,
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

        write_test_document(&real_root, root_title, window_name);

        let var_guard = set_kdl_dir(&linked_root);
        let mut app = test_app();
        let handle = load_document(&mut app);
        (temp, var_guard, app, handle)
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
    fn root_document_reloads_when_window_kdl_changes_under_symlinked_dir() {
        let _env_guard = env_lock().lock().expect("env lock");
        let (temp, _var_guard, mut app, handle) =
            load_symlinked_document("window-reload", "Panel A", "Panel A");

        wait_for(&mut app, Duration::from_secs(10), |app| {
            let assets = app.world().resource::<Assets<SchematicDocumentAsset>>();
            let doc = assets.get(&handle)?;
            first_window_graph_name(doc, assets)
                .filter(|name| *name == "Panel A")
                .map(|_| doc.clone())
        });

        fs::write(
            temp.path().join("real").join("rate-control-panel.kdl"),
            "graph \"drone.gyro\" name=\"Panel B\"\n",
        )
        .expect("update window kdl");

        let reloaded = wait_for(&mut app, Duration::from_secs(10), |app| {
            let assets = app.world().resource::<Assets<SchematicDocumentAsset>>();
            let doc = assets.get(&handle)?;
            first_window_graph_name(doc, assets)
                .filter(|name| *name == "Panel B")
                .map(|_| doc.clone())
        });

        assert!(reloaded.windows.len() == 1);
        drop(app);
    }
}
