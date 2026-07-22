mod commands;
mod loader;
mod messages;
mod operations;
mod systems;
mod types;

pub use commands::*;
pub use messages::*;
pub use operations::{
    apply_initial_kdl_path, reload_sticky_kdl_when_eql_ready, sync_document_from_config,
    sync_document_skybox,
};
#[cfg(all(not(target_family = "wasm"), target_family = "unix"))]
pub use operations::fetch_active_schematic_kdl;
pub(crate) use operations::{
    fetch_schematic_index, plan_db_save, schematic_name_from_key, schematic_save_key_from_name,
    upload_db_save_plan,
};
pub use types::*;

use bevy::prelude::*;

pub(crate) fn plugin(app: &mut App) {
    app.configure_sets(
        PreUpdate,
        (KdlDocumentSet::Commands, KdlDocumentSet::AssetEvents).chain(),
    )
    .init_resource::<InitialKdlPath>()
    .init_resource::<LastSyncedActiveKey>()
    .init_resource::<LastSyncedAssetsRevision>()
    .init_resource::<LastActiveSchematicContent>()
    .init_resource::<PendingActiveSchematic>()
    .init_resource::<systems::ActiveSchematicFetch>()
    .init_resource::<CurrentDocument>()
    .init_asset::<SchematicDocumentAsset>()
    .init_asset_loader::<SchematicDocumentLoader>()
    .add_message::<OpenDocumentRequest>()
    .add_message::<OpenDocumentFromActiveRequest>()
    .add_message::<DocumentLoaded>()
    .add_message::<DocumentCommandFailed>()
    .add_message::<DocumentReloaded>()
    .add_message::<DocumentLoadFailed>()
    .add_message::<DocumentCleared>()
    .add_systems(
        PreUpdate,
        (
            systems::handle_open_document_requests,
            systems::handle_open_document_from_active_requests,
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
        CurrentDocument, DocumentCleared, DocumentLoaded, DocumentReloaded,
        LastActiveSchematicContent, LastSyncedActiveKey, LastSyncedAssetsRevision,
        OpenDocumentFromActiveRequest, OpenDocumentRequest, PendingActiveSchematic,
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

    struct ChdirGuard {
        previous: PathBuf,
    }

    impl Drop for ChdirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.previous);
        }
    }

    fn chdir_to(path: &Path) -> ChdirGuard {
        let previous = std::env::current_dir().expect("current dir");
        std::env::set_current_dir(path).expect("chdir");
        ChdirGuard { previous }
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
        query
            .iter(world)
            .next()
            .and_then(|skybox| skybox.image.clone())
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
            explicit: false,
        });
        app.update();
    }

    #[test]
    fn loaded_document_with_skybox_requests_activation() {
        let mut app = skybox_message_test_app();
        app.world_mut().resource_mut::<SkyboxCache>().active = Some("desert_night".to_string());
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
            explicit: false,
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
            explicit: false,
        });
        app.update();

        let messages = &app.world().resource::<SeenSkyboxMessages>().0;
        assert!(matches!(messages.as_slice(), [SetActiveSkybox::Clear]));
        assert!(app.world().resource::<SkyboxCache>().active.is_none());
    }

    #[test]
    fn sticky_clear_filters_loaded_skybox_unless_open_is_explicit() {
        // The DB carries an explicit clear (`skybox.active=""`).
        let mut sticky_clear = DbConfig::default();
        sticky_clear
            .metadata
            .insert("skybox.active".to_string(), String::new());

        let document = SchematicDocumentAsset {
            root: impeller2_wkt::Schematic {
                skybox: Some(SkyboxConfig {
                    name: "grand_canyon".to_string(),
                }),
                ..default()
            },
            windows: Vec::new(),
        };

        // A background (re)load must not resurrect the cleared skybox...
        let mut app = skybox_message_test_app();
        app.insert_resource(sticky_clear.clone());
        app.world_mut().write_message(DocumentLoaded {
            save_path: None,
            document: document.clone(),
            explicit: false,
        });
        app.update();
        assert!(matches!(
            app.world().resource::<SeenSkyboxMessages>().0.as_slice(),
            [SetActiveSkybox::Clear]
        ));

        // ...but a user-initiated open re-applies the document's skybox.
        let mut app = skybox_message_test_app();
        app.insert_resource(sticky_clear);
        app.world_mut().write_message(DocumentLoaded {
            save_path: None,
            document,
            explicit: true,
        });
        app.update();
        assert!(matches!(
            app.world().resource::<SeenSkyboxMessages>().0.as_slice(),
            [SetActiveSkybox::ByName(name)] if name == "grand_canyon"
        ));
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
        let first = register_runtime_test_skybox(&mut app, "desert_night");
        let second = register_runtime_test_skybox(&mut app, "grand_canyon");

        load_test_document(&mut app, Some("desert_night"));
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

    #[derive(Resource, Clone)]
    struct SyncPath(Option<PathBuf>);

    fn sync_path(path: Res<SyncPath>) -> Option<PathBuf> {
        path.0.clone()
    }

    #[test]
    fn config_sync_opens_given_path_when_not_loaded() {
        let temp = TempTestDir::new("config-sync-open");
        let path = temp.path().join("drone.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");

        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .insert_resource(SyncPath(Some(path.clone())))
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenOpenDocumentRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    sync_path.pipe(super::operations::sync_document_from_config),
                    collect_open_document_requests,
                )
                    .chain(),
            );
        app.update();

        assert_eq!(
            app.world().resource::<SeenOpenDocumentRequests>().0,
            vec![path]
        );
    }

    #[derive(Resource, Default)]
    struct SeenActiveRequests(Vec<String>);

    fn collect_active_requests(
        mut reader: MessageReader<OpenDocumentFromActiveRequest>,
        mut seen: ResMut<SeenActiveRequests>,
    ) {
        seen.0
            .extend(reader.read().map(|request| request.key.clone()));
    }

    /// `(key, only_if_changed, explicit)` of each emitted active-open request.
    #[derive(Resource, Default)]
    struct SeenActiveRequestFlags(Vec<(String, bool, bool)>);

    fn collect_active_request_flags(
        mut reader: MessageReader<OpenDocumentFromActiveRequest>,
        mut seen: ResMut<SeenActiveRequestFlags>,
    ) {
        seen.0.extend(reader.read().map(|request| {
            (
                request.key.clone(),
                request.only_if_changed,
                request.explicit,
            )
        }));
    }

    #[test]
    fn config_sync_reloads_when_active_key_changes_with_equal_content() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "same active key must not reload"
        );

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/b.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/b.kdl".to_string()],
            "changing the active key must reload even when the KDL is identical"
        );
    }

    #[test]
    fn config_sync_reloads_when_asset_revision_bumps_at_same_key() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        // Already loaded schematics/a.kdl at the current revision.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "same key + unchanged revision must not reload"
        );

        // Another client overwrote the bytes at the same key: revision bumps.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/a.kdl".to_string()],
            "a byte change under an unchanged active key must reload (Bug 1)"
        );
    }

    #[test]
    fn config_sync_reloads_same_key_after_active_cleared() {
        // Reproduces the DB-driven clear: `schematic.active` disappears from
        // DbConfig (e.g. a follower snapshot from a source without a pointer),
        // then the *same* key is re-set at an unchanged `assets.revision`. The
        // clear must drop the sync baselines, otherwise sync sees
        // `last_synced == active` + unchanged revision, skips the fetch, and
        // strands the editor on the emptied view.
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        // schematics/a.kdl is loaded and baselined at the current revision.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        {
            let revision = app.world().resource::<DbConfig>().assets_revision();
            app.world_mut()
                .resource_mut::<LastSyncedAssetsRevision>()
                .revision = Some(revision);
        }
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "same key + unchanged revision must not reload"
        );

        // The active pointer disappears: the document is cleared.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .metadata
            .remove("schematic.active");
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "clearing the pointer must not emit a load"
        );
        assert!(
            app.world().resource::<LastSyncedActiveKey>().0.is_none(),
            "the clear must drop the last-synced key baseline"
        );

        // The same key comes back at the same revision: it must reload.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/a.kdl".to_string()],
            "re-setting the same key after a clear must reload the schematic"
        );
    }

    #[test]
    fn revision_bump_refetch_is_content_gated_and_tracked() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequestFlags>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_request_flags,
                )
                    .chain(),
            );

        // A first load through a key change is a plain (unconditional) reload.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequestFlags>().0,
            vec![("schematics/a.kdl".to_string(), false, false)],
            "a key change must reload unconditionally"
        );

        // Loaded: key synced, revision baseline adopted.
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        app.world_mut()
            .resource_mut::<LastSyncedAssetsRevision>()
            .revision = Some(0);

        // A revision bump at the unchanged key (e.g. a skybox cubemap upload)
        // requests a content-gated refetch, so the fetch handler can skip the
        // disruptive reload when the schematic bytes are unchanged (Bug 1/2).
        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequestFlags>().0[1..],
            [("schematics/a.kdl".to_string(), true, false)],
            "a revision-only bump must request a content-gated refetch"
        );
        // The baseline is NOT adopted at request time: a refetch that reads
        // stale bytes (a follower mid-mirror) must stay reloadable. Only the
        // fetch handler consumes it once the refetch concludes. `requested`
        // tracks the dispatched target so the same fetch isn't re-requested
        // every frame in the meantime.
        {
            let revision = app.world().resource::<LastSyncedAssetsRevision>();
            assert_eq!(
                revision.revision,
                Some(0),
                "the baseline must stay un-adopted until the refetch concludes"
            );
            assert_eq!(revision.requested, Some(1));
        }
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequestFlags>().0.len(),
            2,
            "the tracked in-flight target must not re-request the same fetch"
        );
    }

    #[test]
    fn pin_confirmed_open_is_marked_explicit() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequestFlags>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_request_flags,
                )
                    .chain(),
            );

        // "Open Schematic main" while main is already active: pin + reset
        // last_synced (what `open_schematic_item` does), then the DB echoes.
        {
            let mut pending = app.world_mut().resource_mut::<PendingActiveSchematic>();
            pending.pin(
                "schematics/main.kdl".to_string(),
                Some("schematics/main.kdl".to_string()),
            );
        }
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/main.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequestFlags>().0,
            vec![("schematics/main.kdl".to_string(), false, true)],
            "a pin-confirmed load is a user-initiated (explicit) open"
        );
    }

    #[test]
    fn last_active_content_matches_ignores_formatting_only_changes() {
        let mut content = LastActiveSchematicContent::default();
        // Hand-written formatting (as ingested from a source tree)...
        content.record("schematics/main.kdl", "viewport   {\n\n}\n");

        // ...matches the normalized serialization of the same schematic.
        let normalized = {
            use impeller2_kdl::{FromKdl, ToKdl};
            Schematic::from_kdl("viewport {\n}\n").unwrap().to_kdl()
        };
        assert!(content.matches("schematics/main.kdl", &normalized));
        assert!(content.matches("schematics/main.kdl", "viewport {\n}\n"));

        // A different key or genuinely different content must not match.
        assert!(!content.matches("schematics/other.kdl", "viewport {\n}\n"));
        assert!(!content.matches(
            "schematics/main.kdl",
            "viewport {\n}\nskybox name=\"desert_night\"\n"
        ));

        // Unparsable content never matches, so the caller falls back to a
        // full reload (which surfaces the parse error through its own path).
        content.record("schematics/main.kdl", "not-a-schematic {{{");
        assert!(!content.matches("schematics/main.kdl", "not-a-schematic {{{"));
    }

    #[test]
    fn config_sync_suppresses_reload_for_local_save_revision_bump() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        // A local save just landed: we have a revision baseline and want to
        // adopt the next (our own) bump without reloading.
        {
            let mut revision = app.world_mut().resource_mut::<LastSyncedAssetsRevision>();
            revision.revision = Some(0);
            revision.suppress_next = true;
        }

        // Our own save's echoed bump must NOT reload the bytes we just wrote.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "a locally initiated save's revision bump must not reload"
        );
        {
            let revision = app.world().resource::<LastSyncedAssetsRevision>();
            assert_eq!(revision.revision, Some(1), "baseline adopts the bump");
            assert!(!revision.suppress_next, "suppression is one-shot");
        }

        // A later external bump reloads as usual.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/a.kdl".to_string()],
            "an external bump after the suppressed one must reload"
        );
    }

    #[test]
    fn config_sync_skips_reload_while_local_save_in_flight() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        // A multi-`PUT` save this client started is still uploading.
        app.insert_resource(
            crate::ui::command_palette::palette_items::SchematicSaveInFlight::saving_stub(),
        );
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        app.world_mut()
            .resource_mut::<LastSyncedAssetsRevision>()
            .revision = Some(0);

        // Each in-flight PUT bumps the revision; none may reload the tree we are
        // still writing (RFD #724, Bug 1).
        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();
        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "in-flight save PUTs must not reload a partially written schematic"
        );
        assert_eq!(
            app.world().resource::<LastSyncedAssetsRevision>().revision,
            Some(2),
            "baseline tracks our own bumps so completion leaves no stale delta"
        );
    }

    #[test]
    fn config_sync_reloads_on_active_key_change() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/a.kdl".to_string()],
            "an active key must fetch over HTTP"
        );

        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/b.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec![
                "schematics/a.kdl".to_string(),
                "schematics/b.kdl".to_string()
            ],
            "changing the active key must reload"
        );
    }

    #[test]
    fn config_sync_releases_pin_when_active_moves_elsewhere() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        // Baseline: the editor is on schematics/a.kdl and has optimistically
        // pinned schematics/pinned.kdl (e.g. Open Schematic…), superseding a.kdl.
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/a.kdl".to_string());
        {
            let mut pending = app.world_mut().resource_mut::<PendingActiveSchematic>();
            pending.pin(
                "schematics/pinned.kdl".to_string(),
                Some("schematics/a.kdl".to_string()),
            );
        }

        // A stale echo still showing the superseded pointer must not reload, and
        // the pin must remain while we wait for our requested key.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/a.kdl");
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "the superseded pointer must not reload while the pin waits"
        );
        assert_eq!(
            app.world()
                .resource::<PendingActiveSchematic>()
                .target
                .as_deref(),
            Some("schematics/pinned.kdl"),
            "the pin must persist while the DB still shows the superseded key"
        );

        // The active pointer then moves to a third key (external repoint / a
        // failed local one). The pin must release and sync must follow the DB
        // rather than stranding until restart.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/external.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/external.kdl".to_string()],
            "an external move off the superseded key must be followed"
        );
        assert!(
            app.world()
                .resource::<PendingActiveSchematic>()
                .target
                .is_none(),
            "a pin the DB will never confirm must be released"
        );
    }

    #[test]
    fn config_sync_reloads_same_active_key_when_last_synced_reset() {
        // Reproduces: Save (main) → Clear Schematic (local, empties the view) →
        // Open main. "Clear Schematic" resets only the local document, so
        // `last_synced == active` and, at an unchanged revision, sync would skip
        // the reload and leave the empty view. "Open Schematic" resets
        // `LastSyncedActiveKey` so re-selecting the already-active key reloads it.
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    (|| None::<PathBuf>).pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );

        // main is the active, already-synced key (as after Save + a local Clear).
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/main.kdl".to_string());
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/main.kdl");
        app.update();
        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "same key at an unchanged revision must not reload on its own"
        );

        // "Open Schematic main" pins the key and resets last_synced (what
        // `open_schematic_item` does), then the DB echoes active=main. Sync must
        // now reload the same key, restoring the view cleared locally.
        {
            let mut pending = app.world_mut().resource_mut::<PendingActiveSchematic>();
            pending.pin(
                "schematics/main.kdl".to_string(),
                Some("schematics/main.kdl".to_string()),
            );
        }
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 = None;
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/main.kdl");
        app.update();
        assert_eq!(
            app.world().resource::<SeenActiveRequests>().0,
            vec!["schematics/main.kdl".to_string()],
            "opening resets last_synced so the already-active key reloads after a local clear"
        );
    }

    #[test]
    fn config_sync_skips_given_path_when_already_loaded() {
        let temp = TempTestDir::new("config-sync-skip-current");
        let path = temp.path().join("drone.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");
        let resolved_path = impeller2_kdl::env::schematic_file(&path);

        let given = path.clone();
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .insert_resource(SyncPath(Some(given)))
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenOpenDocumentRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    sync_path.pipe(super::operations::sync_document_from_config),
                    collect_open_document_requests,
                )
                    .chain(),
            );
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

    /// When EQL metadata settles, sticky `--kdl` must re-open so panels that
    /// failed `ComponentNotFound` on the empty/partial first open can recompile.
    #[test]
    fn reload_sticky_kdl_when_eql_ready_reopens_once() {
        use std::sync::Arc;

        use bevy::time::TimeUpdateStrategy;
        use impeller2::schema::Schema;
        use impeller2::types::{ComponentId, PrimType, Timestamp};

        let temp = TempTestDir::new("sticky-kdl-eql-ready");
        let path = temp.path().join("local.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");

        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(super::InitialKdlPath(Some(path.clone())))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_millis(
                100,
            )))
            .init_resource::<crate::EqlContext>()
            .init_resource::<SeenOpenDocumentRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_systems(
                Update,
                (
                    super::operations::reload_sticky_kdl_when_eql_ready,
                    collect_open_document_requests,
                )
                    .chain(),
            );

        app.update();
        assert!(
            app.world()
                .resource::<SeenOpenDocumentRequests>()
                .0
                .is_empty(),
            "must not reload while EQL context is empty"
        );

        let component = Arc::new(eql::Component::new(
            "body.WORLD_POS".to_string(),
            ComponentId::new("body.WORLD_POS"),
            Schema::new(PrimType::F64, [7usize]).unwrap(),
        ));
        app.world_mut().resource_mut::<crate::EqlContext>().0 =
            eql::Context::from_leaves([component], Timestamp(0), Timestamp(1000));

        // Fingerprint change arms the settle timer; must not reopen immediately.
        app.update();
        assert!(
            app.world()
                .resource::<SeenOpenDocumentRequests>()
                .0
                .is_empty(),
            "must wait for EQL metadata settle before reopening"
        );

        // 0.35s settle; ManualDuration advances 100ms per update.
        for _ in 0..4 {
            app.update();
        }
        assert_eq!(
            app.world().resource::<SeenOpenDocumentRequests>().0,
            vec![path.clone()],
            "must reopen sticky --kdl once after metadata settles"
        );

        app.update();
        assert_eq!(
            app.world().resource::<SeenOpenDocumentRequests>().0.len(),
            1,
            "must not reopen every frame after EQL is ready"
        );
    }

    /// Sticky CLI `--kdl` (via `given_path` every sync): gaining `schematic.active`
    /// must not open the DB schematic over the local file.
    #[test]
    fn sticky_given_path_blocks_schematic_active_open() {
        let temp = TempTestDir::new("sticky-kdl-blocks-active");
        let path = temp.path().join("local.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");
        let resolved_path = impeller2_kdl::env::schematic_file(&path);

        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .insert_resource(SyncPath(Some(path)))
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .init_resource::<SeenOpenDocumentRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    sync_path.pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                    collect_open_document_requests,
                )
                    .chain(),
            );
        {
            let mut current_document = app.world_mut().resource_mut::<CurrentDocument>();
            current_document.handle = Some(Handle::<SchematicDocumentAsset>::default());
            current_document.save_path = Some(resolved_path);
        }

        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/main.kdl");
        app.update();

        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "sticky --kdl must not open schematic.active"
        );
        assert!(
            app.world()
                .resource::<SeenOpenDocumentRequests>()
                .0
                .is_empty(),
            "already-loaded sticky path must not re-open"
        );
    }

    /// Sticky CLI `--kdl` must still win when `assets.revision` bumps (unrelated
    /// asset write) while the local file is the on-screen document.
    #[test]
    fn sticky_given_path_blocks_revision_bump_reload() {
        let temp = TempTestDir::new("sticky-kdl-blocks-revision");
        let path = temp.path().join("local.kdl");
        fs::write(&path, "timeline\n").expect("write kdl");
        let resolved_path = impeller2_kdl::env::schematic_file(&path);

        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(DbConfig::default())
            .insert_resource(SyncPath(Some(path)))
            .init_resource::<CurrentDocument>()
            .init_resource::<LastSyncedActiveKey>()
            .init_resource::<LastSyncedAssetsRevision>()
            .init_resource::<PendingActiveSchematic>()
            .init_resource::<SeenActiveRequests>()
            .add_message::<OpenDocumentRequest>()
            .add_message::<OpenDocumentFromActiveRequest>()
            .add_message::<DocumentCleared>()
            .add_systems(
                Update,
                (
                    sync_path.pipe(super::operations::sync_document_from_config),
                    collect_active_requests,
                )
                    .chain(),
            );
        {
            let mut current_document = app.world_mut().resource_mut::<CurrentDocument>();
            current_document.handle = Some(Handle::<SchematicDocumentAsset>::default());
            current_document.save_path = Some(resolved_path);
        }
        // Poison last_synced as if a local load wrongly adopted schematic.active
        // (pre-fix behavior); sticky path must still block the refetch.
        app.world_mut()
            .resource_mut::<DbConfig>()
            .set_schematic_active("schematics/main.kdl");
        app.world_mut().resource_mut::<LastSyncedActiveKey>().0 =
            Some("schematics/main.kdl".to_string());
        app.world_mut()
            .resource_mut::<LastSyncedAssetsRevision>()
            .revision = Some(0);

        app.world_mut()
            .resource_mut::<DbConfig>()
            .bump_assets_revision();
        app.update();

        assert!(
            app.world().resource::<SeenActiveRequests>().0.is_empty(),
            "sticky --kdl must not reload from DB on assets.revision bump"
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
    ) -> (TempTestDir, ChdirGuard, App, Handle<SchematicDocumentAsset>) {
        use std::os::unix::fs::symlink;

        let temp = TempTestDir::new(name);
        let real_root = temp.path().join("real");
        fs::create_dir_all(&real_root).expect("create real root");
        let linked_root = temp.path().join("linked");
        symlink(&real_root, &linked_root).expect("create symlink root");

        write_test_document(&real_root, root_title, window_name);

        let _dir_guard = chdir_to(&linked_root);
        let mut app = test_app();
        let handle = load_document(&mut app);
        (temp, _dir_guard, app, handle)
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
