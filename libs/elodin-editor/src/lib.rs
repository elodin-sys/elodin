#![recursion_limit = "256"]

use std::{collections::HashMap, ops::Range, sync::Arc, time::Duration};

use crate::plugins::{editor_cam_input, editor_cam_touch};
#[cfg(feature = "big_space")]
use crate::spatial::{FloatingOrigin, FloatingOriginSettings, GridCell};
use bevy::material::AlphaMode;
use bevy::{
    DefaultPlugins,
    asset::{UnapprovedPathMode, embedded_asset},
    camera::RenderTarget,
    diagnostic::{DiagnosticsPlugin, FrameTimeDiagnosticsPlugin},
    ecs::observer::Observer,
    ecs::system::{NonSendMarker, SystemParam},
    light::DirectionalLightShadowMap,
    log::LogPlugin,
    math::{DQuat, DVec3},
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
    window::{PrimaryWindow, WindowRef, WindowResolution},
    winit::{WINIT_WINDOWS, WinitSettings},
};
use bevy_editor_cam::{SyncCameraPosition, controller::component::EditorCam};
#[cfg(feature = "inspector")]
use bevy_egui::EguiContext;
use bevy_egui::{EguiContextSettings, EguiGlobalSettings, EguiPlugin};
use bevy_geo_frames::GeoFramePlugin;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use bevy_picking::{PickingSettings, PickingSystems, mesh_picking::update_hits};
use impeller2::types::{ComponentId, OwnedPacket};
use impeller2::types::{Msg, Timestamp};
use impeller2_bevy::{
    ComponentMetadataRegistry, ComponentPathRegistry, ComponentSchemaRegistry, ComponentValueMap,
    ConnectionAddr, EntityMap, MsgRequestIdHandlers, PacketHandlerInput, PacketHandlers,
    PacketIdHandlers, RequestIdHandlers,
};
use impeller2_wkt::{CurrentTimestamp, NewConnection, Object3D, WorldPos};
use impeller2_wkt::{EarliestTimestamp, LastUpdated};
use nox::Tensor;
use object_3d::create_object_3d_entity;
use plugins::frustum::FrustumPlugin;
use plugins::frustum_intersection::FrustumIntersectionPlugin;
use plugins::gizmos::GizmoPlugin;
use plugins::navigation_gizmo::NavigationGizmoPlugin;
use plugins::render_layer_alloc;
use plugins::view_cube::{ViewCubeConfig, ViewCubePlugin};
use ui::{
    UI_ORDER_BASE,
    colors::{ColorExt, get_scheme},
    create_egui_context, default_present_mode,
    inspector::viewport::{set_viewport_pos, sync_viewport_focus_pick_targets},
    plot::{CollectedGraphData, gpu::LineHandle},
    tiles,
    utils::FriendlyEpoch,
};

/// Global coordinate frame resource set by the schematic's top-level `coordinate` node.
/// Individual elements (viewport, object_3d, line_3d, vector_arrow) use this as a fallback
/// when they don't specify their own frame.
#[derive(Resource, Default, Clone, Copy, Debug, Reflect)]
#[reflect(Resource)]
pub struct Coordinate(pub Option<GeoFrame>);

mod embedded_lfs;
pub mod icon_rasterizer;
pub mod iter;
pub(crate) use embedded_lfs::embedded_lfs_asset;
pub mod object_3d;
mod offset_parse;
pub mod plugins;
pub mod rim_glow_material;
pub mod sensor_camera;
mod skybox_db_assets;
mod skybox_generation;
#[cfg(feature = "big_space")]
pub(crate) mod spatial;
#[cfg(not(feature = "big_space"))]
#[path = "spatial_fallback.rs"]
pub(crate) mod spatial;
pub mod ui;
pub mod vector_arrow;

#[cfg(all(not(target_family = "wasm"), target_family = "unix"))]
pub mod headless;
#[cfg(not(target_family = "wasm"))]
pub mod run;

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) fn skybox_asset_plugin() -> bevy_ai_skybox::prelude::SkyboxAssetPlugin {
    let assets_dir = plugins::env_asset_source::resolve_assets_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("assets"));
    bevy_ai_skybox::prelude::SkyboxAssetPlugin {
        cache_dir: assets_dir.join("skyboxes"),
        asset_dir: std::path::PathBuf::from("skyboxes"),
        manifest_file: std::path::PathBuf::from("manifest.ron"),
        default_skybox: None,
        apply_to_all_cameras: false,
        // Keep the existing baked EnvironmentMapLight as the cheap lighting fallback.
        // Runtime filtering via GeneratedEnvironmentMapLight is too expensive for
        // multi-viewport editor sessions and sensor cameras in this asset-only slice.
        env_lighting: false,
        watch_manifest: false,
        manifest_poll_secs: 1.0,
    }
}

pub(crate) fn skybox_generation_plugin() -> bevy_ai_skybox::prelude::BlockadeSkyboxPlugin {
    bevy_ai_skybox::prelude::BlockadeSkyboxPlugin {
        default_resolution: bevy_ai_skybox::prelude::SkyboxResolution::EightK,
        ..Default::default()
    }
}

#[cfg(all(not(target_family = "wasm"), target_family = "unix"))]
pub(crate) fn skybox_asset_plugin_headless() -> bevy_ai_skybox::prelude::SkyboxAssetPlugin {
    let mut plugin = skybox_asset_plugin();
    plugin.watch_manifest = true;
    plugin.manifest_poll_secs = 0.25;
    plugin
}

#[cfg(feature = "inspector")]
#[derive(Component)]
struct InspectorWindow;

struct EmbeddedAssetPlugin;

impl Plugin for EmbeddedAssetPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "assets/logo.png");
        embedded_asset!(app, "assets/icons/play.png");
        embedded_asset!(app, "assets/icons/pause.png");
        embedded_asset!(app, "assets/icons/scrub.png");
        embedded_asset!(app, "assets/icons/jump_to_end.png");
        embedded_asset!(app, "assets/icons/jump_to_start.png");
        embedded_asset!(app, "assets/icons/frame_forward.png");
        embedded_asset!(app, "assets/icons/frame_back.png");
        embedded_asset!(app, "assets/icons/search.png");
        embedded_asset!(app, "assets/icons/add.png");
        embedded_asset!(app, "assets/icons/subtract.png");
        embedded_asset!(app, "assets/icons/close.png");
        embedded_asset!(app, "assets/icons/chart.png");
        embedded_asset!(app, "assets/icons/left-side-bar.png");
        embedded_asset!(app, "assets/icons/right-side-bar.png");
        embedded_asset!(app, "assets/icons/fullscreen.png");
        embedded_asset!(app, "assets/icons/exit-fullscreen.png");
        embedded_asset!(app, "assets/icons/setting.png");
        embedded_asset!(app, "assets/icons/lightning.png");
        embedded_asset!(app, "assets/icons/link.png");
        embedded_asset!(app, "assets/icons/loop.png");
        embedded_asset!(app, "assets/icons/tile_3d_viewer.png");
        embedded_asset!(app, "assets/icons/tile_graph.png");
        embedded_asset!(app, "assets/icons/ip-addr.png");
        embedded_asset!(app, "assets/icons/folder.png");
        embedded_asset!(app, "assets/logo-full.png");
        embedded_asset!(app, "assets/icons/chevron_right.png");
        embedded_asset!(app, "assets/icons/vertical-chevrons.png");
        embedded_asset!(app, "assets/icons/container.png");
        embedded_asset!(app, "assets/icons/plot.png");
        embedded_asset!(app, "assets/icons/viewport.png");
        embedded_asset!(app, "assets/icons/entity.png");
        // Font for ViewCube labels
        embedded_asset!(app, "assets/fonts/Roboto-Bold.ttf");
        // Axes Cube 3D model
        embedded_lfs_asset!(app, "assets/axes-cube.glb");
    }
}

#[derive(Default)]
pub struct EditorPlugin {
    window_resolution: WindowResolution,
}

/// The positions of camera of object_3d are sync'd in `PreUpdate`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PositionSync;

impl EditorPlugin {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            window_resolution: WindowResolution::new(width as u32, height as u32),
        }
    }
}

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        // Must run before anything can spawn a `WorldPos` entity.
        register_world_pos_components(app);
        let composite_alpha_mode = if cfg!(target_os = "macos") {
            bevy::window::CompositeAlphaMode::PostMultiplied
        } else {
            bevy::window::CompositeAlphaMode::Opaque
        };
        let winit_settings = if cfg!(feature = "tracy") {
            WinitSettings::continuous()
        } else if cfg!(target_os = "macos") {
            WinitSettings {
                focused_mode: bevy::winit::UpdateMode::Reactive {
                    wait: Duration::from_millis(16),
                    react_to_device_events: true,
                    react_to_user_events: true,
                    react_to_window_events: true,
                },
                unfocused_mode: bevy::winit::UpdateMode::Reactive {
                    wait: Duration::from_millis(32),
                    react_to_device_events: false,
                    react_to_user_events: true,
                    react_to_window_events: true,
                },
            }
        } else {
            WinitSettings::game()
        };
        app
            // .insert_resource(AssetMetaCheck::Never)
            .add_plugins(plugins::WebAssetPlugin)
            .add_plugins(plugins::env_asset_source::plugin)
            .add_plugins(plugins::kdl_asset_source::plugin)
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Elodin".into(),
                            present_mode: default_present_mode(),
                            canvas: Some("#editor".to_string()),
                            resolution: self.window_resolution.clone(),
                            resize_constraints: WindowResizeConstraints {
                                min_width: 400.,
                                min_height: 400.,
                                ..Default::default()
                            },
                            composite_alpha_mode,
                            prevent_default_event_handling: true,
                            decorations: true,
                            visible: cfg!(target_os = "linux"),
                            ..default()
                        }),
                        ..default()
                    })
                    .set(AssetPlugin {
                        watch_for_changes_override: Some(true),
                        unapproved_path_mode: UnapprovedPathMode::Allow,
                        // NOTE: `Processed` interferes with WebAssetPlugin.
                        // mode: AssetMode::Processed,
                        ..default()
                    })
                    .disable::<TransformPlugin>()
                    .disable::<DiagnosticsPlugin>()
                    .disable::<LogPlugin>()
                    // Pulled into DefaultPlugins by the `bevy_dev_tools`
                    // cargo feature (needed for the native infinite grid);
                    // keep 0.18 behavior — no F1 render-debug keybind.
                    .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>()
                    .build(),
            )
            .add_plugins(plugins::kdl_document::plugin)
            .add_plugins(skybox_asset_plugin())
            .add_plugins(skybox_generation_plugin())
            .init_resource::<skybox_db_assets::DbSkyboxAssetMirror>()
            .init_resource::<skybox_db_assets::DbSkyboxSyncInFlight>()
            // Note: we added this because bevy 0.17.3 changed its behavior
            // which broke bevy_editor_cam. See here:
            // https://github.com/aevyrie/bevy_editor_cam/issues/61
            .insert_resource(PickingSettings {
                is_window_picking_enabled: false,
                ..Default::default()
            })
            .insert_resource(winit_settings)
            .add_plugins(bevy_framepace::FramepacePlugin)
            .insert_resource(
                bevy_framepace::FramepaceSettings::default()
                    .with_limiter(bevy_framepace::Limiter::Off),
            )
            //.add_plugins(DefaultPickingPlugins)
            .add_plugins(
                bevy_editor_cam::DefaultEditorCamPlugins
                    .build()
                    .disable::<bevy_editor_cam::input::DefaultInputPlugin>(),
            )
            .add_plugins(editor_cam_input::EditorCamInputPlugin)
            .add_plugins(EmbeddedAssetPlugin)
            .add_plugins(EguiPlugin::default())
            .add_plugins(bevy::dev_tools::infinite_grid::InfiniteGridPlugin)
            .add_plugins(render_layer_alloc::plugin)
            .add_plugins(NavigationGizmoPlugin)
            .add_plugins(ViewCubePlugin {
                config: ViewCubeConfig::editor_mode(),
            })
            .add_plugins(impeller2_bevy::Impeller2Plugin)
            .add_plugins(FrustumPlugin)
            .add_plugins(FrustumIntersectionPlugin)
            .add_plugins(GizmoPlugin);
        #[cfg(not(target_family = "wasm"))]
        app.add_plugins(plugins::thruster_particles::ThrusterParticlesPlugin);
        app.add_plugins(plugins::scene_environment::SceneEnvironmentPlugin);
        #[cfg(not(target_family = "wasm"))]
        app.add_plugins(plugins::screenshot::EnvScreenshotPlugin);
        app.add_plugins(ui::UiPlugin)
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_plugins(WireframePlugin::default())
            .add_plugins(editor_cam_touch::EditorCamTouchPlugin)
            .add_plugins(crate::ui::plot::PlotPlugin)
            .add_plugins(crate::plugins::LogicalKeyPlugin)
            .add_systems(Startup, setup_egui_global_system)
            .add_systems(Startup, setup_window_icon)
            //.add_systems(Startup, spawn_clear_bg)
            .add_systems(Startup, setup_clear_state)
            .init_resource::<SeriesStoreSession>()
            .add_systems(Update, sync_series_store_session_from_db_config)
            .add_systems(Update, setup_egui_context)
            .add_systems(Update, organize_observer_entities)
            //.add_systems(Update, make_entities_selectable)
            .add_systems(
                PreUpdate,
                sanitize_editor_cam_anchor_depth.before(SyncCameraPosition),
            )
            .add_systems(
                PreUpdate,
                sync_viewport_focus_pick_targets
                    .before(update_hits)
                    .in_set(PickingSystems::Backend),
            )
            .add_systems(
                PreUpdate,
                (
                    clamp_current_time,
                    advance_playback,
                    follow_latest,
                    // Update selection after playback advances to keep line_3d and object_3d in sync.
                    set_selected_range,
                    impeller2_bevy::apply_cached_data,
                    // Keep Object3D WorldPos in lock-step with cached component values
                    // before transforms are synchronized for rendering.
                    object_3d::update_object_3d_system,
                    #[cfg(feature = "big_space")]
                    set_floating_origin,
                    sync_object_3d,
                    set_viewport_pos,
                    sync_pos,
                    #[cfg(not(feature = "big_space"))]
                    bevy_geo_frames::apply_transforms,
                    bevy_geo_frames::apply_geo_rotation,
                    #[cfg(feature = "big_space")]
                    spatial::apply_big_translation,
                )
                    .chain()
                    .after(impeller2_bevy::sink)
                    .in_set(PositionSync),
            )
            .add_systems(
                Update,
                impeller2_bevy::backfill_cache.after(crate::ui::plot::update_series_fetch_priority),
            )
            .add_systems(Update, ui::data_overview::trigger_time_range_queries)
            .add_systems(Update, update_eql_context)
            .add_systems(Update, set_eql_context_range.after(update_eql_context))
            .add_systems(
                Update,
                plugins::kdl_document::reload_sticky_kdl_when_eql_ready.after(update_eql_context),
            )
            .add_systems(Startup, spawn_ui_cam)
            .add_systems(Update, ui::video_stream::connect_streams)
            .init_resource::<skybox_generation::LocallyPushedSkyboxActive>()
            .add_systems(
                Update,
                skybox_generation::sync_generated_skybox_to_schematic,
            )
            .add_systems(Update, skybox_generation::on_document_loaded)
            .add_systems(Update, skybox_generation::record_reloaded_schematic_key)
            .add_systems(Update, skybox_generation::push_skybox_active_on_pending)
            .add_systems(Update, skybox_generation::decay_skybox_status_message)
            .add_systems(
                Update,
                ui::video_stream::invalidate_sensor_frames_on_db_skybox_change,
            )
            .add_systems(Update, ui::log_stream::connect_streams)
            .add_systems(PostUpdate, ui::video_stream::set_visibility)
            .init_resource::<skybox_db_assets::DbSkyboxUploaded>()
            .add_systems(
                PostUpdate,
                skybox_db_assets::sync_db_skybox_assets_from_config,
            )
            .add_systems(
                PostUpdate,
                skybox_db_assets::upload_active_skybox_assets_to_db,
            )
            .add_systems(PostUpdate, set_clear_color)
            .insert_resource(WireframeConfig {
                global: false,
                default_color: Color::WHITE,
                ..default()
            })
            .insert_resource(ClearColor(get_scheme().bg_secondary.into_bevy()))
            .insert_resource(TimeRangeBehavior::default())
            .insert_resource(SelectedTimeRange(Timestamp(i64::MIN)..Timestamp(i64::MAX)))
            .insert_resource(FetchTimeRange(Timestamp(i64::MIN)..Timestamp(i64::MAX)))
            .insert_resource(FullTimeRange(Timestamp(0)..Timestamp(1_000_000)))
            .init_resource::<EqlContext>()
            .init_resource::<Coordinate>()
            .init_resource::<SyncedObject3d>()
            .init_resource::<ui::data_overview::ComponentTimeRanges>()
            .add_plugins(bevy_mat3_material::Mat3MaterialPlugin)
            .add_plugins(rim_glow_material::RimGlowMaterialPlugin)
            .add_plugins(object_3d::Object3DPlugin)
            .add_plugins(plugins::world_mesh::EditorWorldMeshPlugin)
            .add_plugins(GeoFramePlugin {
                apply_transforms: false,
                ..default()
            })
            .init_resource::<sensor_camera::SensorCameraConfigs>()
            .init_resource::<sensor_camera::SensorCamerasSpawned>()
            .init_resource::<sensor_camera::SensorCameraFrustumSourcesSpawned>()
            .add_systems(PreUpdate, sensor_camera::load_sensor_configs_from_db)
            .add_systems(
                PreUpdate,
                sensor_camera::spawn_sensor_camera_frustum_sources
                    .run_if(sensor_camera::should_spawn_sensor_camera_frustum_sources),
            )
            .add_systems(
                PreUpdate,
                sensor_camera::update_sensor_camera_frustum_source_transforms.after(PositionSync),
            )
            .add_systems(Update, sensor_camera::patch_sensor_view_dims)
            .add_systems(Update, throttle_for_sensor_cameras);

        app.add_systems(PreUpdate, warn_missing_geo.before(PositionSync));
        #[cfg(feature = "big_space")]
        app.add_plugins(spatial::FloatingOriginPlugin::new(16_000., 100.));
        if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
            app.add_systems(Update, handle_drag_resize);
        }

        #[cfg(feature = "debug")]
        app.add_plugins(spatial::debug::FloatingOriginDebugPlugin::default());

        #[cfg(not(target_family = "wasm"))]
        app.add_plugins(crate::ui::startup_window::StartupPlugin);

        #[cfg(target_os = "macos")]
        app.add_systems(Update, setup_titlebar);

        #[cfg(feature = "inspector")]
        {
            app.add_plugins(bevy_inspector_egui::DefaultInspectorConfigPlugin)
                .add_systems(Startup, setup_egui_inspector)
                .add_systems(Update, run_egui_inspector);
        }

        // For adding features incompatible with wasm:
        embedded_asset!(app, "./assets/diffuse.ktx2");
        embedded_asset!(app, "./assets/specular.ktx2");
        if cfg!(not(target_arch = "wasm32")) {
            app.insert_resource(DirectionalLightShadowMap { size: 8192 });
        }
        app.configure_sets(
            PreUpdate,
            PositionSync.before(bevy_editor_cam::SyncCameraPosition),
        );
        app.configure_sets(
            PostUpdate,
            bevy_editor_cam::SyncCameraPosition.after(bevy::transform::TransformSystems::Propagate),
        );
    }
}

/// Reduce editor GPU overhead when sensor cameras share the same GPU.
///
/// On macOS (unified memory) the headless render-server and editor compete
/// for the same GPU, causing thermal throttling.  We apply aggressive
/// throttling: ~24 fps, shadows off, minimal shadow map.
///
/// On Linux/Windows a discrete GPU is typical, so we apply a lighter
/// policy: ~30 fps with reduced (but not disabled) shadows.
/// True when any loaded window has a `sensor_view` panel in its tile tree.
///
/// The active schematic is no longer mirrored in DB metadata (RFD #724, Phase 4
/// dropped `schematic.content`), so scanning the KDL text for `"sensor_view"` is
/// gone. A `sensor_view` panel becomes a `Pane::SensorView` tile at load time,
/// so inspecting the live tile trees is the metadata-free equivalent: a
/// sensor-view schematic is treated as sensor-camera work even when
/// `SensorCameraConfigs` is empty and the `sensor_cameras` metadata is absent.
fn has_sensor_view_pane(windows: &Query<&crate::ui::tiles::WindowState>) -> bool {
    windows.iter().any(|window| {
        window.tile_state.tree.tiles.iter().any(|(_, tile)| {
            matches!(
                tile,
                egui_tiles::Tile::Pane(crate::ui::tiles::Pane::SensorView(_))
            )
        })
    })
}

#[cfg(target_os = "macos")]
fn throttle_for_sensor_cameras(
    configs: Res<sensor_camera::SensorCameraConfigs>,
    db_config: Res<impeller2_wkt::DbConfig>,
    mut settings: ResMut<bevy_framepace::FramepaceSettings>,
    mut shadow_map: ResMut<DirectionalLightShadowMap>,
    mut dir_lights: Query<&mut DirectionalLight>,
    windows: Query<&crate::ui::tiles::WindowState>,
    mut applied: Local<bool>,
) {
    if *applied {
        return;
    }
    let detected = !configs.0.is_empty()
        || db_config.metadata.contains_key("sensor_cameras")
        || has_sensor_view_pane(&windows);
    if !detected {
        return;
    }
    settings.limiter = bevy_framepace::Limiter::Manual(Duration::from_millis(42));
    shadow_map.size = 256;
    for mut light in dir_lights.iter_mut() {
        light.shadow_maps_enabled = false;
    }
    tracing::info!(
        "Sensor cameras detected — editor throttled to ~24 fps, shadows off (macOS GPU sharing)"
    );
    *applied = true;
}

#[cfg(not(target_os = "macos"))]
fn throttle_for_sensor_cameras(
    configs: Res<sensor_camera::SensorCameraConfigs>,
    db_config: Res<impeller2_wkt::DbConfig>,
    mut settings: ResMut<bevy_framepace::FramepaceSettings>,
    mut shadow_map: ResMut<DirectionalLightShadowMap>,
    windows: Query<&crate::ui::tiles::WindowState>,
    mut applied: Local<bool>,
) {
    if *applied {
        return;
    }
    let detected = !configs.0.is_empty()
        || db_config.metadata.contains_key("sensor_cameras")
        || has_sensor_view_pane(&windows);
    if !detected {
        return;
    }
    settings.limiter = bevy_framepace::Limiter::Manual(Duration::from_millis(33));
    shadow_map.size = 1024;
    tracing::info!(
        "Sensor cameras detected — editor throttled to ~30 fps, shadow map reduced (GPU sharing)"
    );
    *applied = true;
}

#[cfg(feature = "inspector")]
fn setup_egui_inspector(mut commands: Commands) {
    let window = Window {
        title: "World Inspector".to_string(),
        resolution: WindowResolution::new(640, 480),
        ..Default::default()
    };

    let window_ent = commands.spawn((window, InspectorWindow));
    let window_id = window_ent.id();

    let egui_context = create_egui_context();

    commands.entity(window_id).insert((
        Camera2d,
        Camera::default(),
        RenderTarget::Window(WindowRef::Entity(window_id)),
        egui_context,
    ));
}

#[cfg(feature = "inspector")]
fn run_egui_inspector(world: &mut World) {
    let egui_context = world
        .query_filtered::<&mut EguiContext, With<InspectorWindow>>()
        .single(world);

    let Ok(egui_context) = egui_context else {
        return;
    };
    let mut egui_context = egui_context.clone();

    egui::CentralPanel::default().show(egui_context.get_mut(), |ui| {
        egui::ScrollArea::both().show(ui, |ui| {
            bevy_inspector_egui::bevy_inspector::ui_for_world(world, ui);
            ui.allocate_space(ui.available_size());
        });
    });
}

fn setup_egui_global_system(mut egui_global_settings: ResMut<EguiGlobalSettings>) {
    egui_global_settings.auto_create_primary_context = false;
}

fn setup_egui_context(mut contexts: Query<&mut EguiContextSettings>) {
    for mut context in &mut contexts {
        context.capture_pointer_input = false;
        // Workaround for https://github.com/emilk/egui/issues/5008
        // On Linux, IME activation via set_ime_allowed(true) causes the compositor to
        // capture Backspace/arrow key events, preventing them from reaching TextEdit.
        #[cfg(target_os = "linux")]
        {
            context.enable_ime = false;
        }
    }
}

#[derive(Component)]
struct ObserverRoot;

fn organize_observer_entities(
    mut commands: Commands,
    mut root: Local<Option<Entity>>,
    observers: Query<Entity, (With<Observer>, Without<ChildOf>)>,
) {
    let root =
        *root.get_or_insert_with(|| commands.spawn((ObserverRoot, Name::new("observers"))).id());
    for observer in &observers {
        commands.entity(observer).insert(ChildOf(root));
    }
}

#[derive(Component, Clone)]
pub struct MainCamera;

#[derive(Component, Clone, Copy, Debug, Reflect)]
pub struct GridHandle {
    pub layer: usize,
}

fn spawn_ui_cam(mut commands: Commands, mut query: Query<Entity, With<PrimaryWindow>>) {
    let primary_window_ent = query
        .single_mut()
        .expect("failed to get single primary window");

    let egui_context = create_egui_context();

    commands.entity(primary_window_ent).insert((
        Camera2d,
        Camera {
            order: UI_ORDER_BASE,
            ..Default::default()
        },
        RenderTarget::Window(WindowRef::Entity(primary_window_ent)),
        egui_context,
    ));
}

fn set_clear_color(mut clear_color: ResMut<ClearColor>) {
    clear_color.0 = get_scheme().bg_secondary.into_bevy();
}

// NOTE(sphw): enabling this causes weird flickering issues when spawning too many 2d cameras
// This issue (https://github.com/bevyengine/bevy/issues/18897) looks to be the same thing
//
// fn spawn_clear_bg(mut commands: Commands) {
//     commands.spawn((Camera2d, IsDefaultUiCamera));
//     let bg_color = Color::Srgba(Srgba::hex("#0C0C0C").unwrap());
//     // root node
//     commands
//         .spawn(Node {
//             width: Val::Percent(100.0),
//             height: Val::Percent(100.0),
//             justify_content: JustifyContent::Stretch,
//             flex_direction: FlexDirection::Column,
//             ..default()
//         })
//         .with_children(|parent| {
//             parent
//                 .spawn(Node {
//                     height: Val::Px(56.0),
//                     ..default()
//                 })
//                 .insert(BackgroundColor(if cfg!(target_os = "macos") {
//                     Color::NONE
//                 } else {
//                     bg_color
//                 }));

//             parent
//                 .spawn(Node {
//                     height: Val::Percent(100.0),
//                     ..default()
//                 })
//                 .insert(BackgroundColor(bg_color));
//         });
// }

/// Keep the floating origin glued to the active main camera so big_space
/// renders the rest of the scene at low precision relative to it.
///
/// Since the migration to big_space 0.12 the main camera no longer
/// carries a `GridCell`: it is parented under the viewport entity, which
/// is the grid anchor. The system therefore:
///
/// 1. Reads the camera's local `Transform` and looks up its parent
///    viewport's `Transform` + `GridCell`.
/// 2. Composes them into an absolute world-space position via
///    `grid_position_double`.
/// 3. Re-projects that absolute position onto the grid to obtain the
///    `(origin_cell, origin_translation)` pair the floating origin must
///    take. Cameras spawned without a parent (e.g. tests) fall back to
///    the default cell at the origin.
#[cfg(feature = "big_space")]
#[allow(clippy::type_complexity)]
fn set_floating_origin(
    query: Query<
        (&Transform, Option<&ChildOf>),
        (With<MainCamera>, With<EditorCam>, Without<FloatingOrigin>),
    >,
    parent_query: Query<(&Transform, &GridCell), (Without<MainCamera>, Without<FloatingOrigin>)>,
    mut floating_origin: Query<(&mut Transform, &mut GridCell), With<FloatingOrigin>>,
    floating_origin_settings: Res<FloatingOriginSettings>,
) {
    let Some((camera_transform, parent)) = query.iter().next() else {
        return;
    };
    let (base_transform, base_cell) = parent
        .and_then(|parent| parent_query.get(parent.parent()).ok())
        .map(|(parent_transform, parent_cell)| {
            (
                parent_transform.mul_transform(*camera_transform),
                *parent_cell,
            )
        })
        .unwrap_or((*camera_transform, crate::spatial::GridCell::default()));
    let absolute = floating_origin_settings.grid_position_double(&base_cell, &base_transform);
    let (origin_cell, origin_translation) = floating_origin_settings.translation_to_grid(absolute);
    // Keep `FloatingOrigin` rotation at identity. Upstream big_space (post the
    // `no_prop_rot_v0.17` fork) propagates the floating origin's rotation to
    // every descendant's `GlobalTransform`. If we mirror the camera's combined
    // rotation onto the origin, the world rotates twice (once via propagation,
    // once via the camera's own view) and ViewCube snaps send the scene off
    // screen — most visibly on Linux. Only the high-precision translation
    // needs to track the active camera.
    let origin_transform = Transform {
        translation: origin_translation,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    for (mut origin, mut cell) in floating_origin.iter_mut() {
        *origin = origin_transform;
        *cell = origin_cell;
    }
}

#[cfg(target_os = "macos")]
#[derive(Component)]
struct SetupTitlebar;

#[cfg(target_os = "macos")]
fn setup_titlebar(
    windows: Query<Entity, Without<SetupTitlebar>>,
    mut commands: Commands,
    _non_send_marker: NonSendMarker,
) {
    use objc2::rc::Retained;
    use objc2::{ClassType, msg_send, msg_send_id};
    use objc2_app_kit::{NSColor, NSToolbar, NSWindow, NSWindowStyleMask, NSWindowToolbarStyle};

    WINIT_WINDOWS.with_borrow(|winit_windows| {
        for id in &windows {
            let Some(window) = winit_windows.get_window(id) else {
                continue;
            };
            window.set_blur(true);
            use raw_window_handle::HasRawWindowHandle;
            let handle = window.raw_window_handle();
            let raw_window_handle::RawWindowHandle::AppKit(handle) = handle else {
                error!("non AppKit window on macOS");
                continue;
            };
            let window: *mut NSWindow = handle.ns_window.cast();
            if window.is_null() {
                continue;
            }
            unsafe {
                let window = &*window;

                // Create a simple toolbar without delegate for now
                // The delegate isn't strictly necessary for basic toolbar functionality
                use objc2_foundation::NSString;
                let identifier = NSString::from_str("MainToolbar");
                let toolbar: Retained<NSToolbar> = msg_send_id![
                    msg_send_id![NSToolbar::class(), alloc],
                    initWithIdentifier: &*identifier
                ];

                // Keep the native titlebar visible and readable.
                window.setTitlebarAppearsTransparent(false);
                let color = NSColor::windowBackgroundColor();
                window.setBackgroundColor(Some(&color));

                window.setStyleMask(
                    NSWindowStyleMask::FullSizeContentView
                        | NSWindowStyleMask::Resizable
                        | NSWindowStyleMask::Titled
                        | NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Miniaturizable
                        | NSWindowStyleMask::UnifiedTitleAndToolbar,
                );
                window.setToolbarStyle(NSWindowToolbarStyle::UnifiedCompact);
                // Keep the native title visible in the toolbar/titlebar area.
                let _: () = msg_send![window, setTitleVisibility: 0u64]; // NSWindowTitleVisible = 0
                window.setToolbar(Some(&toolbar));
                commands.entity(id).insert(SetupTitlebar);
            }
        }
    });
}

fn handle_drag_resize(
    windows: Query<(Entity, &Window, &bevy::window::PrimaryWindow)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut just_set_cursor: Local<bool>,
    _non_send_marker: NonSendMarker,
) {
    WINIT_WINDOWS.with_borrow(|winit_windows| {
        for (id, window, _) in &windows {
            let Some(cursor_pos) = window.physical_cursor_position() else {
                continue;
            };
            let size = window.physical_size().as_vec2();
            let window = winit_windows.get_window(id).unwrap();
            const RESIZE_ZONE: f32 = 5.0;
            let resize_west = cursor_pos.x < RESIZE_ZONE;
            let resize_east = cursor_pos.x > size.x - RESIZE_ZONE;
            let resize_north = cursor_pos.y < RESIZE_ZONE;
            let resize_south = cursor_pos.y > size.y - RESIZE_ZONE;
            let resize_dir = match (resize_west, resize_east, resize_north, resize_south) {
                (true, _, true, _) => Some(winit::window::ResizeDirection::NorthWest),
                (_, true, true, _) => Some(winit::window::ResizeDirection::NorthEast),

                (true, _, _, true) => Some(winit::window::ResizeDirection::SouthWest),
                (_, true, _, true) => Some(winit::window::ResizeDirection::SouthEast),
                (true, _, _, _) => Some(winit::window::ResizeDirection::West),
                (_, true, _, _) => Some(winit::window::ResizeDirection::East),
                (_, _, true, _) => Some(winit::window::ResizeDirection::North),
                (_, _, _, true) => Some(winit::window::ResizeDirection::South),
                _ => None,
            };
            if let Some(resize_dir) = resize_dir {
                if mouse_buttons.pressed(MouseButton::Left) {
                    let _ = window.drag_resize_window(resize_dir);
                }
                window.set_cursor(winit::window::CursorIcon::from(resize_dir));
                *just_set_cursor = true
            } else if *just_set_cursor {
                *just_set_cursor = false;
                window.set_cursor(winit::window::CursorIcon::Default);
            }
        }
    });
}

fn setup_window_icon(
    windows: Query<(Entity, &bevy::window::PrimaryWindow)>,
    _non_send_marker: NonSendMarker,
) {
    // There was an API change for WINIT_WINDOWS in bevy 0.17. The pre-0.17
    // code had the following comment:
    //
    // "this is load bearing, because it ensures that there is at least one
    // window spawned"
    //
    // We're not certain why it was originally necessary, or if this port
    // accomplishes the same goal, but keeping it in the spirit of Chesterton's
    // Fence.
    WINIT_WINDOWS.with_borrow(|_winit_windows| {
        #[cfg(target_os = "macos")]
        set_icon_mac();

        if !windows.is_empty() {
            #[cfg(target_os = "windows")]
            set_icon_windows();
        }
    });
}

#[cfg(target_os = "windows")]
fn set_icon_windows() {
    use winapi::um::winuser;
    let png_bytes = include_bytes!("../assets/win-512x512@2x.png");
    let unscaled_image =
        image::ImageReader::with_format(std::io::Cursor::new(png_bytes), image::ImageFormat::Png)
            .decode()
            .unwrap();
    let window_handle = unsafe { winuser::GetActiveWindow() };
    if window_handle.is_null() {
        return;
    }

    unsafe {
        let margins = winapi::um::uxtheme::MARGINS {
            cxLeftWidth: 0,
            cxRightWidth: 0,
            cyTopHeight: -40,
            cyBottomHeight: 40,
        };
        winapi::um::dwmapi::DwmExtendFrameIntoClientArea(window_handle, &margins);
        let mut rect: winapi::shared::windef::RECT = std::mem::zeroed();
        winapi::um::winuser::GetWindowRect(window_handle, &mut rect);
    }

    fn resize_image(image: &image::DynamicImage, size: usize) -> Option<Vec<u8>> {
        let image_scaled =
            image::imageops::resize(image, size as _, size as _, image::imageops::Lanczos3);
        let mut image_scaled_bytes: Vec<u8> = Vec::new();
        if image_scaled
            .write_to(
                &mut std::io::Cursor::new(&mut image_scaled_bytes),
                image::ImageFormat::Png,
            )
            .is_ok()
        {
            Some(image_scaled_bytes)
        } else {
            None
        }
    }

    let icon_size_big = unsafe { winuser::GetSystemMetrics(winuser::SM_CXICON) };
    let Some(big_image_bytes) = resize_image(&unscaled_image, icon_size_big as _) else {
        return;
    };

    unsafe {
        let icon = winuser::CreateIconFromResourceEx(
            big_image_bytes.as_ptr() as *mut _,
            big_image_bytes.len() as u32,
            1,          // Means this is an icon, not a cursor.
            0x00030000, // Version number of the HICON
            icon_size_big,
            icon_size_big,
            winuser::LR_DEFAULTCOLOR,
        );
        winuser::SendMessageW(
            window_handle,
            winuser::WM_SETICON,
            winuser::ICON_BIG as usize,
            icon as isize,
        );
    }

    let icon_size_small = unsafe { winuser::GetSystemMetrics(winuser::SM_CXSMICON) };
    let Some(small_image_bytes) = resize_image(&unscaled_image, icon_size_small as _) else {
        return;
    };

    unsafe {
        let icon = winuser::CreateIconFromResourceEx(
            small_image_bytes.as_ptr() as *mut _,
            small_image_bytes.len() as u32,
            1,          // Means this is an icon, not a cursor.
            0x00030000, // Version number of the HICON
            icon_size_small,
            icon_size_small,
            winuser::LR_DEFAULTCOLOR,
        );
        winuser::SendMessageW(
            window_handle,
            winuser::WM_SETICON,
            winuser::ICON_SMALL as usize,
            icon as isize,
        );
    }
}

/// source: https://github.com/emilk/egui/blob/15370bbea0b468cf719a75cc6d1e39eb00c420d8/crates/eframe/src/native/app_icon.rs#L199C1-L268C2
#[cfg(target_os = "macos")]
fn set_icon_mac() {
    use objc2::rc::Retained;
    use objc2::{ClassType, msg_send, msg_send_id};
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSData;

    let png_bytes = include_bytes!("../assets/512x512@2x.png");

    unsafe {
        // Get unowned reference to shared app singleton (+0 retain count)
        // Do NOT wrap in Retained - sharedApplication returns a non-owning reference
        let app: *mut NSApplication = msg_send![NSApplication::class(), sharedApplication];
        if app.is_null() {
            return;
        }

        // Create NSData from bytes
        let data = NSData::with_bytes(png_bytes);

        // Create NSImage from NSData (this is +1, owned by Retained)
        let app_icon: Retained<NSImage> = msg_send_id![NSImage::alloc(), initWithData: &*data];

        // Set the icon using the unowned app pointer
        let _: () = msg_send![app, setApplicationIconImage: &*app_icon];
    }
}

/// `WorldPos` entities are positioned exclusively by the geo pipeline
/// (`sync_pos` -> `GeoPosition`/`GeoRotation` -> `Transform`/`GridCell`), so
/// inserting `WorldPos` must always bring the pipeline components along.
/// Spawners that need a non-default frame insert their own `Geo*` components,
/// which take precedence over these defaults.
pub fn register_world_pos_components(app: &mut App) {
    app.register_required_components_with::<WorldPos, GeoPosition>(|| {
        GeoPosition(GeoFrame::default(), DVec3::ZERO)
    });
    app.register_required_components_with::<WorldPos, GeoRotation>(|| {
        GeoRotation::relative(GeoFrame::default(), DQuat::IDENTITY)
    });
    app.register_required_components::<WorldPos, Transform>();
    #[cfg(feature = "big_space")]
    app.register_required_components::<WorldPos, crate::spatial::GridCell>();
}

/// `WorldPos` requires the `Geo*` components ([`register_world_pos_components`]),
/// so this should never fire; an entity that escapes the geo pipeline is never
/// positioned.
#[allow(clippy::type_complexity)]
pub fn warn_missing_geo(
    query: Query<
        (Entity, Option<&Name>),
        (
            With<WorldPos>,
            Or<(Without<GeoPosition>, Without<GeoRotation>)>,
        ),
    >,
) {
    for (entity, name) in query.iter() {
        bevy::log::warn_once!(
            "{entity} ({name:?}) has WorldPos without GeoPosition/GeoRotation; it will not be positioned"
        );
    }
}

/// Accessors for `WorldPos` simulation coordinates.
///
/// Positioning goes exclusively through the geo pipeline: read `pos()`/`att()`
/// and convert with `GeoPosition`/`GeoRotation` + `GeoContext`. The `bevy_*`
/// methods hard-code the ENU-Plane mapping and exist only as legacy oracles
/// for tests; they silently disagree with `Present::Sphere` and non-ENU
/// frames.
pub trait WorldPosExt {
    /// Legacy ENU-Plane position swizzle `(x, z, -y)`. Test oracle only;
    /// use `GeoPosition::to_bevy` instead.
    fn bevy_pos(&self) -> DVec3;
    /// Legacy ENU-Plane attitude swizzle. Test oracle only; use
    /// `GeoRotation::to_bevy` instead.
    fn bevy_att(&self) -> DQuat;

    /// Position in simulation coordinates.
    fn pos(&self) -> DVec3;
    /// Attitude in simulation coordinates.
    fn att(&self) -> DQuat;
}

impl WorldPosExt for WorldPos {
    fn bevy_pos(&self) -> DVec3 {
        let [x, y, z] = self.pos.parts().map(Tensor::into_buf);
        DVec3::new(x, z, -y)
    }

    fn pos(&self) -> DVec3 {
        let [x, y, z] = self.pos.parts().map(Tensor::into_buf);
        DVec3::new(x, y, z)
    }

    fn bevy_att(&self) -> DQuat {
        let [i, j, k, w] = self.att.parts().map(Tensor::into_buf);
        let x = i;
        let y = k;
        let z = -j;
        DQuat::from_xyzw(x, y, z, w)
    }

    fn att(&self) -> DQuat {
        let [i, j, k, w] = self.att.parts().map(Tensor::into_buf);
        DQuat::from_xyzw(i, j, k, w)
    }
}

pub fn advance_playback(
    time: Res<Time>,
    mut current_ts: ResMut<CurrentTimestamp>,
    paused: Res<ui::Paused>,
    speed: Res<ui::timeline::PlaybackSpeed>,
    last_updated: Res<LastUpdated>,
    earliest: Res<EarliestTimestamp>,
) {
    if paused.0 {
        return;
    }
    if earliest.0 >= last_updated.0 {
        return;
    }
    let delta_micros = (time.delta_secs_f64() * speed.0 * 1_000_000.0) as i64;
    let new_ts = Timestamp(current_ts.0.0.saturating_add(delta_micros));
    current_ts.0 = Timestamp(new_ts.0.clamp(earliest.0.0, last_updated.0.0));
}

pub fn follow_latest(
    mut current_ts: ResMut<CurrentTimestamp>,
    latest: Res<LastUpdated>,
    earliest: Res<EarliestTimestamp>,
    latest_follow: Res<ui::timeline::LatestFollow>,
    replay: Option<Res<ReplayMode>>,
) {
    if replay.is_some() || !latest_follow.0 {
        return;
    }
    if earliest.0 >= latest.0 {
        return;
    }
    current_ts.0 = latest.0;
}

/// Sync `WorldPos` (sim coordinates) into the entity's `GeoPosition` and
/// `GeoRotation`. The geo pipeline (`apply_transforms`/`apply_big_translation`
/// plus `apply_geo_rotation`) then produces the Bevy `Transform`/`GridCell`.
///
/// Every `WorldPos` entity gets its `Geo*` components at insertion
/// ([`register_world_pos_components`]) or from its spawner, so there is no
/// direct `WorldPos` -> `Transform` path.
pub fn sync_pos(
    mut query: Query<(&mut GeoPosition, &mut GeoRotation, &WorldPos), Changed<WorldPos>>,
) {
    query
        .iter_mut()
        .for_each(|(mut geo_pos, mut geo_rot, world_pos)| {
            // TODONT: AI, do not change this. It is what it should be.
            geo_pos.1 = world_pos.pos();
            geo_rot.1 = world_pos.att();
        });
}

fn sanitize_editor_cam_anchor_depth(mut cams: Query<(Entity, &mut EditorCam)>) {
    const DEFAULT_DEPTH: f64 = -2.0;
    for (entity, mut cam) in cams.iter_mut() {
        if cam.last_anchor_depth.is_finite() {
            continue;
        }
        warn!(
            "Resetting invalid camera anchor depth (entity {:?}) from {} to {}",
            entity, cam.last_anchor_depth, DEFAULT_DEPTH
        );
        cam.last_anchor_depth = DEFAULT_DEPTH;
    }
}

pub trait BevyExt {
    type Bevy;
    fn into_bevy(self) -> Self::Bevy;
}
impl BevyExt for impeller2_wkt::Mesh {
    type Bevy = Mesh;

    fn into_bevy(self) -> Self::Bevy {
        match self {
            impeller2_wkt::Mesh::Sphere { radius } => {
                bevy::math::primitives::Sphere { radius }.into()
            }
            impeller2_wkt::Mesh::Box { x, y, z } => {
                bevy::math::primitives::Cuboid::new(x, y, z).into()
            }
            impeller2_wkt::Mesh::Cylinder { radius, height } => {
                bevy::math::primitives::Cylinder::new(radius, height).into()
            }
            impeller2_wkt::Mesh::Plane { width, depth } => {
                bevy::math::primitives::Plane3d::default()
                    .mesh()
                    .size(width, depth)
                    .into()
            }
        }
    }
}

impl BevyExt for impeller2_wkt::Material {
    type Bevy = StandardMaterial;

    fn into_bevy(self) -> Self::Bevy {
        let base_color = Color::srgba(
            self.base_color.r,
            self.base_color.g,
            self.base_color.b,
            self.base_color.a,
        );
        let alpha_mode = if self.base_color.a < 1.0 {
            AlphaMode::Blend
        } else {
            AlphaMode::Opaque
        };
        let emissivity = self.emissivity.clamp(0.0, 1.0);
        let boost = 4.0 * emissivity;
        let boosted_color = Color::srgba(
            self.base_color.r * boost,
            self.base_color.g * boost,
            self.base_color.b * boost,
            self.base_color.a,
        );
        let emissive = if emissivity > 0.0 {
            boosted_color
        } else {
            Color::BLACK
        };

        bevy::prelude::StandardMaterial {
            base_color,
            alpha_mode,
            emissive: emissive.into(),
            ..Default::default()
        }
    }
}

#[derive(Default, Resource)]
pub struct SyncedObject3d(HashMap<Entity, Entity>);

#[allow(clippy::too_many_arguments)]
pub fn sync_object_3d(
    query: Query<(Entity, &ComponentId), With<impeller2_wkt::WorldPos>>,
    meshes: Query<&impeller2_wkt::Mesh>,
    materials: Query<&impeller2_wkt::Material>,
    glbs: Query<&impeller2_wkt::Glb>,
    mut synced_object_3d: ResMut<SyncedObject3d>,
    entity_map: ResMut<EntityMap>,
    path_reg: Res<ComponentPathRegistry>,
    ctx: Res<EqlContext>,
    mut material_assets: ResMut<Assets<StandardMaterial>>,
    mut mesh_assets: ResMut<Assets<Mesh>>,
    mut mat3_material_assets: ResMut<Assets<bevy_mat3_material::Mat3Material>>,
    mut commands: Commands,
    assets: Res<AssetServer>,
    geo_context: Res<GeoContext>,
    connection_addr: Option<Res<impeller2_bevy::ConnectionAddr>>,
) {
    let connection_addr = connection_addr.as_ref().map(|addr| addr.0);
    for (entity, id) in &query {
        if synced_object_3d.0.contains_key(&entity) {
            continue;
        }
        let Some(path) = path_reg.get(id) else {
            continue;
        };
        let parent = path.path.first().unwrap();

        let glb = entity_map
            .get(&ComponentId::from_pair(&parent.name, "asset_handle_glb"))
            .and_then(|e| glbs.get(*e).ok());
        let mesh = entity_map
            .get(&ComponentId::from_pair(&parent.name, "asset_handle_mesh"))
            .and_then(|e| meshes.get(*e).ok());
        let material = entity_map
            .get(&ComponentId::from_pair(
                &parent.name,
                "asset_handle_material",
            ))
            .and_then(|e| materials.get(*e).ok());

        let mesh_source = match (glb, mesh, material) {
            (Some(glb), _, _) => impeller2_wkt::Object3DMesh::glb(glb.0.clone()),
            (_, Some(mesh), Some(mat)) => impeller2_wkt::Object3DMesh::Mesh {
                mesh: mesh.clone(),
                material: mat.clone(),
            },
            _ => continue,
        };

        let eql = format!("{}.world_pos", &parent.name);
        let Ok(expr) = ctx.0.parse_str(&eql) else {
            continue;
        };

        if let Ok(object_entity) = create_object_3d_entity(
            &mut commands,
            Object3D {
                eql,
                mesh: mesh_source,
                icon: None,
                thrusters: Vec::new(),
                mesh_visibility_range: None,
                frame: None,
                frame_orientation: None,
                orientation: Default::default(),
                node_id: Default::default(),
            },
            expr,
            &ctx.0,
            &mut material_assets,
            &mut mesh_assets,
            &mut mat3_material_assets,
            &assets,
            &geo_context,
            connection_addr,
        ) {
            synced_object_3d.0.insert(entity, object_entity);
        }
    }
}

pub fn setup_clear_state(mut packet_handlers: ResMut<PacketHandlers>, mut commands: Commands) {
    // Timestamp reset must run before clear_state mutates SeriesStoreSession.addr,
    // so soft/hard is decided against the pre-reconnect session identity.
    let sys = commands.register_system(reset_timestamps_on_new_connection);
    packet_handlers.0.push(sys);
    let sys = commands.register_system(clear_state_new_connection);
    packet_handlers.0.push(sys);
}

/// Tracks which DB the SeriesStore currently mirrors (addr + recording start).
#[derive(Resource, Debug, Clone, Default, PartialEq, Eq)]
pub struct SeriesStoreSession {
    pub addr: Option<std::net::SocketAddr>,
    pub start_ts: Option<i64>,
}

impl SeriesStoreSession {
    pub fn identity(addr: Option<std::net::SocketAddr>, start_ts: Option<i64>) -> Self {
        Self { addr, start_ts }
    }

    pub fn matches(&self, addr: Option<std::net::SocketAddr>, start_ts: Option<i64>) -> bool {
        self.addr == addr && self.start_ts == start_ts
    }
}

fn cancel_in_flight_series_requests(
    commands: &mut Commands,
    msg_handlers: &mut MsgRequestIdHandlers,
    req_handlers: &mut RequestIdHandlers,
    packet_id_handlers: &mut PacketIdHandlers,
    visible_prefetch: &mut crate::ui::plot::data::VisiblePrefetchState,
) {
    for (_, system_id) in msg_handlers.0.drain() {
        commands.unregister_system(system_id);
    }
    for (_, system_id) in req_handlers.0.drain() {
        commands.unregister_system(system_id);
    }
    for (_, system_id) in packet_id_handlers.0.drain() {
        commands.unregister_system(system_id);
    }
    visible_prefetch.clear_in_flight();
}

/// Despawn freestanding object_3d visuals tracked by [`SyncedObject3d`].
/// Source impeller entities are despawned separately; visuals are not their children.
fn despawn_synced_object_3d(commands: &mut Commands, synced_glbs: &mut SyncedObject3d) {
    for (_, visual) in synced_glbs.0.drain() {
        if let Ok(mut entity_commands) = commands.get_entity(visual) {
            entity_commands.despawn();
        }
    }
}

fn hard_clear_series_store(
    telemetry_cache: &mut impeller2_bevy::TelemetryCache,
    backfill_state: &mut impeller2_bevy::BackfillState,
    series_load: &mut impeller2_bevy::SeriesStoreLoadState,
    plot_sync: &mut crate::ui::plot::data::PlotSyncState,
    visible_prefetch: &mut crate::ui::plot::data::VisiblePrefetchState,
) {
    *telemetry_cache = impeller2_bevy::TelemetryCache::default();
    *backfill_state = impeller2_bevy::BackfillState::default();
    *series_load = impeller2_bevy::SeriesStoreLoadState::default();
    *plot_sync = crate::ui::plot::data::PlotSyncState::default();
    *visible_prefetch = crate::ui::plot::data::VisiblePrefetchState::default();
}

/// Soft reconnect: keep SeriesStore samples; re-arm begin→end backfill for catch-up.
fn soft_rearm_series_store_catchup(
    backfill_state: &mut impeller2_bevy::BackfillState,
    series_load: &mut impeller2_bevy::SeriesStoreLoadState,
) {
    *backfill_state = impeller2_bevy::BackfillState::default();
    series_load.components_started = 0;
    series_load.components_complete = 0;
    series_load.complete = false;
    // Keep samples_loaded as a lower bound of what remains in RAM.
}

fn series_store_soft_reconnect(
    session: &SeriesStoreSession,
    addr: Option<std::net::SocketAddr>,
) -> bool {
    match session.addr {
        None => true,
        Some(prev) => Some(prev) == addr,
    }
}

#[inline]
fn should_hard_clear_ui(soft: bool) -> bool {
    !soft
}

/// Soft/hard SeriesStore policy for `NewConnection`. Always runs (before any UI early-return).
#[allow(clippy::too_many_arguments)]
fn apply_series_store_on_reconnect(
    soft: bool,
    addr: Option<std::net::SocketAddr>,
    session: &mut SeriesStoreSession,
    telemetry_cache: &mut impeller2_bevy::TelemetryCache,
    backfill_state: &mut impeller2_bevy::BackfillState,
    series_load: &mut impeller2_bevy::SeriesStoreLoadState,
    plot_sync: &mut crate::ui::plot::data::PlotSyncState,
    visible_prefetch: &mut crate::ui::plot::data::VisiblePrefetchState,
) {
    if soft {
        soft_rearm_series_store_catchup(backfill_state, series_load);
        session.addr = addr;
        // start_ts confirmed when DbConfig arrives.
    } else {
        hard_clear_series_store(
            telemetry_cache,
            backfill_state,
            series_load,
            plot_sync,
            visible_prefetch,
        );
        *session = SeriesStoreSession::identity(addr, None);
    }
}

/// Clears DB sync baselines on hard reconnect. Does **not** clear
/// [`InitialKdlPath`](plugins::kdl_document::InitialKdlPath): a CLI `--kdl`
/// override must stay sticky so reconnect re-applies the local file instead of
/// only `schematic.active`.
fn invalidate_schematic_sync_baselines(
    last_synced_key: &mut plugins::kdl_document::LastSyncedActiveKey,
    last_synced_revision: &mut plugins::kdl_document::LastSyncedAssetsRevision,
) {
    last_synced_key.0 = None;
    *last_synced_revision = plugins::kdl_document::LastSyncedAssetsRevision::default();
}

/// Tear down tile UI / plot lines / Object3D sync map (hard reconnect only).
fn hard_clear_editor_ui(
    commands: &mut Commands,
    windows_state: &mut Query<(Entity, &mut tiles::WindowState)>,
    primary_window: Option<Entity>,
    line_entities: &[Entity],
    synced_glbs: &mut SyncedObject3d,
    graph_data: &mut CollectedGraphData,
) {
    for line in line_entities {
        if let Ok(mut entity_commands) = commands.get_entity(*line) {
            entity_commands.despawn();
        }
    }
    despawn_synced_object_3d(commands, synced_glbs);
    *graph_data = CollectedGraphData::default();

    let Some(primary_id) = primary_window else {
        return;
    };
    if let Ok(mut primary_state) = windows_state.get_mut(primary_id) {
        primary_state.1.tile_state.clear(commands);
    }
    let secondaries: Vec<(Entity, Vec<Entity>)> = windows_state
        .iter()
        .filter(|(entity, _)| *entity != primary_id)
        .map(|(entity, state)| (entity, state.graph_entities.clone()))
        .collect();
    for (entity, graphs) in secondaries {
        for graph in graphs {
            commands.entity(graph).despawn();
        }
        commands.entity(entity).despawn();
    }
}

/// Bundled UI hard-clear params for identity-mismatch path (SystemParam limit).
#[derive(SystemParam)]
struct EditorUiHardClear<'w, 's> {
    commands: Commands<'w, 's>,
    windows_state: Query<'w, 's, (Entity, &'static mut tiles::WindowState)>,
    primary_window: Option<Single<'w, 's, Entity, With<PrimaryWindow>>>,
    lines: Query<'w, 's, Entity, With<LineHandle>>,
    synced_glbs: ResMut<'w, SyncedObject3d>,
    graph_data: ResMut<'w, CollectedGraphData>,
    last_synced_key: ResMut<'w, plugins::kdl_document::LastSyncedActiveKey>,
    last_synced_revision: ResMut<'w, plugins::kdl_document::LastSyncedAssetsRevision>,
}

impl EditorUiHardClear<'_, '_> {
    fn clear_ui_and_invalidate_schematic_sync(&mut self) {
        let primary = self.primary_window.as_ref().map(|p| **p);
        let line_entities: Vec<Entity> = self.lines.iter().collect();
        hard_clear_editor_ui(
            &mut self.commands,
            &mut self.windows_state,
            primary,
            &line_entities,
            &mut self.synced_glbs,
            &mut self.graph_data,
        );
        invalidate_schematic_sync_baselines(
            &mut self.last_synced_key,
            &mut self.last_synced_revision,
        );
    }
}

/// Bundled so reconnect systems stay within Bevy's SystemParam limit.
#[derive(SystemParam)]
struct SeriesStoreReconnect<'w> {
    session: ResMut<'w, SeriesStoreSession>,
    connection_addr: Option<Res<'w, ConnectionAddr>>,
    msg_handlers: ResMut<'w, MsgRequestIdHandlers>,
    req_handlers: ResMut<'w, RequestIdHandlers>,
    packet_id_handlers: ResMut<'w, PacketIdHandlers>,
    telemetry_cache: ResMut<'w, impeller2_bevy::TelemetryCache>,
    backfill_state: ResMut<'w, impeller2_bevy::BackfillState>,
    series_load: ResMut<'w, impeller2_bevy::SeriesStoreLoadState>,
    plot_sync: ResMut<'w, crate::ui::plot::data::PlotSyncState>,
    visible_prefetch: ResMut<'w, crate::ui::plot::data::VisiblePrefetchState>,
}

/// When DbConfig identity differs from the preserved session, wipe SeriesStore + UI.
pub(crate) fn sync_series_store_session_from_db_config(
    config: Res<impeller2_wkt::DbConfig>,
    mut series: SeriesStoreReconnect,
    mut editor_ui: EditorUiHardClear,
    mut current: ResMut<CurrentTimestamp>,
) {
    if !config.is_changed() {
        return;
    }
    let addr = series.connection_addr.as_ref().map(|a| a.0);
    let start_ts = config.time_start_timestamp_micros();
    // start_ts not confirmed yet for this connection — adopt without wiping soft-preserved store.
    if series.session.start_ts.is_none() {
        *series.session = SeriesStoreSession::identity(addr, start_ts);
        return;
    }
    if series.session.matches(addr, start_ts) {
        return;
    }
    cancel_in_flight_series_requests(
        &mut editor_ui.commands,
        &mut series.msg_handlers,
        &mut series.req_handlers,
        &mut series.packet_id_handlers,
        &mut series.visible_prefetch,
    );
    hard_clear_series_store(
        &mut series.telemetry_cache,
        &mut series.backfill_state,
        &mut series.series_load,
        &mut series.plot_sync,
        &mut series.visible_prefetch,
    );
    editor_ui.clear_ui_and_invalidate_schematic_sync();
    // Stale playhead from the previous recording must not survive identity change.
    current.0 = Timestamp::EPOCH;
    *series.session = SeriesStoreSession::identity(addr, start_ts);
}

#[allow(clippy::too_many_arguments)]
fn clear_state_new_connection(
    PacketHandlerInput { packet, .. }: PacketHandlerInput,
    mut entity_map: ResMut<EntityMap>,
    mut value_map: Query<&mut ComponentValueMap>,
    mut eql_context: ResMut<EqlContext>,
    mut component_time_ranges: ResMut<ui::data_overview::ComponentTimeRanges>,
    mut series: SeriesStoreReconnect,
    mut editor_ui: EditorUiHardClear,
) {
    match packet {
        OwnedPacket::Msg(m) if m.id == NewConnection::ID => {}
        _ => return,
    }

    // SeriesStore ops run before any UI early-return so a missing primary
    // window cannot leave handlers/cache in a half-torn-down state.
    cancel_in_flight_series_requests(
        &mut editor_ui.commands,
        &mut series.msg_handlers,
        &mut series.req_handlers,
        &mut series.packet_id_handlers,
        &mut series.visible_prefetch,
    );

    let addr = series.connection_addr.as_ref().map(|a| a.0);
    let soft = series_store_soft_reconnect(&series.session, addr);
    apply_series_store_on_reconnect(
        soft,
        addr,
        &mut series.session,
        &mut series.telemetry_cache,
        &mut series.backfill_state,
        &mut series.series_load,
        &mut series.plot_sync,
        &mut series.visible_prefetch,
    );

    eql_context.0.component_parts.clear();
    // Clear cached component time ranges so they will be re-queried
    component_time_ranges.ranges.clear();
    component_time_ranges.row_counts.clear();
    component_time_ranges.sparklines.clear();
    component_time_ranges.tables_to_query.clear();
    component_time_ranges.row_settings.clear();
    component_time_ranges.pending_queries = 0;
    component_time_ranges.total_queries = 0;
    component_time_ranges.completed_queries = 0;
    component_time_ranges.current_batch = 0;
    component_time_ranges.state = ui::data_overview::TimeRangeQueryState::NotStarted;
    entity_map.0.retain(|_, entity| {
        if let Ok(mut entity_commands) = editor_ui.commands.get_entity(*entity) {
            entity_commands.despawn();
        }
        false
    });
    value_map.iter_mut().for_each(|mut map| {
        map.0.clear();
    });
    // Object3D visuals are freestanding (not children of impeller sources).
    // Always despawn them when EntityMap is wiped so sync_object_3d does not
    // leave duplicates after sources respawn.
    despawn_synced_object_3d(&mut editor_ui.commands, &mut editor_ui.synced_glbs);

    // Soft: keep tiles, LineHandles, CollectedGraphData, LastSynced*.
    // Hard: wipe UI and force schematic re-sync on next DbConfig.
    if should_hard_clear_ui(soft) {
        editor_ui.clear_ui_and_invalidate_schematic_sync();
    }
}

/// Reset earliest/latest for rehydrate; soft reconnect preserves the playhead.
fn apply_timestamp_bounds_reset_on_reconnect(
    soft: bool,
    earliest: &mut EarliestTimestamp,
    latest: &mut LastUpdated,
    current: &mut CurrentTimestamp,
) {
    *earliest = EarliestTimestamp(Timestamp(i64::MAX));
    *latest = LastUpdated(Timestamp(i64::MIN));
    if !soft {
        current.0 = Timestamp::EPOCH;
    }
}

/// Reset timestamp resources so the next connection initializes correctly.
/// Registered as a packet handler alongside `clear_state_new_connection`.
fn reset_timestamps_on_new_connection(
    PacketHandlerInput { packet, .. }: PacketHandlerInput,
    mut earliest: ResMut<EarliestTimestamp>,
    mut latest: ResMut<LastUpdated>,
    mut current: ResMut<CurrentTimestamp>,
    connection_addr: Option<Res<ConnectionAddr>>,
    session: Res<SeriesStoreSession>,
) {
    match packet {
        OwnedPacket::Msg(m) if m.id == NewConnection::ID => {}
        _ => return,
    }
    let addr = connection_addr.as_ref().map(|a| a.0);
    let soft = series_store_soft_reconnect(&session, addr);
    apply_timestamp_bounds_reset_on_reconnect(soft, &mut earliest, &mut latest, &mut current);
}

#[derive(Resource, Clone)]
pub struct SelectedTimeRange(pub Range<Timestamp>);
impl Default for SelectedTimeRange {
    fn default() -> Self {
        Self(Timestamp(i64::MIN)..Timestamp(i64::MAX))
    }
}

/// Quantized window used for plot fetch / EQL / request dedup.
/// Display camera uses continuous [`SelectedTimeRange`] so trailing windows don't lurch at 10 Hz.
#[derive(Resource, Clone)]
pub struct FetchTimeRange(pub Range<Timestamp>);
impl Default for FetchTimeRange {
    fn default() -> Self {
        Self(Timestamp(i64::MIN)..Timestamp(i64::MAX))
    }
}

#[derive(Resource, Clone)]
pub struct FullTimeRange(pub Range<Timestamp>);

/// When present, the Editor operates in replay mode: the timeline reveals data
/// progressively as `CurrentTimestamp` advances, simulating a live session from
/// a recorded database.
#[derive(Resource, Default)]
pub struct ReplayMode;

/// Quantize step for trailing `LAST_*` **fetch** windows (not display camera).
pub(crate) const TRAILING_RANGE_QUANTUM_MICROS: i64 = 100_000; // 100 ms

/// Short windows (5 / 15 / 30 s presets): prefer full-fidelity GPU stride and Y hysteresis.
pub(crate) const SHORT_WINDOW_ACCURACY_MICROS: i64 = 30_000_000; // 30 s

pub(crate) fn is_short_accuracy_window(range: &Range<Timestamp>) -> bool {
    range.end.0.saturating_sub(range.start.0).max(0) <= SHORT_WINDOW_ACCURACY_MICROS
}

pub(crate) fn floor_timestamp_quantum(ts: Timestamp, quantum_micros: i64) -> Timestamp {
    if quantum_micros <= 0 {
        return ts;
    }
    Timestamp(
        ts.0.div_euclid(quantum_micros)
            .saturating_mul(quantum_micros),
    )
}

pub(crate) fn ceil_timestamp_quantum(ts: Timestamp, quantum_micros: i64) -> Timestamp {
    if quantum_micros <= 0 {
        return ts;
    }
    let q = quantum_micros;
    let floored = ts.0.div_euclid(q).saturating_mul(q);
    if floored == ts.0 {
        ts
    } else {
        Timestamp(floored.saturating_add(q))
    }
}

/// Snap a range onto the GPU/visible-range quantum grid (`end > start`).
pub(crate) fn quantize_visible_range(
    range: Range<Timestamp>,
    quantum_micros: i64,
) -> Range<Timestamp> {
    let start = floor_timestamp_quantum(range.start, quantum_micros);
    let end = ceil_timestamp_quantum(range.end, quantum_micros);
    if end <= start {
        start..Timestamp(start.0.saturating_add(quantum_micros.max(1)))
    } else {
        start..end
    }
}

/// Snap a trailing selected window onto the quantum grid while keeping `end > start`.
pub(crate) fn quantize_trailing_range(
    range: Range<Timestamp>,
    quantum_micros: i64,
) -> Range<Timestamp> {
    let mut start = floor_timestamp_quantum(range.start, quantum_micros);
    let mut end = ceil_timestamp_quantum(range.end, quantum_micros);
    if end <= start {
        end = Timestamp(start.0.saturating_add(quantum_micros.max(1)));
    }
    // Prefer not extending past the unquantized end by more than one quantum when
    // the raw window was already valid.
    if end.0 > range.end.0.saturating_add(quantum_micros) {
        end = ceil_timestamp_quantum(range.end, quantum_micros);
    }
    if end <= start {
        start = Timestamp(end.0.saturating_sub(quantum_micros.max(1)));
    }
    start..end
}

/// Resolve timeline-bar (`FullTimeRange`) and plot-window (`SelectedTimeRange`)
/// anchors.
///
/// - Non-replay: the timeline bar spans the full DB (`earliest..LastUpdated`).
/// - Replay: both bar and non-trailing selection progressive-reveal to the playhead.
/// - Trailing `LAST_*` presets always end at `min(LastUpdated, CurrentTimestamp)`
///   so recordings without `--replay` still window relative to the playhead.
///
/// Display selected range is **continuous** (no fetch quantum). Callers that need
/// fetch/EQL stability should run [`quantize_trailing_range`] for trailing windows.
fn resolve_time_range_anchors(
    earliest: Timestamp,
    db_latest: Timestamp,
    current: Timestamp,
    behavior: TimeRangeBehavior,
    replay: bool,
) -> (Range<Timestamp>, Result<Range<Timestamp>, TimeRangeError>) {
    let playhead_capped = db_latest.min(current);
    let full_latest = if replay { playhead_capped } else { db_latest };

    let full_range = if earliest < full_latest {
        earliest..full_latest
    } else if earliest < db_latest {
        earliest..db_latest
    } else {
        // No usable span yet; caller keeps the previous FullTimeRange.
        earliest..earliest
    };

    let selected_latest = if behavior.is_trailing_window() {
        playhead_capped
    } else {
        full_latest
    };
    let selected = behavior.calculate_selected_range(earliest, selected_latest);
    (full_range, selected)
}

#[allow(clippy::too_many_arguments)]
pub fn set_selected_range(
    mut selected_range: ResMut<SelectedTimeRange>,
    mut fetch_range: ResMut<FetchTimeRange>,
    mut full_range: ResMut<FullTimeRange>,
    earliest: Res<EarliestTimestamp>,
    latest: Res<LastUpdated>,
    current_ts: Res<CurrentTimestamp>,
    behavior: Res<TimeRangeBehavior>,
    replay: Option<Res<ReplayMode>>,
) {
    let (resolved_full, selected) = resolve_time_range_anchors(
        earliest.0,
        latest.0,
        current_ts.0,
        *behavior,
        replay.is_some(),
    );

    if resolved_full.start < resolved_full.end && full_range.0 != resolved_full {
        full_range.0 = resolved_full;
    }

    match selected {
        Ok(range) => {
            if selected_range.0 != range {
                selected_range.0 = range.clone();
            }
            let fetch = if behavior.is_trailing_window() {
                quantize_trailing_range(range, TRAILING_RANGE_QUANTUM_MICROS)
            } else {
                range
            };
            if fetch_range.0 != fetch {
                fetch_range.0 = fetch;
            }
        }
        Err(TimeRangeError::NoData) => {
            if selected_range.0.start.0 == i64::MIN || selected_range.0.end.0 == i64::MAX {
                selected_range.0 = Timestamp(0)..Timestamp(1_000_000);
                fetch_range.0 = selected_range.0.clone();
            }
        }
        Err(TimeRangeError::InvalidRange { start, end }) => {
            bevy::log::warn!(
                "Time range selection skipped because start ({start:?}) is not before end ({end:?})"
            );
        }
    }
}

#[derive(Resource, PartialEq, Eq, Clone, Copy, Debug)]
pub struct TimeRangeBehavior {
    start: Offset,
    end: Offset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeRangeError {
    NoData,
    InvalidRange { start: Timestamp, end: Timestamp },
}

#[derive(Resource, PartialEq, Eq, Clone, Copy, Debug)]

enum Offset {
    Earliest(Duration),
    Latest(Duration),
    Fixed(Timestamp),
}

impl std::fmt::Display for Offset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Offset::Earliest(duration) => {
                let d = hifitime::Duration::from(*duration)
                    .to_string()
                    .to_uppercase();
                write!(f, "+{d}")
            }
            Offset::Latest(duration) => {
                let d = hifitime::Duration::from(*duration)
                    .to_string()
                    .to_uppercase();
                write!(f, "-{d}")
            }
            Offset::Fixed(timestamp) => {
                let timestamp = FriendlyEpoch(hifitime::Epoch::from(*timestamp));
                write!(f, "{timestamp}")
            }
        }
    }
}

impl std::fmt::Display for TimeRangeBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.start, self.end) {
            (Offset::Earliest(start), Offset::Latest(end)) if end.is_zero() && start.is_zero() => {
                write!(f, "FULL RANGE")
            }
            (Offset::Latest(start), Offset::Latest(end)) if end.is_zero() => {
                let start = hifitime::Duration::from(start).to_string().to_uppercase();
                write!(f, "LAST {start}")
            }
            (start, end) => {
                write!(f, "{start} ↔ {end}")
            }
        }
    }
}

impl Default for TimeRangeBehavior {
    fn default() -> Self {
        Self::FULL
    }
}

impl TimeRangeBehavior {
    const FULL: Self = TimeRangeBehavior {
        start: Offset::Earliest(Duration::ZERO),
        end: Offset::Latest(Duration::ZERO),
    };
    const LAST_5S: Self = Self::last(Duration::from_secs(5));
    const LAST_15S: Self = Self::last(Duration::from_secs(15));
    const LAST_30S: Self = Self::last(Duration::from_secs(30));
    const LAST_1M: Self = Self::last(Duration::from_secs(60));
    const LAST_5M: Self = Self::last(Duration::from_secs(60 * 5));
    const LAST_15M: Self = Self::last(Duration::from_secs(60 * 15));
    const LAST_30M: Self = Self::last(Duration::from_secs(60 * 30));
    const LAST_1H: Self = Self::last(Duration::from_secs(60 * 60));
    const LAST_6H: Self = Self::last(Duration::from_secs(60 * 60 * 6));
    const LAST_12H: Self = Self::last(Duration::from_secs(60 * 60 * 12));
    const LAST_24H: Self = Self::last(Duration::from_secs(60 * 60 * 24));

    pub const fn last(duration: Duration) -> Self {
        TimeRangeBehavior {
            start: Offset::Latest(duration),
            end: Offset::Latest(Duration::ZERO),
        }
    }

    /// `LAST_*` presets: both ends are [`Offset::Latest`] (e.g. last 5s →
    /// `Latest(5s)..Latest(0)`). Trailing windows follow the playhead.
    pub(crate) const fn is_trailing_window(self) -> bool {
        matches!(
            (self.start, self.end),
            (Offset::Latest(_), Offset::Latest(_))
        )
    }

    /// Parse a schematic `timeline range=...` value into a time-window behavior.
    pub fn from_schematic_range(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "" | "full" | "full_range" | "fullrange" => Some(Self::FULL),
            "last_5s" | "5s" => Some(Self::LAST_5S),
            "last_15s" | "15s" => Some(Self::LAST_15S),
            "last_30s" | "30s" => Some(Self::LAST_30S),
            "last_1m" | "1m" | "last_60s" | "60s" => Some(Self::LAST_1M),
            "last_5m" | "5m" => Some(Self::LAST_5M),
            "last_15m" | "15m" => Some(Self::LAST_15M),
            "last_30m" | "30m" => Some(Self::LAST_30M),
            "last_1h" | "1h" => Some(Self::LAST_1H),
            "last_6h" | "6h" => Some(Self::LAST_6H),
            "last_12h" | "12h" => Some(Self::LAST_12H),
            "last_24h" | "24h" => Some(Self::LAST_24H),
            other => {
                let secs = other
                    .strip_prefix("last_")
                    .unwrap_or(other)
                    .strip_suffix('s')
                    .and_then(|n| n.parse::<u64>().ok())?;
                Some(Self::last(Duration::from_secs(secs)))
            }
        }
    }

    /// Serialize the current behavior for `timeline range=...` when non-default.
    pub fn to_schematic_range(self) -> Option<String> {
        match (self.start, self.end) {
            (Offset::Earliest(start), Offset::Latest(end)) if start.is_zero() && end.is_zero() => {
                None
            }
            (Offset::Latest(start), Offset::Latest(end)) if end.is_zero() => {
                let secs = start.as_secs();
                Some(match secs {
                    5 => "last_5s".to_string(),
                    15 => "last_15s".to_string(),
                    30 => "last_30s".to_string(),
                    60 => "last_1m".to_string(),
                    300 => "last_5m".to_string(),
                    900 => "last_15m".to_string(),
                    1800 => "last_30m".to_string(),
                    3600 => "last_1h".to_string(),
                    21600 => "last_6h".to_string(),
                    43200 => "last_12h".to_string(),
                    86400 => "last_24h".to_string(),
                    other => format!("last_{other}s"),
                })
            }
            _ => None,
        }
    }

    fn calculate_selected_range(
        &self,
        earliest: Timestamp,
        latest: Timestamp,
    ) -> Result<Range<Timestamp>, TimeRangeError> {
        if earliest >= latest {
            return Err(TimeRangeError::NoData);
        }

        let start = self.start.resolve(earliest, latest);
        let end = self.end.resolve(earliest, latest);

        if start >= end {
            return Err(TimeRangeError::InvalidRange { start, end });
        }

        let clamped = clamp_range(earliest..latest, start..end);
        if clamped.start >= clamped.end {
            return Err(TimeRangeError::InvalidRange {
                start: clamped.start,
                end: clamped.end,
            });
        }

        Ok(clamped)
    }

    pub fn is_subset(&self, earliest: Timestamp, latest: Timestamp) -> bool {
        let start = self.start.resolve(earliest, latest);
        let end = self.end.resolve(earliest, latest);
        let full_range = earliest..=latest;
        full_range.contains(&start) && full_range.contains(&end)
    }
}

impl Offset {
    pub(crate) fn resolve(&self, earliest: Timestamp, latest: Timestamp) -> Timestamp {
        match self {
            Offset::Earliest(duration) => earliest + *duration,
            Offset::Latest(duration) => latest - *duration,
            Offset::Fixed(timestamp) => *timestamp,
        }
    }
}

#[cfg(test)]
mod time_range_tests {
    use super::*;

    fn timestamp_secs(sec: i64) -> Timestamp {
        Timestamp(sec * 1_000_000)
    }

    #[test]
    fn calculate_selected_range_accepts_valid_bounds() {
        let behavior = TimeRangeBehavior::FULL;
        let earliest = timestamp_secs(0);
        let latest = timestamp_secs(10);

        let result = behavior.calculate_selected_range(earliest, latest);
        assert!(matches!(result, Ok(ref range) if range.start == earliest && range.end == latest));
    }

    #[test]
    fn calculate_selected_range_rejects_inverted_bounds() {
        let behavior = TimeRangeBehavior {
            start: Offset::Earliest(Duration::from_secs(10)),
            end: Offset::Latest(Duration::ZERO),
        };
        let earliest = timestamp_secs(0);
        let latest = timestamp_secs(5);

        let result = behavior.calculate_selected_range(earliest, latest);
        assert!(matches!(result, Err(TimeRangeError::InvalidRange { .. })));
    }

    #[test]
    fn calculate_selected_range_handles_no_data() {
        let behavior = TimeRangeBehavior::FULL;
        let earliest = timestamp_secs(0);
        let latest = timestamp_secs(0);

        let result = behavior.calculate_selected_range(earliest, latest);
        assert!(matches!(result, Err(TimeRangeError::NoData)));
    }

    #[test]
    fn last_30s_window_at_mid_playhead_position() {
        let behavior = TimeRangeBehavior::last(Duration::from_secs(30));
        let earliest = timestamp_secs(0);
        let playhead = timestamp_secs(40);

        let range = behavior
            .calculate_selected_range(earliest, playhead)
            .expect("valid trailing window");
        assert_eq!(range.start, timestamp_secs(10));
        assert_eq!(range.end, timestamp_secs(40));
    }

    #[test]
    fn last_30s_window_at_early_playhead_clamps_to_earliest() {
        let behavior = TimeRangeBehavior::last(Duration::from_secs(30));
        let earliest = timestamp_secs(0);
        let playhead = timestamp_secs(15);

        let range = behavior
            .calculate_selected_range(earliest, playhead)
            .expect("valid trailing window");
        assert_eq!(range.start, timestamp_secs(0));
        assert_eq!(range.end, timestamp_secs(15));
    }

    #[test]
    fn full_range_with_capped_latest_uses_that_latest() {
        let behavior = TimeRangeBehavior::FULL;
        let earliest = timestamp_secs(0);
        let capped_latest = timestamp_secs(20);

        let range = behavior
            .calculate_selected_range(earliest, capped_latest)
            .expect("valid full window");
        assert_eq!(range.start, earliest);
        assert_eq!(range.end, capped_latest);
    }

    #[test]
    fn non_replay_last_5s_mid_playhead_keeps_full_timeline() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let playhead = timestamp_secs(40);
        let (full, selected) = resolve_time_range_anchors(
            earliest,
            db_latest,
            playhead,
            TimeRangeBehavior::LAST_5S,
            false,
        );
        assert_eq!(full, earliest..db_latest);
        let selected = selected.expect("valid last_5s window");
        assert_eq!(selected.start, timestamp_secs(35));
        assert_eq!(selected.end, playhead);
    }

    #[test]
    fn non_replay_last_5s_early_playhead_clamps_to_earliest() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let playhead = timestamp_secs(3);
        let (full, selected) = resolve_time_range_anchors(
            earliest,
            db_latest,
            playhead,
            TimeRangeBehavior::LAST_5S,
            false,
        );
        assert_eq!(full, earliest..db_latest);
        let selected = selected.expect("valid last_5s window");
        assert_eq!(selected.start, earliest);
        assert_eq!(selected.end, playhead);
    }

    #[test]
    fn non_replay_full_mid_playhead_keeps_full_db_selection() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let playhead = timestamp_secs(40);
        let (full, selected) = resolve_time_range_anchors(
            earliest,
            db_latest,
            playhead,
            TimeRangeBehavior::FULL,
            false,
        );
        assert_eq!(full, earliest..db_latest);
        let selected = selected.expect("valid full window");
        assert_eq!(selected.start, earliest);
        assert_eq!(selected.end, db_latest);
    }

    #[test]
    fn non_replay_last_5s_live_like_playhead_at_db_end() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let (full, selected) = resolve_time_range_anchors(
            earliest,
            db_latest,
            db_latest,
            TimeRangeBehavior::LAST_5S,
            false,
        );
        assert_eq!(full, earliest..db_latest);
        let selected = selected.expect("valid last_5s window");
        assert_eq!(selected.start, timestamp_secs(95));
        assert_eq!(selected.end, db_latest);
    }

    #[test]
    fn replay_full_progressive_reveals_both_ranges() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let playhead = timestamp_secs(20);
        let (full, selected) = resolve_time_range_anchors(
            earliest,
            db_latest,
            playhead,
            TimeRangeBehavior::FULL,
            true,
        );
        assert_eq!(full, earliest..playhead);
        let selected = selected.expect("valid replay full window");
        assert_eq!(selected.start, earliest);
        assert_eq!(selected.end, playhead);
    }

    #[test]
    fn is_trailing_window_detects_last_presets_only() {
        assert!(TimeRangeBehavior::LAST_5S.is_trailing_window());
        assert!(TimeRangeBehavior::last(Duration::from_secs(7)).is_trailing_window());
        assert!(!TimeRangeBehavior::FULL.is_trailing_window());
    }

    #[test]
    fn trailing_display_range_tracks_playhead_continuously() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let (_, a) = resolve_time_range_anchors(
            earliest,
            db_latest,
            Timestamp(40_050_000),
            TimeRangeBehavior::LAST_5S,
            false,
        );
        let (_, b) = resolve_time_range_anchors(
            earliest,
            db_latest,
            Timestamp(40_090_000),
            TimeRangeBehavior::LAST_5S,
            false,
        );
        let a = a.expect("a");
        let b = b.expect("b");
        // Display window follows playhead without 100 ms quantization.
        assert_ne!(a.end, b.end);
        assert_eq!(b.end.0 - a.end.0, 40_000);
    }

    #[test]
    fn trailing_fetch_quantize_stable_within_100ms() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let (_, a) = resolve_time_range_anchors(
            earliest,
            db_latest,
            Timestamp(40_050_000),
            TimeRangeBehavior::LAST_5S,
            false,
        );
        let (_, b) = resolve_time_range_anchors(
            earliest,
            db_latest,
            Timestamp(40_090_000),
            TimeRangeBehavior::LAST_5S,
            false,
        );
        let fa = quantize_trailing_range(a.expect("a"), TRAILING_RANGE_QUANTUM_MICROS);
        let fb = quantize_trailing_range(b.expect("b"), TRAILING_RANGE_QUANTUM_MICROS);
        assert_eq!(fa, fb);
    }

    #[test]
    fn trailing_fetch_quantize_advances_across_100ms() {
        let earliest = timestamp_secs(0);
        let db_latest = timestamp_secs(100);
        let (_, a) = resolve_time_range_anchors(
            earliest,
            db_latest,
            Timestamp(40_000_000),
            TimeRangeBehavior::LAST_5S,
            false,
        );
        let (_, b) = resolve_time_range_anchors(
            earliest,
            db_latest,
            Timestamp(40_100_000),
            TimeRangeBehavior::LAST_5S,
            false,
        );
        let fa = quantize_trailing_range(a.expect("a"), TRAILING_RANGE_QUANTUM_MICROS);
        let fb = quantize_trailing_range(b.expect("b"), TRAILING_RANGE_QUANTUM_MICROS);
        assert_ne!(fa.end, fb.end);
        assert_eq!(fb.end.0 - fa.end.0, TRAILING_RANGE_QUANTUM_MICROS);
    }

    #[test]
    fn is_short_accuracy_window_for_last_presets() {
        assert!(is_short_accuracy_window(
            &(Timestamp(0)..Timestamp(5_000_000))
        ));
        assert!(is_short_accuracy_window(
            &(Timestamp(0)..Timestamp(30_000_000))
        ));
        assert!(!is_short_accuracy_window(
            &(Timestamp(0)..Timestamp(30_000_001))
        ));
    }

    #[test]
    fn quantize_trailing_range_keeps_end_after_start() {
        let range = quantize_trailing_range(
            Timestamp(100)..Timestamp(150),
            TRAILING_RANGE_QUANTUM_MICROS,
        );
        assert!(range.end > range.start);
    }

    #[test]
    fn quantize_visible_range_stable_within_quantum() {
        let a = quantize_visible_range(
            Timestamp(1_050_000)..Timestamp(6_050_000),
            TRAILING_RANGE_QUANTUM_MICROS,
        );
        let b = quantize_visible_range(
            Timestamp(1_099_000)..Timestamp(6_099_000),
            TRAILING_RANGE_QUANTUM_MICROS,
        );
        assert_eq!(a, b);
        assert_eq!(a.start.0 % TRAILING_RANGE_QUANTUM_MICROS, 0);
        assert_eq!(a.end.0 % TRAILING_RANGE_QUANTUM_MICROS, 0);
    }

    #[test]
    fn quantize_visible_range_advances_across_quantum() {
        let a = quantize_visible_range(
            Timestamp(1_000_000)..Timestamp(6_000_000),
            TRAILING_RANGE_QUANTUM_MICROS,
        );
        let b = quantize_visible_range(
            Timestamp(1_100_000)..Timestamp(6_100_000),
            TRAILING_RANGE_QUANTUM_MICROS,
        );
        assert_ne!(a, b);
        assert_eq!(b.start.0 - a.start.0, TRAILING_RANGE_QUANTUM_MICROS);
    }
}

fn clamp_range(total_range: Range<Timestamp>, b: Range<Timestamp>) -> Range<Timestamp> {
    let start = total_range.start.max(b.start);
    let end = total_range.end.min(b.end);
    start..end
}

pub fn clamp_current_time(
    earliest: Res<EarliestTimestamp>,
    latest: Res<LastUpdated>,
    mut current_timestamp: ResMut<CurrentTimestamp>,
) {
    if earliest.0 >= latest.0 {
        return;
    }
    let previous_timestamp = current_timestamp.0;
    let new_timestamp = previous_timestamp.clamp(earliest.0, latest.0);
    if new_timestamp != previous_timestamp {
        current_timestamp.0 = new_timestamp;
    }
}

#[derive(Default, Resource)]
pub struct EqlContext(pub eql::Context);

pub fn update_eql_context(
    component_metadata_registry: Res<ComponentMetadataRegistry>,
    component_schema_registry: Res<ComponentSchemaRegistry>,
    path_reg: Res<ComponentPathRegistry>,
    mut eql_context: ResMut<EqlContext>,
) {
    if path_reg.0.is_empty() {
        return;
    }
    // Rebuild only when the registries that feed the context change. Blindly
    // assigning every frame marks `EqlContext` dirty and forces downstream
    // recompiles (viewports / object_3d) for no reason.
    if !path_reg.is_changed()
        && !component_metadata_registry.is_changed()
        && !component_schema_registry.is_changed()
        && !eql_context.0.component_parts.is_empty()
    {
        return;
    }
    let earliest = eql_context.0.earliest_timestamp;
    let latest = eql_context.0.last_timestamp;
    eql_context.0 = eql::Context::from_leaves(
        path_reg.0.iter().filter_map(|(id, path)| {
            let schema = component_schema_registry.0.get(id)?;
            let metadata = component_metadata_registry.0.get(id)?;

            // Exclude timestamp source components from the EQL context.
            if metadata.is_timestamp_source() {
                return None;
            }

            let mut component = eql::Component::new(metadata.name.clone(), path.id, schema.clone());
            if !metadata.element_names().is_empty() {
                component.element_names = metadata
                    .element_names()
                    .split(",")
                    .map(str::to_string)
                    .collect();
            }
            Some(Arc::new(component))
        }),
        earliest,
        latest,
    );
}

pub fn set_eql_context_range(fetch_range: Res<FetchTimeRange>, mut eql: ResMut<EqlContext>) {
    if eql.0.earliest_timestamp == fetch_range.0.start && eql.0.last_timestamp == fetch_range.0.end
    {
        return;
    }
    eql.0.earliest_timestamp = fetch_range.0.start;
    eql.0.last_timestamp = fetch_range.0.end;
}

pub fn dirs() -> directories::ProjectDirs {
    directories::ProjectDirs::from("systems", "elodin", "editor").unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::app::App;
    use bevy::math::{DQuat, EulerRot};
    use impeller2::types::{ComponentId, Timestamp};
    use impeller2_wkt::ComponentValue;

    /// Inserting a bare `WorldPos` must pull in the `Geo*` components
    /// (required components) and end up at the same Bevy pose the old
    /// `sync_pos` fallback (`bevy_pos()`/`bevy_att()`) produced.
    #[test]
    fn world_pos_geo_pipeline_matches_legacy_swizzle() {
        let ctx = GeoContext::default();
        let mut app = App::new();
        app.insert_resource(ctx.clone());
        register_world_pos_components(&mut app);
        app.add_systems(bevy::app::Update, sync_pos);

        let att = DQuat::from_euler(EulerRot::XYZ, 0.3, -0.8, 1.1);
        let world_pos = WorldPos {
            att: nox::Quaternion::new(att.w, att.x, att.y, att.z),
            pos: nox::Vector3::new(1.0, 2.0, 3.0),
        };
        let entity = app.world_mut().spawn(world_pos).id();
        app.update();

        let world = app.world();
        let wp = world.get::<WorldPos>(entity).unwrap();
        let geo_pos = world.get::<GeoPosition>(entity).unwrap();
        let geo_rot = world.get::<GeoRotation>(entity).unwrap();
        assert!(world.get::<Transform>(entity).is_some());

        let pos = geo_pos.to_bevy(&ctx);
        assert!(
            (pos - wp.bevy_pos()).length() < 1e-9,
            "got {pos:?}, expected {:?}",
            wp.bevy_pos()
        );
        let q = geo_rot.to_bevy(&ctx);
        assert!(
            q.dot(wp.bevy_att()).abs() > 1.0 - 1e-9,
            "got {q:?}, expected {:?}",
            wp.bevy_att()
        );
    }

    fn sample_addr(port: u16) -> std::net::SocketAddr {
        std::net::SocketAddr::from(([127, 0, 0, 1], port))
    }

    fn populated_cache() -> impeller2_bevy::TelemetryCache {
        let mut cache = impeller2_bevy::TelemetryCache::default();
        cache.insert(
            ComponentId(42),
            Timestamp(1_000),
            ComponentValue::F64(nox::array![1.0f64].to_dyn()),
        );
        cache
    }

    #[test]
    fn soft_reconnect_preserves_telemetry_cache() {
        let addr = sample_addr(2240);
        let mut session = SeriesStoreSession::identity(Some(addr), Some(100));
        let mut cache = populated_cache();
        let gen_before = cache.generation();
        let mut backfill = impeller2_bevy::BackfillState::default();
        let mut series_load = impeller2_bevy::SeriesStoreLoadState {
            components_started: 3,
            components_complete: 3,
            samples_loaded: 10,
            complete: true,
        };
        let mut plot_sync = crate::ui::plot::data::PlotSyncState::default();
        let mut prefetch = crate::ui::plot::data::VisiblePrefetchState::default();

        assert!(series_store_soft_reconnect(&session, Some(addr)));
        apply_series_store_on_reconnect(
            true,
            Some(addr),
            &mut session,
            &mut cache,
            &mut backfill,
            &mut series_load,
            &mut plot_sync,
            &mut prefetch,
        );

        assert_eq!(cache.generation(), gen_before);
        assert!(cache.has_series(&ComponentId(42)));
        assert!(!series_load.complete);
        assert_eq!(series_load.samples_loaded, 10);
        assert_eq!(session.addr, Some(addr));
        assert_eq!(session.start_ts, Some(100));
    }

    #[test]
    fn soft_reconnect_cancels_handlers_and_prefetch() {
        use bevy::ecs::system::{InRef, RunSystemOnce};
        use impeller2::types::PacketId;
        use impeller2_bevy::PacketGrantR;

        let mut app = App::new();
        app.init_resource::<MsgRequestIdHandlers>()
            .init_resource::<RequestIdHandlers>()
            .init_resource::<PacketIdHandlers>()
            .init_resource::<crate::ui::plot::data::VisiblePrefetchState>();

        let msg_sys = app
            .world_mut()
            .register_system(|_: InRef<OwnedPacket<PacketGrantR>>| false);
        let req_sys = app
            .world_mut()
            .register_system(|_: InRef<OwnedPacket<PacketGrantR>>| false);
        let pkt_sys = app
            .world_mut()
            .register_system(|_: InRef<OwnedPacket<PacketGrantR>>| {});
        let packet_id: PacketId = 42u16.to_le_bytes();
        {
            let mut msg = app.world_mut().resource_mut::<MsgRequestIdHandlers>();
            msg.0.insert(7, msg_sys);
            let mut req = app.world_mut().resource_mut::<RequestIdHandlers>();
            req.0.insert(9, req_sys);
            let mut pkt = app.world_mut().resource_mut::<PacketIdHandlers>();
            pkt.0.insert(packet_id, pkt_sys);
            app.world_mut()
                .resource_mut::<crate::ui::plot::data::VisiblePrefetchState>()
                .in_flight
                .insert((ComponentId(1), 0, 1));
        }

        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 mut msg: ResMut<MsgRequestIdHandlers>,
                 mut req: ResMut<RequestIdHandlers>,
                 mut pkt: ResMut<PacketIdHandlers>,
                 mut prefetch: ResMut<crate::ui::plot::data::VisiblePrefetchState>| {
                    cancel_in_flight_series_requests(
                        &mut commands,
                        &mut msg,
                        &mut req,
                        &mut pkt,
                        &mut prefetch,
                    );
                },
            )
            .unwrap();

        assert!(app.world().resource::<MsgRequestIdHandlers>().0.is_empty());
        assert!(app.world().resource::<RequestIdHandlers>().0.is_empty());
        assert!(app.world().resource::<PacketIdHandlers>().0.is_empty());
        assert!(
            app.world()
                .resource::<crate::ui::plot::data::VisiblePrefetchState>()
                .in_flight
                .is_empty()
        );
        // Systems were unregistered — a second unregister must fail.
        assert!(app.world_mut().unregister_system(msg_sys).is_err());
        assert!(app.world_mut().unregister_system(req_sys).is_err());
        assert!(app.world_mut().unregister_system(pkt_sys).is_err());
    }

    #[test]
    fn despawn_synced_object_3d_removes_visuals() {
        use bevy::ecs::system::RunSystemOnce;

        let mut app = App::new();
        app.init_resource::<SyncedObject3d>();
        let visual = app.world_mut().spawn_empty().id();
        let source = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<SyncedObject3d>()
            .0
            .insert(source, visual);

        app.world_mut()
            .run_system_once(
                |mut commands: Commands, mut synced: ResMut<SyncedObject3d>| {
                    despawn_synced_object_3d(&mut commands, &mut synced);
                },
            )
            .unwrap();

        assert!(app.world().resource::<SyncedObject3d>().0.is_empty());
        assert!(app.world().get_entity(visual).is_err());
    }

    #[test]
    fn series_store_policy_runs_even_when_primary_window_missing() {
        // Mirrors clear_state_new_connection ordering: SeriesStore soft path
        // completes before the primary-window early return.
        let addr = sample_addr(2240);
        let mut session = SeriesStoreSession::identity(Some(addr), Some(50));
        let mut cache = populated_cache();
        let mut backfill = impeller2_bevy::BackfillState::default();
        let mut series_load = impeller2_bevy::SeriesStoreLoadState::default();
        let mut plot_sync = crate::ui::plot::data::PlotSyncState::default();
        let mut prefetch = crate::ui::plot::data::VisiblePrefetchState::default();
        prefetch.in_flight.insert((ComponentId(1), 0, 1));
        prefetch.clear_in_flight();

        let soft = series_store_soft_reconnect(&session, Some(addr));
        apply_series_store_on_reconnect(
            soft,
            Some(addr),
            &mut session,
            &mut cache,
            &mut backfill,
            &mut series_load,
            &mut plot_sync,
            &mut prefetch,
        );
        let primary_window: Option<()> = None;
        if primary_window.is_none() {
            // early return — SeriesStore work already applied
        }

        assert!(cache.has_series(&ComponentId(42)));
        assert!(prefetch.in_flight.is_empty());
    }

    #[test]
    fn hard_clear_on_addr_change() {
        let old = sample_addr(2240);
        let new = sample_addr(2241);
        let mut session = SeriesStoreSession::identity(Some(old), Some(100));
        let mut cache = populated_cache();
        let mut backfill = impeller2_bevy::BackfillState::default();
        let mut series_load = impeller2_bevy::SeriesStoreLoadState {
            samples_loaded: 99,
            complete: true,
            ..Default::default()
        };
        let mut plot_sync = crate::ui::plot::data::PlotSyncState::default();
        let mut prefetch = crate::ui::plot::data::VisiblePrefetchState::default();

        assert!(!series_store_soft_reconnect(&session, Some(new)));
        apply_series_store_on_reconnect(
            false,
            Some(new),
            &mut session,
            &mut cache,
            &mut backfill,
            &mut series_load,
            &mut plot_sync,
            &mut prefetch,
        );

        assert!(!cache.has_series(&ComponentId(42)));
        assert_eq!(series_load.samples_loaded, 0);
        assert_eq!(session, SeriesStoreSession::identity(Some(new), None));
    }

    fn sync_app(addr: std::net::SocketAddr) -> App {
        let mut app = App::new();
        app.init_resource::<impeller2_wkt::DbConfig>()
            .init_resource::<SeriesStoreSession>()
            .init_resource::<MsgRequestIdHandlers>()
            .init_resource::<RequestIdHandlers>()
            .init_resource::<PacketIdHandlers>()
            .init_resource::<impeller2_bevy::TelemetryCache>()
            .init_resource::<impeller2_bevy::BackfillState>()
            .init_resource::<impeller2_bevy::SeriesStoreLoadState>()
            .init_resource::<crate::ui::plot::data::PlotSyncState>()
            .init_resource::<crate::ui::plot::data::VisiblePrefetchState>()
            .init_resource::<SyncedObject3d>()
            .init_resource::<CollectedGraphData>()
            .init_resource::<plugins::kdl_document::LastSyncedActiveKey>()
            .init_resource::<plugins::kdl_document::LastSyncedAssetsRevision>()
            .insert_resource(CurrentTimestamp(Timestamp(9_000)))
            .insert_resource(ConnectionAddr(addr))
            .add_systems(bevy::app::Update, sync_series_store_session_from_db_config);
        app
    }

    #[test]
    fn soft_timestamp_reset_preserves_playhead() {
        let mut earliest = EarliestTimestamp(Timestamp(100));
        let mut latest = LastUpdated(Timestamp(10_000));
        let mut current = CurrentTimestamp(Timestamp(5_000));
        apply_timestamp_bounds_reset_on_reconnect(true, &mut earliest, &mut latest, &mut current);
        assert_eq!(earliest.0, Timestamp(i64::MAX));
        assert_eq!(latest.0, Timestamp(i64::MIN));
        assert_eq!(current.0, Timestamp(5_000));
    }

    #[test]
    fn hard_timestamp_reset_zeroes_playhead() {
        let mut earliest = EarliestTimestamp(Timestamp(100));
        let mut latest = LastUpdated(Timestamp(10_000));
        let mut current = CurrentTimestamp(Timestamp(5_000));
        apply_timestamp_bounds_reset_on_reconnect(false, &mut earliest, &mut latest, &mut current);
        assert_eq!(earliest.0, Timestamp(i64::MAX));
        assert_eq!(latest.0, Timestamp(i64::MIN));
        assert_eq!(current.0, Timestamp::EPOCH);
    }

    #[test]
    fn hard_clear_when_db_config_identity_changes() {
        use bevy::ecs::system::InRef;
        use impeller2_bevy::PacketGrantR;

        let addr = sample_addr(2240);
        let mut app = sync_app(addr);

        let msg_sys = app
            .world_mut()
            .register_system(|_: InRef<OwnedPacket<PacketGrantR>>| false);
        let req_sys = app
            .world_mut()
            .register_system(|_: InRef<OwnedPacket<PacketGrantR>>| false);
        let pkt_sys = app
            .world_mut()
            .register_system(|_: InRef<OwnedPacket<PacketGrantR>>| {});
        {
            let mut session = app.world_mut().resource_mut::<SeriesStoreSession>();
            *session = SeriesStoreSession::identity(Some(addr), Some(100));
            let mut cache = app
                .world_mut()
                .resource_mut::<impeller2_bevy::TelemetryCache>();
            *cache = populated_cache();
            app.world_mut()
                .resource_mut::<MsgRequestIdHandlers>()
                .0
                .insert(1, msg_sys);
            app.world_mut()
                .resource_mut::<RequestIdHandlers>()
                .0
                .insert(2, req_sys);
            app.world_mut()
                .resource_mut::<PacketIdHandlers>()
                .0
                .insert(3u16.to_le_bytes(), pkt_sys);
            app.world_mut()
                .resource_mut::<crate::ui::plot::data::VisiblePrefetchState>()
                .in_flight
                .insert((ComponentId(1), 0, 1));
            app.world_mut()
                .resource_mut::<plugins::kdl_document::LastSyncedActiveKey>()
                .0 = Some("schematics/main.kdl".into());
            app.world_mut()
                .resource_mut::<plugins::kdl_document::LastSyncedAssetsRevision>()
                .revision = Some(3);
        }

        {
            let mut config = app.world_mut().resource_mut::<impeller2_wkt::DbConfig>();
            config.set_time_start_timestamp_micros(200);
        }
        app.update();

        let cache = app.world().resource::<impeller2_bevy::TelemetryCache>();
        assert!(!cache.has_series(&ComponentId(42)));
        let session = app.world().resource::<SeriesStoreSession>();
        assert_eq!(session.start_ts, Some(200));
        assert!(app.world().resource::<MsgRequestIdHandlers>().0.is_empty());
        assert!(app.world().resource::<RequestIdHandlers>().0.is_empty());
        assert!(app.world().resource::<PacketIdHandlers>().0.is_empty());
        assert!(
            app.world()
                .resource::<crate::ui::plot::data::VisiblePrefetchState>()
                .in_flight
                .is_empty()
        );
        assert!(
            app.world()
                .resource::<plugins::kdl_document::LastSyncedActiveKey>()
                .0
                .is_none()
        );
        assert!(
            app.world()
                .resource::<plugins::kdl_document::LastSyncedAssetsRevision>()
                .revision
                .is_none()
        );
        assert_eq!(
            app.world().resource::<CurrentTimestamp>().0,
            Timestamp::EPOCH
        );
        assert!(app.world_mut().unregister_system(msg_sys).is_err());
        assert!(app.world_mut().unregister_system(req_sys).is_err());
        assert!(app.world_mut().unregister_system(pkt_sys).is_err());
    }

    #[test]
    fn db_config_adopts_start_ts_without_clearing_soft_store() {
        let addr = sample_addr(2240);
        let mut app = sync_app(addr);

        {
            let mut session = app.world_mut().resource_mut::<SeriesStoreSession>();
            // Soft reconnect left start_ts unconfirmed.
            *session = SeriesStoreSession::identity(Some(addr), None);
            let mut cache = app
                .world_mut()
                .resource_mut::<impeller2_bevy::TelemetryCache>();
            *cache = populated_cache();
            app.world_mut()
                .resource_mut::<plugins::kdl_document::LastSyncedActiveKey>()
                .0 = Some("schematics/main.kdl".into());
        }
        {
            let mut config = app.world_mut().resource_mut::<impeller2_wkt::DbConfig>();
            config.set_time_start_timestamp_micros(100);
        }
        app.update();

        assert!(
            app.world()
                .resource::<impeller2_bevy::TelemetryCache>()
                .has_series(&ComponentId(42))
        );
        assert_eq!(
            app.world().resource::<SeriesStoreSession>().start_ts,
            Some(100)
        );
        // Soft adopt must not invalidate schematic sync (would force blank reload).
        assert_eq!(
            app.world()
                .resource::<plugins::kdl_document::LastSyncedActiveKey>()
                .0
                .as_deref(),
            Some("schematics/main.kdl")
        );
        // Soft adopt must not zero the playhead.
        assert_eq!(
            app.world().resource::<CurrentTimestamp>().0,
            Timestamp(9_000)
        );
    }

    #[test]
    fn soft_reconnect_skips_ui_hard_clear_policy() {
        assert!(!should_hard_clear_ui(true));
        assert!(should_hard_clear_ui(false));
    }

    #[test]
    fn invalidate_schematic_sync_baselines_clears_last_synced() {
        let mut key =
            plugins::kdl_document::LastSyncedActiveKey(Some("schematics/main.kdl".into()));
        let mut revision = plugins::kdl_document::LastSyncedAssetsRevision {
            revision: Some(7),
            suppress_next: true,
            requested: Some(7),
        };
        invalidate_schematic_sync_baselines(&mut key, &mut revision);
        assert!(key.0.is_none());
        assert!(revision.revision.is_none());
        assert!(!revision.suppress_next);
        assert!(revision.requested.is_none());
    }

    #[test]
    fn hard_clear_editor_ui_clears_synced_object3d_and_graph_data() {
        use bevy::ecs::system::RunSystemOnce;

        let mut app = App::new();
        app.init_resource::<SyncedObject3d>()
            .init_resource::<CollectedGraphData>();

        let a = app.world_mut().spawn_empty().id();
        let b = app.world_mut().spawn_empty().id();
        app.world_mut()
            .resource_mut::<SyncedObject3d>()
            .0
            .insert(a, b);
        app.world_mut()
            .resource_mut::<CollectedGraphData>()
            .components
            .insert(
                ComponentId(1),
                crate::ui::plot::data::PlotDataComponent::new("x", vec![]),
            );

        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 mut windows: Query<(Entity, &mut tiles::WindowState)>,
                 mut synced: ResMut<SyncedObject3d>,
                 mut graph_data: ResMut<CollectedGraphData>| {
                    hard_clear_editor_ui(
                        &mut commands,
                        &mut windows,
                        None,
                        &[],
                        &mut synced,
                        &mut graph_data,
                    );
                },
            )
            .unwrap();

        assert!(app.world().resource::<SyncedObject3d>().0.is_empty());
        assert!(
            app.world()
                .resource::<CollectedGraphData>()
                .components
                .is_empty()
        );
    }
}
