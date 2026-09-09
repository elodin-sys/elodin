use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{
    HeadlessMode, SensorCamera, SensorCameraConfigs, SensorCameraPlugin, SensorCameraRenderMetrics,
    SensorCamerasSpawned, SensorReadbackStatus, TEMP_MAP_SUFFIX, set_cameras_active,
    set_readback_armed, update_auto_agc,
};
use crate::sensor_h264::SensorH264Encoder;
use crate::{EqlContext, PositionSync, sync_pos};
use bevy::core_pipeline::Skybox;
use bevy::tasks::{IoTaskPool, Task, futures_lite::future};
use bevy::{
    a11y::AccessibilityPlugin,
    animation::AnimationPlugin,
    app::{App, AppExit, Plugin, Startup},
    asset::{AssetPlugin, Assets, UnapprovedPathMode},
    audio::AudioPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore},
    ecs::system::SystemParam,
    gilrs::GilrsPlugin,
    input::InputPlugin,
    log::LogPlugin,
    math::{EulerRot, Quat},
    picking::{InteractionPlugin, PickingPlugin, input::PointerInputPlugin},
    prelude::*,
    render::{RenderApp, pipelined_rendering::PipelinedRenderingPlugin},
    sprite::SpritePlugin,
    sprite_render::SpriteRenderPlugin,
    state::app::StatesPlugin,
    text::TextPlugin,
    transform::TransformPlugin,
    ui::UiPlugin,
    ui_render::UiRenderPlugin,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_ai_skybox::prelude::{
    PrimarySkybox, SetActiveSkybox, SkyboxAssetSettings, SkyboxCache, SkyboxFailed,
};
use bevy_geo_frames::GeoContext;
use bevy_geo_frames::GeoFramePlugin;
use bevy_mat3_material::Mat3Material;
use impeller2::types::{ComponentId, LenPacket, Timestamp, msg_id};
use impeller2_bevy::{
    ConnectionAddr, ConnectionStatus, CurrentStreamId, MsgPacketTx, PacketTx, SeriesFetchPriority,
    ThreadConnectionStatus,
};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{
    CurrentTimestamp, DbConfig, DumpMetadata, LastUpdated, MsgMetadata, SchematicElem,
    SetMsgMetadata, SetStreamFilter, opaque_bytes_msg_schema,
};

const SENSOR_CAMERA_CONFIG_WARMUP_CYCLES: usize = 600;
const SENSOR_CAMERA_PRIME_CYCLES: usize = 60;

/// A headless Bevy app dedicated to sensor camera rendering.
///
/// Used by both `elodin run` (as a sibling s10 process) and `elodin editor`
/// (also as a sibling s10 process). Connects to the simulation's DB via TCP,
/// subscribes to `LastUpdated`, and emits one rendered frame per camera every
/// `1 / fps` µs of sim time. Frames are pushed to the DB as
/// `MsgWithTimestamp` packets via the existing TCP connection.
///
/// There is no UDS, no request-response protocol, and no sim-side blocking.
pub struct HeadlessEditorPlugin;

impl Plugin for HeadlessEditorPlugin {
    fn build(&self, app: &mut App) {
        // Must run before anything can spawn a `WorldPos` entity.
        crate::register_world_pos_components(app);
        crate::plugins::gpu_info::install_gpu_panic_handler();
        app.add_plugins(crate::plugins::WebAssetPlugin)
            .add_plugins(crate::plugins::env_asset_source::plugin)
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: None,
                        exit_condition: ExitCondition::DontExit,
                        ..default()
                    })
                    .disable::<WinitPlugin>()
                    .disable::<LogPlugin>()
                    .disable::<PipelinedRenderingPlugin>()
                    .disable::<TransformPlugin>()
                    .disable::<DiagnosticsPlugin>()
                    .disable::<InputPlugin>()
                    .disable::<AccessibilityPlugin>()
                    .disable::<AnimationPlugin>()
                    .disable::<AudioPlugin>()
                    .disable::<GilrsPlugin>()
                    .disable::<SpritePlugin>()
                    .disable::<SpriteRenderPlugin>()
                    .disable::<TextPlugin>()
                    .disable::<UiPlugin>()
                    .disable::<UiRenderPlugin>()
                    // GizmoPlugin must stay enabled since Bevy 0.19:
                    // bevy_light's LightGizmoPlugin (nested in LightPlugin,
                    // not individually disableable) registers gizmo systems
                    // that require GizmoPlugin's resources.
                    .disable::<StatesPlugin>()
                    .disable::<PointerInputPlugin>()
                    .disable::<PickingPlugin>()
                    .disable::<InteractionPlugin>()
                    // In DefaultPlugins since Bevy 0.19; its
                    // `dispatch_focused_input` systems read input messages
                    // that are never initialized with `InputPlugin` disabled.
                    .disable::<bevy::input_focus::InputDispatchPlugin>()
                    // Pulled into DefaultPlugins by the `bevy_dev_tools`
                    // cargo feature (needed for the native infinite grid);
                    // its `handle_input` reads `ButtonInput<KeyCode>` which
                    // doesn't exist without `InputPlugin`.
                    .disable::<bevy::dev_tools::render_debug::RenderDebugOverlayPlugin>()
                    .set(AssetPlugin {
                        watch_for_changes_override: Some(true),
                        unapproved_path_mode: UnapprovedPathMode::Allow,
                        ..default()
                    }),
            )
            .add_plugins(crate::skybox_asset_plugin_headless())
            .add_plugins(impeller2_bevy::Impeller2Plugin)
            .add_plugins(bevy::dev_tools::infinite_grid::InfiniteGridPlugin)
            .add_plugins(bevy::pbr::wireframe::WireframePlugin::default())
            .add_plugins(bevy_mat3_material::Mat3MaterialPlugin)
            .add_plugins(crate::plugins::world_mesh::EditorWorldMeshPlugin)
            .add_plugins(crate::rim_glow_material::RimGlowMaterialPlugin);
        app.add_plugins(crate::plugins::scene_environment::SceneEnvironmentPlugin);
        #[cfg(not(target_family = "wasm"))]
        {
            crate::register_earth_embedded_assets(app);
            crate::register_ibl_embedded_assets(app);
            app.add_plugins(bevy_hanabi::HanabiPlugin);
            app.add_plugins(
                crate::plugins::cinematic_earth::earth_night_material::EarthNightMaterialPlugin,
            );
            app.add_plugins(crate::plugins::cinematic_earth::CinematicEarthPlugin);
        }
        app.add_plugins(GeoFramePlugin {
            apply_transforms: false,
            ..default()
        })
        .add_plugins(SensorCameraPlugin)
        .init_resource::<DiagnosticsStore>()
        .init_resource::<HeadlessMode>()
        .add_systems(
            PreUpdate,
            (
                impeller2_bevy::apply_cached_data,
                crate::object_3d::update_object_3d_system,
                crate::sync_object_3d,
                // `sync_pos` writes `WorldPos` into `GeoPosition`/`GeoRotation`;
                // the geo systems below propagate those into `Transform`. Running
                // them in this order keeps each tick's plane pose in lock-step
                // with the sensor camera's pose (which reads the TelemetryCache
                // directly), preventing one-frame jitter in `sensor_view`.
                sync_pos,
                #[cfg(not(feature = "big_space"))]
                bevy_geo_frames::apply_transforms,
                bevy_geo_frames::apply_geo_rotation,
                #[cfg(feature = "big_space")]
                crate::spatial::apply_big_translation,
            )
                .chain()
                .after(impeller2_bevy::sink)
                .in_set(PositionSync),
        )
        .add_systems(Startup, setup_headless_lighting)
        .add_systems(Update, disable_headless_fallback_for_cinematic)
        .add_systems(
            Update,
            apply_cinematic_sensor_environment.after(load_headless_scene),
        )
        .add_systems(
            Update,
            disable_lwir_shadows.after(apply_cinematic_sensor_environment),
        )
        .init_resource::<crate::EqlContext>()
        .init_resource::<crate::Coordinate>()
        .init_resource::<crate::SyncedObject3d>()
        .init_resource::<HeadlessSchematicSkybox>()
        .init_resource::<HeadlessSkyboxRenderGate>()
        .init_resource::<crate::skybox_db_assets::DbSkyboxAssetMirror>()
        .init_resource::<crate::skybox_db_assets::DbSkyboxSyncInFlight>()
        // Same SeriesStore subscription path as the interactive editor: empty
        // allowlist means Option D admits no live/backfill samples, freezing
        // sensor cameras and object_3d at spawn defaults.
        .add_systems(Update, crate::ui::plot::update_series_fetch_priority)
        .add_systems(
            Update,
            sync_headless_stream_filter.after(crate::ui::plot::update_series_fetch_priority),
        )
        .add_systems(
            Update,
            impeller2_bevy::backfill_cache.after(crate::ui::plot::update_series_fetch_priority),
        )
        .add_systems(Update, crate::update_eql_context)
        .add_systems(Update, poll_headless_db_config)
        .add_systems(
            Update,
            crate::skybox_db_assets::sync_db_skybox_assets_from_config.before(sync_headless_skybox),
        )
        .add_systems(Update, sync_headless_skybox)
        .add_systems(Update, load_headless_scene)
        .set_runner(render_server_runner);

        app.add_systems(PreUpdate, crate::warn_missing_geo.before(PositionSync));
        #[cfg(feature = "big_space")]
        app.add_plugins(crate::spatial::FloatingOriginPlugin::new(16_000., 100.));
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<HeadlessMode>()
                .init_resource::<SensorCameraRenderMetrics>();
        }
    }
}

#[derive(Default)]
struct SentStreamFilter {
    component_ids: HashSet<ComponentId>,
    frequency: Option<u64>,
    connected: bool,
}

fn sync_headless_stream_filter(
    priority: Res<SeriesFetchPriority>,
    stream_id: Res<CurrentStreamId>,
    packet_tx: Res<PacketTx>,
    connection: Res<ThreadConnectionStatus>,
    configs: Res<SensorCameraConfigs>,
    mut sent: Local<SentStreamFilter>,
) {
    let connected = connection.status() == ConnectionStatus::Success;
    if !connected {
        sent.connected = false;
        return;
    }
    let frequency = Some(
        configs
            .0
            .iter()
            .map(|config| config.fps.ceil().max(1.0) as u64)
            .max()
            .unwrap_or(60),
    );
    // Empty high-priority set means "schematic not ready yet", not "subscribe
    // to nothing". Sending [] would drop every pose until cameras appear.
    if priority.high.is_empty() {
        sent.connected = true;
        sent.component_ids.clear();
        sent.frequency = frequency;
        return;
    }
    if sent.connected && sent.component_ids == priority.high && sent.frequency == frequency {
        return;
    }
    packet_tx.send_msg(SetStreamFilter {
        id: stream_id.0,
        component_ids: priority.high.iter().copied().collect(),
        frequency,
    });
    tracing::info!(
        components = priority.high.len(),
        "render server stream filter updated"
    );
    sent.component_ids.clone_from(&priority.high);
    sent.frequency = frequency;
    sent.connected = true;
}

// ---------------------------------------------------------------------------
// Scene loading
// ---------------------------------------------------------------------------

#[derive(Component)]
struct HeadlessFallbackLight;

fn setup_headless_lighting(mut commands: Commands) {
    commands.insert_resource(bevy::light::DirectionalLightShadowMap { size: 256 });
    commands.spawn((
        HeadlessFallbackLight,
        DirectionalLight {
            illuminance: 10_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));
}

fn disable_headless_fallback_for_cinematic(
    configs: Res<SensorCameraConfigs>,
    lights: Query<Entity, With<HeadlessFallbackLight>>,
    mut commands: Commands,
) {
    if !configs.0.iter().any(|config| config.cinematic) {
        return;
    }
    for entity in &lights {
        commands.entity(entity).despawn();
    }
}

fn disable_lwir_shadows(
    configs: Res<SensorCameraConfigs>,
    mut lights: Query<&mut DirectionalLight>,
) {
    if configs.0.is_empty() || configs.0.iter().any(|config| config.effect != "lwir") {
        return;
    }
    for mut light in &mut lights {
        light.shadow_maps_enabled = false;
    }
}

fn apply_cinematic_sensor_environment(
    configs: Res<SensorCameraConfigs>,
    mut scene_environment: ResMut<crate::plugins::scene_environment::SceneEnvironment>,
) {
    let Some(camera) = configs.0.iter().find(|config| config.cinematic) else {
        return;
    };
    let desired = camera.resolved_environment();
    if scene_environment.0 != desired {
        scene_environment.0 = desired;
    }
}

/// Loads the active schematic's scene and keeps it in sync with the DB.
///
/// Mirrors the interactive editor's config sync (`sync_document_from_config`):
/// the scene reloads when `schematic.active` repoints to another key, and
/// refetches when `assets.revision` bumps under the same key — comparing bytes
/// so a bump from an unrelated asset write (mesh/skybox `PUT`) doesn't tear the
/// scene down for nothing (RFD #724).
#[allow(clippy::too_many_arguments)]
fn load_headless_scene(
    config: Res<DbConfig>,
    mut pending: Local<HeadlessSchematicLoad>,
    mut schematic_skybox: ResMut<HeadlessSchematicSkybox>,
    mut commands: Commands,
    eql: Res<EqlContext>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mat3_materials: ResMut<Assets<Mat3Material>>,
    mut world_mesh_materials: ResMut<Assets<bevy_world_mesh::prelude::WorldMeshMaterial>>,
    asset_server: Res<AssetServer>,
    connection_addr: Option<Res<ConnectionAddr>>,
    mut geo_context: ResMut<GeoContext>,
    mut coordinate: ResMut<crate::Coordinate>,
    mut scene_environment: ResMut<crate::plugins::scene_environment::SceneEnvironment>,
    configs: Res<SensorCameraConfigs>,
) {
    // Poll an in-flight fetch. The blocking HTTP request runs on the IO pool
    // (RFD #724): a slow/unreachable DB Asset Server never freezes the app.
    let (content, key, revision) = if let Some(task) = pending.task.as_mut() {
        let Some(result) = future::block_on(future::poll_once(task)) else {
            return;
        };
        pending.task = None;
        let Some((key, revision)) = pending.fetch_target.take() else {
            return;
        };
        match result {
            Ok(content) => (content, key, revision),
            Err(err) => {
                tracing::debug!("Headless scene load waiting for active schematic: {err}");
                pending.next_attempt = Some(Instant::now() + Duration::from_millis(400));
                return;
            }
        }
    } else {
        if pending.next_attempt.is_some_and(|at| Instant::now() < at) {
            return;
        }
        // Wait for the EQL context to have component paths registered before
        // attempting to parse object_3d expressions — otherwise the schematic
        // loads during warm-up with an empty context and all objects silently fail.
        if eql.0.component_parts.is_empty() {
            return;
        }
        let Some(key) = config.schematic_active().map(str::to_owned) else {
            // The active pointer was cleared: tear down the loaded scene so the
            // renderer doesn't keep a schematic the DB no longer designates.
            scene_environment.0 = None;
            if let Some(previous) = pending.loaded.take() {
                despawn_headless_scene(&mut commands, &previous);
                schematic_skybox.0 = None;
                coordinate.0 = None;
            }
            return;
        };
        let revision = config.assets_revision();
        if pending
            .loaded
            .as_ref()
            .is_some_and(|loaded| loaded.key == key && loaded.revision == revision)
        {
            return;
        }
        let Some(addr) = connection_addr.as_ref().map(|addr| addr.0) else {
            return;
        };
        pending.fetch_target = Some((key.clone(), revision));
        pending.task = Some(IoTaskPool::get().spawn(async move {
            crate::plugins::kdl_document::fetch_active_schematic_kdl(&key, Some(addr))
        }));
        return;
    };
    // Unchanged bytes under the same key: the revision bump came from an
    // unrelated asset write. Adopt the new baseline without a respawn.
    if let Some(loaded) = pending.loaded.as_mut()
        && loaded.key == key
        && loaded.content == content
    {
        loaded.revision = revision;
        return;
    }
    let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(&content).inspect_err(|e| {
        tracing::warn!("Failed to parse schematic KDL: {e}");
    }) else {
        // Bytes fetched but unparsable: back off before retrying so permanently
        // invalid active schematic bytes don't spin a tight fetch loop each
        // frame (RFD #724). A later valid byte change still gets picked up.
        // The previously loaded scene (if any) stays up in the meantime.
        pending.next_attempt = Some(Instant::now() + Duration::from_millis(400));
        return;
    };
    if let Err(err) =
        impeller2_wkt::validate_single_cinematic_environment(Some(&schematic), &configs.0)
    {
        tracing::error!("{err}");
        pending.next_attempt = Some(Instant::now() + Duration::from_millis(400));
        return;
    };
    // Parse succeeded: replace the previous scene with the new one.
    if let Some(previous) = pending.loaded.take() {
        despawn_headless_scene(&mut commands, &previous);
    }
    let connection_addr = connection_addr.as_ref().map(|addr| addr.0);
    schematic_skybox.0 = Some(schematic.skybox.as_ref().map(|skybox| skybox.name.clone()));
    let fallback_frame = schematic.frame;
    coordinate.0 = schematic.frame;
    scene_environment.0 = schematic.environment.clone().map(|mut environment| {
        if !configs.0.iter().any(|config| config.cinematic) {
            environment.earth = None;
            environment.atmosphere = None;
        }
        environment
    });

    geo_context.origin = crate::ui::schematic::schematic_geo_origin(&schematic);

    let mut entities = Vec::new();
    for elem in &schematic.elems {
        match elem {
            SchematicElem::Object3d(obj) => {
                if !obj.sensor_visible {
                    continue;
                }
                let mut obj = obj.clone();
                if obj.frame.is_none() {
                    obj.frame = fallback_frame;
                }
                let Ok(expr) = eql.0.parse_str(&obj.eql) else {
                    tracing::warn!("Failed to parse EQL for object_3d: {}", obj.eql);
                    continue;
                };
                if let Ok(entity) = create_object_3d_entity(
                    &mut commands,
                    obj,
                    expr,
                    &eql.0,
                    &mut materials,
                    &mut meshes,
                    &mut mat3_materials,
                    &asset_server,
                    &geo_context,
                    connection_addr,
                    // Headless render server is always DB-driven; no --kdl
                    // local-iteration override.
                    None,
                ) {
                    entities.push(entity);
                }
            }
            SchematicElem::WorldMesh(world_mesh) => {
                let mut world_mesh = world_mesh.clone();
                if world_mesh.frame.is_none() {
                    world_mesh.frame = fallback_frame;
                }
                entities.push(crate::plugins::world_mesh::spawn_world_mesh_terrain(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut world_mesh_materials,
                    &world_mesh,
                    &geo_context,
                ));
            }
            _ => {}
        }
    }
    tracing::debug!(
        "Headless scene loaded: {} elements from schematic {key} (revision {revision})",
        schematic.elems.len()
    );
    pending.loaded = Some(LoadedHeadlessScene {
        key,
        revision,
        content,
        entities,
    });
}

fn despawn_headless_scene(commands: &mut Commands, scene: &LoadedHeadlessScene) {
    for entity in &scene.entities {
        commands.entity(*entity).despawn();
    }
}

#[derive(Resource, Default, Debug, Clone)]
struct HeadlessSchematicSkybox(Option<Option<String>>);

#[derive(Default)]
struct HeadlessSchematicLoad {
    next_attempt: Option<Instant>,
    /// In-flight async fetch of the active schematic's KDL. Keeps the bounded —
    /// but potentially multi-second — HTTP request off the main thread so a slow
    /// or unreachable DB Asset Server never stalls the headless app each retry.
    task: Option<Task<Result<String, String>>>,
    /// `(schematic.active, assets.revision)` captured when `task` was spawned,
    /// adopted as the new baseline once the fetch result is applied.
    fetch_target: Option<(String, u64)>,
    /// The scene currently applied, or `None` before the first load.
    loaded: Option<LoadedHeadlessScene>,
}

/// Baseline of the last applied scene, used to decide when to reload: a change
/// of active key always reloads; a revision bump under the same key refetches
/// and reloads only when the schematic bytes actually differ.
struct LoadedHeadlessScene {
    key: String,
    revision: u64,
    content: String,
    /// Root entities spawned from the schematic, despawned on reload.
    entities: Vec<Entity>,
}

const SKYBOX_TRANSITION_WARMUP_FRAMES: u8 = 2;

#[derive(Debug, Resource)]
struct HeadlessSkyboxRenderGate {
    desired: Option<Option<String>>,
    applied: bool,
    warmup_remaining: u8,
    activation_dispatched: bool,
    /// Desired skybox we stopped waiting for after a load failure.
    skipped_desired: Option<Option<String>>,
}

impl Default for HeadlessSkyboxRenderGate {
    fn default() -> Self {
        Self {
            desired: None,
            applied: true,
            warmup_remaining: 0,
            activation_dispatched: false,
            skipped_desired: None,
        }
    }
}

type AiSkyboxCameraQuery<'w, 's> = Query<
    'w,
    's,
    (
        Option<&'static PrimarySkybox>,
        Option<&'static Skybox>,
        Has<crate::plugins::cinematic_earth::CinematicSkybox>,
    ),
    With<Camera3d>,
>;

fn is_ai_skybox_target(
    apply_to_all_cameras: bool,
    has_primary: bool,
    has_cinematic_skybox: bool,
) -> bool {
    !has_cinematic_skybox && (apply_to_all_cameras || has_primary)
}

fn headless_skybox_applied(
    desired: &Option<String>,
    cache: &SkyboxCache,
    settings: &SkyboxAssetSettings,
    cameras: &AiSkyboxCameraQuery,
) -> bool {
    let targets: Vec<_> = cameras
        .iter()
        .filter(|(primary, _, cinematic)| {
            is_ai_skybox_target(settings.apply_to_all_cameras, primary.is_some(), *cinematic)
        })
        .collect();

    if targets.is_empty() {
        return match desired {
            None => cache.active.is_none(),
            Some(_) => false,
        };
    }

    match desired {
        None => targets.iter().all(|(_, skybox, _)| skybox.is_none()),
        Some(name) => {
            cache.active.as_deref() == Some(name.as_str())
                && targets.iter().all(|(_, skybox, _)| skybox.is_some())
        }
    }
}

fn skybox_failure_matches_gate(gate_desired: &Option<Option<String>>, failed_name: &str) -> bool {
    matches!(gate_desired, Some(Some(name)) if name == failed_name)
}

fn clear_applied_in_cache(desired: &Option<String>, cache: &SkyboxCache) -> bool {
    desired.is_none() && cache.active.is_none()
}

#[derive(SystemParam)]
struct SyncHeadlessSkyboxParams<'w, 's> {
    config: Res<'w, DbConfig>,
    cache: Res<'w, SkyboxCache>,
    settings: Res<'w, SkyboxAssetSettings>,
    cameras: AiSkyboxCameraQuery<'w, 's>,
    render_gate: ResMut<'w, HeadlessSkyboxRenderGate>,
    skybox_writer: MessageWriter<'w, SetActiveSkybox>,
    failed: MessageReader<'w, 's, SkyboxFailed>,
    connection_addr: Option<Res<'w, ConnectionAddr>>,
    mirror: Res<'w, crate::skybox_db_assets::DbSkyboxAssetMirror>,
    in_flight: Res<'w, crate::skybox_db_assets::DbSkyboxSyncInFlight>,
    schematic_skybox: Res<'w, HeadlessSchematicSkybox>,
}

fn sync_headless_skybox(params: SyncHeadlessSkyboxParams) {
    let SyncHeadlessSkyboxParams {
        config,
        cache,
        settings,
        cameras,
        mut render_gate,
        mut skybox_writer,
        mut failed,
        connection_addr,
        mirror,
        in_flight,
        schematic_skybox,
    } = params;

    let desired = config
        .skybox_active_desired()
        .or_else(|| schematic_skybox.0.clone());

    for event in failed.read() {
        if !skybox_failure_matches_gate(&render_gate.desired, &event.name) {
            continue;
        }
        tracing::warn!(
            "render server: skybox `{}` failed to load ({}); continuing without skybox",
            event.name,
            event.error
        );
        render_gate.applied = true;
        render_gate.warmup_remaining = SKYBOX_TRANSITION_WARMUP_FRAMES;
        render_gate.activation_dispatched = false;
        render_gate.skipped_desired = render_gate.desired.clone();
    }

    let Some(desired) = desired else {
        render_gate.desired = None;
        render_gate.applied = true;
        render_gate.warmup_remaining = 0;
        render_gate.skipped_desired = None;
        return;
    };

    if render_gate.desired.as_ref() != Some(&desired) {
        render_gate.desired = Some(desired.clone());
        render_gate.applied = false;
        render_gate.warmup_remaining = 0;
        render_gate.activation_dispatched = false;
        render_gate.skipped_desired = None;
    }

    if headless_skybox_applied(&desired, &cache, &settings, &cameras) {
        if !render_gate.applied {
            render_gate.applied = true;
            render_gate.warmup_remaining = SKYBOX_TRANSITION_WARMUP_FRAMES;
        }
        render_gate.skipped_desired = None;
        return;
    }

    if render_gate.skipped_desired.as_ref() == Some(&desired) {
        render_gate.applied = true;
        return;
    }

    if render_gate.activation_dispatched && clear_applied_in_cache(&desired, &cache) {
        render_gate.applied = true;
        render_gate.warmup_remaining = SKYBOX_TRANSITION_WARMUP_FRAMES;
        return;
    }

    if let (Some(connection_addr), Some(name)) = (connection_addr.as_deref(), &desired) {
        if crate::skybox_db_assets::db_skybox_mirror_pending(
            connection_addr.0,
            name,
            &mirror,
            &in_flight,
        ) {
            render_gate.applied = false;
            return;
        }
        if crate::skybox_db_assets::db_skybox_mirror_synced(connection_addr.0, name, &mirror) {
            render_gate.applied = false;
            if cache.active.as_deref() == Some(name.as_str()) {
                skybox_writer.write(SetActiveSkybox::ByName(name.clone()));
                render_gate.activation_dispatched = true;
                return;
            }
            if render_gate.activation_dispatched {
                return;
            }
            // Assets are mirrored; sync_db re-activates when cache.active is stale.
            render_gate.activation_dispatched = true;
            return;
        }
    }

    render_gate.applied = false;
    if render_gate.activation_dispatched {
        return;
    }
    render_gate.activation_dispatched = true;
    match &desired {
        Some(name) => skybox_writer.write(SetActiveSkybox::ByName(name.clone())),
        None => skybox_writer.write(SetActiveSkybox::Clear),
    };
}

fn poll_headless_db_config(mut last_poll: Local<Option<Instant>>, packet_tx: Res<PacketTx>) {
    let now = Instant::now();
    if last_poll.is_some_and(|last| now.duration_since(last) < Duration::from_millis(200)) {
        return;
    }
    *last_poll = Some(now);
    packet_tx.send_msg(DumpMetadata);
}

// ---------------------------------------------------------------------------
// Custom Bevy runner
// ---------------------------------------------------------------------------

fn run_headless_update(app: &mut App) {
    app.update();
}

fn run_headless_main_update(app: &mut App) {
    app.main_mut().update();
}

enum SkyboxEmissionGate {
    Ready,
    WaitingForApply,
    Warming,
}

fn consume_skybox_emission_gate(app: &mut App) -> SkyboxEmissionGate {
    let mut render_gate = app.world_mut().resource_mut::<HeadlessSkyboxRenderGate>();
    if !render_gate.applied {
        return SkyboxEmissionGate::WaitingForApply;
    }
    if render_gate.warmup_remaining > 0 {
        render_gate.warmup_remaining -= 1;
        return SkyboxEmissionGate::Warming;
    }
    SkyboxEmissionGate::Ready
}

struct SensorEncoderJob {
    timestamp: Timestamp,
    rgba: Vec<u8>,
}

struct SensorH264Worker {
    sender: Option<std::sync::mpsc::SyncSender<SensorEncoderJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl SensorH264Worker {
    fn new(
        camera_name: String,
        width: u32,
        height: u32,
        fps: f32,
        msg_tx: MsgPacketTx,
    ) -> Result<Self, std::io::Error> {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<SensorEncoderJob>(64);
        let thread_name = format!("h264-{camera_name}");
        let handle = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                let mut encoder = match SensorH264Encoder::new(width, height, fps) {
                    Ok(encoder) => encoder,
                    Err(err) => {
                        tracing::error!("sensor camera {camera_name}: {err}");
                        return;
                    }
                };
                msg_tx.send_msg(SetMsgMetadata {
                    id: msg_id(&camera_name),
                    metadata: MsgMetadata {
                        name: camera_name.clone(),
                        schema: opaque_bytes_msg_schema(),
                        metadata: HashMap::new(),
                    },
                });
                while let Ok(job) = receiver.recv() {
                    match encoder.encode(&job.rgba, job.timestamp) {
                        Ok(frames) => send_encoded_h264(&camera_name, &msg_tx, frames),
                        Err(err) => tracing::warn!("sensor camera {camera_name}: {err}"),
                    }
                }
                match encoder.flush() {
                    Ok(frames) => send_encoded_h264(&camera_name, &msg_tx, frames),
                    Err(err) => tracing::warn!("sensor camera {camera_name}: flush: {err}"),
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    fn is_alive(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
    }

    fn send(&self, timestamp: Timestamp, rgba: Vec<u8>) -> bool {
        let Some(sender) = &self.sender else {
            return false;
        };
        match sender.try_send(SensorEncoderJob { timestamp, rgba }) {
            Ok(()) => true,
            Err(std::sync::mpsc::TrySendError::Full(_)) => {
                tracing::debug!("sensor camera encoder queue full; dropping frame");
                true
            }
            Err(std::sync::mpsc::TrySendError::Disconnected(_)) => false,
        }
    }
}

fn send_encoded_h264(camera_name: &str, msg_tx: &MsgPacketTx, frames: Vec<(Timestamp, Vec<u8>)>) {
    for (timestamp, encoded) in frames {
        if encoded.is_empty() {
            continue;
        }
        let mut packet =
            LenPacket::msg_with_timestamp(msg_id(camera_name), timestamp, encoded.len());
        packet.extend_from_slice(&encoded);
        if msg_tx.0.try_send(Some(packet)).is_err() {
            tracing::debug!(
                "render server: MsgPacketTx queue full; dropping frame for {camera_name}"
            );
        }
    }
}

impl Drop for SensorH264Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn rgba_to_gray8(
    rgba: &[u8],
    width: u32,
    height: u32,
    monochrome_lwir: bool,
) -> Result<Vec<u8>, String> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "sensor frame dimensions overflow".to_string())?;
    let expected = pixels
        .checked_mul(4)
        .ok_or_else(|| "sensor frame dimensions overflow".to_string())?;
    if rgba.len() != expected {
        return Err(format!(
            "unexpected RGBA frame size {} (expected {})",
            rgba.len(),
            expected
        ));
    }
    if monochrome_lwir {
        return Ok(rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| pixel[0])
            .collect());
    }
    Ok(rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| {
            ((77 * u32::from(pixel[0])
                + 150 * u32::from(pixel[1])
                + 29 * u32::from(pixel[2])
                + 128)
                >> 8) as u8
        })
        .collect())
}

fn dispatch_sensor_frames(
    app: &App,
    frames: Vec<(String, Timestamp, Vec<u8>)>,
    encoders: &mut HashMap<String, SensorH264Worker>,
) -> Vec<(String, Timestamp, Vec<u8>)> {
    let configs = app.world().resource::<SensorCameraConfigs>();
    let msg_tx = app.world().get_resource::<MsgPacketTx>().cloned();
    frames
        .into_iter()
        .filter_map(|(camera_name, timestamp, bytes)| {
            let Some(config) = configs
                .0
                .iter()
                .find(|config| config.camera_name == camera_name)
            else {
                return Some((camera_name, timestamp, bytes));
            };
            if config.format == "gray8" {
                let monochrome_lwir = config.effect == "lwir"
                    && config.effect_param_str(&["palette"], "white_hot") != "ironbow";
                return match rgba_to_gray8(&bytes, config.width, config.height, monochrome_lwir) {
                    Ok(gray) => Some((camera_name, timestamp, gray)),
                    Err(err) => {
                        tracing::warn!("sensor camera {camera_name}: {err}");
                        None
                    }
                };
            }
            if config.format != "h264" {
                return Some((camera_name, timestamp, bytes));
            }
            let Some(msg_tx) = msg_tx.clone() else {
                tracing::warn!("render server: MsgPacketTx not available; dropping frame");
                return None;
            };
            if encoders
                .get(&camera_name)
                .is_some_and(|worker| !worker.is_alive())
            {
                encoders.remove(&camera_name);
            }
            if !encoders.contains_key(&camera_name) {
                match SensorH264Worker::new(
                    camera_name.clone(),
                    config.width,
                    config.height,
                    config.fps,
                    msg_tx,
                ) {
                    Ok(worker) => {
                        encoders.insert(camera_name.clone(), worker);
                    }
                    Err(err) => {
                        tracing::error!("sensor camera {camera_name}: {err}");
                        return None;
                    }
                }
            }
            if encoders
                .get(&camera_name)
                .is_none_or(|worker| !worker.send(timestamp, bytes))
            {
                encoders.remove(&camera_name);
            }
            None
        })
        .collect()
}

/// Per-camera scheduling state for the autonomous render loop.
struct CameraSchedule {
    name: String,
    /// Frame interval in microseconds of sim time, derived from `fps`.
    interval_us: i64,
    /// Sim timestamp of the most recently emitted frame for this camera, or
    /// `None` if no frame has been emitted yet.
    last_rendered: Option<Timestamp>,
}

/// Latest cadence-aligned timestamp at or before `sim_ts`. Overdue intervals
/// are skipped so a slow renderer stays live instead of accumulating debt.
fn next_due_ts(
    last_rendered: Option<Timestamp>,
    sim_ts: Timestamp,
    interval_us: i64,
) -> Option<Timestamp> {
    match last_rendered {
        None => Some(sim_ts),
        Some(_) if interval_us <= 0 => Some(sim_ts),
        Some(previous) => {
            let elapsed = sim_ts.0.saturating_sub(previous.0);
            if elapsed >= interval_us {
                Some(Timestamp(
                    previous.0 + (elapsed / interval_us) * interval_us,
                ))
            } else {
                None
            }
        }
    }
}

fn build_schedules(app: &App) -> Vec<CameraSchedule> {
    app.world()
        .resource::<SensorCameraConfigs>()
        .0
        .iter()
        .map(|c| {
            let fps = c.fps.max(1.0e-6);
            CameraSchedule {
                name: c.camera_name.clone(),
                interval_us: (1_000_000.0 / fps as f64).round() as i64,
                last_rendered: None,
            }
        })
        .collect()
}

fn prime_sensor_cameras(app: &mut App, names: &[String]) {
    enable_all_sensor_cameras(app.world_mut());
    for _ in 0..SENSOR_CAMERA_PRIME_CYCLES {
        run_headless_update(app);
        std::thread::sleep(Duration::from_millis(10));
    }
    set_cameras_active(app.world_mut(), names, false);
}

/// Autonomous render-server runner. Replaces the previous request-response
/// loop with a continuous renderer paced by the DB's `LastUpdated` signal.
fn render_server_runner(mut app: App) -> AppExit {
    app.finish();
    app.cleanup();

    // Warm-up: pump updates until DB metadata arrives, sensor camera configs
    // are loaded, and sensor camera entities are spawned. Then run a few
    // priming cycles with readback armed so the GPU shader cache is warm
    // before we start emitting frames. Steady state after warm-up is "all
    // sensor cameras inactive"; `render_and_emit` flips the due set on for
    // each scheduled frame so we don't spend GPU time rendering scenes
    // nobody is going to read.
    let mut cameras_warmed = false;
    for i in 0..SENSOR_CAMERA_CONFIG_WARMUP_CYCLES {
        run_headless_update(&mut app);
        if app.world().resource::<SensorCamerasSpawned>().0 {
            let names: Vec<String> = build_schedules(&app).into_iter().map(|s| s.name).collect();
            prime_sensor_cameras(&mut app, &names);
            tracing::info!(
                "Sensor cameras spawned and primed after {i} warm-up cycles ({} cameras)",
                names.len()
            );
            cameras_warmed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    if !cameras_warmed {
        tracing::warn!("render server: no sensor cameras configured in DB after warm-up; idling");
    }

    let mut schedules = build_schedules(&app);
    let mut encoders = HashMap::new();
    let mut rendered_frames = 0u64;
    let mut render_time = Duration::ZERO;

    loop {
        if let Some(exit) = app.should_exit() {
            flush_pending_sensor_frames(&mut app, &mut encoders);
            encoders.clear();
            return exit;
        }
        // Readback finishes after the schedule advances; drain here so a
        // paused/stopped sim still publishes the last in-flight frames.
        emit_completed_frames(&mut app, &mut encoders);

        if !cameras_warmed {
            run_headless_update(&mut app);
        }

        // If sensor cameras spawned only after the warm-up loop bailed out,
        // pick them up now. We still briefly flip everything active so the
        // pipelines exist, then drop back to inactive so the per-render
        // gating in `render_and_emit` is the only source of truth.
        if !cameras_warmed && app.world().resource::<SensorCamerasSpawned>().0 {
            schedules = build_schedules(&app);
            let names: Vec<String> = schedules.iter().map(|s| s.name.clone()).collect();
            prime_sensor_cameras(&mut app, &names);
            cameras_warmed = true;
            tracing::info!(
                "Sensor cameras late-spawned; render server now scheduling {} cameras",
                schedules.len()
            );
        }

        if schedules.is_empty() {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }

        let sim_ts = app.world().resource::<LastUpdated>().0;
        if sim_ts.0 == i64::MIN {
            run_headless_main_update(&mut app);
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }

        let next_render = schedules
            .iter()
            .filter_map(|schedule| {
                next_due_ts(schedule.last_rendered, sim_ts, schedule.interval_us)
            })
            .min_by_key(|timestamp| timestamp.0);
        let Some(render_ts) = next_render else {
            run_headless_main_update(&mut app);
            let next_wait_us = schedules
                .iter()
                .filter_map(|s| {
                    s.last_rendered
                        .map(|prev| s.interval_us - (sim_ts.0 - prev.0))
                })
                .filter(|w| *w > 0)
                .min()
                .unwrap_or(1_000);
            std::thread::sleep(Duration::from_micros(next_wait_us.clamp(250, 1_000) as u64));
            continue;
        };
        let due_names: Vec<String> = schedules
            .iter()
            .filter(|schedule| {
                next_due_ts(schedule.last_rendered, sim_ts, schedule.interval_us) == Some(render_ts)
            })
            .map(|schedule| schedule.name.clone())
            .collect();

        match consume_skybox_emission_gate(&mut app) {
            SkyboxEmissionGate::Ready => {}
            SkyboxEmissionGate::WaitingForApply => {
                // Priming renders are required while the cubemap loads and
                // `apply_skybox_to_camera` attaches the `Skybox` component.
                // Skipping render here can stall `HeadlessSkyboxRenderGate.applied`.
                render_without_emit(&mut app, render_ts, &due_names);
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            SkyboxEmissionGate::Warming => {
                render_without_emit(&mut app, render_ts, &due_names);
                continue;
            }
        }

        let render_start = Instant::now();
        render_and_emit(&mut app, render_ts, &due_names, &mut encoders);
        render_time += render_start.elapsed();
        rendered_frames += 1;
        if rendered_frames.is_multiple_of(120) {
            tracing::debug!(
                average_ms = render_time.as_secs_f64() * 1000.0 / rendered_frames as f64,
                "sensor camera render cadence"
            );
        }

        // Mark every emitted camera as rendered at `render_ts`. (If a frame was
        // dropped by `collect_frames` we still advance — the renderer's
        // schedule is independent of whether the emit succeeded.)
        for s in schedules.iter_mut() {
            if due_names.iter().any(|n| n == &s.name) {
                s.last_rendered = Some(render_ts);
            }
        }
    }
}

/// Set the timestamp resource, activate the due cameras, arm readback, run
/// one update cycle, collect frames, push them to the DB, and tear down.
///
/// Activating only `due_names` keeps Bevy's render extract from issuing a 3D
/// pass for cameras whose configured `fps` interval has not yet elapsed.
/// Combined with `ReadbackArmed`, this means each `due` camera does one
/// render plus one GPU->CPU copy per scheduled frame, and idle cameras cost
/// nothing per polling iteration.
fn render_and_emit(
    app: &mut App,
    sim_ts: Timestamp,
    due_names: &[String],
    encoders: &mut HashMap<String, SensorH264Worker>,
) {
    app.world_mut().resource_mut::<CurrentTimestamp>().0 = sim_ts;
    set_cameras_active(app.world_mut(), due_names, true);
    set_readback_armed(app.world_mut(), due_names, true);
    run_headless_update(app);
    set_readback_armed(app.world_mut(), due_names, false);
    set_cameras_active(app.world_mut(), due_names, false);

    emit_completed_frames(app, encoders);
}

fn emit_completed_frames(app: &mut App, encoders: &mut HashMap<String, SensorH264Worker>) {
    let frames = collect_frames(app);
    let (temp_frames, db_frames): (Vec<_>, Vec<_>) = frames
        .into_iter()
        .partition(|(name, _, _)| name.ends_with(TEMP_MAP_SUFFIX));
    let temp_frames: Vec<_> = temp_frames
        .into_iter()
        .map(|(name, _, bytes)| (name, bytes))
        .collect();
    update_auto_agc(app.world_mut(), &temp_frames);
    let db_frames = dispatch_sensor_frames(app, db_frames, encoders);
    push_frames_to_db(app, &db_frames);
}

fn flush_pending_sensor_frames(app: &mut App, encoders: &mut HashMap<String, SensorH264Worker>) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while app.world().resource::<SensorReadbackStatus>().pending() > 0 {
        if Instant::now() >= deadline {
            tracing::warn!(
                pending = app.world().resource::<SensorReadbackStatus>().pending(),
                "timed out waiting for in-flight sensor readbacks"
            );
            break;
        }
        emit_completed_frames(app, encoders);
        std::thread::sleep(Duration::from_millis(1));
    }
    emit_completed_frames(app, encoders);
}

fn render_without_emit(app: &mut App, sim_ts: Timestamp, due_names: &[String]) {
    app.world_mut().resource_mut::<CurrentTimestamp>().0 = sim_ts;

    set_cameras_active(app.world_mut(), due_names, true);
    run_headless_update(app);
    set_cameras_active(app.world_mut(), due_names, false);
}

/// Push rendered frames to the DB as `MsgWithTimestamp` packets via the
/// existing TCP connection (managed by `TcpImpellerPlugin`).
fn push_frames_to_db(app: &App, frames: &[(String, Timestamp, Vec<u8>)]) {
    let Some(tx) = app.world().get_resource::<MsgPacketTx>() else {
        tracing::warn!(
            "render server: MsgPacketTx not available; dropping {} frame(s)",
            frames.len()
        );
        return;
    };
    for (camera_name, timestamp, bytes) in frames {
        let id = msg_id(camera_name);
        let mut pkt = LenPacket::msg_with_timestamp(id, *timestamp, bytes.len());
        pkt.extend_from_slice(bytes);
        if tx.0.try_send(Some(pkt)).is_err() {
            tracing::debug!(
                "render server: MsgPacketTx queue full; dropping frame for {camera_name}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Enable all sensor cameras (used during warm-up).
fn enable_all_sensor_cameras(world: &mut World) {
    let mut query = world.query::<(&SensorCamera, &mut Camera)>();
    for (_, mut camera) in query.iter_mut(world) {
        camera.is_active = true;
    }
}

fn collect_frames(app: &App) -> Vec<(String, Timestamp, Vec<u8>)> {
    let world = app.world();
    let frame_rx = world.resource::<crate::sensor_camera::SensorFrameReceiver>();

    let mut frames = Vec::new();
    while let Ok((camera_name, timestamp, frame_bytes, _, _)) = frame_rx.0.try_recv() {
        frames.push((camera_name, timestamp, frame_bytes));
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::{is_ai_skybox_target, next_due_ts, rgba_to_gray8};
    use impeller2::types::Timestamp;

    #[test]
    fn cinematic_earth_skybox_is_not_an_ai_skybox_target() {
        assert!(!is_ai_skybox_target(false, true, true));
        assert!(is_ai_skybox_target(false, true, false));
        assert!(!is_ai_skybox_target(false, false, false));
    }

    #[test]
    fn gray8_repack_handles_lwir_and_rgb() {
        let rgba = [10, 20, 30, 255, 200, 100, 50, 255];
        assert_eq!(rgba_to_gray8(&rgba, 2, 1, true).unwrap(), [10, 200]);
        assert_eq!(rgba_to_gray8(&rgba, 2, 1, false).unwrap(), [18, 124]);
    }

    #[test]
    fn next_due_stays_on_cadence_and_skips_overdue_intervals() {
        let t0 = Timestamp(1_000_000);
        let interval = 16_667;
        assert_eq!(next_due_ts(None, t0, interval), Some(t0));
        assert_eq!(
            next_due_ts(Some(t0), Timestamp(t0.0 + interval - 1), interval),
            None
        );
        assert_eq!(
            next_due_ts(Some(t0), Timestamp(t0.0 + interval), interval),
            Some(Timestamp(t0.0 + interval))
        );
        assert_eq!(
            next_due_ts(Some(t0), Timestamp(t0.0 + 3 * interval + 100), interval),
            Some(Timestamp(t0.0 + 3 * interval))
        );
    }
}
