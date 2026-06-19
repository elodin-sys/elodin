use std::collections::HashMap;
use std::time::{Duration, Instant};

use bevy::core_pipeline::Skybox;
use bevy::{
    a11y::AccessibilityPlugin,
    animation::AnimationPlugin,
    app::{App, AppExit, Plugin, Startup},
    asset::{AssetPlugin, Assets, UnapprovedPathMode},
    audio::AudioPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore},
    ecs::system::SystemParam,
    gilrs::GilrsPlugin,
    gizmos::GizmoPlugin,
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
use bevy_mat3_material::Mat3Material;
use impeller2::types::{LenPacket, Timestamp, msg_id};
use impeller2_bevy::{ConnectionAddr, MsgPacketTx, PacketTx};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{
    CurrentTimestamp, DbConfig, DumpMetadata, LastUpdated, Schematic, SchematicElem,
};

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{
    HeadlessMode, SensorCamera, SensorCameraConfigs, SensorCameraPlugin, SensorCameraRenderMetrics,
    SensorCamerasSpawned, set_cameras_active, set_readback_armed,
};
use crate::{EqlContext, PositionSync, sync_pos};
use bevy_geo_frames::GeoFramePlugin;

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
                    .disable::<GizmoPlugin>()
                    .disable::<StatesPlugin>()
                    .disable::<PointerInputPlugin>()
                    .disable::<PickingPlugin>()
                    .disable::<InteractionPlugin>()
                    .set(AssetPlugin {
                        watch_for_changes_override: Some(true),
                        unapproved_path_mode: UnapprovedPathMode::Allow,
                        ..default()
                    }),
            )
            .add_plugins(crate::skybox_asset_plugin_headless())
            .add_plugins(impeller2_bevy::Impeller2Plugin)
            .add_plugins(bevy_infinite_grid::InfiniteGridPlugin)
            .add_plugins(bevy::pbr::wireframe::WireframePlugin::default())
            .add_plugins(bevy_mat3_material::Mat3MaterialPlugin)
            .add_plugins(crate::plugins::world_mesh::EditorWorldMeshPlugin)
            .add_plugins(crate::rim_glow_material::RimGlowMaterialPlugin)
            .add_plugins(GeoFramePlugin {
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
            .init_resource::<crate::EqlContext>()
            .init_resource::<crate::SyncedObject3d>()
            .init_resource::<HeadlessSkyboxRenderGate>()
            .init_resource::<crate::skybox_db_assets::DbSkyboxAssetMirror>()
            .init_resource::<crate::skybox_db_assets::DbSkyboxSyncInFlight>()
            .add_systems(Update, crate::update_eql_context)
            .add_systems(Update, poll_headless_db_config)
            .add_systems(
                Update,
                crate::skybox_db_assets::sync_db_skybox_assets_from_config
                    .before(sync_headless_skybox),
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

// ---------------------------------------------------------------------------
// Scene loading
// ---------------------------------------------------------------------------

fn setup_headless_lighting(mut commands: Commands) {
    commands.insert_resource(bevy::light::DirectionalLightShadowMap { size: 256 });
    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));
}

#[allow(clippy::too_many_arguments)]
fn load_headless_scene(
    config: Res<DbConfig>,
    mut loaded: Local<bool>,
    mut commands: Commands,
    eql: Res<EqlContext>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mat3_materials: ResMut<Assets<Mat3Material>>,
    mut world_mesh_materials: ResMut<Assets<bevy_world_mesh::prelude::WorldMeshMaterial>>,
    asset_server: Res<AssetServer>,
    connection_addr: Option<Res<ConnectionAddr>>,
    mut geo_context: ResMut<GeoContext>,
) {
    if *loaded {
        return;
    }
    // Wait for the EQL context to have component paths registered before
    // attempting to parse object_3d expressions — otherwise the schematic
    // loads during warm-up with an empty context and all objects silently fail.
    if eql.0.component_parts.is_empty() {
        return;
    }
    let Some(content) = config.schematic_content() else {
        return;
    };
    let Ok(schematic) = impeller2_wkt::Schematic::from_kdl(content).inspect_err(|e| {
        tracing::warn!("Failed to parse schematic KDL: {e}");
    }) else {
        return;
    };
    let connection_addr = connection_addr.as_ref().map(|addr| addr.0);
    let fallback_frame = schematic.frame;

    if let Some(o) = schematic.origin {
        geo_context.origin =
            bevy_geo_frames::GeoOrigin::new_from_degrees(o.latitude, o.longitude, o.altitude);
    }

    for elem in &schematic.elems {
        match elem {
            SchematicElem::Object3d(obj) => {
                let mut obj = obj.clone();
                if obj.frame.is_none() {
                    obj.frame = fallback_frame;
                }
                let Ok(expr) = eql.0.parse_str(&obj.eql) else {
                    tracing::warn!("Failed to parse EQL for object_3d: {}", obj.eql);
                    continue;
                };
                let _ = create_object_3d_entity(
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
                );
            }
            SchematicElem::WorldMesh(world_mesh) => {
                let mut world_mesh = world_mesh.clone();
                if world_mesh.frame.is_none() {
                    world_mesh.frame = fallback_frame;
                }
                crate::plugins::world_mesh::spawn_world_mesh_terrain(
                    &mut commands,
                    &mut meshes,
                    &mut materials,
                    &mut world_mesh_materials,
                    &world_mesh,
                );
            }
            _ => {}
        }
    }
    tracing::debug!(
        "Headless scene loaded: {} elements from schematic",
        schematic.elems.len()
    );
    *loaded = true;
}

fn desired_skybox_from_config(config: &DbConfig) -> Option<Option<String>> {
    if let Some(desired) = config.skybox_active_desired() {
        return Some(desired);
    }

    let content = config.schematic_content()?;
    let schematic = Schematic::from_kdl(content)
        .inspect_err(|e| tracing::warn!("Failed to parse schematic KDL while syncing skybox: {e}"))
        .ok()?;
    Some(schematic.skybox.as_ref().map(|skybox| skybox.name.clone()))
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

fn headless_skybox_applied(
    desired: &Option<String>,
    cache: &SkyboxCache,
    settings: &SkyboxAssetSettings,
    cameras: &Query<(Option<&PrimarySkybox>, Option<&Skybox>), With<Camera3d>>,
) -> bool {
    let targets: Vec<_> = cameras
        .iter()
        .filter(|(primary, _)| settings.apply_to_all_cameras || primary.is_some())
        .collect();

    if targets.is_empty() {
        return match desired {
            None => cache.active.is_none(),
            Some(_) => false,
        };
    }

    match desired {
        None => targets.iter().all(|(_, skybox)| skybox.is_none()),
        Some(name) => {
            cache.active.as_deref() == Some(name.as_str())
                && targets.iter().all(|(_, skybox)| skybox.is_some())
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
    cameras:
        Query<'w, 's, (Option<&'static PrimarySkybox>, Option<&'static Skybox>), With<Camera3d>>,
    render_gate: ResMut<'w, HeadlessSkyboxRenderGate>,
    skybox_writer: MessageWriter<'w, SetActiveSkybox>,
    failed: MessageReader<'w, 's, SkyboxFailed>,
    connection_addr: Option<Res<'w, ConnectionAddr>>,
    mirror: Res<'w, crate::skybox_db_assets::DbSkyboxAssetMirror>,
    in_flight: Res<'w, crate::skybox_db_assets::DbSkyboxSyncInFlight>,
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
    } = params;

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

    let Some(desired) = desired_skybox_from_config(&config) else {
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

fn drain_stale_frames(app: &App) {
    let rx = app
        .world()
        .resource::<crate::sensor_camera::SensorFrameReceiver>();
    while rx.0.try_recv().is_ok() {}
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
    for i in 0..120 {
        run_headless_update(&mut app);
        if app.world().resource::<SensorCamerasSpawned>().0 {
            enable_all_sensor_cameras(app.world_mut());
            let names: Vec<String> = build_schedules(&app).into_iter().map(|s| s.name).collect();
            set_readback_armed(app.world_mut(), &names, true);
            for _ in 0..4 {
                run_headless_update(&mut app);
            }
            drain_stale_frames(&app);
            set_readback_armed(app.world_mut(), &names, false);
            set_cameras_active(app.world_mut(), &names, false);
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

    loop {
        if let Some(exit) = app.should_exit() {
            return exit;
        }

        // Pump one update first. This:
        //   1. Drains the impeller2 TCP packet queue into the TelemetryCache,
        //      so component poses for the latest sim_ts are visible to the
        //      camera-transform system before we decide which cameras to render.
        //   2. Receives the next `LastUpdated` packet from the DB (set when
        //      the sim bumps `db.last_updated`).
        run_headless_update(&mut app);

        // If sensor cameras spawned only after the warm-up loop bailed out,
        // pick them up now. We still briefly flip everything active so the
        // pipelines exist, then drop back to inactive so the per-render
        // gating in `render_and_emit` is the only source of truth.
        if !cameras_warmed && app.world().resource::<SensorCamerasSpawned>().0 {
            enable_all_sensor_cameras(app.world_mut());
            schedules = build_schedules(&app);
            let names: Vec<String> = schedules.iter().map(|s| s.name.clone()).collect();
            set_cameras_active(app.world_mut(), &names, false);
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
            // Sim hasn't published a `LastUpdated` yet.
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }

        // Pick the cameras whose frame interval has elapsed.
        let due_names: Vec<String> = schedules
            .iter()
            .filter(|s| match s.last_rendered {
                None => true,
                Some(prev) => sim_ts.0 - prev.0 >= s.interval_us,
            })
            .map(|s| s.name.clone())
            .collect();

        if due_names.is_empty() {
            // Nothing to render right now; sleep briefly to avoid a busy loop.
            // Choose the sleep based on the shortest remaining wait across all
            // cameras, capped at 5 ms so we stay responsive to new
            // `LastUpdated` events.
            //
            // If a skybox transition is in flight, pump another update so asset
            // loading / apply systems can progress even when no camera is due.
            if !app.world().resource::<HeadlessSkyboxRenderGate>().applied {
                run_headless_update(&mut app);
            }
            let next_wait_us = schedules
                .iter()
                .filter_map(|s| {
                    s.last_rendered
                        .map(|prev| s.interval_us - (sim_ts.0 - prev.0))
                })
                .filter(|w| *w > 0)
                .min()
                .unwrap_or(5_000);
            std::thread::sleep(Duration::from_micros(next_wait_us.clamp(500, 5_000) as u64));
            continue;
        }

        match consume_skybox_emission_gate(&mut app) {
            SkyboxEmissionGate::Ready => {}
            SkyboxEmissionGate::WaitingForApply => {
                // Priming renders are required while the cubemap loads and
                // `apply_skybox_to_camera` attaches the `Skybox` component.
                // Skipping render here can stall `HeadlessSkyboxRenderGate.applied`.
                render_without_emit(&mut app, sim_ts, &due_names);
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            SkyboxEmissionGate::Warming => {
                render_without_emit(&mut app, sim_ts, &due_names);
                continue;
            }
        }

        render_and_emit(&mut app, sim_ts, &due_names);

        // Mark every emitted camera as rendered at `sim_ts`. (If a frame was
        // dropped by `collect_frames` we still advance — the renderer's
        // schedule is independent of whether the emit succeeded.)
        for s in schedules.iter_mut() {
            if due_names.iter().any(|n| n == &s.name) {
                s.last_rendered = Some(sim_ts);
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
fn render_and_emit(app: &mut App, sim_ts: Timestamp, due_names: &[String]) {
    app.world_mut().resource_mut::<CurrentTimestamp>().0 = sim_ts;

    // Drain any stale frames from the previous pass before arming.
    drain_stale_frames(app);
    set_cameras_active(app.world_mut(), due_names, true);
    set_readback_armed(app.world_mut(), due_names, true);

    // The render-graph + GPU readback runs inside `app.update()`.
    run_headless_update(app);

    let mut frames = collect_frames(app, due_names);
    // The readback ping-pong may need one more update before all frames are
    // mapped on slow GPUs / first runs after a warm reload. We deliberately
    // leave the cameras active for this retry — a rare second render is
    // cheaper than wiring up readback-only updates.
    if frames.len() < due_names.len() {
        run_headless_update(app);
        let more = collect_frames(app, due_names);
        for (name, data) in more {
            if !frames.iter().any(|(n, _)| n == &name) {
                frames.push((name, data));
            }
        }
    }

    push_frames_to_db(app, sim_ts, &frames);
    set_readback_armed(app.world_mut(), due_names, false);
    set_cameras_active(app.world_mut(), due_names, false);
}

fn render_without_emit(app: &mut App, sim_ts: Timestamp, due_names: &[String]) {
    app.world_mut().resource_mut::<CurrentTimestamp>().0 = sim_ts;

    set_cameras_active(app.world_mut(), due_names, true);
    run_headless_update(app);
    set_cameras_active(app.world_mut(), due_names, false);
}

/// Push rendered frames to the DB as `MsgWithTimestamp` packets via the
/// existing TCP connection (managed by `TcpImpellerPlugin`).
fn push_frames_to_db(app: &App, sim_ts: Timestamp, frames: &[(String, Vec<u8>)]) {
    let Some(tx) = app.world().get_resource::<MsgPacketTx>() else {
        tracing::warn!(
            "render server: MsgPacketTx not available; dropping {} frame(s)",
            frames.len()
        );
        return;
    };
    for (camera_name, bytes) in frames {
        let id = msg_id(camera_name);
        let mut pkt = LenPacket::msg_with_timestamp(id, sim_ts, bytes.len());
        pkt.extend_from_slice(bytes);
        if tx.0.try_send(Some(pkt)).is_err() {
            tracing::warn!(
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

/// Collect rendered frames from the frame receiver, matching requested camera names.
fn collect_frames(app: &App, camera_names: &[String]) -> Vec<(String, Vec<u8>)> {
    let world = app.world();
    let frame_rx = world.resource::<crate::sensor_camera::SensorFrameReceiver>();

    let mut frames_map: HashMap<String, Vec<u8>> = HashMap::new();

    while let Ok((camera_name, frame_bytes, _, _)) = frame_rx.0.try_recv() {
        frames_map.insert(camera_name, frame_bytes);
    }

    camera_names
        .iter()
        .filter_map(|name| frames_map.remove(name).map(|bytes| (name.clone(), bytes)))
        .collect()
}
