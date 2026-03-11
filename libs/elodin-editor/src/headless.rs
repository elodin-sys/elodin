use std::time::{Duration, Instant};

use bevy::{
    a11y::AccessibilityPlugin,
    animation::AnimationPlugin,
    app::{App, AppExit, AppLabel, Plugin, Startup},
    asset::{AssetPlugin, Assets, UnapprovedPathMode},
    audio::AudioPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore},
    gilrs::GilrsPlugin,
    gizmos::GizmoPlugin,
    input::InputPlugin,
    log::LogPlugin,
    math::{EulerRot, Quat},
    picking::{InteractionPlugin, PickingPlugin, input::PointerInputPlugin},
    prelude::*,
    render::{Render, RenderApp, RenderSystems, pipelined_rendering::PipelinedRenderingPlugin},
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
use bevy_mat3_material::Mat3Material;
use big_space::{FloatingOrigin, GridCell};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{CurrentTimestamp, DbConfig, SchematicElem};
use render_bridge::RenderBridgeServer;

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{
    HeadlessMode, SensorCameraPlugin, SensorCameraRenderMetrics, SensorCamerasSpawned,
    set_all_sensor_cameras_active, set_readback_armed, set_sensor_cameras_active,
};
use crate::{EqlContext, PositionSync, sync_pos};

const RENDER_TARGET_MS: f64 = 5.0;
const RENDER_CRITICAL_MS: f64 = 8.0;

/// A headless Bevy app dedicated to sensor camera rendering.
///
/// Used by both `elodin run` (main thread) and `elodin editor` (background
/// thread). Connects to the simulation's DB via TCP and renders sensor camera
/// frames on demand when the simulation calls `ctx.render_camera()`.
///
/// The custom runner (`headless_sensor_runner`) listens on a Unix domain
/// socket for render requests from the simulation subprocess, waits for
/// entity data to arrive, enables the requested camera, renders, then
/// writes the frame to the DB and responds over the socket.
pub struct HeadlessEditorPlugin;

impl Plugin for HeadlessEditorPlugin {
    fn build(&self, app: &mut App) {
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
                        unapproved_path_mode: UnapprovedPathMode::Allow,
                        ..default()
                    }),
            )
            .add_plugins(impeller2_bevy::Impeller2Plugin)
            .add_plugins(big_space::FloatingOriginPlugin::<i128>::new(16_000., 100.))
            .add_plugins(bevy_mat3_material::Mat3MaterialPlugin)
            .add_plugins(crate::object_3d::Object3DPlugin)
            .add_plugins(SensorCameraPlugin)
            .init_resource::<DiagnosticsStore>()
            .init_resource::<HeadlessMode>()
            .init_resource::<HeadlessMainScheduleTimingState>()
            .init_resource::<HeadlessMainScheduleMetrics>()
            .add_systems(
                PreUpdate,
                (
                    record_after_sink,
                    crate::setup_cell,
                    record_after_setup_cell,
                    impeller2_bevy::apply_cached_data,
                    record_after_apply_cached_data,
                    crate::object_3d::update_object_3d_system,
                    record_after_update_object_3d_system,
                    crate::queue_object_3d_sync_candidates,
                    record_after_queue_object_3d_sync_candidates,
                    crate::sync_object_3d,
                    record_after_sync_object_3d,
                    sync_pos,
                    record_after_sync_pos,
                )
                    .chain()
                    .after(impeller2_bevy::sink)
                    .in_set(PositionSync),
            )
            .add_systems(Startup, setup_floating_origin)
            .add_systems(Startup, setup_headless_lighting)
            .init_resource::<crate::EqlContext>()
            .init_resource::<crate::PendingObject3dSync>()
            .init_resource::<crate::SyncedObject3d>()
            .add_systems(
                Update,
                (
                    record_before_update_eql_context,
                    crate::update_eql_context,
                    record_after_update_eql_context,
                    load_headless_scene,
                    record_after_load_headless_scene,
                )
                    .chain(),
            )
            .set_runner(headless_sensor_runner);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<HeadlessMode>()
                .init_resource::<HeadlessRenderScheduleTimingState>()
                .init_resource::<HeadlessRenderScheduleMetrics>()
                .add_systems(
                    Render,
                    (
                        record_after_extract_commands
                            .after(RenderSystems::ExtractCommands)
                            .before(RenderSystems::PrepareMeshes),
                        record_after_prepare_meshes
                            .after(RenderSystems::PrepareMeshes)
                            .before(RenderSystems::ManageViews),
                        record_after_manage_views
                            .after(RenderSystems::ManageViews)
                            .before(RenderSystems::Queue),
                        record_after_queue
                            .after(RenderSystems::Queue)
                            .before(RenderSystems::PhaseSort),
                        record_after_phase_sort
                            .after(RenderSystems::PhaseSort)
                            .before(RenderSystems::Prepare),
                        record_after_prepare_resources
                            .after(RenderSystems::PrepareResources)
                            .before(RenderSystems::PrepareResourcesCollectPhaseBuffers),
                        record_after_prepare_resources_collect_phase_buffers
                            .after(RenderSystems::PrepareResourcesCollectPhaseBuffers)
                            .before(RenderSystems::PrepareResourcesFlush),
                        record_after_prepare_resources_flush
                            .after(RenderSystems::PrepareResourcesFlush)
                            .before(RenderSystems::PrepareBindGroups),
                        record_after_prepare_bind_groups
                            .after(RenderSystems::PrepareBindGroups)
                            .before(record_after_prepare),
                        record_after_prepare
                            .after(RenderSystems::Prepare)
                            .before(RenderSystems::Render),
                        record_after_render
                            .after(RenderSystems::Render)
                            .before(RenderSystems::Cleanup),
                        record_after_cleanup
                            .after(RenderSystems::Cleanup)
                            .before(RenderSystems::PostCleanup),
                        finalize_headless_render_schedule_metrics.after(RenderSystems::PostCleanup),
                    ),
                );
        }
    }
}

fn setup_floating_origin(mut commands: Commands) {
    commands.spawn((
        FloatingOrigin,
        GridCell::<i128>::default(),
        Transform::default(),
        GlobalTransform::default(),
    ));
}

// ---------------------------------------------------------------------------
// Scene loading
// ---------------------------------------------------------------------------

fn setup_headless_lighting(mut commands: Commands) {
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
    asset_server: Res<AssetServer>,
) {
    if *loaded {
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

    for elem in &schematic.elems {
        if let SchematicElem::Object3d(obj) = elem {
            let Ok(expr) = eql.0.parse_str(&obj.eql) else {
                tracing::warn!("Failed to parse EQL for object_3d: {}", obj.eql);
                continue;
            };
            create_object_3d_entity(
                &mut commands,
                obj.clone(),
                expr,
                &eql.0,
                &mut materials,
                &mut meshes,
                &mut mat3_materials,
                &asset_server,
            );
        }
    }
    tracing::debug!(
        "Headless scene loaded: {} elements from schematic",
        schematic.elems.len()
    );
    *loaded = true;
}

// ---------------------------------------------------------------------------
// Custom runner
// ---------------------------------------------------------------------------

fn drain_stale_frames(app: &App) {
    let rx = app
        .world()
        .resource::<crate::sensor_camera::SensorFrameReceiver>();
    while rx.0.try_recv().is_ok() {}
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Clone, Copy, Debug, Default)]
struct HeadlessUpdateBreakdown {
    main_schedule_ms: f64,
    render_extract_ms: f64,
    render_app_ms: f64,
    main_clear_trackers_ms: f64,
    main_schedule: HeadlessMainScheduleMetrics,
    render_schedule: HeadlessRenderScheduleMetrics,
}

impl HeadlessUpdateBreakdown {
    fn total_ms(self) -> f64 {
        self.main_schedule_ms
            + self.render_extract_ms
            + self.render_app_ms
            + self.main_clear_trackers_ms
    }
}

#[derive(Clone, Copy, Debug, Default, Resource)]
struct HeadlessMainScheduleMetrics {
    setup_cell_ms: f64,
    apply_cached_data_ms: f64,
    update_object_3d_system_ms: f64,
    queue_object_3d_sync_candidates_ms: f64,
    sync_object_3d_ms: f64,
    sync_pos_ms: f64,
    update_eql_context_ms: f64,
    load_headless_scene_ms: f64,
    other_ms: f64,
}

#[derive(Debug, Default, Resource)]
struct HeadlessMainScheduleTimingState {
    after_sink: Option<Instant>,
    after_setup_cell: Option<Instant>,
    after_apply_cached_data: Option<Instant>,
    after_update_object_3d_system: Option<Instant>,
    after_queue_object_3d_sync_candidates: Option<Instant>,
    after_sync_object_3d: Option<Instant>,
    after_sync_pos: Option<Instant>,
    before_update_eql_context: Option<Instant>,
    after_update_eql_context: Option<Instant>,
    after_load_headless_scene: Option<Instant>,
}

#[derive(Clone, Copy, Debug, Default, Resource)]
struct HeadlessRenderScheduleMetrics {
    extract_commands_ms: f64,
    prepare_meshes_ms: f64,
    manage_views_ms: f64,
    queue_ms: f64,
    phase_sort_ms: f64,
    prepare_resources_ms: f64,
    prepare_resources_collect_phase_buffers_ms: f64,
    prepare_resources_flush_ms: f64,
    prepare_bind_groups_ms: f64,
    prepare_other_ms: f64,
    prepare_ms: f64,
    render_ms: f64,
    cleanup_ms: f64,
    post_cleanup_ms: f64,
}

#[derive(Debug, Default, Resource)]
struct HeadlessRenderScheduleTimingState {
    frame_start: Option<Instant>,
    after_extract_commands: Option<Instant>,
    after_prepare_meshes: Option<Instant>,
    after_manage_views: Option<Instant>,
    after_queue: Option<Instant>,
    after_phase_sort: Option<Instant>,
    after_prepare_resources: Option<Instant>,
    after_prepare_resources_collect_phase_buffers: Option<Instant>,
    after_prepare_resources_flush: Option<Instant>,
    after_prepare_bind_groups: Option<Instant>,
    after_prepare: Option<Instant>,
    after_render: Option<Instant>,
    after_cleanup: Option<Instant>,
}

fn mark_now(slot: &mut Option<Instant>) {
    *slot = Some(Instant::now());
}

fn elapsed_between(start: Option<Instant>, end: Option<Instant>) -> f64 {
    match (start, end) {
        (Some(start), Some(end)) => end.saturating_duration_since(start).as_secs_f64() * 1000.0,
        _ => 0.0,
    }
}

fn reset_headless_main_schedule_timing(world: &mut World) {
    *world.resource_mut::<HeadlessMainScheduleTimingState>() =
        HeadlessMainScheduleTimingState::default();
    *world.resource_mut::<HeadlessMainScheduleMetrics>() = HeadlessMainScheduleMetrics::default();
}

fn snapshot_headless_main_schedule_metrics(
    world: &mut World,
    total_main_schedule_ms: f64,
) -> HeadlessMainScheduleMetrics {
    let state = world.resource::<HeadlessMainScheduleTimingState>();
    let mut metrics = HeadlessMainScheduleMetrics {
        setup_cell_ms: elapsed_between(state.after_sink, state.after_setup_cell),
        apply_cached_data_ms: elapsed_between(
            state.after_setup_cell,
            state.after_apply_cached_data,
        ),
        update_object_3d_system_ms: elapsed_between(
            state.after_apply_cached_data,
            state.after_update_object_3d_system,
        ),
        queue_object_3d_sync_candidates_ms: elapsed_between(
            state.after_update_object_3d_system,
            state.after_queue_object_3d_sync_candidates,
        ),
        sync_object_3d_ms: elapsed_between(
            state.after_queue_object_3d_sync_candidates,
            state.after_sync_object_3d,
        ),
        sync_pos_ms: elapsed_between(state.after_sync_object_3d, state.after_sync_pos),
        update_eql_context_ms: elapsed_between(
            state.before_update_eql_context,
            state.after_update_eql_context,
        ),
        load_headless_scene_ms: elapsed_between(
            state.after_update_eql_context,
            state.after_load_headless_scene,
        ),
        ..default()
    };
    let measured_ms = metrics.setup_cell_ms
        + metrics.apply_cached_data_ms
        + metrics.update_object_3d_system_ms
        + metrics.queue_object_3d_sync_candidates_ms
        + metrics.sync_object_3d_ms
        + metrics.sync_pos_ms
        + metrics.update_eql_context_ms
        + metrics.load_headless_scene_ms;
    metrics.other_ms = (total_main_schedule_ms - measured_ms).max(0.0);
    metrics
}

fn reset_headless_render_schedule_timing(world: &mut World) {
    let mut state = world.resource_mut::<HeadlessRenderScheduleTimingState>();
    *state = HeadlessRenderScheduleTimingState {
        frame_start: Some(Instant::now()),
        ..default()
    };
    *world.resource_mut::<HeadlessRenderScheduleMetrics>() =
        HeadlessRenderScheduleMetrics::default();
}

fn record_after_extract_commands(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_extract_commands);
}

fn record_after_sink(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_sink);
}

fn record_after_setup_cell(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_setup_cell);
}

fn record_after_apply_cached_data(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_apply_cached_data);
}

fn record_after_update_object_3d_system(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_update_object_3d_system);
}

fn record_after_queue_object_3d_sync_candidates(
    mut state: ResMut<HeadlessMainScheduleTimingState>,
) {
    mark_now(&mut state.after_queue_object_3d_sync_candidates);
}

fn record_after_sync_object_3d(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_sync_object_3d);
}

fn record_after_sync_pos(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_sync_pos);
}

fn record_before_update_eql_context(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.before_update_eql_context);
}

fn record_after_update_eql_context(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_update_eql_context);
}

fn record_after_load_headless_scene(mut state: ResMut<HeadlessMainScheduleTimingState>) {
    mark_now(&mut state.after_load_headless_scene);
}

fn record_after_prepare_meshes(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_meshes);
}

fn record_after_manage_views(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_manage_views);
}

fn record_after_queue(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_queue);
}

fn record_after_phase_sort(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_phase_sort);
}

fn record_after_prepare_resources(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_resources);
}

fn record_after_prepare_resources_collect_phase_buffers(
    mut state: ResMut<HeadlessRenderScheduleTimingState>,
) {
    mark_now(&mut state.after_prepare_resources_collect_phase_buffers);
}

fn record_after_prepare_resources_flush(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_resources_flush);
}

fn record_after_prepare_bind_groups(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_bind_groups);
}

fn record_after_prepare(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare);
}

fn record_after_render(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_render);
}

fn record_after_cleanup(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_cleanup);
}

fn finalize_headless_render_schedule_metrics(
    state: Res<HeadlessRenderScheduleTimingState>,
    mut metrics: ResMut<HeadlessRenderScheduleMetrics>,
) {
    let frame_end = Instant::now();
    metrics.extract_commands_ms = elapsed_between(state.frame_start, state.after_extract_commands);
    metrics.prepare_meshes_ms =
        elapsed_between(state.after_extract_commands, state.after_prepare_meshes);
    metrics.manage_views_ms = elapsed_between(state.after_prepare_meshes, state.after_manage_views);
    metrics.queue_ms = elapsed_between(state.after_manage_views, state.after_queue);
    metrics.phase_sort_ms = elapsed_between(state.after_queue, state.after_phase_sort);
    metrics.prepare_resources_ms =
        elapsed_between(state.after_phase_sort, state.after_prepare_resources);
    metrics.prepare_resources_collect_phase_buffers_ms = elapsed_between(
        state.after_prepare_resources,
        state.after_prepare_resources_collect_phase_buffers,
    );
    metrics.prepare_resources_flush_ms = elapsed_between(
        state.after_prepare_resources_collect_phase_buffers,
        state.after_prepare_resources_flush,
    );
    metrics.prepare_bind_groups_ms = elapsed_between(
        state.after_prepare_resources_flush,
        state.after_prepare_bind_groups,
    );
    metrics.prepare_other_ms =
        elapsed_between(state.after_prepare_bind_groups, state.after_prepare);
    metrics.prepare_ms = elapsed_between(state.after_phase_sort, state.after_prepare);
    metrics.render_ms = elapsed_between(state.after_prepare, state.after_render);
    metrics.cleanup_ms = elapsed_between(state.after_render, state.after_cleanup);
    metrics.post_cleanup_ms = elapsed_between(state.after_cleanup, Some(frame_end));
}

fn run_headless_update(app: &mut App) -> HeadlessUpdateBreakdown {
    let mut breakdown = HeadlessUpdateBreakdown::default();
    let sub_apps = app.sub_apps_mut();
    let (main_app, render_sub_apps) = (&mut sub_apps.main, &mut sub_apps.sub_apps);

    reset_headless_main_schedule_timing(main_app.world_mut());
    let main_schedule_start = Instant::now();
    main_app.run_default_schedule();
    breakdown.main_schedule_ms = elapsed_ms(main_schedule_start);
    breakdown.main_schedule =
        snapshot_headless_main_schedule_metrics(main_app.world_mut(), breakdown.main_schedule_ms);

    if let Some(render_app) = render_sub_apps.get_mut(&RenderApp.intern()) {
        let render_extract_start = Instant::now();
        render_app.extract(main_app.world_mut());
        breakdown.render_extract_ms = elapsed_ms(render_extract_start);

        let render_app_start = Instant::now();
        reset_headless_render_schedule_timing(render_app.world_mut());
        render_app.world_mut().run_schedule(Render);
        breakdown.render_app_ms = elapsed_ms(render_app_start);
        breakdown.render_schedule = *render_app
            .world()
            .resource::<HeadlessRenderScheduleMetrics>();
    }

    let clear_trackers_start = Instant::now();
    main_app.world_mut().clear_trackers();
    breakdown.main_clear_trackers_ms = elapsed_ms(clear_trackers_start);

    breakdown
}

fn headless_sensor_runner(mut app: App) -> AppExit {
    app.finish();
    app.cleanup();

    let server = match RenderBridgeServer::bind() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("Failed to bind render bridge socket: {e}");
            return AppExit::Error(1.try_into().unwrap());
        }
    };

    // Warm-up: run updates until DB metadata is loaded and sensor cameras are spawned.
    let mut cameras_enabled = false;
    for i in 0..120 {
        run_headless_update(&mut app);
        let cameras_ready = app.world().resource::<SensorCamerasSpawned>().0;
        if cameras_ready && !cameras_enabled {
            set_all_sensor_cameras_active(app.world_mut(), true);
            cameras_enabled = true;
            tracing::info!("Sensor cameras spawned and enabled after {i} warm-up cycles");
            for _ in 0..4 {
                run_headless_update(&mut app);
            }
            set_all_sensor_cameras_active(app.world_mut(), false);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    tracing::info!(
        "Render server ready (cameras_enabled={cameras_enabled}), waiting for client connection..."
    );

    // Accept persistent client connection (blocking).
    if let Err(e) = server.accept_client() {
        tracing::error!("Failed to accept client connection: {e}");
        return AppExit::Error(1.try_into().unwrap());
    }
    tracing::debug!("Client connected");

    // Main loop: blocking read on persistent connection, single update per request.
    loop {
        if let Some(exit) = app.should_exit() {
            return exit;
        }

        // Block until a batch request arrives (or connection closes).
        let Some(request) = server.recv_batch() else {
            tracing::info!("Client disconnected, exiting render server");
            return AppExit::Success;
        };

        let request_start = Instant::now();

        // Set timestamp for this request.
        let setup_start = Instant::now();
        app.world_mut().resource_mut::<CurrentTimestamp>().0 = request.timestamp;

        // Check if cameras are ready.
        let cameras_ready = app.world().resource::<SensorCamerasSpawned>().0;

        // If cameras just became ready (spawned during main loop), enable them now.
        if cameras_ready && !cameras_enabled {
            set_all_sensor_cameras_active(app.world_mut(), true);
            cameras_enabled = true;
            tracing::info!("Sensor cameras late-enabled during main loop");
            for _ in 0..4 {
                run_headless_update(&mut app);
            }
            set_all_sensor_cameras_active(app.world_mut(), false);
        }

        if cameras_ready {
            set_sensor_cameras_active(app.world_mut(), &request.camera_names, true);
            drain_stale_frames(&app);
            set_readback_armed(app.world_mut(), &request.camera_names, true);
        }
        let setup_ms = elapsed_ms(setup_start);

        // With PipelinedRenderingPlugin disabled, Extract + Render run synchronously in one update.
        let update0_breakdown = run_headless_update(&mut app);
        let update0_ms = update0_breakdown.total_ms();
        let render_metrics = app
            .get_sub_app_mut(RenderApp)
            .map(|render_app| *render_app.world().resource::<SensorCameraRenderMetrics>())
            .unwrap_or_default();

        if cameras_ready {
            let collect0_start = Instant::now();
            let mut frames = collect_frames(&app, &request.camera_names);
            let collect0_ms = elapsed_ms(collect0_start);
            let frames_after_update0 = frames.len();
            let (fallback_used, fallback_update_ms, fallback_breakdown, collect1_ms) =
                if frames.len() < request.camera_names.len() {
                    let fallback_breakdown = run_headless_update(&mut app);
                    let fallback_update_ms = fallback_breakdown.total_ms();
                    let collect1_start = Instant::now();
                    let more = collect_frames(&app, &request.camera_names);
                    for (name, data) in more {
                        if !frames.iter().any(|(existing, _)| existing == &name) {
                            frames.push((name, data));
                        }
                    }
                    (
                        true,
                        fallback_update_ms,
                        fallback_breakdown,
                        elapsed_ms(collect1_start),
                    )
                } else {
                    (false, 0.0, HeadlessUpdateBreakdown::default(), 0.0)
                };
            let final_frame_count = frames.len();

            set_readback_armed(app.world_mut(), &request.camera_names, false);
            set_sensor_cameras_active(app.world_mut(), &request.camera_names, false);

            let respond_start = Instant::now();
            let respond_metrics = match server.respond_batch(request.timestamp, &frames) {
                Ok(metrics) => metrics,
                Err(e) => {
                    tracing::warn!("Render bridge write failed, client disconnected: {e}");
                    break;
                }
            };
            let respond_ms = elapsed_ms(respond_start);
            let total_request_ms = elapsed_ms(request_start);
            if total_request_ms > RENDER_CRITICAL_MS {
                tracing::warn!(
                    total_request_ms,
                    camera_count = request.camera_names.len(),
                    setup_ms,
                    update0_ms,
                    update0_main_schedule_ms = update0_breakdown.main_schedule_ms,
                    update0_main_setup_cell_ms = update0_breakdown.main_schedule.setup_cell_ms,
                    update0_main_apply_cached_data_ms =
                        update0_breakdown.main_schedule.apply_cached_data_ms,
                    update0_main_update_object_3d_system_ms =
                        update0_breakdown.main_schedule.update_object_3d_system_ms,
                    update0_main_queue_object_3d_sync_candidates_ms = update0_breakdown
                        .main_schedule
                        .queue_object_3d_sync_candidates_ms,
                    update0_main_sync_object_3d_ms =
                        update0_breakdown.main_schedule.sync_object_3d_ms,
                    update0_main_sync_pos_ms = update0_breakdown.main_schedule.sync_pos_ms,
                    update0_main_update_eql_context_ms =
                        update0_breakdown.main_schedule.update_eql_context_ms,
                    update0_main_load_headless_scene_ms =
                        update0_breakdown.main_schedule.load_headless_scene_ms,
                    update0_main_other_ms = update0_breakdown.main_schedule.other_ms,
                    update0_render_extract_ms = update0_breakdown.render_extract_ms,
                    update0_render_app_ms = update0_breakdown.render_app_ms,
                    update0_render_extract_commands_ms =
                        update0_breakdown.render_schedule.extract_commands_ms,
                    update0_render_prepare_meshes_ms =
                        update0_breakdown.render_schedule.prepare_meshes_ms,
                    update0_render_manage_views_ms =
                        update0_breakdown.render_schedule.manage_views_ms,
                    update0_render_queue_ms = update0_breakdown.render_schedule.queue_ms,
                    update0_render_phase_sort_ms = update0_breakdown.render_schedule.phase_sort_ms,
                    update0_render_prepare_resources_ms =
                        update0_breakdown.render_schedule.prepare_resources_ms,
                    update0_render_prepare_resources_collect_phase_buffers_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_collect_phase_buffers_ms,
                    update0_render_prepare_resources_flush_ms =
                        update0_breakdown.render_schedule.prepare_resources_flush_ms,
                    update0_render_prepare_bind_groups_ms =
                        update0_breakdown.render_schedule.prepare_bind_groups_ms,
                    update0_render_prepare_other_ms =
                        update0_breakdown.render_schedule.prepare_other_ms,
                    update0_render_prepare_ms = update0_breakdown.render_schedule.prepare_ms,
                    update0_render_render_ms = update0_breakdown.render_schedule.render_ms,
                    update0_render_cleanup_ms = update0_breakdown.render_schedule.cleanup_ms,
                    update0_render_post_cleanup_ms =
                        update0_breakdown.render_schedule.post_cleanup_ms,
                    update0_main_clear_trackers_ms = update0_breakdown.main_clear_trackers_ms,
                    collect0_ms,
                    fallback_used,
                    fallback_update_ms,
                    fallback_main_schedule_ms = fallback_breakdown.main_schedule_ms,
                    fallback_main_setup_cell_ms = fallback_breakdown.main_schedule.setup_cell_ms,
                    fallback_main_apply_cached_data_ms =
                        fallback_breakdown.main_schedule.apply_cached_data_ms,
                    fallback_main_update_object_3d_system_ms =
                        fallback_breakdown.main_schedule.update_object_3d_system_ms,
                    fallback_main_queue_object_3d_sync_candidates_ms = fallback_breakdown
                        .main_schedule
                        .queue_object_3d_sync_candidates_ms,
                    fallback_main_sync_object_3d_ms =
                        fallback_breakdown.main_schedule.sync_object_3d_ms,
                    fallback_main_sync_pos_ms = fallback_breakdown.main_schedule.sync_pos_ms,
                    fallback_main_update_eql_context_ms =
                        fallback_breakdown.main_schedule.update_eql_context_ms,
                    fallback_main_load_headless_scene_ms =
                        fallback_breakdown.main_schedule.load_headless_scene_ms,
                    fallback_main_other_ms = fallback_breakdown.main_schedule.other_ms,
                    fallback_render_extract_ms = fallback_breakdown.render_extract_ms,
                    fallback_render_app_ms = fallback_breakdown.render_app_ms,
                    fallback_render_extract_commands_ms =
                        fallback_breakdown.render_schedule.extract_commands_ms,
                    fallback_render_prepare_meshes_ms =
                        fallback_breakdown.render_schedule.prepare_meshes_ms,
                    fallback_render_manage_views_ms =
                        fallback_breakdown.render_schedule.manage_views_ms,
                    fallback_render_queue_ms = fallback_breakdown.render_schedule.queue_ms,
                    fallback_render_phase_sort_ms =
                        fallback_breakdown.render_schedule.phase_sort_ms,
                    fallback_render_prepare_resources_ms =
                        fallback_breakdown.render_schedule.prepare_resources_ms,
                    fallback_render_prepare_resources_collect_phase_buffers_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_collect_phase_buffers_ms,
                    fallback_render_prepare_resources_flush_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_flush_ms,
                    fallback_render_prepare_bind_groups_ms =
                        fallback_breakdown.render_schedule.prepare_bind_groups_ms,
                    fallback_render_prepare_other_ms =
                        fallback_breakdown.render_schedule.prepare_other_ms,
                    fallback_render_prepare_ms = fallback_breakdown.render_schedule.prepare_ms,
                    fallback_render_render_ms = fallback_breakdown.render_schedule.render_ms,
                    fallback_render_cleanup_ms = fallback_breakdown.render_schedule.cleanup_ms,
                    fallback_render_post_cleanup_ms =
                        fallback_breakdown.render_schedule.post_cleanup_ms,
                    fallback_main_clear_trackers_ms = fallback_breakdown.main_clear_trackers_ms,
                    collect1_ms,
                    respond_ms,
                    respond_header_write_ms = respond_metrics.response_header_write_ms,
                    respond_frame_header_write_ms = respond_metrics.frame_header_write_ms,
                    respond_frame_bytes_write_ms = respond_metrics.frame_bytes_write_ms,
                    respond_flush_ms = respond_metrics.flush_ms,
                    respond_frame_count = respond_metrics.frame_count,
                    respond_total_bytes = respond_metrics.total_bytes,
                    image_copy_driver_ms = render_metrics.image_copy_driver_ms,
                    image_copy_count = render_metrics.image_copy_count,
                    receive_image_poll_wait_ms = render_metrics.receive_image_poll_wait_ms,
                    receive_image_from_buffer_ms = render_metrics.receive_image_from_buffer_ms,
                    readback_camera_count = render_metrics.readback_camera_count,
                    frames_after_update0,
                    final_frame_count,
                    "Render request exceeded critical latency budget"
                );
            } else if total_request_ms > RENDER_TARGET_MS {
                tracing::info!(
                    total_request_ms,
                    camera_count = request.camera_names.len(),
                    setup_ms,
                    update0_ms,
                    update0_main_schedule_ms = update0_breakdown.main_schedule_ms,
                    update0_main_setup_cell_ms = update0_breakdown.main_schedule.setup_cell_ms,
                    update0_main_apply_cached_data_ms =
                        update0_breakdown.main_schedule.apply_cached_data_ms,
                    update0_main_update_object_3d_system_ms =
                        update0_breakdown.main_schedule.update_object_3d_system_ms,
                    update0_main_queue_object_3d_sync_candidates_ms = update0_breakdown
                        .main_schedule
                        .queue_object_3d_sync_candidates_ms,
                    update0_main_sync_object_3d_ms =
                        update0_breakdown.main_schedule.sync_object_3d_ms,
                    update0_main_sync_pos_ms = update0_breakdown.main_schedule.sync_pos_ms,
                    update0_main_update_eql_context_ms =
                        update0_breakdown.main_schedule.update_eql_context_ms,
                    update0_main_load_headless_scene_ms =
                        update0_breakdown.main_schedule.load_headless_scene_ms,
                    update0_main_other_ms = update0_breakdown.main_schedule.other_ms,
                    update0_render_extract_ms = update0_breakdown.render_extract_ms,
                    update0_render_app_ms = update0_breakdown.render_app_ms,
                    update0_render_extract_commands_ms =
                        update0_breakdown.render_schedule.extract_commands_ms,
                    update0_render_prepare_meshes_ms =
                        update0_breakdown.render_schedule.prepare_meshes_ms,
                    update0_render_manage_views_ms =
                        update0_breakdown.render_schedule.manage_views_ms,
                    update0_render_queue_ms = update0_breakdown.render_schedule.queue_ms,
                    update0_render_phase_sort_ms = update0_breakdown.render_schedule.phase_sort_ms,
                    update0_render_prepare_resources_ms =
                        update0_breakdown.render_schedule.prepare_resources_ms,
                    update0_render_prepare_resources_collect_phase_buffers_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_collect_phase_buffers_ms,
                    update0_render_prepare_resources_flush_ms =
                        update0_breakdown.render_schedule.prepare_resources_flush_ms,
                    update0_render_prepare_bind_groups_ms =
                        update0_breakdown.render_schedule.prepare_bind_groups_ms,
                    update0_render_prepare_other_ms =
                        update0_breakdown.render_schedule.prepare_other_ms,
                    update0_render_prepare_ms = update0_breakdown.render_schedule.prepare_ms,
                    update0_render_render_ms = update0_breakdown.render_schedule.render_ms,
                    update0_render_cleanup_ms = update0_breakdown.render_schedule.cleanup_ms,
                    update0_render_post_cleanup_ms =
                        update0_breakdown.render_schedule.post_cleanup_ms,
                    update0_main_clear_trackers_ms = update0_breakdown.main_clear_trackers_ms,
                    collect0_ms,
                    fallback_used,
                    fallback_update_ms,
                    fallback_main_schedule_ms = fallback_breakdown.main_schedule_ms,
                    fallback_main_setup_cell_ms = fallback_breakdown.main_schedule.setup_cell_ms,
                    fallback_main_apply_cached_data_ms =
                        fallback_breakdown.main_schedule.apply_cached_data_ms,
                    fallback_main_update_object_3d_system_ms =
                        fallback_breakdown.main_schedule.update_object_3d_system_ms,
                    fallback_main_queue_object_3d_sync_candidates_ms = fallback_breakdown
                        .main_schedule
                        .queue_object_3d_sync_candidates_ms,
                    fallback_main_sync_object_3d_ms =
                        fallback_breakdown.main_schedule.sync_object_3d_ms,
                    fallback_main_sync_pos_ms = fallback_breakdown.main_schedule.sync_pos_ms,
                    fallback_main_update_eql_context_ms =
                        fallback_breakdown.main_schedule.update_eql_context_ms,
                    fallback_main_load_headless_scene_ms =
                        fallback_breakdown.main_schedule.load_headless_scene_ms,
                    fallback_main_other_ms = fallback_breakdown.main_schedule.other_ms,
                    fallback_render_extract_ms = fallback_breakdown.render_extract_ms,
                    fallback_render_app_ms = fallback_breakdown.render_app_ms,
                    fallback_render_extract_commands_ms =
                        fallback_breakdown.render_schedule.extract_commands_ms,
                    fallback_render_prepare_meshes_ms =
                        fallback_breakdown.render_schedule.prepare_meshes_ms,
                    fallback_render_manage_views_ms =
                        fallback_breakdown.render_schedule.manage_views_ms,
                    fallback_render_queue_ms = fallback_breakdown.render_schedule.queue_ms,
                    fallback_render_phase_sort_ms =
                        fallback_breakdown.render_schedule.phase_sort_ms,
                    fallback_render_prepare_resources_ms =
                        fallback_breakdown.render_schedule.prepare_resources_ms,
                    fallback_render_prepare_resources_collect_phase_buffers_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_collect_phase_buffers_ms,
                    fallback_render_prepare_resources_flush_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_flush_ms,
                    fallback_render_prepare_bind_groups_ms =
                        fallback_breakdown.render_schedule.prepare_bind_groups_ms,
                    fallback_render_prepare_other_ms =
                        fallback_breakdown.render_schedule.prepare_other_ms,
                    fallback_render_prepare_ms = fallback_breakdown.render_schedule.prepare_ms,
                    fallback_render_render_ms = fallback_breakdown.render_schedule.render_ms,
                    fallback_render_cleanup_ms = fallback_breakdown.render_schedule.cleanup_ms,
                    fallback_render_post_cleanup_ms =
                        fallback_breakdown.render_schedule.post_cleanup_ms,
                    fallback_main_clear_trackers_ms = fallback_breakdown.main_clear_trackers_ms,
                    collect1_ms,
                    respond_ms,
                    respond_header_write_ms = respond_metrics.response_header_write_ms,
                    respond_frame_header_write_ms = respond_metrics.frame_header_write_ms,
                    respond_frame_bytes_write_ms = respond_metrics.frame_bytes_write_ms,
                    respond_flush_ms = respond_metrics.flush_ms,
                    respond_frame_count = respond_metrics.frame_count,
                    respond_total_bytes = respond_metrics.total_bytes,
                    image_copy_driver_ms = render_metrics.image_copy_driver_ms,
                    image_copy_count = render_metrics.image_copy_count,
                    receive_image_poll_wait_ms = render_metrics.receive_image_poll_wait_ms,
                    receive_image_from_buffer_ms = render_metrics.receive_image_from_buffer_ms,
                    readback_camera_count = render_metrics.readback_camera_count,
                    frames_after_update0,
                    final_frame_count,
                    "Render request exceeded target latency"
                );
            }
        } else if let Err(e) = server.respond_empty() {
            tracing::warn!("Render bridge write failed, client disconnected: {e}");
            break;
        }
    }
    AppExit::Success
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect rendered frames from the frame receiver, matching requested camera names.
fn collect_frames(app: &App, camera_names: &[String]) -> Vec<(String, Vec<u8>)> {
    let world = app.world();
    let frame_rx = world.resource::<crate::sensor_camera::SensorFrameReceiver>();

    let mut frames_map: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();

    // Drain all queued frames, keeping the latest for each camera.
    while let Ok((camera_name, frame_bytes, _, _)) = frame_rx.0.try_recv() {
        frames_map.insert(camera_name, frame_bytes);
    }

    // Return frames in the order they were requested.
    camera_names
        .iter()
        .filter_map(|name| frames_map.remove(name).map(|bytes| (name.clone(), bytes)))
        .collect()
}
