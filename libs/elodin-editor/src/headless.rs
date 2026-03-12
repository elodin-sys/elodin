use std::{
    fmt::Write as _,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use bevy::ecs::system::{RunSystemError, ScheduleSystem, System, SystemParamValidationError};
use bevy::{
    a11y::AccessibilityPlugin,
    animation::AnimationPlugin,
    app::{App, AppExit, AppLabel, Plugin, Startup},
    asset::{AssetPlugin, Assets, UnapprovedPathMode},
    audio::AudioPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore, FrameCount},
    ecs::system::IntoSystem,
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
                .init_resource::<HeadlessPrepareResourcesProbe>()
                .init_resource::<HeadlessViewUniformPrepareMetrics>()
                .add_systems(
                    Render,
                    (
                        (
                            record_after_extract_commands
                                .after(RenderSystems::ExtractCommands)
                                .before(RenderSystems::PrepareMeshes),
                            record_after_prepare_meshes
                                .after(RenderSystems::PrepareMeshes)
                                .before(RenderSystems::ManageViews),
                            record_after_clear_view_attachments
                                .after(bevy::render::view::clear_view_attachments)
                                .before(bevy::render::view::prepare_view_attachments),
                            record_after_prepare_view_attachments
                                .after(bevy::render::view::prepare_view_attachments)
                                .before(bevy::render::view::prepare_view_targets),
                            record_after_prepare_view_targets
                                .after(bevy::render::view::prepare_view_targets)
                                .before(record_after_manage_views),
                            record_after_manage_views
                                .after(RenderSystems::ManageViews)
                                .before(RenderSystems::Queue),
                            record_after_queue_meshes
                                .after(RenderSystems::QueueMeshes)
                                .before(RenderSystems::QueueSweep),
                            record_after_queue_sweep
                                .after(RenderSystems::QueueSweep)
                                .before(record_after_queue),
                            record_after_queue
                                .after(RenderSystems::Queue)
                                .before(RenderSystems::PhaseSort),
                            record_after_phase_sort
                                .after(RenderSystems::PhaseSort)
                                .before(RenderSystems::Prepare),
                        ),
                        (
                            record_after_prepare_view_uniforms
                                .after(bevy::render::view::prepare_view_uniforms)
                                .before(bevy::pbr::prepare_clusters),
                            record_after_prepare_clusters
                                .after(bevy::pbr::prepare_clusters)
                                .before(
                                    bevy::core_pipeline::core_3d::prepare_core_3d_depth_textures,
                                ),
                            record_after_prepare_core_3d_depth_textures
                                .after(
                                    bevy::core_pipeline::core_3d::prepare_core_3d_depth_textures,
                                )
                                .before(
                                    bevy::core_pipeline::core_3d::prepare_core_3d_transmission_textures,
                                ),
                            record_after_prepare_core_3d_transmission_textures
                                .after(
                                    bevy::core_pipeline::core_3d::prepare_core_3d_transmission_textures,
                                )
                                .before(bevy::core_pipeline::core_3d::prepare_prepass_textures),
                            record_after_prepare_prepass_textures
                                .after(bevy::core_pipeline::core_3d::prepare_prepass_textures)
                                .before(record_after_prepare_resources),
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
                        ),
                        (
                            record_after_render
                                .after(RenderSystems::Render)
                                .before(RenderSystems::Cleanup),
                            record_after_cleanup
                                .after(RenderSystems::Cleanup)
                                .before(RenderSystems::PostCleanup),
                            finalize_headless_render_schedule_metrics
                                .after(RenderSystems::PostCleanup),
                        ),
                    ),
                );
            install_headless_prepare_resources_probes(render_app);
            install_headless_prepare_view_uniforms_override(render_app);
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
    apply_cached_data: impeller2_bevy::ApplyCachedDataMetrics,
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
    manage_views_clear_view_attachments_ms: f64,
    manage_views_prepare_view_attachments_ms: f64,
    manage_views_prepare_view_targets_ms: f64,
    manage_views_other_ms: f64,
    manage_views_ms: f64,
    queue_before_sweep_ms: f64,
    queue_sweep_ms: f64,
    queue_after_sweep_ms: f64,
    queue_ms: f64,
    phase_sort_ms: f64,
    prepare_resources_view_uniforms_ms: f64,
    prepare_resources_view_uniforms_clear_ms: f64,
    prepare_resources_view_uniforms_build_ms: f64,
    prepare_resources_view_uniforms_write_buffer_ms: f64,
    prepare_resources_globals_buffer_ms: f64,
    prepare_resources_uniform_components_ms: f64,
    prepare_resources_gpu_component_array_buffers_ms: f64,
    prepare_resources_gpu_readback_prepare_buffers_ms: f64,
    prepare_resources_before_view_uniforms_other_ms: f64,
    prepare_resources_clusters_ms: f64,
    prepare_resources_core_3d_depth_textures_ms: f64,
    prepare_resources_core_3d_transmission_textures_ms: f64,
    prepare_resources_prepass_textures_ms: f64,
    prepare_resources_other_ms: f64,
    prepare_resources_ms: f64,
    prepare_resources_collect_phase_buffers_ms: f64,
    prepare_resources_flush_ms: f64,
    prepare_bind_groups_ms: f64,
    prepare_other_ms: f64,
    prepare_ms: f64,
    render_ms: f64,
    cleanup_ms: f64,
    post_cleanup_ms: f64,
    extracted_camera_count: usize,
    extracted_view_count: usize,
    view_uniform_offset_count: usize,
    extracted_cluster_config_count: usize,
    view_cluster_bindings_count: usize,
    view_target_count: usize,
    view_depth_texture_count: usize,
    view_prepass_texture_count: usize,
    view_transmission_texture_count: usize,
    no_indirect_drawing_view_count: usize,
    occlusion_culling_view_count: usize,
}

#[derive(Clone, Copy, Debug, Default, Resource)]
struct HeadlessViewUniformPrepareMetrics {
    clear_ms: f64,
    build_ms: f64,
    write_buffer_ms: f64,
}

#[derive(Debug, Clone, Copy)]
enum HeadlessPrepareResourcesProbeKind {
    GlobalsBuffer,
    UniformComponents,
    GpuComponentArrayBuffers,
    GpuReadbackPrepareBuffers,
}

#[derive(Debug, Default)]
struct HeadlessPrepareResourcesProbeInner {
    globals_buffer_ns: AtomicU64,
    uniform_components_ns: AtomicU64,
    gpu_component_array_buffers_ns: AtomicU64,
    gpu_readback_prepare_buffers_ns: AtomicU64,
}

impl HeadlessPrepareResourcesProbeInner {
    fn reset(&self) {
        self.globals_buffer_ns.store(0, Ordering::Relaxed);
        self.uniform_components_ns.store(0, Ordering::Relaxed);
        self.gpu_component_array_buffers_ns
            .store(0, Ordering::Relaxed);
        self.gpu_readback_prepare_buffers_ns
            .store(0, Ordering::Relaxed);
    }

    fn record(&self, kind: HeadlessPrepareResourcesProbeKind, elapsed: Duration) {
        let elapsed_ns = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        match kind {
            HeadlessPrepareResourcesProbeKind::GlobalsBuffer => {
                self.globals_buffer_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
            }
            HeadlessPrepareResourcesProbeKind::UniformComponents => {
                self.uniform_components_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
            }
            HeadlessPrepareResourcesProbeKind::GpuComponentArrayBuffers => {
                self.gpu_component_array_buffers_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
            }
            HeadlessPrepareResourcesProbeKind::GpuReadbackPrepareBuffers => {
                self.gpu_readback_prepare_buffers_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
            }
        }
    }

    fn snapshot(&self) -> HeadlessPrepareResourcesProbeSnapshot {
        HeadlessPrepareResourcesProbeSnapshot {
            globals_buffer_ms: self.globals_buffer_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            uniform_components_ms: self.uniform_components_ns.load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            gpu_component_array_buffers_ms: self
                .gpu_component_array_buffers_ns
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0,
            gpu_readback_prepare_buffers_ms: self
                .gpu_readback_prepare_buffers_ns
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct HeadlessPrepareResourcesProbeSnapshot {
    globals_buffer_ms: f64,
    uniform_components_ms: f64,
    gpu_component_array_buffers_ms: f64,
    gpu_readback_prepare_buffers_ms: f64,
}

#[derive(Resource, Clone, Default)]
struct HeadlessPrepareResourcesProbe(Arc<HeadlessPrepareResourcesProbeInner>);

struct HeadlessTimedScheduleSystem {
    inner: ScheduleSystem,
    kind: HeadlessPrepareResourcesProbeKind,
    probe: Arc<HeadlessPrepareResourcesProbeInner>,
}

impl HeadlessTimedScheduleSystem {
    fn new(
        inner: ScheduleSystem,
        kind: HeadlessPrepareResourcesProbeKind,
        probe: Arc<HeadlessPrepareResourcesProbeInner>,
    ) -> Self {
        Self { inner, kind, probe }
    }
}

impl System for HeadlessTimedScheduleSystem {
    type In = ();
    type Out = ();

    fn name(&self) -> bevy::utils::prelude::DebugName {
        self.inner.name()
    }

    fn type_id(&self) -> std::any::TypeId {
        self.inner.type_id()
    }

    fn flags(&self) -> bevy::ecs::system::SystemStateFlags {
        self.inner.flags()
    }

    unsafe fn run_unsafe(
        &mut self,
        input: bevy::prelude::SystemIn<'_, Self>,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell,
    ) -> Result<Self::Out, RunSystemError> {
        let start = Instant::now();
        let result = unsafe { self.inner.run_unsafe(input, world) };
        self.probe.record(self.kind, start.elapsed());
        result
    }

    fn apply_deferred(&mut self, world: &mut World) {
        self.inner.apply_deferred(world);
    }

    fn queue_deferred(&mut self, world: bevy::ecs::world::DeferredWorld) {
        self.inner.queue_deferred(world);
    }

    unsafe fn validate_param_unsafe(
        &mut self,
        world: bevy::ecs::world::unsafe_world_cell::UnsafeWorldCell,
    ) -> Result<(), SystemParamValidationError> {
        unsafe { self.inner.validate_param_unsafe(world) }
    }

    fn initialize(&mut self, world: &mut World) -> bevy::ecs::query::FilteredAccessSet {
        self.inner.initialize(world)
    }

    fn check_change_tick(&mut self, check: bevy::ecs::component::CheckChangeTicks) {
        self.inner.check_change_tick(check);
    }

    fn default_system_sets(&self) -> Vec<bevy::ecs::schedule::InternedSystemSet> {
        self.inner.default_system_sets()
    }

    fn get_last_run(&self) -> bevy::ecs::component::Tick {
        self.inner.get_last_run()
    }

    fn set_last_run(&mut self, last_run: bevy::ecs::component::Tick) {
        self.inner.set_last_run(last_run);
    }
}

#[derive(Debug, Default, Resource)]
struct HeadlessRenderScheduleTimingState {
    frame_start: Option<Instant>,
    after_extract_commands: Option<Instant>,
    after_prepare_meshes: Option<Instant>,
    after_clear_view_attachments: Option<Instant>,
    after_prepare_view_attachments: Option<Instant>,
    after_prepare_view_targets: Option<Instant>,
    after_manage_views: Option<Instant>,
    after_queue_sweep: Option<Instant>,
    after_queue_meshes: Option<Instant>,
    after_queue: Option<Instant>,
    after_phase_sort: Option<Instant>,
    after_prepare_view_uniforms: Option<Instant>,
    after_prepare_clusters: Option<Instant>,
    after_prepare_core_3d_depth_textures: Option<Instant>,
    after_prepare_core_3d_transmission_textures: Option<Instant>,
    after_prepare_prepass_textures: Option<Instant>,
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
    let (
        setup_cell_ms,
        apply_cached_data_ms,
        update_object_3d_system_ms,
        queue_object_3d_sync_candidates_ms,
        sync_object_3d_ms,
        sync_pos_ms,
        update_eql_context_ms,
        load_headless_scene_ms,
    ) = {
        let state = world.resource::<HeadlessMainScheduleTimingState>();
        (
            elapsed_between(state.after_sink, state.after_setup_cell),
            elapsed_between(state.after_setup_cell, state.after_apply_cached_data),
            elapsed_between(
                state.after_apply_cached_data,
                state.after_update_object_3d_system,
            ),
            elapsed_between(
                state.after_update_object_3d_system,
                state.after_queue_object_3d_sync_candidates,
            ),
            elapsed_between(
                state.after_queue_object_3d_sync_candidates,
                state.after_sync_object_3d,
            ),
            elapsed_between(state.after_sync_object_3d, state.after_sync_pos),
            elapsed_between(
                state.before_update_eql_context,
                state.after_update_eql_context,
            ),
            elapsed_between(
                state.after_update_eql_context,
                state.after_load_headless_scene,
            ),
        )
    };
    let mut metrics = HeadlessMainScheduleMetrics {
        setup_cell_ms,
        apply_cached_data_ms,
        update_object_3d_system_ms,
        queue_object_3d_sync_candidates_ms,
        sync_object_3d_ms,
        sync_pos_ms,
        update_eql_context_ms,
        load_headless_scene_ms,
        apply_cached_data: *world.resource::<impeller2_bevy::ApplyCachedDataMetrics>(),
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

fn format_apply_cached_data_top_components(
    world: &World,
    metrics: impeller2_bevy::ApplyCachedDataMetrics,
) -> String {
    let Some(metadata_reg) = world.get_resource::<impeller2_bevy::ComponentMetadataRegistry>()
    else {
        return String::new();
    };
    let mut out = String::new();
    for hot in metrics.top_components {
        let Some(component_id) = hot.component_id else {
            continue;
        };
        if hot.bytes == 0 {
            continue;
        }
        let name = metadata_reg
            .get_metadata(&component_id)
            .map(|metadata| metadata.name.as_str())
            .unwrap_or("unknown");
        if !out.is_empty() {
            out.push(',');
        }
        let _ = write!(&mut out, "{}({})={}", name, component_id, hot.bytes);
    }
    out
}

fn reset_headless_render_schedule_timing(world: &mut World) {
    let mut state = world.resource_mut::<HeadlessRenderScheduleTimingState>();
    *state = HeadlessRenderScheduleTimingState {
        frame_start: Some(Instant::now()),
        ..default()
    };
    *world.resource_mut::<HeadlessRenderScheduleMetrics>() =
        HeadlessRenderScheduleMetrics::default();
    *world.resource_mut::<HeadlessViewUniformPrepareMetrics>() =
        HeadlessViewUniformPrepareMetrics::default();
    world.resource::<HeadlessPrepareResourcesProbe>().0.reset();
}

fn install_headless_prepare_view_uniforms_override(render_app: &mut bevy::app::SubApp) {
    render_app.edit_schedule(Render, |schedule| {
        let Some(system_key) = schedule
            .graph()
            .systems
            .iter()
            .find_map(|(key, system, _)| {
                system
                    .name()
                    .ends_with("prepare_view_uniforms")
                    .then_some(key)
            })
        else {
            tracing::warn!(
                "Failed to find bevy_render::view::prepare_view_uniforms for headless override"
            );
            return;
        };
        let Some(system) = schedule.graph_mut().systems.get_mut(system_key) else {
            tracing::warn!(
                "Failed to mutate prepare_view_uniforms system slot for headless override"
            );
            return;
        };
        system.system = Box::new(
            IntoSystem::into_system(headless_prepare_view_uniforms)
                .with_name("bevy_render::view::prepare_view_uniforms"),
        );
    });
}

fn noop_schedule_system() {}

fn prepare_resources_probe_kind_for_name(name: &str) -> Option<HeadlessPrepareResourcesProbeKind> {
    if name.ends_with("prepare_globals_buffer") {
        Some(HeadlessPrepareResourcesProbeKind::GlobalsBuffer)
    } else if name.contains("prepare_uniform_components") {
        Some(HeadlessPrepareResourcesProbeKind::UniformComponents)
    } else if name.contains("prepare_gpu_component_array_buffers") {
        Some(HeadlessPrepareResourcesProbeKind::GpuComponentArrayBuffers)
    } else if name.contains("gpu_readback") && name.ends_with("prepare_buffers") {
        Some(HeadlessPrepareResourcesProbeKind::GpuReadbackPrepareBuffers)
    } else {
        None
    }
}

fn install_headless_prepare_resources_probes(render_app: &mut bevy::app::SubApp) {
    let probe = render_app
        .world()
        .resource::<HeadlessPrepareResourcesProbe>()
        .0
        .clone();
    render_app.edit_schedule(Render, |schedule| {
        let system_keys: Vec<_> = schedule
            .graph()
            .systems
            .iter()
            .filter_map(|(key, system, _)| {
                prepare_resources_probe_kind_for_name(&system.name().to_string())
                    .map(|kind| (key, kind))
            })
            .collect();

        for (system_key, kind) in system_keys {
            let Some(system) = schedule.graph_mut().systems.get_mut(system_key) else {
                continue;
            };
            let inner = std::mem::replace(
                &mut system.system,
                Box::new(
                    IntoSystem::into_system(noop_schedule_system)
                        .with_name("headless::noop_prepare_resources_probe_placeholder"),
                ),
            );
            system.system = Box::new(HeadlessTimedScheduleSystem::new(inner, kind, probe.clone()));
        }
    });
}

fn headless_prepare_view_uniforms(
    mut commands: Commands,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
    mut view_uniforms: ResMut<bevy::render::view::ViewUniforms>,
    views: Query<(
        Entity,
        Option<&bevy::render::camera::ExtractedCamera>,
        &bevy::render::view::ExtractedView,
        Option<&bevy::camera::primitives::Frustum>,
        Option<&bevy::render::camera::TemporalJitter>,
        Option<&bevy::render::camera::MipBias>,
        Option<&bevy::camera::MainPassResolutionOverride>,
    )>,
    frame_count: Res<FrameCount>,
    mut metrics: ResMut<HeadlessViewUniformPrepareMetrics>,
) {
    let clear_start = Instant::now();
    view_uniforms.uniforms.clear();
    metrics.clear_ms = elapsed_ms(clear_start);

    if views.is_empty() {
        return;
    }

    let frame_count = frame_count.0;
    let default_exposure = bevy::camera::Exposure::default().exposure();
    let build_start = Instant::now();
    for (
        entity,
        extracted_camera,
        extracted_view,
        frustum,
        temporal_jitter,
        mip_bias,
        resolution_override,
    ) in &views
    {
        let viewport = extracted_view.viewport.as_vec4();
        let mut main_pass_viewport = viewport;
        if let Some(resolution_override) = resolution_override {
            main_pass_viewport.z = resolution_override.0.x as f32;
            main_pass_viewport.w = resolution_override.0.y as f32;
        }

        let unjittered_projection = extracted_view.clip_from_view;
        let mut clip_from_view = unjittered_projection;
        if let Some(temporal_jitter) = temporal_jitter {
            temporal_jitter.jitter_projection(
                &mut clip_from_view,
                Vec2::new(main_pass_viewport.z, main_pass_viewport.w),
            );
        }

        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let clip_from_world = if temporal_jitter.is_some() {
            clip_from_view * view_from_world
        } else {
            extracted_view
                .clip_from_world
                .unwrap_or_else(|| clip_from_view * view_from_world)
        };
        let frustum = frustum
            .map(|frustum| frustum.half_spaces.map(|half_space| half_space.normal_d()))
            .unwrap_or([Vec4::ZERO; 6]);

        let offset = view_uniforms
            .uniforms
            .push(&bevy::render::view::ViewUniform {
                clip_from_world,
                unjittered_clip_from_world: unjittered_projection * view_from_world,
                world_from_clip: world_from_view * view_from_clip,
                world_from_view,
                view_from_world,
                clip_from_view,
                view_from_clip,
                world_position: extracted_view.world_from_view.translation(),
                exposure: extracted_camera
                    .map(|camera| camera.exposure)
                    .unwrap_or(default_exposure),
                viewport,
                main_pass_viewport,
                frustum,
                color_grading: extracted_view.color_grading.clone().into(),
                mip_bias: mip_bias.map_or(0.0, |mip_bias| mip_bias.0),
                frame_count,
            });
        commands
            .entity(entity)
            .insert(bevy::render::view::ViewUniformOffset { offset });
    }
    metrics.build_ms = elapsed_ms(build_start);

    let write_buffer_start = Instant::now();
    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
    metrics.write_buffer_ms = elapsed_ms(write_buffer_start);
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

fn record_after_clear_view_attachments(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_clear_view_attachments);
}

fn record_after_prepare_view_attachments(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_view_attachments);
}

fn record_after_prepare_view_targets(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_view_targets);
}

fn record_after_manage_views(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_manage_views);
}

fn record_after_queue_meshes(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_queue_meshes);
}

fn record_after_queue_sweep(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_queue_sweep);
}

fn record_after_queue(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_queue);
}

fn record_after_phase_sort(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_phase_sort);
}

fn record_after_prepare_view_uniforms(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_view_uniforms);
}

fn record_after_prepare_clusters(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_clusters);
}

fn record_after_prepare_core_3d_depth_textures(
    mut state: ResMut<HeadlessRenderScheduleTimingState>,
) {
    mark_now(&mut state.after_prepare_core_3d_depth_textures);
}

fn record_after_prepare_core_3d_transmission_textures(
    mut state: ResMut<HeadlessRenderScheduleTimingState>,
) {
    mark_now(&mut state.after_prepare_core_3d_transmission_textures);
}

fn record_after_prepare_prepass_textures(mut state: ResMut<HeadlessRenderScheduleTimingState>) {
    mark_now(&mut state.after_prepare_prepass_textures);
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
    view_uniform_prepare_metrics: Res<HeadlessViewUniformPrepareMetrics>,
    prepare_resources_probe: Res<HeadlessPrepareResourcesProbe>,
    extracted_cameras: Query<(), With<bevy::render::camera::ExtractedCamera>>,
    extracted_views: Query<(), With<bevy::render::view::ExtractedView>>,
    view_uniform_offsets: Query<(), With<bevy::render::view::ViewUniformOffset>>,
    extracted_cluster_configs: Query<(), With<bevy::pbr::ExtractedClusterConfig>>,
    view_cluster_bindings: Query<(), With<bevy::pbr::ViewClusterBindings>>,
    view_targets: Query<(), With<bevy::render::view::ViewTarget>>,
    view_depth_textures: Query<(), With<bevy::render::view::ViewDepthTexture>>,
    view_prepass_textures: Query<(), With<bevy::core_pipeline::prepass::ViewPrepassTextures>>,
    view_transmission_textures: Query<
        (),
        With<bevy::core_pipeline::core_3d::ViewTransmissionTexture>,
    >,
    no_indirect_drawing_views: Query<(), With<bevy::render::view::NoIndirectDrawing>>,
    occlusion_culling_views: Query<
        (),
        With<bevy::render::experimental::occlusion_culling::OcclusionCulling>,
    >,
) {
    let frame_end = Instant::now();
    let prepare_resources_probe = prepare_resources_probe.0.snapshot();
    metrics.extract_commands_ms = elapsed_between(state.frame_start, state.after_extract_commands);
    metrics.prepare_meshes_ms =
        elapsed_between(state.after_extract_commands, state.after_prepare_meshes);
    metrics.manage_views_clear_view_attachments_ms = elapsed_between(
        state.after_prepare_meshes,
        state.after_clear_view_attachments,
    );
    metrics.manage_views_prepare_view_attachments_ms = elapsed_between(
        state.after_clear_view_attachments,
        state.after_prepare_view_attachments,
    );
    metrics.manage_views_prepare_view_targets_ms = elapsed_between(
        state.after_prepare_view_attachments,
        state.after_prepare_view_targets,
    );
    metrics.manage_views_other_ms =
        elapsed_between(state.after_prepare_view_targets, state.after_manage_views);
    metrics.manage_views_ms = elapsed_between(state.after_prepare_meshes, state.after_manage_views);
    metrics.queue_before_sweep_ms =
        elapsed_between(state.after_manage_views, state.after_queue_meshes);
    metrics.queue_sweep_ms = elapsed_between(state.after_queue_meshes, state.after_queue_sweep);
    metrics.queue_after_sweep_ms = elapsed_between(state.after_queue_sweep, state.after_queue);
    metrics.queue_ms = elapsed_between(state.after_manage_views, state.after_queue);
    metrics.phase_sort_ms = elapsed_between(state.after_queue, state.after_phase_sort);
    metrics.prepare_resources_view_uniforms_ms =
        elapsed_between(state.after_phase_sort, state.after_prepare_view_uniforms);
    metrics.prepare_resources_view_uniforms_clear_ms = view_uniform_prepare_metrics.clear_ms;
    metrics.prepare_resources_view_uniforms_build_ms = view_uniform_prepare_metrics.build_ms;
    metrics.prepare_resources_view_uniforms_write_buffer_ms =
        view_uniform_prepare_metrics.write_buffer_ms;
    metrics.prepare_resources_globals_buffer_ms = prepare_resources_probe.globals_buffer_ms;
    metrics.prepare_resources_uniform_components_ms = prepare_resources_probe.uniform_components_ms;
    metrics.prepare_resources_gpu_component_array_buffers_ms =
        prepare_resources_probe.gpu_component_array_buffers_ms;
    metrics.prepare_resources_gpu_readback_prepare_buffers_ms =
        prepare_resources_probe.gpu_readback_prepare_buffers_ms;
    let view_uniforms_inner_ms = metrics.prepare_resources_view_uniforms_clear_ms
        + metrics.prepare_resources_view_uniforms_build_ms
        + metrics.prepare_resources_view_uniforms_write_buffer_ms;
    let before_view_uniforms_known_ms = metrics.prepare_resources_globals_buffer_ms
        + metrics.prepare_resources_uniform_components_ms
        + metrics.prepare_resources_gpu_component_array_buffers_ms
        + metrics.prepare_resources_gpu_readback_prepare_buffers_ms
        + view_uniforms_inner_ms;
    metrics.prepare_resources_before_view_uniforms_other_ms =
        (metrics.prepare_resources_view_uniforms_ms - before_view_uniforms_known_ms).max(0.0);
    metrics.prepare_resources_clusters_ms = elapsed_between(
        state.after_prepare_view_uniforms,
        state.after_prepare_clusters,
    );
    metrics.prepare_resources_core_3d_depth_textures_ms = elapsed_between(
        state.after_prepare_clusters,
        state.after_prepare_core_3d_depth_textures,
    );
    metrics.prepare_resources_core_3d_transmission_textures_ms = elapsed_between(
        state.after_prepare_core_3d_depth_textures,
        state.after_prepare_core_3d_transmission_textures,
    );
    metrics.prepare_resources_prepass_textures_ms = elapsed_between(
        state.after_prepare_core_3d_transmission_textures,
        state.after_prepare_prepass_textures,
    );
    metrics.prepare_resources_ms =
        elapsed_between(state.after_phase_sort, state.after_prepare_resources);
    let prepare_resources_measured_ms = metrics.prepare_resources_view_uniforms_ms
        + metrics.prepare_resources_clusters_ms
        + metrics.prepare_resources_core_3d_depth_textures_ms
        + metrics.prepare_resources_core_3d_transmission_textures_ms
        + metrics.prepare_resources_prepass_textures_ms;
    metrics.prepare_resources_other_ms =
        (metrics.prepare_resources_ms - prepare_resources_measured_ms).max(0.0);
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
    metrics.extracted_camera_count = extracted_cameras.iter().len();
    metrics.extracted_view_count = extracted_views.iter().len();
    metrics.view_uniform_offset_count = view_uniform_offsets.iter().len();
    metrics.extracted_cluster_config_count = extracted_cluster_configs.iter().len();
    metrics.view_cluster_bindings_count = view_cluster_bindings.iter().len();
    metrics.view_target_count = view_targets.iter().len();
    metrics.view_depth_texture_count = view_depth_textures.iter().len();
    metrics.view_prepass_texture_count = view_prepass_textures.iter().len();
    metrics.view_transmission_texture_count = view_transmission_textures.iter().len();
    metrics.no_indirect_drawing_view_count = no_indirect_drawing_views.iter().len();
    metrics.occlusion_culling_view_count = occlusion_culling_views.iter().len();
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
                let update0_main_apply_cached_data_top_components =
                    format_apply_cached_data_top_components(
                        app.world(),
                        update0_breakdown.main_schedule.apply_cached_data,
                    );
                let fallback_main_apply_cached_data_top_components =
                    format_apply_cached_data_top_components(
                        app.world(),
                        fallback_breakdown.main_schedule.apply_cached_data,
                    );
                tracing::warn!(
                    total_request_ms,
                    camera_count = request.camera_names.len(),
                    setup_ms,
                    update0_ms,
                    update0_main_schedule_ms = update0_breakdown.main_schedule_ms,
                    update0_main_setup_cell_ms = update0_breakdown.main_schedule.setup_cell_ms,
                    update0_main_apply_cached_data_ms =
                        update0_breakdown.main_schedule.apply_cached_data_ms,
                    update0_main_apply_cached_data_skipped =
                        update0_breakdown.main_schedule.apply_cached_data.skipped,
                    update0_main_apply_cached_data_scanned_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .scanned_component_count,
                    update0_main_apply_cached_data_applied_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .applied_component_count,
                    update0_main_apply_cached_data_missing_entity_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .missing_entity_count,
                    update0_main_apply_cached_data_total_bytes = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .total_bytes,
                    update0_main_apply_cached_data_inplace_copy_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .inplace_copy_component_count,
                    update0_main_apply_cached_data_cloned_replace_component_count =
                        update0_breakdown
                            .main_schedule
                            .apply_cached_data
                            .cloned_replace_component_count,
                    update0_main_apply_cached_data_inserted_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .inserted_component_count,
                    update0_main_apply_cached_data_adapter_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .adapter_count,
                    update0_main_apply_cached_data_top_components,
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
                    update0_render_manage_views_clear_view_attachments_ms = update0_breakdown
                        .render_schedule
                        .manage_views_clear_view_attachments_ms,
                    update0_render_manage_views_prepare_view_attachments_ms = update0_breakdown
                        .render_schedule
                        .manage_views_prepare_view_attachments_ms,
                    update0_render_manage_views_prepare_view_targets_ms = update0_breakdown
                        .render_schedule
                        .manage_views_prepare_view_targets_ms,
                    update0_render_manage_views_other_ms =
                        update0_breakdown.render_schedule.manage_views_other_ms,
                    update0_render_manage_views_ms =
                        update0_breakdown.render_schedule.manage_views_ms,
                    update0_render_queue_before_sweep_ms =
                        update0_breakdown.render_schedule.queue_before_sweep_ms,
                    update0_render_queue_sweep_ms =
                        update0_breakdown.render_schedule.queue_sweep_ms,
                    update0_render_queue_after_sweep_ms =
                        update0_breakdown.render_schedule.queue_after_sweep_ms,
                    update0_render_queue_ms = update0_breakdown.render_schedule.queue_ms,
                    update0_render_phase_sort_ms = update0_breakdown.render_schedule.phase_sort_ms,
                    update0_render_prepare_resources_ms =
                        update0_breakdown.render_schedule.prepare_resources_ms,
                    update0_render_prepare_resources_view_uniforms_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_ms,
                    update0_render_prepare_resources_view_uniforms_clear_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_clear_ms,
                    update0_render_prepare_resources_view_uniforms_build_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_build_ms,
                    update0_render_prepare_resources_view_uniforms_write_buffer_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_view_uniforms_write_buffer_ms,
                    update0_render_prepare_resources_globals_buffer_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_globals_buffer_ms,
                    update0_render_prepare_resources_uniform_components_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_uniform_components_ms,
                    update0_render_prepare_resources_gpu_component_array_buffers_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_gpu_component_array_buffers_ms,
                    update0_render_prepare_resources_gpu_readback_prepare_buffers_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_gpu_readback_prepare_buffers_ms,
                    update0_render_prepare_resources_before_view_uniforms_other_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_before_view_uniforms_other_ms,
                    update0_render_prepare_resources_clusters_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_clusters_ms,
                    update0_render_prepare_resources_core_3d_depth_textures_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_core_3d_depth_textures_ms,
                    update0_render_prepare_resources_core_3d_transmission_textures_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_core_3d_transmission_textures_ms,
                    update0_render_prepare_resources_prepass_textures_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_prepass_textures_ms,
                    update0_render_prepare_resources_other_ms =
                        update0_breakdown.render_schedule.prepare_resources_other_ms,
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
                    update0_render_extracted_camera_count =
                        update0_breakdown.render_schedule.extracted_camera_count,
                    update0_render_extracted_view_count =
                        update0_breakdown.render_schedule.extracted_view_count,
                    update0_render_view_uniform_offset_count =
                        update0_breakdown.render_schedule.view_uniform_offset_count,
                    update0_render_extracted_cluster_config_count = update0_breakdown
                        .render_schedule
                        .extracted_cluster_config_count,
                    update0_render_view_cluster_bindings_count = update0_breakdown
                        .render_schedule
                        .view_cluster_bindings_count,
                    update0_render_view_target_count =
                        update0_breakdown.render_schedule.view_target_count,
                    update0_render_view_depth_texture_count =
                        update0_breakdown.render_schedule.view_depth_texture_count,
                    update0_render_view_prepass_texture_count =
                        update0_breakdown.render_schedule.view_prepass_texture_count,
                    update0_render_view_transmission_texture_count = update0_breakdown
                        .render_schedule
                        .view_transmission_texture_count,
                    update0_render_no_indirect_drawing_view_count = update0_breakdown
                        .render_schedule
                        .no_indirect_drawing_view_count,
                    update0_render_occlusion_culling_view_count = update0_breakdown
                        .render_schedule
                        .occlusion_culling_view_count,
                    update0_main_clear_trackers_ms = update0_breakdown.main_clear_trackers_ms,
                    collect0_ms,
                    fallback_used,
                    fallback_update_ms,
                    fallback_main_schedule_ms = fallback_breakdown.main_schedule_ms,
                    fallback_main_setup_cell_ms = fallback_breakdown.main_schedule.setup_cell_ms,
                    fallback_main_apply_cached_data_ms =
                        fallback_breakdown.main_schedule.apply_cached_data_ms,
                    fallback_main_apply_cached_data_skipped =
                        fallback_breakdown.main_schedule.apply_cached_data.skipped,
                    fallback_main_apply_cached_data_scanned_component_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .scanned_component_count,
                    fallback_main_apply_cached_data_applied_component_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .applied_component_count,
                    fallback_main_apply_cached_data_missing_entity_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .missing_entity_count,
                    fallback_main_apply_cached_data_total_bytes = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .total_bytes,
                    fallback_main_apply_cached_data_inplace_copy_component_count =
                        fallback_breakdown
                            .main_schedule
                            .apply_cached_data
                            .inplace_copy_component_count,
                    fallback_main_apply_cached_data_cloned_replace_component_count =
                        fallback_breakdown
                            .main_schedule
                            .apply_cached_data
                            .cloned_replace_component_count,
                    fallback_main_apply_cached_data_inserted_component_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .inserted_component_count,
                    fallback_main_apply_cached_data_adapter_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .adapter_count,
                    fallback_main_apply_cached_data_top_components,
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
                    fallback_render_manage_views_clear_view_attachments_ms = fallback_breakdown
                        .render_schedule
                        .manage_views_clear_view_attachments_ms,
                    fallback_render_manage_views_prepare_view_attachments_ms = fallback_breakdown
                        .render_schedule
                        .manage_views_prepare_view_attachments_ms,
                    fallback_render_manage_views_prepare_view_targets_ms = fallback_breakdown
                        .render_schedule
                        .manage_views_prepare_view_targets_ms,
                    fallback_render_manage_views_other_ms =
                        fallback_breakdown.render_schedule.manage_views_other_ms,
                    fallback_render_manage_views_ms =
                        fallback_breakdown.render_schedule.manage_views_ms,
                    fallback_render_queue_before_sweep_ms =
                        fallback_breakdown.render_schedule.queue_before_sweep_ms,
                    fallback_render_queue_sweep_ms =
                        fallback_breakdown.render_schedule.queue_sweep_ms,
                    fallback_render_queue_after_sweep_ms =
                        fallback_breakdown.render_schedule.queue_after_sweep_ms,
                    fallback_render_queue_ms = fallback_breakdown.render_schedule.queue_ms,
                    fallback_render_phase_sort_ms =
                        fallback_breakdown.render_schedule.phase_sort_ms,
                    fallback_render_prepare_resources_ms =
                        fallback_breakdown.render_schedule.prepare_resources_ms,
                    fallback_render_prepare_resources_view_uniforms_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_ms,
                    fallback_render_prepare_resources_view_uniforms_clear_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_clear_ms,
                    fallback_render_prepare_resources_view_uniforms_build_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_build_ms,
                    fallback_render_prepare_resources_view_uniforms_write_buffer_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_view_uniforms_write_buffer_ms,
                    fallback_render_prepare_resources_globals_buffer_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_globals_buffer_ms,
                    fallback_render_prepare_resources_uniform_components_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_uniform_components_ms,
                    fallback_render_prepare_resources_gpu_component_array_buffers_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_gpu_component_array_buffers_ms,
                    fallback_render_prepare_resources_gpu_readback_prepare_buffers_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_gpu_readback_prepare_buffers_ms,
                    fallback_render_prepare_resources_before_view_uniforms_other_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_before_view_uniforms_other_ms,
                    fallback_render_prepare_resources_clusters_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_clusters_ms,
                    fallback_render_prepare_resources_core_3d_depth_textures_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_core_3d_depth_textures_ms,
                    fallback_render_prepare_resources_core_3d_transmission_textures_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_core_3d_transmission_textures_ms,
                    fallback_render_prepare_resources_prepass_textures_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_prepass_textures_ms,
                    fallback_render_prepare_resources_other_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_other_ms,
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
                    fallback_render_extracted_camera_count =
                        fallback_breakdown.render_schedule.extracted_camera_count,
                    fallback_render_extracted_view_count =
                        fallback_breakdown.render_schedule.extracted_view_count,
                    fallback_render_view_uniform_offset_count =
                        fallback_breakdown.render_schedule.view_uniform_offset_count,
                    fallback_render_extracted_cluster_config_count = fallback_breakdown
                        .render_schedule
                        .extracted_cluster_config_count,
                    fallback_render_view_cluster_bindings_count = fallback_breakdown
                        .render_schedule
                        .view_cluster_bindings_count,
                    fallback_render_view_target_count =
                        fallback_breakdown.render_schedule.view_target_count,
                    fallback_render_view_depth_texture_count =
                        fallback_breakdown.render_schedule.view_depth_texture_count,
                    fallback_render_view_prepass_texture_count = fallback_breakdown
                        .render_schedule
                        .view_prepass_texture_count,
                    fallback_render_view_transmission_texture_count = fallback_breakdown
                        .render_schedule
                        .view_transmission_texture_count,
                    fallback_render_no_indirect_drawing_view_count = fallback_breakdown
                        .render_schedule
                        .no_indirect_drawing_view_count,
                    fallback_render_occlusion_culling_view_count = fallback_breakdown
                        .render_schedule
                        .occlusion_culling_view_count,
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
                let update0_main_apply_cached_data_top_components =
                    format_apply_cached_data_top_components(
                        app.world(),
                        update0_breakdown.main_schedule.apply_cached_data,
                    );
                let fallback_main_apply_cached_data_top_components =
                    format_apply_cached_data_top_components(
                        app.world(),
                        fallback_breakdown.main_schedule.apply_cached_data,
                    );
                tracing::info!(
                    total_request_ms,
                    camera_count = request.camera_names.len(),
                    setup_ms,
                    update0_ms,
                    update0_main_schedule_ms = update0_breakdown.main_schedule_ms,
                    update0_main_setup_cell_ms = update0_breakdown.main_schedule.setup_cell_ms,
                    update0_main_apply_cached_data_ms =
                        update0_breakdown.main_schedule.apply_cached_data_ms,
                    update0_main_apply_cached_data_skipped =
                        update0_breakdown.main_schedule.apply_cached_data.skipped,
                    update0_main_apply_cached_data_scanned_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .scanned_component_count,
                    update0_main_apply_cached_data_applied_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .applied_component_count,
                    update0_main_apply_cached_data_missing_entity_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .missing_entity_count,
                    update0_main_apply_cached_data_total_bytes = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .total_bytes,
                    update0_main_apply_cached_data_inplace_copy_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .inplace_copy_component_count,
                    update0_main_apply_cached_data_cloned_replace_component_count =
                        update0_breakdown
                            .main_schedule
                            .apply_cached_data
                            .cloned_replace_component_count,
                    update0_main_apply_cached_data_inserted_component_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .inserted_component_count,
                    update0_main_apply_cached_data_adapter_count = update0_breakdown
                        .main_schedule
                        .apply_cached_data
                        .adapter_count,
                    update0_main_apply_cached_data_top_components,
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
                    update0_render_prepare_resources_view_uniforms_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_ms,
                    update0_render_prepare_resources_view_uniforms_clear_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_clear_ms,
                    update0_render_prepare_resources_view_uniforms_build_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_build_ms,
                    update0_render_prepare_resources_view_uniforms_write_buffer_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_view_uniforms_write_buffer_ms,
                    update0_render_prepare_resources_globals_buffer_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_globals_buffer_ms,
                    update0_render_prepare_resources_uniform_components_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_uniform_components_ms,
                    update0_render_prepare_resources_gpu_component_array_buffers_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_gpu_component_array_buffers_ms,
                    update0_render_prepare_resources_gpu_readback_prepare_buffers_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_gpu_readback_prepare_buffers_ms,
                    update0_render_prepare_resources_before_view_uniforms_other_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_before_view_uniforms_other_ms,
                    update0_render_prepare_resources_clusters_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_clusters_ms,
                    update0_render_prepare_resources_core_3d_depth_textures_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_core_3d_depth_textures_ms,
                    update0_render_prepare_resources_core_3d_transmission_textures_ms =
                        update0_breakdown
                            .render_schedule
                            .prepare_resources_core_3d_transmission_textures_ms,
                    update0_render_prepare_resources_prepass_textures_ms = update0_breakdown
                        .render_schedule
                        .prepare_resources_prepass_textures_ms,
                    update0_render_prepare_resources_other_ms =
                        update0_breakdown.render_schedule.prepare_resources_other_ms,
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
                    update0_render_extracted_camera_count =
                        update0_breakdown.render_schedule.extracted_camera_count,
                    update0_render_extracted_view_count =
                        update0_breakdown.render_schedule.extracted_view_count,
                    update0_render_view_uniform_offset_count =
                        update0_breakdown.render_schedule.view_uniform_offset_count,
                    update0_render_extracted_cluster_config_count = update0_breakdown
                        .render_schedule
                        .extracted_cluster_config_count,
                    update0_render_view_cluster_bindings_count = update0_breakdown
                        .render_schedule
                        .view_cluster_bindings_count,
                    update0_render_view_target_count =
                        update0_breakdown.render_schedule.view_target_count,
                    update0_render_view_depth_texture_count =
                        update0_breakdown.render_schedule.view_depth_texture_count,
                    update0_render_view_prepass_texture_count =
                        update0_breakdown.render_schedule.view_prepass_texture_count,
                    update0_render_view_transmission_texture_count = update0_breakdown
                        .render_schedule
                        .view_transmission_texture_count,
                    update0_render_no_indirect_drawing_view_count = update0_breakdown
                        .render_schedule
                        .no_indirect_drawing_view_count,
                    update0_render_occlusion_culling_view_count = update0_breakdown
                        .render_schedule
                        .occlusion_culling_view_count,
                    update0_main_clear_trackers_ms = update0_breakdown.main_clear_trackers_ms,
                    collect0_ms,
                    fallback_used,
                    fallback_update_ms,
                    fallback_main_schedule_ms = fallback_breakdown.main_schedule_ms,
                    fallback_main_setup_cell_ms = fallback_breakdown.main_schedule.setup_cell_ms,
                    fallback_main_apply_cached_data_ms =
                        fallback_breakdown.main_schedule.apply_cached_data_ms,
                    fallback_main_apply_cached_data_skipped =
                        fallback_breakdown.main_schedule.apply_cached_data.skipped,
                    fallback_main_apply_cached_data_scanned_component_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .scanned_component_count,
                    fallback_main_apply_cached_data_applied_component_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .applied_component_count,
                    fallback_main_apply_cached_data_missing_entity_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .missing_entity_count,
                    fallback_main_apply_cached_data_total_bytes = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .total_bytes,
                    fallback_main_apply_cached_data_inplace_copy_component_count =
                        fallback_breakdown
                            .main_schedule
                            .apply_cached_data
                            .inplace_copy_component_count,
                    fallback_main_apply_cached_data_cloned_replace_component_count =
                        fallback_breakdown
                            .main_schedule
                            .apply_cached_data
                            .cloned_replace_component_count,
                    fallback_main_apply_cached_data_inserted_component_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .inserted_component_count,
                    fallback_main_apply_cached_data_adapter_count = fallback_breakdown
                        .main_schedule
                        .apply_cached_data
                        .adapter_count,
                    fallback_main_apply_cached_data_top_components,
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
                    fallback_render_prepare_resources_view_uniforms_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_ms,
                    fallback_render_prepare_resources_view_uniforms_clear_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_clear_ms,
                    fallback_render_prepare_resources_view_uniforms_build_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_view_uniforms_build_ms,
                    fallback_render_prepare_resources_view_uniforms_write_buffer_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_view_uniforms_write_buffer_ms,
                    fallback_render_prepare_resources_globals_buffer_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_globals_buffer_ms,
                    fallback_render_prepare_resources_uniform_components_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_uniform_components_ms,
                    fallback_render_prepare_resources_gpu_component_array_buffers_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_gpu_component_array_buffers_ms,
                    fallback_render_prepare_resources_gpu_readback_prepare_buffers_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_gpu_readback_prepare_buffers_ms,
                    fallback_render_prepare_resources_before_view_uniforms_other_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_before_view_uniforms_other_ms,
                    fallback_render_prepare_resources_clusters_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_clusters_ms,
                    fallback_render_prepare_resources_core_3d_depth_textures_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_core_3d_depth_textures_ms,
                    fallback_render_prepare_resources_core_3d_transmission_textures_ms =
                        fallback_breakdown
                            .render_schedule
                            .prepare_resources_core_3d_transmission_textures_ms,
                    fallback_render_prepare_resources_prepass_textures_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_prepass_textures_ms,
                    fallback_render_prepare_resources_other_ms = fallback_breakdown
                        .render_schedule
                        .prepare_resources_other_ms,
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
                    fallback_render_extracted_camera_count =
                        fallback_breakdown.render_schedule.extracted_camera_count,
                    fallback_render_extracted_view_count =
                        fallback_breakdown.render_schedule.extracted_view_count,
                    fallback_render_view_uniform_offset_count =
                        fallback_breakdown.render_schedule.view_uniform_offset_count,
                    fallback_render_extracted_cluster_config_count = fallback_breakdown
                        .render_schedule
                        .extracted_cluster_config_count,
                    fallback_render_view_cluster_bindings_count = fallback_breakdown
                        .render_schedule
                        .view_cluster_bindings_count,
                    fallback_render_view_target_count =
                        fallback_breakdown.render_schedule.view_target_count,
                    fallback_render_view_depth_texture_count =
                        fallback_breakdown.render_schedule.view_depth_texture_count,
                    fallback_render_view_prepass_texture_count = fallback_breakdown
                        .render_schedule
                        .view_prepass_texture_count,
                    fallback_render_view_transmission_texture_count = fallback_breakdown
                        .render_schedule
                        .view_transmission_texture_count,
                    fallback_render_no_indirect_drawing_view_count = fallback_breakdown
                        .render_schedule
                        .no_indirect_drawing_view_count,
                    fallback_render_occlusion_culling_view_count = fallback_breakdown
                        .render_schedule
                        .occlusion_culling_view_count,
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
