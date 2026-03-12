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
use bevy_mat3_material::Mat3Material;
use big_space::{FloatingOrigin, GridCell};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{CurrentTimestamp, DbConfig, SchematicElem};
use render_bridge::RenderBridgeServer;

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{
    HeadlessMode, SensorCamera, SensorCameraPlugin, SensorCameraRenderMetrics,
    SensorCamerasSpawned, set_readback_armed,
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
            .add_systems(
                PreUpdate,
                (
                    impeller2_bevy::apply_cached_data,
                    crate::object_3d::update_object_3d_system,
                    crate::sync_object_3d,
                    sync_pos,
                )
                    .chain()
                    .after(impeller2_bevy::sink)
                    .in_set(PositionSync),
            )
            .add_systems(PreUpdate, crate::setup_cell.after(impeller2_bevy::sink))
            .add_systems(Startup, setup_floating_origin)
            .add_systems(Startup, setup_headless_lighting)
            .init_resource::<crate::EqlContext>()
            .init_resource::<crate::SyncedObject3d>()
            .add_systems(Update, crate::update_eql_context)
            .add_systems(Update, load_headless_scene)
            .set_runner(headless_sensor_runner);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<HeadlessMode>()
                .init_resource::<SensorCameraRenderMetrics>();
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

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Clone, Copy, Debug, Default)]
struct HeadlessUpdateBreakdown {
    main_schedule_ms: f64,
    render_extract_ms: f64,
    render_app_ms: f64,
    main_clear_trackers_ms: f64,
}

impl HeadlessUpdateBreakdown {
    fn total_ms(self) -> f64 {
        self.main_schedule_ms
            + self.render_extract_ms
            + self.render_app_ms
            + self.main_clear_trackers_ms
    }
}

fn run_headless_update(app: &mut App) -> HeadlessUpdateBreakdown {
    let _update_span = tracing::info_span!("headless_update").entered();
    let mut breakdown = HeadlessUpdateBreakdown::default();
    let sub_apps = app.sub_apps_mut();
    let (main_app, render_sub_apps) = (&mut sub_apps.main, &mut sub_apps.sub_apps);

    {
        let _span = tracing::info_span!("headless_main_schedule").entered();
        let main_schedule_start = Instant::now();
        main_app.run_default_schedule();
        breakdown.main_schedule_ms = elapsed_ms(main_schedule_start);
    }

    if let Some(render_app) = render_sub_apps.get_mut(&RenderApp.intern()) {
        {
            let _span = tracing::info_span!("headless_render_extract").entered();
            let render_extract_start = Instant::now();
            render_app.extract(main_app.world_mut());
            breakdown.render_extract_ms = elapsed_ms(render_extract_start);
        }

        {
            let _span = tracing::info_span!("headless_render_app").entered();
            let render_app_start = Instant::now();
            render_app.update();
            breakdown.render_app_ms = elapsed_ms(render_app_start);
        }
    }

    {
        let _span = tracing::info_span!("headless_clear_trackers").entered();
        let clear_trackers_start = Instant::now();
        main_app.world_mut().clear_trackers();
        breakdown.main_clear_trackers_ms = elapsed_ms(clear_trackers_start);
    }

    breakdown
}

fn drain_stale_frames(app: &App) {
    let rx = app
        .world()
        .resource::<crate::sensor_camera::SensorFrameReceiver>();
    while rx.0.try_recv().is_ok() {}
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
            enable_all_sensor_cameras(app.world_mut());
            cameras_enabled = true;
            tracing::info!("Sensor cameras spawned and enabled after {i} warm-up cycles");
            for _ in 0..4 {
                run_headless_update(&mut app);
            }
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
        let request_span = tracing::info_span!(
            "sensor_render_request",
            camera_count = request.camera_names.len(),
            timestamp = request.timestamp.0,
            setup_ms = tracing::field::Empty,
            update0_ms = tracing::field::Empty,
            collect0_ms = tracing::field::Empty,
            fallback_used = tracing::field::Empty,
            fallback_update_ms = tracing::field::Empty,
            collect1_ms = tracing::field::Empty,
            respond_ms = tracing::field::Empty,
            total_request_ms = tracing::field::Empty,
            frames_after_update0 = tracing::field::Empty,
            final_frame_count = tracing::field::Empty,
            image_copy_driver_ms = tracing::field::Empty,
            receive_image_poll_wait_ms = tracing::field::Empty,
            receive_image_from_buffer_ms = tracing::field::Empty,
            respond_frame_bytes_write_ms = tracing::field::Empty,
        );
        let _request_span = request_span.enter();

        // Set timestamp for this request.
        app.world_mut().resource_mut::<CurrentTimestamp>().0 = request.timestamp;

        // Check if cameras are ready.
        let cameras_ready = app.world().resource::<SensorCamerasSpawned>().0;

        // If cameras just became ready (spawned during main loop), enable them now.
        if cameras_ready && !cameras_enabled {
            enable_all_sensor_cameras(app.world_mut());
            cameras_enabled = true;
            tracing::info!("Sensor cameras late-enabled during main loop");
            for _ in 0..4 {
                run_headless_update(&mut app);
            }
        }

        let setup_ms = {
            let _span = tracing::info_span!("headless_request_setup").entered();
            let setup_start = Instant::now();
            if cameras_ready {
                drain_stale_frames(&app);
                set_readback_armed(app.world_mut(), &request.camera_names, true);
            }
            elapsed_ms(setup_start)
        };
        request_span.record("setup_ms", setup_ms);

        // With PipelinedRenderingPlugin disabled, Extract + Render run synchronously in one update.
        let update0_breakdown = run_headless_update(&mut app);
        let update0_ms = update0_breakdown.total_ms();
        request_span.record("update0_ms", update0_ms);
        let render_metrics = app
            .get_sub_app_mut(RenderApp)
            .map(|render_app| *render_app.world().resource::<SensorCameraRenderMetrics>())
            .unwrap_or_default();
        request_span.record("image_copy_driver_ms", render_metrics.image_copy_driver_ms);
        request_span.record(
            "receive_image_poll_wait_ms",
            render_metrics.receive_image_poll_wait_ms,
        );
        request_span.record(
            "receive_image_from_buffer_ms",
            render_metrics.receive_image_from_buffer_ms,
        );

        if cameras_ready {
            let (mut frames, collect0_ms) = {
                let _span = tracing::info_span!("headless_collect_frames_update0").entered();
                let collect0_start = Instant::now();
                let frames = collect_frames(&app, &request.camera_names);
                (frames, elapsed_ms(collect0_start))
            };
            request_span.record("collect0_ms", collect0_ms);
            let frames_after_update0 = frames.len();
            let (fallback_used, fallback_update_ms, collect1_ms) = if frames.len()
                < request.camera_names.len()
            {
                let fallback_update_ms = {
                    let _span = tracing::info_span!("headless_fallback_update").entered();
                    run_headless_update(&mut app).total_ms()
                };
                let collect1_ms = {
                    let _span = tracing::info_span!("headless_collect_frames_fallback").entered();
                    let collect1_start = Instant::now();
                    let more = collect_frames(&app, &request.camera_names);
                    for (name, data) in more {
                        if !frames.iter().any(|(n, _)| n == &name) {
                            frames.push((name, data));
                        }
                    }
                    elapsed_ms(collect1_start)
                };
                (true, fallback_update_ms, collect1_ms)
            } else {
                (false, 0.0, 0.0)
            };
            request_span.record("fallback_used", fallback_used);
            request_span.record("fallback_update_ms", fallback_update_ms);
            request_span.record("collect1_ms", collect1_ms);
            request_span.record("frames_after_update0", frames_after_update0);

            set_readback_armed(app.world_mut(), &request.camera_names, false);
            let final_frame_count = frames.len();
            request_span.record("final_frame_count", final_frame_count);

            let (respond_metrics, respond_ms) = {
                let _span = tracing::info_span!("headless_respond_batch").entered();
                let respond_start = Instant::now();
                let respond_metrics = match server.respond_batch(request.timestamp, &frames) {
                    Ok(metrics) => metrics,
                    Err(e) => {
                        tracing::warn!("Render bridge write failed, client disconnected: {e}");
                        break;
                    }
                };
                (respond_metrics, elapsed_ms(respond_start))
            };
            request_span.record("respond_ms", respond_ms);
            request_span.record(
                "respond_frame_bytes_write_ms",
                respond_metrics.frame_bytes_write_ms,
            );

            let total_request_ms = elapsed_ms(request_start);
            request_span.record("total_request_ms", total_request_ms);
            if total_request_ms > RENDER_CRITICAL_MS {
                tracing::warn!(
                    total_request_ms,
                    camera_count = request.camera_names.len(),
                    final_frame_count,
                    fallback_used,
                    "Render request exceeded critical latency budget"
                );
            } else if total_request_ms > RENDER_TARGET_MS {
                tracing::info!(
                    total_request_ms,
                    camera_count = request.camera_names.len(),
                    final_frame_count,
                    fallback_used,
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

/// Enable all sensor cameras permanently (keeps Bevy render pipeline warm).
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
