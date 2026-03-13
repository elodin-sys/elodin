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
    render_bridge::elapsed_ms(start)
}

fn sensor_camera_probe_logs_enabled() -> bool {
    render_bridge::sensor_camera_probe_logs_enabled()
}

fn run_headless_update(app: &mut App) {
    if !sensor_camera_probe_logs_enabled() {
        app.update();
        return;
    }

    let _update_span = tracing::info_span!("headless_update").entered();
    let sub_apps = app.sub_apps_mut();
    let (main_app, render_sub_apps) = (&mut sub_apps.main, &mut sub_apps.sub_apps);

    {
        let _span = tracing::info_span!("headless_main_schedule").entered();
        main_app.run_default_schedule();
    }

    if let Some(render_app) = render_sub_apps.get_mut(&RenderApp.intern()) {
        {
            let _span = tracing::info_span!("headless_render_extract").entered();
            render_app.extract(main_app.world_mut());
        }
        {
            let _span = tracing::info_span!("headless_render_app").entered();
            render_app.update();
        }
    }

    {
        let _span = tracing::info_span!("headless_clear_trackers").entered();
        main_app.world_mut().clear_trackers();
    }
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

        let profiling = sensor_camera_probe_logs_enabled();
        let request_start = Instant::now();

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

        if cameras_ready {
            drain_stale_frames(&app);
            set_readback_armed(app.world_mut(), &request.camera_names, true);
        }

        // With PipelinedRenderingPlugin disabled, Extract + Render run synchronously.
        run_headless_update(&mut app);

        if cameras_ready {
            let mut frames = collect_frames(&app, &request.camera_names);
            let fallback_used = if frames.len() < request.camera_names.len() {
                run_headless_update(&mut app);
                let more = collect_frames(&app, &request.camera_names);
                for (name, data) in more {
                    if !frames.iter().any(|(n, _)| n == &name) {
                        frames.push((name, data));
                    }
                }
                true
            } else {
                false
            };

            set_readback_armed(app.world_mut(), &request.camera_names, false);

            if profiling {
                match server.respond_batch_with_metrics(request.timestamp, &frames) {
                    Ok(respond_metrics) => {
                        let total_request_ms = elapsed_ms(request_start);
                        tracing::info!(
                            total_request_ms,
                            camera_count = request.camera_names.len(),
                            final_frame_count = frames.len(),
                            fallback_used,
                            respond_frame_bytes_write_ms = respond_metrics.frame_bytes_write_ms,
                            "sensor_camera_probe_request"
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Render bridge write failed, client disconnected: {e}");
                        break;
                    }
                }
            } else if let Err(e) = server.respond_batch(request.timestamp, &frames) {
                tracing::warn!("Render bridge write failed, client disconnected: {e}");
                break;
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
