use std::time::{Duration, Instant};

use bevy::{
    a11y::AccessibilityPlugin,
    animation::AnimationPlugin,
    app::{App, AppExit, Plugin, Startup},
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
use elodin_db::render_bridge::RenderBridgeServer;
use impeller2_kdl::FromKdl;
use impeller2_wkt::{CurrentTimestamp, DbConfig, SchematicElem};

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{
    HeadlessMode, SensorCamera, SensorCameraPlugin, SensorCamerasSpawned, set_readback_armed,
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
            render_app.init_resource::<HeadlessMode>();
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
        app.update();
        let cameras_ready = app.world().resource::<SensorCamerasSpawned>().0;
        if cameras_ready && !cameras_enabled {
            enable_all_sensor_cameras(app.world_mut());
            cameras_enabled = true;
            tracing::info!("Sensor cameras spawned and enabled after {i} warm-up cycles");
            for _ in 0..4 {
                app.update();
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
                app.update();
            }
        }

        if cameras_ready {
            drain_stale_frames(&app);
            set_readback_armed(app.world_mut(), &request.camera_names, true);
        }

        // With PipelinedRenderingPlugin disabled, Extract + Render run synchronously in one update.
        let update0_start = Instant::now();
        app.update();
        tracing::debug!(
            render_latency_ms = update0_start.elapsed().as_secs_f64() * 1000.0,
            update_index = 0
        );

        if cameras_ready {
            let mut frames = collect_frames(&app, &request.camera_names);
            if frames.is_empty() {
                let fallback_start = Instant::now();
                app.update();
                tracing::debug!(
                    render_latency_ms = fallback_start.elapsed().as_secs_f64() * 1000.0,
                    update_index = 1,
                    fallback = true
                );
                frames = collect_frames(&app, &request.camera_names);
            }

            set_readback_armed(app.world_mut(), &request.camera_names, false);

            if let Err(e) = server.respond_batch(request.timestamp, &frames) {
                tracing::warn!("Render bridge write failed, client disconnected: {e}");
                break;
            }
            tracing::debug!(total_request_ms = request_start.elapsed().as_secs_f64() * 1000.0);
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
