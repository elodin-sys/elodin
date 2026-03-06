use std::os::unix::net::UnixStream;
use std::time::Duration;

use bevy::{
    app::{App, AppExit, Plugin, Startup},
    asset::{AssetPlugin, Assets, UnapprovedPathMode},
    log::LogPlugin,
    math::{EulerRot, Quat},
    prelude::*,
    window::{ExitCondition, WindowPlugin},
    winit::WinitPlugin,
};
use bevy_mat3_material::Mat3Material;
use big_space::{FloatingOrigin, GridCell};
use elodin_db::render_bridge::{RenderBridgeServer, RenderRequest};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{CurrentTimestamp, DbConfig, LastUpdated, SchematicElem};

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{SensorCamera, SensorCameraConfigs, SensorCameraPlugin};
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
            .add_systems(
                PreUpdate,
                advance_headless_timestamp.after(impeller2_bevy::sink),
            )
            .add_systems(
                PreUpdate,
                (
                    impeller2_bevy::apply_cached_data,
                    crate::object_3d::update_object_3d_system,
                    crate::sync_object_3d,
                    sync_pos,
                )
                    .chain()
                    .after(advance_headless_timestamp)
                    .in_set(PositionSync),
            )
            .add_systems(
                PreUpdate,
                crate::setup_cell.after(impeller2_bevy::sink),
            )
            .add_systems(Startup, setup_floating_origin)
            .add_systems(Startup, setup_headless_lighting)
            .init_resource::<crate::EqlContext>()
            .init_resource::<crate::SyncedObject3d>()
            .add_systems(Update, crate::update_eql_context)
            .add_systems(Update, load_headless_scene)
            .set_runner(headless_sensor_runner);
    }
}

fn advance_headless_timestamp(
    last_updated: Res<LastUpdated>,
    mut current_ts: ResMut<CurrentTimestamp>,
) {
    if last_updated.0 > current_ts.0 {
        current_ts.0 = last_updated.0;
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
    tracing::info!(
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

    // Warm-up: run a few updates so the TCP connection is established,
    // DB metadata is loaded, and sensor cameras are spawned.
    for _ in 0..20 {
        app.update();
        std::thread::sleep(Duration::from_millis(50));
    }

    let mut pending: Option<(RenderRequest, UnixStream)> = None;
    let mut armed_frames = 0u32;

    loop {
        if let Some(exit) = app.should_exit() {
            return exit;
        }

        app.update();

        // Phase 1: camera is armed — check if the render produced a frame.
        // Bevy's render pipeline may need 2-3 frames after camera activation
        // to cache the pipeline and produce actual pixel data. We run up to
        // 4 update cycles before giving up.
        if armed_frames > 0 {
            let has_frame = {
                let rx = app
                    .world()
                    .resource::<crate::sensor_camera::SensorFrameReceiver>();
                !rx.0.is_empty()
            };
            if has_frame || armed_frames >= 4 {
                if let Some((request, stream)) = pending.take() {
                    send_frame_response(&app, &request, stream);
                }
                disable_all_sensor_cameras(app.world_mut());
                armed_frames = 0;
            } else {
                armed_frames += 1;
            }
            continue;
        }

        // Phase 2: check for new render requests via the UDS.
        if pending.is_none() {
            if let Some(req) = server.try_recv() {
                drain_stale_frames(&app);
                app.world_mut().resource_mut::<CurrentTimestamp>().0 = req.0.timestamp;
                enable_sensor_camera(app.world_mut(), &req.0.camera_name);
                armed_frames = 1;
                pending = Some(req);
            }
        }

        if pending.is_none() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn send_frame_response(
    app: &App,
    request: &RenderRequest,
    stream: std::os::unix::net::UnixStream,
) {
    let world = app.world();
    let frame_rx = world.resource::<crate::sensor_camera::SensorFrameReceiver>();

    // Drain all queued frames, keeping only the one matching the requested camera.
    let mut matched_frame: Option<(String, Vec<u8>, u32, u32)> = None;
    while let Ok(frame) = frame_rx.0.try_recv() {
        if frame.0 == request.camera_name {
            matched_frame = Some(frame);
        }
    }

    if let Some((camera_name, frame_bytes, _, _)) = matched_frame {
        RenderBridgeServer::respond_with_frame(
            stream,
            &camera_name,
            request.timestamp,
            &frame_bytes,
        );
    } else {
        RenderBridgeServer::respond_empty(stream);
    }
}

fn enable_sensor_camera(world: &mut World, camera_name: &str) {
    let target_index = {
        let configs = world.resource::<SensorCameraConfigs>();
        configs
            .0
            .iter()
            .position(|c| c.camera_name == camera_name)
    };
    let Some(target_index) = target_index else {
        tracing::warn!("render_camera: unknown camera '{camera_name}'");
        return;
    };

    let mut query = world.query::<(&SensorCamera, &mut Camera)>();
    for (sensor, mut camera) in query.iter_mut(world) {
        camera.is_active = sensor.config_index == target_index;
    }
}

fn disable_all_sensor_cameras(world: &mut World) {
    let mut query = world.query::<(&SensorCamera, &mut Camera)>();
    for (_, mut camera) in query.iter_mut(world) {
        camera.is_active = false;
    }
}
