use std::time::Duration;

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
use bevy_geo_frames::GeoContext;
use bevy_mat3_material::Mat3Material;
use big_space::{FloatingOrigin, GridCell};
use impeller2_kdl::FromKdl;
use impeller2_wkt::{CurrentTimestamp, DbConfig, SchematicElem};
use render_bridge::{BatchRenderRequest, RenderBridgeServer};

use crate::object_3d::create_object_3d_entity;
use crate::sensor_camera::{
    HeadlessMode, SensorCamera, SensorCameraConfigs, SensorCameraPlugin, SensorCameraRenderMetrics,
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
    asset_server: Res<AssetServer>,
    geo_context: Res<GeoContext>,
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
                &geo_context,
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

fn run_headless_update(app: &mut App) {
    app.update();
}

fn drain_stale_frames(app: &App) {
    let rx = app
        .world()
        .resource::<crate::sensor_camera::SensorFrameReceiver>();
    while rx.0.try_recv().is_ok() {}
}

// ---------------------------------------------------------------------------
// Custom Bevy runner
// ---------------------------------------------------------------------------

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

            let names = all_camera_names(&app);
            set_readback_armed(app.world_mut(), &names, true);
            for _ in 0..4 {
                run_headless_update(&mut app);
            }
            drain_stale_frames(&app);
            set_readback_armed(app.world_mut(), &names, false);

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

    // Post-connect GPU warm-up.
    if cameras_enabled {
        let names = all_camera_names(&app);
        set_readback_armed(app.world_mut(), &names, true);
        run_headless_update(&mut app);
        drain_stale_frames(&app);
        set_readback_armed(app.world_mut(), &names, false);
        tracing::debug!("Post-connect GPU warm-up complete");
    }

    loop {
        if let Some(exit) = app.should_exit() {
            return exit;
        }

        let Some(request) = server.recv_batch() else {
            tracing::info!("Client disconnected, exiting render server");
            return AppExit::Success;
        };

        render_frame(&mut app, &request, &mut cameras_enabled);
        let frames = collect_and_disarm(&mut app, &request);

        if let Err(e) = server.respond_batch(request.timestamp, &frames) {
            tracing::warn!("Render bridge write failed, client disconnected: {e}");
            break;
        }
    }
    AppExit::Success
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_camera_names(app: &App) -> Vec<String> {
    app.world()
        .resource::<SensorCameraConfigs>()
        .0
        .iter()
        .map(|c| c.camera_name.clone())
        .collect()
}

/// Set timestamp, late-enable cameras if needed, arm readback, and run one
/// headless update cycle.
fn render_frame(app: &mut App, request: &BatchRenderRequest, cameras_enabled: &mut bool) {
    app.world_mut().resource_mut::<CurrentTimestamp>().0 = request.timestamp;

    let cameras_ready = app.world().resource::<SensorCamerasSpawned>().0;
    if cameras_ready && !*cameras_enabled {
        enable_all_sensor_cameras(app.world_mut());
        *cameras_enabled = true;
        tracing::info!("Sensor cameras late-enabled during render");
        let names = all_camera_names(app);
        set_readback_armed(app.world_mut(), &names, true);
        for _ in 0..4 {
            run_headless_update(app);
        }
        drain_stale_frames(app);
        set_readback_armed(app.world_mut(), &names, false);
    }

    if cameras_ready {
        drain_stale_frames(app);
        set_readback_armed(app.world_mut(), &request.camera_names, true);
    }

    run_headless_update(app);
}

/// Collect rendered frames and disarm readback. Returns empty vec if cameras
/// are not ready.
fn collect_and_disarm(app: &mut App, request: &BatchRenderRequest) -> Vec<(String, Vec<u8>)> {
    let cameras_ready = app.world().resource::<SensorCamerasSpawned>().0;
    if !cameras_ready {
        return Vec::new();
    }

    let mut frames = collect_frames(app, &request.camera_names);
    if frames.len() < request.camera_names.len() {
        run_headless_update(app);
        let more = collect_frames(app, &request.camera_names);
        for (name, data) in more {
            if !frames.iter().any(|(n, _)| n == &name) {
                frames.push((name, data));
            }
        }
    }
    set_readback_armed(app.world_mut(), &request.camera_names, false);
    frames
}

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
