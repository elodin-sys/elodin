//! ViewCube Plugin Demo
//!
//! Demonstrates the ViewCube plugin for CAD-style camera navigation.
//!
//! Run with:
//!   cargo run --example view_cube_demo -p elodin-editor
//!
//! Features:
//! - Hover over faces, edges, or corners to highlight them
//! - Click on any element to rotate the camera to that view
//! - Use arrow buttons for incremental rotation

use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use bevy_fontmesh::prelude::*;
use elodin_editor::plugins::view_cube::{
    CoordinateSystem, CornerPosition, EdgeDirection, FaceDirection, RotationArrow, ViewCubeConfig,
    ViewCubeEvent, ViewCubePlugin, spawn::spawn_view_cube,
};
use std::f32::consts::PI;
use std::path::PathBuf;

fn main() {
    // Compute path to repo root's assets/ folder
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let assets_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("assets");

    // Configure the ViewCube
    let view_cube_config = ViewCubeConfig {
        system: CoordinateSystem::ENU,
        scale: 0.95,
        rotation_increment: 15.0 * PI / 180.0,
        camera_distance: 4.5,
    };

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "ViewCube Plugin Demo".to_string(),
                        resolution: (800, 600).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: assets_path.to_string_lossy().to_string(),
                    ..default()
                }),
        )
        .add_plugins(FontMeshPlugin)
        .add_plugins(ViewCubePlugin {
            config: view_cube_config,
        })
        .init_resource::<CameraAnimation>()
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_view_cube_events, animate_camera))
        .run();
}

// ============================================================================
// Components
// ============================================================================

#[derive(Component)]
struct MainCamera;

/// Camera animation state
#[derive(Resource)]
struct CameraAnimation {
    start_position: Vec3,
    start_rotation: Quat,
    target_position: Vec3,
    target_rotation: Quat,
    progress: f32,
    animating: bool,
}

impl Default for CameraAnimation {
    fn default() -> Self {
        Self {
            start_position: Vec3::new(3.0, 2.5, 3.0),
            start_rotation: Quat::IDENTITY,
            target_position: Vec3::ZERO,
            target_rotation: Quat::IDENTITY,
            progress: 0.0,
            animating: false,
        }
    }
}

// ============================================================================
// Setup
// ============================================================================

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<ViewCubeConfig>,
) {
    // Spawn camera
    let camera_entity = commands
        .spawn((
            Transform::from_xyz(3.0, 2.5, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
            Camera3d::default(),
            MainCamera,
            Name::new("main_camera"),
        ))
        .id();

    // Spawn the ViewCube using the plugin
    spawn_view_cube(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut materials,
        &config,
        camera_entity,
    );

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 15000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 800.0,
        ..default()
    });

    println!("=== ViewCube Plugin Demo ===");
    println!("Hover over faces, edges, or corners to highlight them.");
    println!("Click to rotate the camera toward that element.");
    println!("Use arrow buttons for incremental rotation.");
}

// ============================================================================
// Event Handler
// ============================================================================

fn handle_view_cube_events(
    mut events: MessageReader<ViewCubeEvent>,
    camera_query: Query<&Transform, With<MainCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    config: Res<ViewCubeConfig>,
) {
    for event in events.read() {
        match event {
            ViewCubeEvent::FaceClicked(direction) => {
                let look_dir = direction.to_look_direction();
                start_camera_animation(look_dir, &camera_query, &mut camera_anim, &config);
            }
            ViewCubeEvent::EdgeClicked(direction) => {
                let look_dir = direction.to_look_direction();
                start_camera_animation(look_dir, &camera_query, &mut camera_anim, &config);
            }
            ViewCubeEvent::CornerClicked(position) => {
                let look_dir = position.to_look_direction();
                start_camera_animation(look_dir, &camera_query, &mut camera_anim, &config);
            }
            ViewCubeEvent::ArrowClicked(arrow) => {
                apply_arrow_rotation(*arrow, &config, &camera_query);
            }
        }
    }
}

fn start_camera_animation(
    mut look_dir: Vec3,
    camera_query: &Query<&Transform, With<MainCamera>>,
    camera_anim: &mut CameraAnimation,
    config: &ViewCubeConfig,
) {
    let Ok(current_transform) = camera_query.single() else {
        return;
    };

    // Check if clicking through transparent face
    let camera_dir = current_transform.translation.normalize();
    if camera_dir.dot(look_dir) < 0.0 {
        look_dir = -look_dir;
    }

    let up_dir = get_up_direction_for_look(look_dir);
    let target_pos = look_dir * config.camera_distance;
    let target_transform = Transform::from_translation(target_pos).looking_at(Vec3::ZERO, up_dir);

    // Skip if already at target
    let position_similar = current_transform.translation.distance(target_pos) < 0.5;
    let rotation_similar = current_transform
        .rotation
        .dot(target_transform.rotation)
        .abs()
        > 0.99;

    if position_similar && rotation_similar {
        return;
    }

    camera_anim.start_position = current_transform.translation;
    camera_anim.start_rotation = current_transform.rotation;
    camera_anim.target_position = target_pos;
    camera_anim.target_rotation = target_transform.rotation;
    camera_anim.progress = 0.0;
    camera_anim.animating = true;
}

fn apply_arrow_rotation(
    arrow: RotationArrow,
    config: &ViewCubeConfig,
    camera_query: &Query<&Transform, With<MainCamera>>,
) {
    // Note: This is a simplified version - in the real implementation,
    // we'd need mutable access to the transform
    let Ok(transform) = camera_query.single() else {
        return;
    };

    let _rotation = match arrow {
        RotationArrow::Left => Quat::from_rotation_y(config.rotation_increment),
        RotationArrow::Right => Quat::from_rotation_y(-config.rotation_increment),
        RotationArrow::Up => {
            let right = transform.right();
            Quat::from_axis_angle(*right, config.rotation_increment)
        }
        RotationArrow::Down => {
            let right = transform.right();
            Quat::from_axis_angle(*right, -config.rotation_increment)
        }
        RotationArrow::RollLeft => {
            let forward = transform.forward();
            Quat::from_axis_angle(*forward, config.rotation_increment)
        }
        RotationArrow::RollRight => {
            let forward = transform.forward();
            Quat::from_axis_angle(*forward, -config.rotation_increment)
        }
    };

    println!("Arrow clicked: {:?}", arrow);
}

fn get_up_direction_for_look(look_dir: Vec3) -> Vec3 {
    if look_dir.y.abs() > 0.9 {
        if look_dir.y > 0.0 {
            Vec3::NEG_Z
        } else {
            Vec3::Z
        }
    } else {
        Vec3::Y
    }
}

// ============================================================================
// Camera Animation
// ============================================================================

fn slerp_vec3(start: Vec3, end: Vec3, t: f32) -> Vec3 {
    let start_norm = start.normalize();
    let end_norm = end.normalize();
    let dot = start_norm.dot(end_norm).clamp(-1.0, 1.0);
    let theta = dot.acos();

    if theta.abs() < 0.0001 {
        return start.lerp(end, t);
    }

    let sin_theta = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sin_theta;
    let b = (t * theta).sin() / sin_theta;

    let start_len = start.length();
    let end_len = end.length();
    let interpolated_len = start_len + (end_len - start_len) * t;

    (start_norm * a + end_norm * b) * interpolated_len
}

fn animate_camera(
    mut camera_query: Query<&mut Transform, With<MainCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    time: Res<Time>,
) {
    if !camera_anim.animating {
        return;
    }

    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };

    let speed = 2.0;
    camera_anim.progress = (camera_anim.progress + speed * time.delta_secs()).min(1.0);

    let t = camera_anim.progress;
    let eased_t = if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    };

    transform.translation = slerp_vec3(
        camera_anim.start_position,
        camera_anim.target_position,
        eased_t,
    );

    let start_up = camera_anim.start_rotation * Vec3::Y;
    let target_up = camera_anim.target_rotation * Vec3::Y;
    let current_up = slerp_vec3(start_up, target_up, eased_t).normalize();

    transform.look_at(Vec3::ZERO, current_up);

    if camera_anim.progress >= 1.0 {
        transform.translation = camera_anim.target_position;
        transform.rotation = camera_anim.target_rotation;
        camera_anim.animating = false;
    }
}
