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
//! - Overlay mode: ViewCube appears fixed in top-right corner

use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use elodin_editor::plugins::view_cube::{
    ViewCubeConfig, ViewCubePlugin, ViewCubeTargetCamera, spawn::spawn_view_cube,
};
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

    // Configure ViewCube for overlay mode (like in the editor)
    let config = ViewCubeConfig {
        use_overlay: true,      // Render as overlay in corner
        sync_with_camera: true, // Cube shows world orientation
        auto_rotate: true,      // Plugin handles camera rotation on click
        ..default()
    };

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "ViewCube Plugin Demo - Overlay Mode".to_string(),
                        resolution: (1024, 768).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: assets_path.to_string_lossy().to_string(),
                    ..default()
                }),
        )
        .add_plugins(ViewCubePlugin { config })
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_camera_with_keys)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<ViewCubeConfig>,
) {
    // Spawn main camera
    let camera_entity = commands
        .spawn((
            Transform::from_xyz(5.0, 4.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            Camera3d::default(),
            ViewCubeTargetCamera,
            Name::new("main_camera"),
        ))
        .id();

    // Spawn the ViewCube (in overlay mode, creates its own camera)
    let spawned = spawn_view_cube(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut materials,
        &config,
        camera_entity,
    );

    println!(
        "ViewCube spawned: cube={:?}, camera={:?}",
        spawned.cube_root, spawned.camera
    );

    // Add a reference object in the scene to see the main camera movement
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.3, 0.3),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Name::new("reference_cube"),
    ));

    // Add a floor
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(5.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.4, 0.4),
            ..default()
        })),
        Transform::from_xyz(0.0, -0.5, 0.0),
        Name::new("floor"),
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(5.0, 10.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });

    println!("=== ViewCube Overlay Demo ===");
    println!("Use arrow keys to rotate the main camera.");
    println!("The ViewCube in the top-right shows world orientation.");
    println!("Click on the ViewCube faces to snap the camera.");
}

/// Rotate the main camera with keyboard input
fn rotate_camera_with_keys(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_query: Query<&mut Transform, With<ViewCubeTargetCamera>>,
    time: Res<Time>,
) {
    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };

    let rotation_speed = 1.0 * time.delta_secs();

    if keyboard.pressed(KeyCode::ArrowLeft) {
        let rotation = Quat::from_rotation_y(rotation_speed);
        transform.rotate_around(Vec3::ZERO, rotation);
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        let rotation = Quat::from_rotation_y(-rotation_speed);
        transform.rotate_around(Vec3::ZERO, rotation);
    }
    if keyboard.pressed(KeyCode::ArrowUp) {
        let right = transform.right();
        let rotation = Quat::from_axis_angle(*right, rotation_speed);
        transform.rotate_around(Vec3::ZERO, rotation);
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        let right = transform.right();
        let rotation = Quat::from_axis_angle(*right, -rotation_speed);
        transform.rotate_around(Vec3::ZERO, rotation);
    }
}
