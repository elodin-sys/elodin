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
        .add_plugins(ViewCubePlugin::default()) // auto_rotate = true by default
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<ViewCubeConfig>,
) {
    // Spawn camera with ViewCubeTargetCamera marker
    let camera_entity = commands
        .spawn((
            Transform::from_xyz(3.0, 2.5, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
            Camera3d::default(),
            ViewCubeTargetCamera, // This tells the plugin which camera to control
            Name::new("main_camera"),
        ))
        .id();

    // Spawn the ViewCube
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
