//! Spawning functions for the ViewCube widget

use bevy::ecs::hierarchy::ChildOf;
use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy::render::render_resource::Face;
use bevy_fontmesh::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI};

use super::components::*;
use super::config::*;
use super::theme::ViewCubeColors;

/// Rotation increment per click (15 degrees)
pub const ROTATION_INCREMENT: f32 = 15.0 * PI / 180.0;

// ============================================================================
// Main Spawn Function
// ============================================================================

/// Spawn a complete ViewCube widget
///
/// Returns the root entity of the ViewCube.
/// The ViewCube is spawned at the origin and should be positioned by the caller.
pub fn spawn_view_cube(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    camera_entity: Entity,
) -> Entity {
    // Load the axes-cube.glb
    let scene = asset_server.load("axes-cube.glb#Scene0");

    // Spawn the cube root
    let cube_root = commands
        .spawn((
            SceneRoot(scene),
            Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(config.scale)),
            ViewCubeRoot,
            ViewCubeMeshRoot,
            Name::new("view_cube_root"),
        ))
        .id();

    // Spawn RGB axes extending from the corner of the cube
    spawn_axes(commands, meshes, materials, config);

    // Spawn 3D text labels on cube faces
    spawn_face_labels(commands, asset_server, materials, config);

    // Spawn rotation arrows as children of camera (fixed on screen)
    spawn_rotation_arrows(commands, meshes, materials, camera_entity);

    cube_root
}

// ============================================================================
// Axes
// ============================================================================

/// Spawn RGB axes extending from the bottom-left-back corner of the cube
pub fn spawn_axes(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
) {
    let axis_length = 1.4 * config.scale;
    let axis_radius = 0.035 * config.scale;
    let tip_radius = 0.08 * config.scale;
    let tip_length = 0.2 * config.scale;
    let origin = Vec3::new(-0.55, -0.55, -0.55) * config.scale;

    let axes = config.system.get_axes();
    let axis_configs: [(Vec3, Color, &str); 3] = [
        (
            Vec3::X,
            axes.iter()
                .find(|a| a.direction == Vec3::X)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.9, 0.2, 0.2)),
            "X",
        ),
        (
            Vec3::Y,
            axes.iter()
                .find(|a| a.direction == Vec3::Y || a.direction == Vec3::NEG_Y)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.2, 0.8, 0.2)),
            "Y",
        ),
        (
            Vec3::Z,
            axes.iter()
                .find(|a| a.direction == Vec3::Z)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.2, 0.4, 0.9)),
            "Z",
        ),
    ];

    let shaft_mesh = meshes.add(Cylinder::new(axis_radius, axis_length));
    let tip_mesh = meshes.add(Cone::new(tip_radius, tip_length));

    for (direction, color, name) in axis_configs {
        let material = materials.add(StandardMaterial {
            base_color: color,
            unlit: true,
            ..default()
        });

        let rotation = if direction == Vec3::X {
            Quat::from_rotation_z(-FRAC_PI_2)
        } else if direction == Vec3::Z {
            Quat::from_rotation_x(FRAC_PI_2)
        } else {
            Quat::IDENTITY
        };

        let shaft_pos = origin + direction * (axis_length / 2.0);
        commands.spawn((
            Mesh3d(shaft_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(shaft_pos).with_rotation(rotation),
            Pickable::IGNORE,
            Name::new(format!("axis_{}_shaft", name)),
        ));

        let tip_pos = origin + direction * (axis_length + tip_length / 2.0);
        commands.spawn((
            Mesh3d(tip_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(tip_pos).with_rotation(rotation),
            Pickable::IGNORE,
            Name::new(format!("axis_{}_tip", name)),
        ));
    }
}

// ============================================================================
// Face Labels
// ============================================================================

/// Spawn 3D text labels on cube faces using bevy_fontmesh
#[allow(clippy::needless_update)]
pub fn spawn_face_labels(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
) {
    // Load the font
    let font: Handle<FontMesh> = asset_server.load("fonts/Roboto-Bold.ttf");

    // Label configuration - scaled by global config
    let label_scale = 0.12 * config.scale;
    let label_depth = 0.05 * config.scale;
    let face_offset = 0.52 * config.scale;

    // Get face labels from coordinate system configuration
    let face_labels = config.system.get_face_labels(face_offset);

    for label in face_labels {
        let material = materials.add(StandardMaterial {
            base_color: label.color,
            unlit: true,
            // Cull back faces so text is only visible from the front
            // This prevents seeing reversed text through transparent cube faces
            cull_mode: Some(Face::Back),
            ..default()
        });

        commands.spawn((
            TextMeshBundle {
                text_mesh: TextMesh {
                    text: label.text.to_string(),
                    font: font.clone(),
                    style: TextMeshStyle {
                        depth: label_depth,
                        anchor: TextAnchor::Center,
                        ..default()
                    },
                    ..default()
                },
                material: MeshMaterial3d(material),
                transform: Transform::from_translation(label.position)
                    .with_rotation(label.rotation)
                    .with_scale(Vec3::splat(label_scale)),
                ..default()
            },
            // Add CubeElement so clicking on labels triggers camera rotation
            CubeElement::Face(label.direction),
            Name::new(format!("label_{}", label.text)),
        ));
    }
}

// ============================================================================
// Rotation Arrows
// ============================================================================

fn create_arrow_mesh() -> Mesh {
    Cone::new(0.015, 0.04).into()
}

fn create_roll_arrow_mesh() -> Mesh {
    Capsule3d::new(0.008, 0.025).into()
}

/// Spawn rotation arrows as children of camera (fixed on screen)
pub fn spawn_rotation_arrows(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    camera_entity: Entity,
) {
    let arrow_mesh = meshes.add(create_arrow_mesh());
    let colors = ViewCubeColors::default();
    let arrow_color = colors.arrow_normal;

    let horizontal_distance = 0.5;
    let vertical_distance = 0.4;
    let depth = -1.2;

    let arrows = [
        (
            RotationArrow::Left,
            Vec3::new(-horizontal_distance, 0.0, depth),
            Quat::from_rotation_z(FRAC_PI_2),
        ),
        (
            RotationArrow::Right,
            Vec3::new(horizontal_distance, 0.0, depth),
            Quat::from_rotation_z(-FRAC_PI_2),
        ),
        (
            RotationArrow::Up,
            Vec3::new(0.0, vertical_distance, depth),
            Quat::IDENTITY,
        ),
        (
            RotationArrow::Down,
            Vec3::new(0.0, -vertical_distance, depth),
            Quat::from_rotation_z(PI),
        ),
    ];

    for (direction, position, rotation) in arrows {
        let material = materials.add(StandardMaterial {
            base_color: arrow_color,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });

        commands
            .spawn((
                Mesh3d(arrow_mesh.clone()),
                MeshMaterial3d(material),
                Transform::from_translation(position).with_rotation(rotation),
                direction,
                Name::new(format!("rotation_arrow_{:?}", direction)),
            ))
            .insert(ChildOf(camera_entity));
    }

    // Roll arrows
    let roll_mesh = meshes.add(create_roll_arrow_mesh());
    let roll_offset = 0.15;

    let roll_arrows = [
        (
            RotationArrow::RollLeft,
            Vec3::new(-roll_offset, vertical_distance, depth),
            Quat::from_rotation_z(FRAC_PI_2 * 0.5),
        ),
        (
            RotationArrow::RollRight,
            Vec3::new(roll_offset, vertical_distance, depth),
            Quat::from_rotation_z(-FRAC_PI_2 * 0.5),
        ),
    ];

    for (direction, position, rotation) in roll_arrows {
        let material = materials.add(StandardMaterial {
            base_color: arrow_color,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });

        commands
            .spawn((
                Mesh3d(roll_mesh.clone()),
                MeshMaterial3d(material),
                Transform::from_translation(position).with_rotation(rotation),
                direction,
                Name::new(format!("rotation_arrow_{:?}", direction)),
            ))
            .insert(ChildOf(camera_entity));
    }
}
