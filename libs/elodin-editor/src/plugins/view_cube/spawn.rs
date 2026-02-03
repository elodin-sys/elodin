//! Spawning functions for the ViewCube widget

use bevy::camera::ClearColorConfig;
use bevy::camera::visibility::RenderLayers;
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

/// Result of spawning a ViewCube
pub struct SpawnedViewCube {
    /// The root entity of the ViewCube (cube + labels + axes)
    pub cube_root: Entity,
    /// The dedicated camera entity (only in overlay mode)
    pub camera: Option<Entity>,
}

/// Spawn a complete ViewCube widget
///
/// Returns the root entity of the ViewCube and optionally a dedicated camera.
/// In overlay mode, a dedicated camera is created for rendering the ViewCube
/// as an overlay in the top-right corner.
pub fn spawn_view_cube(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    main_camera_entity: Entity,
) -> SpawnedViewCube {
    let render_layers = if config.use_overlay {
        Some(RenderLayers::layer(config.render_layer as usize))
    } else {
        None
    };

    // Load the axes-cube.glb
    let scene = asset_server.load("axes-cube.glb#Scene0");

    // Spawn the cube root with link to main camera
    let mut cube_root_cmd = commands.spawn((
        SceneRoot(scene),
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(config.scale)),
        ViewCubeRoot,
        ViewCubeMeshRoot,
        ViewCubeLink {
            main_camera: main_camera_entity,
        },
        ViewCubeRenderLayer(config.render_layer as usize),
        Name::new("view_cube_root"),
    ));

    if let Some(layers) = render_layers.clone() {
        cube_root_cmd.insert(layers);
    }

    let cube_root = cube_root_cmd.id();

    // Spawn RGB axes extending from the corner of the cube (as children of cube root)
    spawn_axes(
        commands,
        meshes,
        materials,
        config,
        render_layers.clone(),
        cube_root,
    );

    // Spawn 3D text labels on cube faces (as children of cube root)
    spawn_face_labels(
        commands,
        asset_server,
        materials,
        config,
        render_layers.clone(),
        cube_root,
    );

    // Spawn the dedicated camera for overlay mode
    let camera = if config.use_overlay {
        let gizmo_camera = spawn_overlay_camera(commands, config, main_camera_entity);
        // In overlay mode, arrows are children of the gizmo camera
        spawn_rotation_arrows(commands, meshes, materials, gizmo_camera, render_layers);
        Some(gizmo_camera)
    } else {
        // In standalone mode, arrows are children of the main camera
        spawn_rotation_arrows(
            commands,
            meshes,
            materials,
            main_camera_entity,
            render_layers,
        );
        None
    };

    SpawnedViewCube { cube_root, camera }
}

/// Spawn the dedicated camera for overlay mode
fn spawn_overlay_camera(
    commands: &mut Commands,
    config: &ViewCubeConfig,
    main_camera: Entity,
) -> Entity {
    let render_layers = RenderLayers::layer(config.render_layer as usize);

    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, config.camera_distance).looking_at(Vec3::ZERO, Vec3::Y),
            Camera {
                order: 3, // Match navigation_gizmo camera order
                // NOTE: Don't clear on the ViewCube camera because the
                // MainCamera already cleared the window.
                clear_color: ClearColorConfig::None,
                ..default()
            },
            Camera3d::default(),
            render_layers,
            ViewCubeCamera,
            ViewCubeLink { main_camera },
            Name::new("view_cube_camera"),
        ))
        .id()
}

// ============================================================================
// Axes
// ============================================================================

/// Spawn RGB axes extending from the bottom-left-back corner of the cube
fn spawn_axes(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    render_layers: Option<RenderLayers>,
    parent: Entity,
) {
    // Axes extend beyond the cube for better visibility
    let axis_length = 2.6 * config.scale; // Long enough to be clearly visible
    let axis_radius = 0.08 * config.scale; // Thick for visibility
    let tip_radius = 0.14 * config.scale;
    let tip_length = 0.3 * config.scale;
    // Origin at bottom-front-left corner - axes point towards user in ENU
    let origin = Vec3::new(-0.55, -0.55, 0.55) * config.scale;

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
        // Convert color to LinearRgba for emissive
        let emissive = match color {
            Color::Srgba(c) => LinearRgba::new(c.red * 0.5, c.green * 0.5, c.blue * 0.5, 1.0),
            _ => LinearRgba::new(0.3, 0.3, 0.3, 1.0),
        };
        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive, // Glow for visibility in dark/light modes
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
        let mut shaft_cmd = commands.spawn((
            Mesh3d(shaft_mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(shaft_pos).with_rotation(rotation),
            Pickable::IGNORE,
            ChildOf(parent),
            Name::new(format!("axis_{}_shaft", name)),
        ));
        if let Some(layers) = render_layers.clone() {
            shaft_cmd.insert(layers);
        }

        let tip_pos = origin + direction * (axis_length + tip_length / 2.0);
        let mut tip_cmd = commands.spawn((
            Mesh3d(tip_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(tip_pos).with_rotation(rotation),
            Pickable::IGNORE,
            ChildOf(parent),
            Name::new(format!("axis_{}_tip", name)),
        ));
        if let Some(layers) = render_layers.clone() {
            tip_cmd.insert(layers);
        }
    }
}

// ============================================================================
// Face Labels
// ============================================================================

/// Spawn 3D text labels on cube faces using bevy_fontmesh
#[allow(clippy::needless_update)]
fn spawn_face_labels(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    render_layers: Option<RenderLayers>,
    parent: Entity,
) {
    // Load the embedded font
    let font: Handle<FontMesh> =
        asset_server.load("embedded://elodin_editor/assets/fonts/IBMPlexMono-Medium_ss04.ttf");

    // Label configuration - scaled by global config
    // Large letters that almost fill the face
    let label_scale = 0.35 * config.scale;
    let label_depth = 0.03 * config.scale;
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

        let mut label_cmd = commands.spawn((
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
            ChildOf(parent),
            Name::new(format!("label_{}", label.text)),
        ));
        if let Some(layers) = render_layers.clone() {
            label_cmd.insert(layers);
        }
    }
}

// ============================================================================
// Rotation Arrows
// ============================================================================

fn create_arrow_mesh() -> Mesh {
    // Increased size for better visibility and clickability
    Cone::new(0.035, 0.08).into()
}

fn create_roll_arrow_mesh() -> Mesh {
    // Increased size for better visibility and clickability
    Capsule3d::new(0.018, 0.05).into()
}

/// Spawn rotation arrows as children of camera (fixed on screen)
fn spawn_rotation_arrows(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    camera_entity: Entity,
    render_layers: Option<RenderLayers>,
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

        let mut arrow_cmd = commands.spawn((
            Mesh3d(arrow_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(position).with_rotation(rotation),
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            arrow_cmd.insert(layers);
        }
        arrow_cmd.insert(ChildOf(camera_entity));
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

        let mut arrow_cmd = commands.spawn((
            Mesh3d(roll_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(position).with_rotation(rotation),
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            arrow_cmd.insert(layers);
        }
        arrow_cmd.insert(ChildOf(camera_entity));
    }
}
