//! Spawning functions for the ViewCube widget

use bevy::camera::ClearColorConfig;
use bevy::camera::visibility::RenderLayers;
use bevy::ecs::hierarchy::ChildOf;
use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy_fontmesh::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI};

use super::components::*;
use super::config::*;
use super::theme::ViewCubeColors;

// ============================================================================
// Main Spawn Function
// ============================================================================

/// Result of spawning a ViewCube
pub struct SpawnedViewCube {
    /// The root entity of the ViewCube (cube + labels + axes)
    pub cube_root: Entity,
    /// The dedicated camera entity used for overlay rendering.
    pub camera: Option<Entity>,
}

/// Spawn a complete ViewCube widget
///
/// Returns the root entity of the ViewCube and its dedicated overlay camera.
pub fn spawn_view_cube(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    main_camera_entity: Entity,
) -> SpawnedViewCube {
    let render_layers = Some(RenderLayers::layer(config.render_layer as usize));

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
        asset_server,
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

    let gizmo_camera = spawn_overlay_camera(commands, config, main_camera_entity);
    spawn_rotation_arrows(
        commands,
        asset_server,
        meshes,
        materials,
        gizmo_camera,
        render_layers,
    );

    SpawnedViewCube {
        cube_root,
        camera: Some(gizmo_camera),
    }
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
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    render_layers: Option<RenderLayers>,
    parent: Entity,
) {
    const AXIS_SCALE_BUMP: f32 = 0.95;
    const CUBE_HALF_EXTENT: f32 = 0.5;

    // Axes are children of `view_cube_root` (already scaled by `config.scale`),
    // so keep these in cube-local units to avoid double-scaling.
    let axis_length = CUBE_HALF_EXTENT * 2.0;
    let axis_radius = 0.08 * AXIS_SCALE_BUMP;
    // Origin at the exact bottom-back-left cube corner - each axis lies on an edge
    // X goes right (along bottom-back edge)
    // Y goes up (along back-left edge)
    // Z goes forward (along bottom-left edge)
    let axis_origin = Vec3::splat(-CUBE_HALF_EXTENT);

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
                .unwrap_or(Color::srgb(0.2, 0.4, 0.9)),
            "Y",
        ),
        (
            Vec3::Z,
            axes.iter()
                .find(|a| a.direction == Vec3::Z)
                .map(|a| a.color)
                .unwrap_or(Color::srgb(0.2, 0.8, 0.2)),
            "Z",
        ),
    ];

    let shaft_mesh = meshes.add(Cylinder::new(axis_radius, axis_length));
    let font: Handle<FontMesh> =
        asset_server.load("embedded://elodin_editor/assets/fonts/Roboto-Bold.ttf");
    let axis_label_scale = 0.41;
    let axis_label_depth = 0.005;
    // Small gap between axis end and letter.
    let axis_label_offset = 0.14 * AXIS_SCALE_BUMP;
    let axis_label_distance = axis_length + axis_label_offset;
    // Push labels away from the cube volume (not just along the axis direction).
    let axis_label_outward_offset = 0.11 * AXIS_SCALE_BUMP;

    for (direction, color, name) in axis_configs {
        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive: LinearRgba::BLACK,
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

        let shaft_pos = axis_origin + direction * (axis_length / 2.0);
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

        let axis_tip = axis_origin + direction * axis_label_distance;
        let outward = (axis_tip - direction * axis_tip.dot(direction)).normalize_or_zero();
        let label_pos = axis_tip + outward * axis_label_outward_offset;
        let mut label_cmd = commands.spawn((
            TextMeshBundle {
                text_mesh: TextMesh {
                    text: name.to_string(),
                    font: font.clone(),
                    style: TextMeshStyle {
                        depth: axis_label_depth,
                        anchor: TextAnchor::Center,
                        ..default()
                    },
                },
                material: MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: color,
                    emissive: LinearRgba::BLACK,
                    unlit: true,
                    cull_mode: None,
                    ..default()
                })),
                transform: Transform::from_translation(label_pos)
                    .with_scale(Vec3::splat(axis_label_scale)),
                ..default()
            },
            Pickable::IGNORE,
            AxisLabelBillboard,
            ChildOf(parent),
            Name::new(format!("axis_{}_label", name)),
        ));
        if let Some(layers) = render_layers.clone() {
            label_cmd.insert(layers);
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
    // Load the embedded font for ViewCube labels
    let font: Handle<FontMesh> =
        asset_server.load("embedded://elodin_editor/assets/fonts/Roboto-Bold.ttf");

    // Label transforms are local to the cube root, which already carries `config.scale`.
    // Applying `config.scale` again would push labels inside the cube in editor mode.
    let label_scale = 0.88;
    let label_depth = 0.008;
    let face_offset = 0.535;

    // Get face labels from coordinate system configuration
    let face_labels = config.system.get_face_labels(face_offset);

    for label in face_labels {
        let material = materials.add(StandardMaterial {
            base_color: label.color,
            emissive: LinearRgba::BLACK,
            unlit: true,
            // Keep text visible even if the generated mesh winding differs per face.
            cull_mode: None,
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
            // Labels are purely visual; interaction stays on cube faces/edges/corners.
            Pickable::IGNORE,
            ChildOf(parent),
            Name::new(format!("label_{}", label.text)),
        ));
        if let Some(layers) = render_layers.clone() {
            label_cmd.insert(layers);
        }
    }
}

// ============================================================================
// Rotation Buttons
// ============================================================================

/// Spawn rotation buttons as icon quads (fixed on screen).
fn spawn_rotation_arrows(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    camera_entity: Entity,
    render_layers: Option<RenderLayers>,
) {
    let colors = ViewCubeColors::default();
    let arrow_color = colors.arrow_normal;

    let directional_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/chevron_right.png");
    let roll_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/loop.png");

    // Quads provide a much more reliable click target than thin cone/capsule meshes.
    let button_size = 0.2;
    let directional_mesh = meshes.add(Rectangle::new(button_size, button_size));
    let roll_mesh = meshes.add(Rectangle::new(button_size * 0.92, button_size * 0.92));

    let horizontal_distance = 0.42;
    let vertical_distance = 0.43;
    let vertical_center_offset = -0.03;
    let depth = -1.2;

    // Slightly thicken cardinal arrows (left/right/up/down) without affecting roll arrows.
    let directional_scale = Vec3::splat(1.08);

    let arrows = [
        (
            RotationArrow::Left,
            Vec3::new(-horizontal_distance, vertical_center_offset, depth),
            Quat::from_rotation_z(PI),
            directional_scale,
        ),
        (
            RotationArrow::Right,
            Vec3::new(horizontal_distance, vertical_center_offset, depth),
            Quat::IDENTITY,
            directional_scale,
        ),
        (
            RotationArrow::Up,
            Vec3::new(0.0, vertical_center_offset + vertical_distance, depth),
            Quat::from_rotation_z(FRAC_PI_2),
            directional_scale,
        ),
        (
            RotationArrow::Down,
            Vec3::new(0.0, vertical_center_offset - vertical_distance, depth),
            Quat::from_rotation_z(-FRAC_PI_2),
            directional_scale,
        ),
    ];

    for (direction, position, rotation, scale) in arrows {
        let material = materials.add(StandardMaterial {
            base_color: arrow_color,
            base_color_texture: Some(directional_icon.clone()),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });

        let mut arrow_cmd = commands.spawn((
            Mesh3d(directional_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(position)
                .with_rotation(rotation)
                .with_scale(scale),
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            arrow_cmd.insert(layers);
        }
        arrow_cmd.insert(ChildOf(camera_entity));
    }

    // Keep roll arrows slightly above the top arrow and near left/right verticals.
    let roll_offset = horizontal_distance - 0.03;
    let roll_height = vertical_distance + 0.01;

    let roll_arrows = [
        (
            RotationArrow::RollLeft,
            Vec3::new(-roll_offset, vertical_center_offset + roll_height, depth),
            Quat::from_rotation_z(0.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ),
        (
            RotationArrow::RollRight,
            Vec3::new(roll_offset, vertical_center_offset + roll_height, depth),
            Quat::from_rotation_z(0.0),
            Vec3::ONE,
        ),
    ];

    for (direction, position, rotation, scale) in roll_arrows {
        let material = materials.add(StandardMaterial {
            base_color: arrow_color,
            base_color_texture: Some(roll_icon.clone()),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        });

        let mut arrow_cmd = commands.spawn((
            Mesh3d(roll_mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(position)
                .with_rotation(rotation)
                .with_scale(scale),
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            arrow_cmd.insert(layers);
        }
        arrow_cmd.insert(ChildOf(camera_entity));
    }
}
