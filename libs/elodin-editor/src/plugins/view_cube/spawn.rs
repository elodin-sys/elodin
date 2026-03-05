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

pub(crate) fn plugin(app: &mut App) {
    app.add_systems(PreUpdate, swap_zoom_buttons_on_alt);
}

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

    // Load the axes-cube.glb (embedded)
    let scene = asset_server.load("embedded://elodin_editor/assets/axes-cube.glb#Scene0");

    // Spawn the cube root hidden; apply_render_layers_to_scene will flip to
    // Visibility::Inherited once every descendant has been assigned the correct
    // RenderLayers, preventing the GLB children from briefly appearing on all
    // cameras via the default layer 0.
    let mut cube_root_cmd = commands.spawn((
        SceneRoot(scene),
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(config.scale)),
        Visibility::Hidden,
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
        render_layers.clone(),
    );
    spawn_viewport_action_buttons(
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

fn axis_visual_configs(system: CoordinateSystem) -> [(Vec3, Color, &'static str); 3] {
    let axes = system.get_axes();
    let east_color = axes
        .iter()
        .find(|axis| axis.direction.x.abs() > 0.9)
        .map(|axis| axis.color)
        .unwrap_or(Color::srgb(0.9, 0.2, 0.2));
    let up_color = axes
        .iter()
        .find(|axis| axis.direction.y.abs() > 0.9)
        .map(|axis| axis.color)
        .unwrap_or(Color::srgb(0.2, 0.4, 0.9));
    let north_color = axes
        .iter()
        .find(|axis| axis.direction.z.abs() > 0.9)
        .map(|axis| axis.color)
        .unwrap_or(Color::srgb(0.2, 0.8, 0.2));

    // Visual mapping requested for ENU view cube:
    // - X must point opposite local +X
    // - Y/Z labels are swapped while preserving blue/green axis colors
    [
        (Vec3::NEG_X, east_color, "X"),
        (Vec3::Y, up_color, "Z"),
        (Vec3::Z, north_color, "Y"),
    ]
}

fn axis_origin_for_visual_layout(cube_half_extent: f32, axis_center_offset: f32) -> Vec3 {
    // In synced mode the cube receives a Y-PI correction.
    // Use local corner (+X, -Y, -Z) so it appears at visual (W, bottom, S),
    // then offset outward by axis radius so shafts sit on cube borders.
    Vec3::new(
        cube_half_extent + axis_center_offset,
        -cube_half_extent - axis_center_offset,
        -cube_half_extent - axis_center_offset,
    )
}

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
    const AXIS_SURFACE_GAP: f32 = 0.01;
    const AXIS_OVERHANG: f32 = 0.24;

    // Axes are children of `view_cube_root` (already scaled by `config.scale`),
    // so keep these in cube-local units to avoid double-scaling.
    // Extend a touch beyond cube edges so XYZ labels have more breathing room.
    let axis_length = (CUBE_HALF_EXTENT * 2.0) + AXIS_OVERHANG;
    let axis_radius = 0.04 * AXIS_SCALE_BUMP;
    // `axis_visual_configs` defines final visual axis directions/labels/colors.
    // Keep shafts on cube borders (no extra surface gap).
    let axis_center_offset = axis_radius + AXIS_SURFACE_GAP;
    let axis_origin = axis_origin_for_visual_layout(CUBE_HALF_EXTENT, axis_center_offset);

    let axis_configs = axis_visual_configs(config.system);

    let shaft_mesh = meshes.add(Cylinder::new(axis_radius, axis_length));
    let font: Handle<FontMesh> =
        asset_server.load("embedded://elodin_editor/assets/fonts/Roboto-Bold.ttf");
    let axis_label_scale = 0.37;
    let axis_label_depth = 0.005;
    // Small gap between axis end and letter.
    let axis_label_offset = 0.18 * AXIS_SCALE_BUMP;
    let axis_label_distance = axis_length + axis_label_offset;

    for (direction, color, name) in axis_configs {
        let material = materials.add(StandardMaterial {
            base_color: color,
            emissive: LinearRgba::BLACK,
            unlit: true,
            ..default()
        });

        let rotation = if direction.x.abs() > 0.9 {
            Quat::from_rotation_z(-FRAC_PI_2)
        } else if direction.z.abs() > 0.9 {
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

        let label_pos = axis_origin + direction * axis_label_distance;
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
            AxisLabelBillboard {
                axis_direction: direction,
                base_position: label_pos,
            },
            ChildOf(parent),
            Name::new(format!("axis_{}_label", name)),
        ));
        if let Some(layers) = render_layers.clone() {
            label_cmd.insert(layers);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_visual_configs_match_requested_xyz_rbg_mapping() {
        let axis_configs = axis_visual_configs(CoordinateSystem::ENU);

        assert_eq!(axis_configs[0].0, Vec3::NEG_X);
        assert_eq!(axis_configs[0].2, "X");
        assert_eq!(axis_configs[1].0, Vec3::Y);
        assert_eq!(axis_configs[1].2, "Z");
        assert_eq!(axis_configs[2].0, Vec3::Z);
        assert_eq!(axis_configs[2].2, "Y");

        let logical_axes = CoordinateSystem::ENU.get_axes();
        let east_color = logical_axes
            .iter()
            .find(|axis| axis.direction == Vec3::X)
            .map(|axis| axis.color)
            .expect("east color");
        let up_color = logical_axes
            .iter()
            .find(|axis| axis.direction == Vec3::Y)
            .map(|axis| axis.color)
            .expect("up color");
        let north_color = logical_axes
            .iter()
            .find(|axis| axis.direction == Vec3::Z)
            .map(|axis| axis.color)
            .expect("north color");

        assert_eq!(axis_configs[0].1, east_color);
        assert_eq!(axis_configs[1].1, up_color);
        assert_eq!(axis_configs[2].1, north_color);
    }

    #[test]
    fn axis_origin_layout_maps_to_visual_west_bottom_south_corner() {
        let local_origin = axis_origin_for_visual_layout(0.5, 0.04);
        let correction = ViewCubeConfig::system_axis_correction(CoordinateSystem::ENU);
        let visual_origin = correction * local_origin;

        assert!(
            visual_origin.x < 0.0,
            "origin should be on visual west side"
        );
        assert!(
            visual_origin.y < 0.0,
            "origin should be on visual bottom side"
        );
        assert!(
            visual_origin.z > 0.0,
            "origin should be on visual south side"
        );
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
    let label_scale = 0.6;
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
    let hitbox_size = button_size * 1.55;
    let directional_mesh = meshes.add(Rectangle::new(button_size, button_size));
    let roll_mesh = meshes.add(Rectangle::new(button_size * 0.92, button_size * 0.92));
    let hitbox_mesh = meshes.add(Rectangle::new(hitbox_size, hitbox_size));
    let hitbox_material = materials.add(StandardMaterial {
        // Invisible pick surface used to make interaction forgiving.
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.0),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });

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
            // Use a dedicated child hitbox so users don't need pixel-perfect alignment.
            Pickable::IGNORE,
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            arrow_cmd.insert(layers);
        }
        arrow_cmd.insert(ChildOf(camera_entity));
        let arrow_entity = arrow_cmd.id();

        let mut hitbox_cmd = commands.spawn((
            Mesh3d(hitbox_mesh.clone()),
            MeshMaterial3d(hitbox_material.clone()),
            Transform::from_translation(Vec3::ZERO),
            ChildOf(arrow_entity),
            Name::new(format!("rotation_arrow_{:?}_hitbox", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            hitbox_cmd.insert(layers);
        }
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
            Pickable::IGNORE,
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            arrow_cmd.insert(layers);
        }
        arrow_cmd.insert(ChildOf(camera_entity));
        let arrow_entity = arrow_cmd.id();

        let mut hitbox_cmd = commands.spawn((
            Mesh3d(hitbox_mesh.clone()),
            MeshMaterial3d(hitbox_material.clone()),
            Transform::from_translation(Vec3::ZERO),
            ChildOf(arrow_entity),
            Name::new(format!("rotation_arrow_{:?}_hitbox", direction)),
        ));
        if let Some(layers) = render_layers.clone() {
            hitbox_cmd.insert(layers);
        }
    }
}

#[derive(Debug, Resource)]
struct ZoomIconMaterials {
    icon_material: Handle<StandardMaterial>,
    zoom_in: Handle<Image>,
    zoom_out: Handle<Image>,
}

/// Spawn viewport action buttons as icon quads (fixed on screen).
fn spawn_viewport_action_buttons(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    camera_entity: Entity,
    render_layers: Option<RenderLayers>,
) {
    let colors = ViewCubeColors::default();
    let button_color = colors.arrow_normal;
    let depth = -1.2;
    let reset_button_mesh = meshes.add(Rectangle::new(0.165, 0.165));
    let reset_hitbox_mesh = meshes.add(Rectangle::new(0.26, 0.26));
    let zoom_button_mesh = meshes.add(Annulus::new(0.073, 0.088));
    let zoom_icon_mesh = meshes.add(Rectangle::new(0.102, 0.102));
    let zoom_hitbox_mesh = meshes.add(Rectangle::new(0.26, 0.26));
    let hitbox_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.0),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });

    let reset_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/viewport.png");
    let zoom_out_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/subtract.png");
    let zoom_in_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/add.png");
    // Keep buttons inside the camera frustum (Perspective fov ~= 45deg),
    // while still reading as bottom-left / bottom-right controls.
    let reset_material = materials.add(StandardMaterial {
        base_color: button_color,
        base_color_texture: Some(reset_icon),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });
    let mut reset_cmd = commands.spawn((
        Mesh3d(reset_button_mesh),
        MeshMaterial3d(reset_material),
        Transform::from_translation(Vec3::new(-0.40, -0.39, depth)),
        Pickable::IGNORE,
        ViewportActionButton::Reset,
        Name::new("viewport_action_button_Reset"),
    ));
    if let Some(layers) = render_layers.clone() {
        reset_cmd.insert(layers);
    }
    reset_cmd.insert(ChildOf(camera_entity));
    let reset_button = reset_cmd.id();
    let mut reset_hitbox_cmd = commands.spawn((
        Mesh3d(reset_hitbox_mesh),
        MeshMaterial3d(hitbox_material.clone()),
        Transform::from_translation(Vec3::ZERO),
        ChildOf(reset_button),
        Name::new("viewport_action_button_Reset_hitbox"),
    ));
    if let Some(layers) = render_layers.clone() {
        reset_hitbox_cmd.insert(layers);
    }

    // Circular zoom-out button for clearer visual hierarchy.
    let zoom_material = materials.add(StandardMaterial {
        base_color: button_color,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });
    let zoom_button = commands
        .spawn((
            Mesh3d(zoom_button_mesh),
            MeshMaterial3d(zoom_material),
            Transform::from_translation(Vec3::new(0.40, -0.39, depth)),
            Pickable::IGNORE,
            ViewportActionButton::ZoomOut,
            Name::new("viewport_action_button_ZoomOut"),
        ))
        .id();
    let mut zoom_button_cmd = commands.entity(zoom_button);
    if let Some(layers) = render_layers.clone() {
        zoom_button_cmd.insert(layers.clone());
    }
    zoom_button_cmd.insert(ChildOf(camera_entity));

    let zoom_icon_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(zoom_out_icon.clone()),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });
    commands.insert_resource(ZoomIconMaterials {
        icon_material: zoom_icon_material.clone(),
        zoom_out: zoom_out_icon,
        zoom_in: zoom_in_icon,
    });
    let mut zoom_icon_cmd = commands.spawn((
        Mesh3d(zoom_icon_mesh),
        MeshMaterial3d(zoom_icon_material),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.002)),
        Pickable::IGNORE,
        ChildOf(zoom_button),
        Name::new("viewport_action_button_ZoomOut_icon"),
    ));
    if let Some(layers) = render_layers.clone() {
        zoom_icon_cmd.insert(layers);
    }

    let mut zoom_hitbox_cmd = commands.spawn((
        Mesh3d(zoom_hitbox_mesh),
        MeshMaterial3d(hitbox_material),
        Transform::from_translation(Vec3::ZERO),
        ChildOf(zoom_button),
        Name::new("viewport_action_button_ZoomOut_hitbox"),
    ));
    if let Some(layers) = render_layers.clone() {
        zoom_hitbox_cmd.insert(layers);
    }
}

fn swap_zoom_buttons_on_alt(
    mut buttons: Query<&mut ViewportActionButton>,
    keys: Res<ButtonInput<KeyCode>>,
    zoom_icon_materials: Option<Res<ZoomIconMaterials>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let pressed = if keys.just_pressed(KeyCode::AltLeft) || keys.just_pressed(KeyCode::AltRight) {
        Some(true)
    } else if keys.just_released(KeyCode::AltLeft) || keys.just_released(KeyCode::AltRight) {
        Some(false)
    } else {
        None
    };
    if let Some(pressed) = pressed {
        for mut action in &mut buttons {
            match *action {
                ViewportActionButton::ZoomIn | ViewportActionButton::ZoomOut => {
                    if pressed {
                        // Go to zoom in.
                        *action = ViewportActionButton::ZoomIn;
                    } else {
                        // Go back to zoom out.
                        *action = ViewportActionButton::ZoomOut;
                    }
                }
                _ => (),
            }
        }
        let Some(zoom_icon_materials) = zoom_icon_materials else {
            return;
        };

        let Some(zoom_material) = materials.get_mut(&zoom_icon_materials.icon_material) else {
            return;
        };
        let zoom_icon: Handle<Image> = if pressed {
            zoom_icon_materials.zoom_in.clone()
        } else {
            zoom_icon_materials.zoom_out.clone()
        };
        zoom_material.base_color_texture = Some(zoom_icon);
    }
}
