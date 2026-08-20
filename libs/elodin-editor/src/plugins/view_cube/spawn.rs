//! Spawning functions for the ViewCube widget

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{CameraOutputMode, ClearColorConfig};
use bevy::core_pipeline::tonemapping::{DebandDither, Tonemapping};
use bevy::ecs::hierarchy::ChildOf;
use bevy::picking::prelude::*;
use bevy::prelude::*;
use bevy::render::render_resource::BlendState;
use bevy_fontmesh::prelude::*;
use bevy_geo_frames::GeoFrame;
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, PI};

use super::components::*;
use super::config::*;
use super::theme::ViewCubeColors;
use crate::plugins::navigation_gizmo::{NavGizmoCamera, NavGizmoParent};
use crate::plugins::render_layer_alloc::{
    RenderLayerLease, VIEW_CUBE_RENDER_LAYERS, view_cube_render_layer, view_cube_render_layers,
};

pub(crate) fn plugin(app: &mut App) {
    app.init_resource::<ViewCubeFrames>()
        .add_systems(Startup, spawn_frame_view_cubes)
        .add_systems(
            PreUpdate,
            // Reads keyboard state and is read back by the picking observers, so
            // it has to be pinned between the two.
            sync_zoom_button_mode
                .after(bevy::input::InputSystems)
                .before(bevy::picking::PickingSystems::Hover),
        );
}

// ============================================================================
// Shared frame cubes
// ============================================================================

/// Root entity for each geo-frame view cube (ENU, NED, ECEF).
#[derive(Resource, Default)]
pub struct ViewCubeFrames {
    pub cubes: HashMap<GeoFrame, Entity>,
}

impl ViewCubeFrames {
    pub fn get(&self, frame: GeoFrame) -> Option<Entity> {
        self.cubes.get(&frame).copied()
    }
}

/// Result of spawning a per-viewport ViewCube overlay camera.
pub struct SpawnedViewCubeOverlay {
    pub camera: Entity,
    pub frame: GeoFrame,
    /// Shared frame layer used by the global cube mesh.
    pub frame_layer: usize,
    /// Per-viewport layer for rotation arrows and action buttons.
    pub ui_layer: usize,
}

fn spawn_frame_view_cubes(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<ViewCubeConfig>,
    mut frames: ResMut<ViewCubeFrames>,
) {
    for (frame, _) in VIEW_CUBE_RENDER_LAYERS {
        if frames.cubes.contains_key(&frame) {
            continue;
        }
        let mut frame_config = config.clone();
        frame_config.system = CoordinateSystem(frame);
        let render_layers = view_cube_render_layers(frame);
        let cube_root = spawn_frame_view_cube_mesh(
            &mut commands,
            &asset_server,
            &mut meshes,
            &mut materials,
            &frame_config,
            frame,
            render_layers.clone(),
        );
        commands
            .entity(cube_root)
            .insert(bevy_geo_frames::GeoRotation::absolute(
                frame,
                bevy::math::DQuat::IDENTITY,
            ));
        frames.cubes.insert(frame, cube_root);
    }
}

fn spawn_frame_view_cube_mesh(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    config: &ViewCubeConfig,
    frame: GeoFrame,
    render_layers: RenderLayers,
) -> Entity {
    let scene = asset_server.load("embedded://elodin_editor/assets/axes-cube.glb#Scene0");

    let cube_root_cmd = commands.spawn((
        WorldAssetRoot(scene),
        Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(config.scale)),
        Visibility::Hidden,
        ViewCubeRoot,
        ViewCubeMeshRoot,
        ViewCubeFrame(frame),
        Name::new(format!("view_cube_{frame:?}")),
        render_layers.clone(),
    ));

    let cube_root = cube_root_cmd.id();

    spawn_axes(
        commands,
        asset_server,
        meshes,
        materials,
        config,
        cube_root,
        Some(&render_layers),
    );

    spawn_face_labels(
        commands,
        asset_server,
        materials,
        config,
        Some(&render_layers),
        cube_root,
    );

    cube_root
}

/// Spawn a per-viewport overlay camera that renders the shared frame cube.
#[allow(clippy::too_many_arguments)]
pub fn spawn_view_cube_overlay(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    config: &ViewCubeConfig,
    frame: GeoFrame,
    main_camera_entity: Entity,
    ui_lease: RenderLayerLease,
) -> SpawnedViewCubeOverlay {
    let frame_layers = view_cube_render_layers(frame);
    let frame_layer = view_cube_render_layer(frame);
    let ui_layers = ui_lease.render_layers();
    let ui_layer = ui_lease.layer();
    let camera_layers = frame_layers.union(&ui_layers);

    let gizmo_camera = spawn_overlay_camera(
        commands,
        config,
        frame,
        main_camera_entity,
        camera_layers,
        ui_lease,
    );
    spawn_rotation_arrows(
        commands,
        asset_server,
        meshes,
        materials,
        gizmo_camera,
        Some(&ui_layers),
    );
    spawn_viewport_action_buttons(
        commands,
        asset_server,
        meshes,
        materials,
        gizmo_camera,
        Some(&ui_layers),
    );

    commands.entity(gizmo_camera).insert((
        NavGizmoParent {
            main_camera: main_camera_entity,
        },
        NavGizmoCamera,
    ));

    SpawnedViewCubeOverlay {
        camera: gizmo_camera,
        frame,
        frame_layer,
        ui_layer,
    }
}

/// Spawn the dedicated camera for overlay mode
fn spawn_overlay_camera(
    commands: &mut Commands,
    config: &ViewCubeConfig,
    frame: GeoFrame,
    main_camera: Entity,
    render_layers: RenderLayers,
    ui_lease: RenderLayerLease,
) -> Entity {
    commands
        .spawn((
            Transform::from_xyz(0.0, 0.0, config.camera_distance).looking_at(Vec3::ZERO, Vec3::Y),
            Camera {
                order: 3,
                // Steps 1–3 + DebandDither::Disabled (Msaa::Off and Hdr exclusion break visibility).
                clear_color: ClearColorConfig::Custom(Color::srgba(0.0, 0.0, 0.0, 0.0)),
                output_mode: CameraOutputMode::Write {
                    blend_state: Some(BlendState::ALPHA_BLENDING),
                    clear_color: ClearColorConfig::None,
                },
                ..default()
            },
            Camera3d::default(),
            Tonemapping::None,
            DebandDither::Disabled,
            MeshPickingCamera,
            ViewCubeCamera,
            ViewCubeFrameRef(frame),
            ViewCubeLink { main_camera },
            Name::new(format!("view_cube_camera_{frame:?}")),
            render_layers,
            ui_lease,
        ))
        .id()
}

// ============================================================================
// Axes
// ============================================================================

fn axis_visual_configs(system: CoordinateSystem) -> [(Vec3, Color, &'static str); 3] {
    let axes = system.get_axes();
    [
        (axes[0].direction, axes[0].color, "X"),
        (axes[1].direction, axes[1].color, "Y"),
        (axes[2].direction, axes[2].color, "Z"),
    ]
}

/// Spawn RGB axes extending from the bottom-left-back corner of the cube
fn spawn_axes(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    config: &ViewCubeConfig,
    parent: Entity,
    render_layers: Option<&RenderLayers>,
) {
    const AXIS_SCALE_BUMP: f32 = 0.95;
    const CUBE_HALF_EXTENT: f32 = 0.5;
    const AXIS_SURFACE_GAP: f32 = 0.01;
    const AXIS_OVERHANG: f32 = 0.24;

    let axis_length = (CUBE_HALF_EXTENT * 2.0) + AXIS_OVERHANG;
    let axis_radius = 0.04 * AXIS_SCALE_BUMP;
    let axis_center_offset = axis_radius + AXIS_SURFACE_GAP;
    let axis_configs = axis_visual_configs(config.system);
    let points_to: Vec3 = axis_configs.iter().map(|axis_config| axis_config.0).sum();
    let axis_origin = -points_to * (CUBE_HALF_EXTENT + axis_center_offset);

    let shaft_mesh = meshes.add(Cylinder::new(axis_radius, axis_length));
    let font: Handle<Font> =
        asset_server.load("embedded://elodin_editor/assets/fonts/Roboto-Bold.ttf");
    let axis_label_scale = 0.37;
    let axis_label_depth = 0.005;
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
        if let Some(layers) = render_layers {
            shaft_cmd.insert(layers.clone());
        }

        let label_pos = axis_origin + direction * axis_label_distance;
        let mut label_cmd = commands.spawn((
            TextMesh {
                text: name.to_string(),
                font: font.clone(),
                style: TextMeshStyle {
                    depth: axis_label_depth,
                    anchor: TextAnchor::Center,
                    ..default()
                },
            },
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                emissive: LinearRgba::BLACK,
                unlit: true,
                cull_mode: None,
                ..default()
            })),
            Transform::from_translation(label_pos).with_scale(Vec3::splat(axis_label_scale)),
            Pickable::IGNORE,
            AxisLabelBillboard {
                axis_direction: direction,
                base_position: label_pos,
            },
            ChildOf(parent),
            Name::new(format!("axis_{}_label", name)),
        ));
        if let Some(layers) = render_layers {
            label_cmd.insert(layers.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_geo_frames::GeoFrame;

    #[test]
    fn overlay_camera_layer_mask_includes_frame_and_own_ui_only() {
        use crate::plugins::render_layer_alloc::RenderLayerAllocator;

        let frame_layers = view_cube_render_layers(GeoFrame::ENU);
        let frame_layer = view_cube_render_layer(GeoFrame::ENU);
        let mut alloc = RenderLayerAllocator::default();
        let lease_a = alloc.alloc().expect("ui a");
        let lease_b = alloc.alloc().expect("ui b");
        let camera_a = frame_layers.union(&lease_a.render_layers());
        let camera_b = frame_layers.union(&lease_b.render_layers());

        assert!(camera_a.intersects(&RenderLayers::layer(frame_layer)));
        assert!(camera_a.intersects(&lease_a.render_layers()));
        assert!(!camera_a.intersects(&lease_b.render_layers()));

        assert!(camera_b.intersects(&RenderLayers::layer(frame_layer)));
        assert!(camera_b.intersects(&lease_b.render_layers()));
        assert!(!camera_b.intersects(&lease_a.render_layers()));
    }

    #[test]
    fn enu_axis_visual_configs_match_requested_xyz_rbg_mapping() {
        let axis_configs = axis_visual_configs(CoordinateSystem(GeoFrame::ENU));

        assert_eq!(axis_configs[0].0, Vec3::X);
        assert_eq!(axis_configs[0].2, "X");
        assert_eq!(axis_configs[1].0, Vec3::Y);
        assert_eq!(axis_configs[1].2, "Y");
        assert_eq!(axis_configs[2].0, Vec3::Z);
        assert_eq!(axis_configs[2].2, "Z");
    }

    #[test]
    fn ned_axis_visual_configs_returns_correct_colors() {
        let axis_configs = axis_visual_configs(CoordinateSystem(GeoFrame::NED));

        assert_eq!(axis_configs[0].2, "X");
        assert_eq!(axis_configs[1].2, "Y");
        assert_eq!(axis_configs[2].2, "Z");
    }

    #[test]
    fn ned_has_correct_axis_labels() {
        let axes = CoordinateSystem(GeoFrame::NED).get_axes();

        let north_axis = axes
            .iter()
            .find(|a| a.positive_label == "N")
            .expect("North axis");
        let east_axis = axes
            .iter()
            .find(|a| a.positive_label == "E")
            .expect("East axis");
        let down_axis = axes
            .iter()
            .find(|a| a.positive_label == "D")
            .expect("Down axis");

        assert_eq!(north_axis.direction, Vec3::X);
        assert_eq!(north_axis.negative_label, "S");

        assert_eq!(east_axis.direction, Vec3::Y);
        assert_eq!(east_axis.negative_label, "W");

        assert_eq!(down_axis.direction, Vec3::Z);
        assert_eq!(down_axis.negative_label, "U");
    }
}

// ============================================================================
// Face Labels
// ============================================================================

/// Spawn 3D text labels on cube faces using bevy_fontmesh
#[allow(clippy::needless_update)]
fn spawn_face_labels(
    commands: &mut Commands,
    asset_server: &AssetServer,
    materials: &mut Assets<StandardMaterial>,
    config: &ViewCubeConfig,
    render_layers: Option<&RenderLayers>,
    parent: Entity,
) {
    let font: Handle<Font> =
        asset_server.load("embedded://elodin_editor/assets/fonts/Roboto-Bold.ttf");

    let label_scale = 0.6;
    let label_depth = 0.008;
    let face_offset = 0.535;

    let face_labels = config.system.get_face_labels(face_offset);

    for label in face_labels {
        let material = materials.add(StandardMaterial {
            base_color: label.color,
            emissive: LinearRgba::BLACK,
            unlit: true,
            cull_mode: None,
            ..default()
        });

        let mut label_cmd = commands.spawn((
            TextMesh {
                text: label.text.to_string(),
                font: font.clone(),
                style: TextMeshStyle {
                    depth: label_depth,
                    anchor: TextAnchor::Center,
                    ..default()
                },
            },
            MeshMaterial3d(material),
            Transform::from_translation(label.position)
                .with_rotation(label.rotation)
                .with_scale(Vec3::splat(label_scale)),
            Pickable::IGNORE,
            ChildOf(parent),
            Name::new(format!("label_{}", label.text)),
        ));
        if let Some(layers) = render_layers {
            label_cmd.insert(layers.clone());
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
    render_layers: Option<&RenderLayers>,
) {
    let colors = ViewCubeColors::default();
    let arrow_color = colors.arrow_normal;

    let directional_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/chevron_right.png");
    let roll_icon: Handle<Image> =
        asset_server.load("embedded://elodin_editor/assets/icons/loop.png");

    let button_size = 0.2;
    let hitbox_size = button_size * 1.55;
    let directional_mesh = meshes.add(Rectangle::new(button_size, button_size));
    let roll_mesh = meshes.add(Rectangle::new(button_size * 0.92, button_size * 0.92));
    let hitbox_mesh = meshes.add(Rectangle::new(hitbox_size, hitbox_size));
    let hitbox_material = materials.add(StandardMaterial {
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
            Pickable::IGNORE,
            direction,
            Name::new(format!("rotation_arrow_{:?}", direction)),
        ));
        if let Some(layers) = render_layers {
            arrow_cmd.insert(layers.clone());
        }
        arrow_cmd.insert(ChildOf(camera_entity));
        let arrow_entity = arrow_cmd.id();

        let mut hitbox_cmd = commands.spawn((
            Mesh3d(hitbox_mesh.clone()),
            MeshMaterial3d(hitbox_material.clone()),
            Transform::from_translation(Vec3::ZERO),
            ChildOf(arrow_entity),
            Pickable::default(),
            Name::new(format!("rotation_arrow_{:?}_hitbox", direction)),
        ));
        if let Some(layers) = render_layers {
            hitbox_cmd.insert(layers.clone());
        }
    }

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
        if let Some(layers) = render_layers {
            arrow_cmd.insert(layers.clone());
        }
        arrow_cmd.insert(ChildOf(camera_entity));
        let arrow_entity = arrow_cmd.id();

        let mut hitbox_cmd = commands.spawn((
            Mesh3d(hitbox_mesh.clone()),
            MeshMaterial3d(hitbox_material.clone()),
            Transform::from_translation(Vec3::ZERO),
            ChildOf(arrow_entity),
            Pickable::default(),
            Name::new(format!("rotation_arrow_{:?}_hitbox", direction)),
        ));
        if let Some(layers) = render_layers {
            hitbox_cmd.insert(layers.clone());
        }
    }
}

/// Icon textures of a zoom button, kept on the icon entity: every viewport
/// overlay spawns its own material, so this state cannot be shared globally.
#[derive(Debug, Component)]
struct ZoomButtonIcon {
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
    render_layers: Option<&RenderLayers>,
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
    if let Some(layers) = render_layers {
        reset_cmd.insert(layers.clone());
    }
    reset_cmd.insert(ChildOf(camera_entity));
    let reset_button = reset_cmd.id();
    let mut reset_hitbox_cmd = commands.spawn((
        Mesh3d(reset_hitbox_mesh),
        MeshMaterial3d(hitbox_material.clone()),
        Transform::from_translation(Vec3::ZERO),
        ChildOf(reset_button),
        Pickable::default(),
        Name::new("viewport_action_button_Reset_hitbox"),
    ));
    if let Some(layers) = render_layers {
        reset_hitbox_cmd.insert(layers.clone());
    }

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
            Name::new("viewport_action_button_zoom"),
        ))
        .id();
    let mut zoom_button_cmd = commands.entity(zoom_button);
    if let Some(layers) = render_layers {
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
    let mut zoom_icon_cmd = commands.spawn((
        Mesh3d(zoom_icon_mesh),
        MeshMaterial3d(zoom_icon_material),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.002)),
        Pickable::IGNORE,
        ChildOf(zoom_button),
        ZoomButtonIcon {
            zoom_in: zoom_in_icon,
            zoom_out: zoom_out_icon,
        },
        Name::new("viewport_action_button_zoom_icon"),
    ));
    if let Some(layers) = render_layers {
        zoom_icon_cmd.insert(layers.clone());
    }

    let mut zoom_hitbox_cmd = commands.spawn((
        Mesh3d(zoom_hitbox_mesh),
        MeshMaterial3d(hitbox_material),
        Transform::from_translation(Vec3::ZERO),
        ChildOf(zoom_button),
        Pickable::default(),
        Name::new("viewport_action_button_zoom_hitbox"),
    ));
    if let Some(layers) = render_layers {
        zoom_hitbox_cmd.insert(layers.clone());
    }
}

/// Track whether the zoom button acts as zoom-in (Alt/Option held) or zoom-out.
///
/// Derived from the held state rather than press/release edges: a dropped edge
/// (window focus change, compositor grabbing Alt on Linux, Alt-Tab) would
/// otherwise latch the button in the wrong mode for the rest of the session.
fn sync_zoom_button_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut buttons: Query<&mut ViewportActionButton>,
    icons: Query<(&ZoomButtonIcon, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let zoom_in = keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);
    let mode = if zoom_in {
        ViewportActionButton::ZoomIn
    } else {
        ViewportActionButton::ZoomOut
    };

    for mut action in &mut buttons {
        let is_zoom = matches!(
            *action,
            ViewportActionButton::ZoomIn | ViewportActionButton::ZoomOut
        );
        if is_zoom && *action != mode {
            *action = mode;
        }
    }

    for (icon, handle) in icons.iter() {
        let texture = if zoom_in {
            &icon.zoom_in
        } else {
            &icon.zoom_out
        };
        // Only reach for `get_mut` on a real change, otherwise every frame would
        // flag the material as modified and force a GPU re-upload.
        let stale = materials
            .get(&handle.0)
            .is_some_and(|material| material.base_color_texture.as_ref() != Some(texture));
        if stale && let Some(mut material) = materials.get_mut(&handle.0) {
            material.base_color_texture = Some(texture.clone());
        }
    }
}

#[cfg(test)]
mod zoom_button_tests {
    use super::*;

    struct ZoomButton {
        button: Entity,
        material: Handle<StandardMaterial>,
        zoom_in: Handle<Image>,
        zoom_out: Handle<Image>,
    }

    impl ZoomButton {
        fn spawn(app: &mut App) -> Self {
            let images = app.world().resource::<Assets<Image>>();
            let zoom_in = images.reserve_handle();
            let zoom_out = images.reserve_handle();
            let material = app
                .world_mut()
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial {
                    base_color_texture: Some(zoom_out.clone()),
                    ..default()
                });

            let button = app.world_mut().spawn(ViewportActionButton::ZoomOut).id();
            app.world_mut().spawn((
                MeshMaterial3d(material.clone()),
                ZoomButtonIcon {
                    zoom_in: zoom_in.clone(),
                    zoom_out: zoom_out.clone(),
                },
                ChildOf(button),
            ));

            Self {
                button,
                material,
                zoom_in,
                zoom_out,
            }
        }

        fn action(&self, app: &App) -> ViewportActionButton {
            *app.world()
                .entity(self.button)
                .get::<ViewportActionButton>()
                .unwrap()
        }

        fn icon(&self, app: &App) -> Handle<Image> {
            app.world()
                .resource::<Assets<StandardMaterial>>()
                .get(&self.material)
                .unwrap()
                .base_color_texture
                .clone()
                .unwrap()
        }
    }

    fn app() -> App {
        let mut app = App::new();
        app.init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<Image>>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, sync_zoom_button_mode);
        app
    }

    fn set_alt(app: &mut App, held: bool) {
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        if held {
            keys.press(KeyCode::AltLeft);
        } else {
            keys.release(KeyCode::AltLeft);
        }
    }

    /// Each viewport overlay owns its icon material, so all of them have to
    /// follow the modifier, not just the last one spawned.
    #[test]
    fn every_zoom_button_follows_alt() {
        let mut app = app();
        let first = ZoomButton::spawn(&mut app);
        let second = ZoomButton::spawn(&mut app);

        set_alt(&mut app, true);
        app.update();

        for button in [&first, &second] {
            assert_eq!(button.action(&app), ViewportActionButton::ZoomIn);
            assert_eq!(button.icon(&app), button.zoom_in);
        }

        set_alt(&mut app, false);
        app.update();

        for button in [&first, &second] {
            assert_eq!(button.action(&app), ViewportActionButton::ZoomOut);
            assert_eq!(button.icon(&app), button.zoom_out);
        }
    }

    /// A press edge consumed elsewhere (focus change, window manager) must not
    /// leave the button stuck in the wrong mode.
    #[test]
    fn zoom_mode_recovers_from_missed_key_edges() {
        let mut app = app();
        let button = ZoomButton::spawn(&mut app);

        set_alt(&mut app, true);
        // Clearing drops `just_pressed`/`just_released` while Alt stays held.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();

        assert_eq!(button.action(&app), ViewportActionButton::ZoomIn);
        assert_eq!(button.icon(&app), button.zoom_in);

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.update();

        assert_eq!(button.action(&app), ViewportActionButton::ZoomOut);
        assert_eq!(button.icon(&app), button.zoom_out);
    }

    #[test]
    fn reset_button_is_not_affected_by_alt() {
        let mut app = app();
        let reset = app.world_mut().spawn(ViewportActionButton::Reset).id();

        set_alt(&mut app, true);
        app.update();

        assert_eq!(
            *app.world()
                .entity(reset)
                .get::<ViewportActionButton>()
                .unwrap(),
            ViewportActionButton::Reset
        );
    }
}
