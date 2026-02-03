//! Camera control systems for the ViewCube plugin
//!
//! When `auto_rotate` is enabled, this module handles camera rotation
//! in response to ViewCube events.

use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::math::Dir3;
use bevy::prelude::*;
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::extensions::look_to::LookToTrigger;

use super::components::{
    RotationArrow, ViewCubeCamera, ViewCubeLink, ViewCubeRenderLayer, ViewCubeRoot,
};
use super::config::ViewCubeConfig;
use super::events::ViewCubeEvent;

// ============================================================================
// Components
// ============================================================================

/// Marker component for the camera that should be controlled by the ViewCube
#[derive(Component)]
pub struct ViewCubeTargetCamera;

// ============================================================================
// Resources
// ============================================================================

/// Tracks camera animation state
#[derive(Resource, Default)]
pub struct CameraAnimation {
    pub animating: bool,
    pub progress: f32,
    pub start_position: Vec3,
    pub start_rotation: Quat,
    pub target_position: Vec3,
    pub target_rotation: Quat,
}

// ============================================================================
// Systems
// ============================================================================

/// Handle ViewCube events and rotate the camera accordingly
pub fn handle_view_cube_camera(
    mut events: MessageReader<ViewCubeEvent>,
    mut camera_query: Query<&mut Transform, With<ViewCubeTargetCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    config: Res<ViewCubeConfig>,
) {
    for event in events.read() {
        match event {
            ViewCubeEvent::FaceClicked(direction) => {
                let look_dir = direction.to_look_direction();
                if let Ok(transform) = camera_query.single() {
                    start_camera_animation(look_dir, transform, &mut camera_anim, &config);
                }
            }
            ViewCubeEvent::EdgeClicked(direction) => {
                let look_dir = direction.to_look_direction();
                if let Ok(transform) = camera_query.single() {
                    start_camera_animation(look_dir, transform, &mut camera_anim, &config);
                }
            }
            ViewCubeEvent::CornerClicked(position) => {
                let look_dir = position.to_look_direction();
                if let Ok(transform) = camera_query.single() {
                    start_camera_animation(look_dir, transform, &mut camera_anim, &config);
                }
            }
            ViewCubeEvent::ArrowClicked(arrow) => {
                if let Ok(mut transform) = camera_query.single_mut() {
                    apply_arrow_rotation(*arrow, &config, &mut transform);
                }
            }
        }
    }
}

/// Animate the camera smoothly to its target
pub fn animate_camera(
    mut camera_query: Query<&mut Transform, With<ViewCubeTargetCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    time: Res<Time>,
) {
    if !camera_anim.animating {
        return;
    }

    let Ok(mut transform) = camera_query.single_mut() else {
        return;
    };

    camera_anim.progress += time.delta_secs() * 3.0; // Animation speed

    if camera_anim.progress >= 1.0 {
        transform.translation = camera_anim.target_position;
        transform.rotation = camera_anim.target_rotation;
        camera_anim.animating = false;
        camera_anim.progress = 0.0;
    } else {
        // Smooth interpolation using ease-out curve
        let t = ease_out_cubic(camera_anim.progress);
        transform.translation = camera_anim
            .start_position
            .lerp(camera_anim.target_position, t);
        transform.rotation = camera_anim
            .start_rotation
            .slerp(camera_anim.target_rotation, t);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn start_camera_animation(
    mut look_dir: Vec3,
    current_transform: &Transform,
    camera_anim: &mut CameraAnimation,
    config: &ViewCubeConfig,
) {
    // Check if clicking through transparent face (flip direction if looking from behind)
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

fn apply_arrow_rotation(arrow: RotationArrow, config: &ViewCubeConfig, transform: &mut Transform) {
    let rotation = match arrow {
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

    // Rotate around the origin (where the cube is)
    transform.rotate_around(Vec3::ZERO, rotation);
}

fn get_up_direction_for_look(look_dir: Vec3) -> Vec3 {
    // When looking up or down, use a different up vector to avoid gimbal lock
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

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

// ============================================================================
// Sync System - Makes ViewCube rotate inversely to main camera
// ============================================================================

/// Sync the ViewCube rotation with the main camera.
/// The cube rotates inversely so it always shows the world orientation
/// from the camera's perspective.
pub fn sync_view_cube_rotation(
    main_camera_query: Query<&GlobalTransform, Without<ViewCubeRoot>>,
    mut view_cube_query: Query<(&ViewCubeLink, &mut Transform), With<ViewCubeRoot>>,
) {
    for (link, mut cube_transform) in view_cube_query.iter_mut() {
        let Ok(main_camera_transform) = main_camera_query.get(link.main_camera) else {
            continue;
        };

        // Set cube rotation to inverse of camera rotation
        // This makes the cube appear to show world orientation
        let (_, rotation, _) = main_camera_transform.to_scale_rotation_translation();
        cube_transform.rotation = rotation.conjugate();
    }
}

// ============================================================================
// Overlay Mode Systems
// ============================================================================

/// Apply render layers to entities loaded from the GLB scene.
/// This runs every frame to catch newly loaded entities.
pub fn apply_render_layers_to_scene(
    config: Res<ViewCubeConfig>,
    view_cube_query: Query<(Entity, &ViewCubeRenderLayer), With<ViewCubeRoot>>,
    children_query: Query<&Children>,
    entities_without_layer: Query<Entity, (Without<RenderLayers>, Without<ViewCubeCamera>)>,
    mut commands: Commands,
) {
    if !config.use_overlay {
        return;
    }

    for (cube_root, layer) in view_cube_query.iter() {
        let render_layers = RenderLayers::layer(layer.0);

        // Find all descendants of the cube root that don't have render layers
        apply_layers_recursive(
            cube_root,
            &children_query,
            &entities_without_layer,
            &render_layers,
            &mut commands,
        );
    }
}

fn apply_layers_recursive(
    entity: Entity,
    children_query: &Query<&Children>,
    entities_without_layer: &Query<Entity, (Without<RenderLayers>, Without<ViewCubeCamera>)>,
    render_layers: &RenderLayers,
    commands: &mut Commands,
) {
    // Apply render layer if entity doesn't have one
    if entities_without_layer.get(entity).is_ok() {
        commands.entity(entity).insert(render_layers.clone());
    }

    // Recurse to children
    if let Ok(children) = children_query.get(entity) {
        for child in children.iter() {
            apply_layers_recursive(
                child,
                children_query,
                entities_without_layer,
                render_layers,
                commands,
            );
        }
    }
}

/// Set the viewport for the ViewCube camera (positions it in top-right corner)
/// Simple version: positions based on window size only
pub fn set_view_cube_viewport(
    config: Res<ViewCubeConfig>,
    windows: Query<&Window>,
    mut camera_query: Query<&mut Camera, With<ViewCubeCamera>>,
) {
    if !config.use_overlay || config.follow_main_viewport {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };

    let scale_factor = window.scale_factor();
    let margin = config.overlay_margin * scale_factor;
    let size = (config.overlay_size as f32 * scale_factor) as u32;

    let window_width = window.physical_width();

    // Position in top-right corner
    let pos_x = window_width
        .saturating_sub(size)
        .saturating_sub(margin as u32);
    let pos_y = margin as u32;

    for mut camera in camera_query.iter_mut() {
        camera.viewport = Some(Viewport {
            physical_position: UVec2::new(pos_x, pos_y),
            physical_size: UVec2::new(size, size),
            depth: 0.0..1.0,
        });
    }
}

/// Set the viewport for the ViewCube camera relative to the main camera's viewport.
/// Editor version: respects split views and positions gizmo in each viewport's corner.
pub fn set_view_cube_viewport_editor(
    config: Res<ViewCubeConfig>,
    windows: Query<&Window>,
    mut view_cube_camera_query: Query<(&mut Camera, &ViewCubeLink), With<ViewCubeCamera>>,
    main_camera_query: Query<&Camera, Without<ViewCubeCamera>>,
) {
    if !config.use_overlay || !config.follow_main_viewport {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };

    let scale_factor = window.scale_factor();
    let margin = config.overlay_margin * scale_factor;
    let side_length = config.overlay_size as f32 * scale_factor;

    for (mut view_cube_camera, link) in view_cube_camera_query.iter_mut() {
        let Ok(main_camera) = main_camera_query.get(link.main_camera) else {
            continue;
        };

        // Get main camera's viewport, or use full window if none
        let (viewport_pos, viewport_size) = if let Some(viewport) = &main_camera.viewport {
            (
                viewport.physical_position.as_vec2(),
                viewport.physical_size.as_vec2(),
            )
        } else {
            (
                Vec2::ZERO,
                Vec2::new(
                    window.physical_width() as f32,
                    window.physical_height() as f32,
                ),
            )
        };

        // Position ViewCube in top-right corner of main camera's viewport
        let nav_viewport_pos = Vec2::new(
            (viewport_pos.x + viewport_size.x) - (side_length + margin),
            viewport_pos.y + margin,
        );

        // Clamp to window bounds
        let window_size = window.physical_size();
        let pos_x = nav_viewport_pos.x.max(0.0) as u32;
        let pos_y = nav_viewport_pos.y.max(0.0) as u32;
        let max_w = window_size.x.saturating_sub(pos_x);
        let max_h = window_size.y.saturating_sub(pos_y);

        let (physical_size, is_active) = if main_camera.is_active && max_w > 0 && max_h > 0 {
            (
                UVec2::new(
                    side_length.min(max_w as f32) as u32,
                    side_length.min(max_h as f32) as u32,
                ),
                true,
            )
        } else {
            (UVec2::new(1, 1), false)
        };

        view_cube_camera.is_active = is_active;
        view_cube_camera.viewport = Some(Viewport {
            physical_position: UVec2::new(pos_x, pos_y),
            physical_size,
            depth: 0.0..1.0,
        });
    }
}

// ============================================================================
// LookToTrigger Integration (Editor Mode)
// ============================================================================

/// Handle ViewCube events using LookToTrigger (editor mode).
/// This integrates with bevy_editor_cam for smoother camera control.
pub fn handle_view_cube_look_to(
    mut events: MessageReader<ViewCubeEvent>,
    camera_query: Query<(Entity, &Transform, &EditorCam), With<ViewCubeTargetCamera>>,
    mut look_to: MessageWriter<LookToTrigger>,
) {
    for event in events.read() {
        let direction = match event {
            ViewCubeEvent::FaceClicked(dir) => Some(dir.to_look_direction()),
            ViewCubeEvent::EdgeClicked(dir) => Some(dir.to_look_direction()),
            ViewCubeEvent::CornerClicked(pos) => Some(pos.to_look_direction()),
            ViewCubeEvent::ArrowClicked(_) => None, // Arrows don't use LookToTrigger
        };

        if let Some(look_dir) = direction {
            if let Ok((entity, transform, editor_cam)) = camera_query.single() {
                // Convert Vec3 to Dir3
                if let Ok(dir) = Dir3::new(look_dir) {
                    look_to.write(LookToTrigger::auto_snap_up_direction(
                        dir, entity, transform, editor_cam,
                    ));
                }
            }
        }
    }
}

/// Handle arrow rotations in editor mode using EditorCam orbit.
/// This ensures rotation is around the camera's anchor point.
pub fn handle_view_cube_arrows_editor(
    mut events: MessageReader<ViewCubeEvent>,
    mut camera_query: Query<(&mut Transform, &mut EditorCam), With<ViewCubeTargetCamera>>,
    config: Res<ViewCubeConfig>,
) {
    for event in events.read() {
        if let ViewCubeEvent::ArrowClicked(arrow) = event {
            if let Ok((mut transform, mut editor_cam)) = camera_query.single_mut() {
                // Handle roll separately (EditorCam orbit doesn't support roll)
                if matches!(arrow, RotationArrow::RollLeft | RotationArrow::RollRight) {
                    let roll_angle = if *arrow == RotationArrow::RollLeft {
                        config.rotation_increment
                    } else {
                        -config.rotation_increment
                    };
                    let forward = transform.forward();
                    let roll_rotation = Quat::from_axis_angle(*forward, roll_angle);
                    transform.rotation = roll_rotation * transform.rotation;
                    continue;
                }

                // Get anchor point from camera transform
                let anchor =
                    crate::plugins::camera_anchor::camera_anchor_from_transform(&transform);

                // Calculate screenspace delta based on arrow direction
                // Using a base delta scaled by rotation_increment
                let base_delta = 50.0 * (config.rotation_increment / 0.1);
                let delta = match arrow {
                    RotationArrow::Left => Vec2::new(base_delta, 0.0),
                    RotationArrow::Right => Vec2::new(-base_delta, 0.0),
                    RotationArrow::Up => Vec2::new(0.0, -base_delta),
                    RotationArrow::Down => Vec2::new(0.0, base_delta),
                    _ => continue,
                };

                // Use EditorCam orbit for rotation - single shot
                editor_cam.end_move();
                editor_cam.start_orbit(anchor);
                editor_cam.send_screenspace_input(delta);
                editor_cam.end_move(); // Stop immediately after input
            }
        }
    }
}
