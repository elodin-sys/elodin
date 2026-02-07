//! Camera control systems for the ViewCube plugin.
//!
//! Two modes:
//! - **Standalone** (`auto_rotate`): direct camera animation via `CameraAnimation`.
//! - **Editor** (`use_look_to_trigger`): orbit around the subject using `OrbitState`.
//!
//! The editor mode uses `EditorCam.last_anchor_depth` (updated by mouse interactions)
//! to determine the distance to the subject, with `GlobalTransform` as fallback.

use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy_editor_cam::controller::component::EditorCam;

use super::components::{
    RotationArrow, ViewCubeCamera, ViewCubeLink, ViewCubeRenderLayer, ViewCubeRoot,
};
use super::config::ViewCubeConfig;
use super::events::ViewCubeEvent;

// ============================================================================
// Components
// ============================================================================

/// Marker component for the camera that should be controlled by the ViewCube.
#[derive(Component)]
pub struct ViewCubeTargetCamera;

// ============================================================================
// OrbitState - Central geometry helper
// ============================================================================

/// Captures the camera's orbit geometry relative to a subject.
/// All ViewCube camera operations go through this struct.
struct OrbitState {
    /// The point the camera orbits around (the subject's position).
    pivot: Vec3,
    /// Distance from the camera to the pivot.
    distance: f32,
}

impl OrbitState {
    /// Compute the orbit state from the camera's current transform.
    ///
    /// Uses `EditorCam.last_anchor_depth` (most accurate, updated by mouse interactions)
    /// with `GlobalTransform.translation().length()` as fallback (big_space compatible).
    fn from_camera(
        transform: &Transform,
        global_transform: &GlobalTransform,
        editor_cam: Option<&EditorCam>,
    ) -> Self {
        let distance = editor_cam
            .and_then(|ec| {
                let d = ec.last_anchor_depth.abs() as f32;
                if d > 0.5 { Some(d) } else { None }
            })
            .unwrap_or_else(|| global_transform.translation().length().max(1.0));

        // The pivot (subject) is at `distance` along the camera's forward ray
        let pivot = transform.translation + *transform.forward() * distance;

        Self { pivot, distance }
    }

    /// Snap the camera to view from a specific direction (face/edge/corner click).
    /// Sets camera rotation to look along `-look_dir` (perpendicular to the face),
    /// then repositions at `distance` from the pivot along `look_dir`.
    fn snap_to_direction(&self, transform: &mut Transform, look_dir: Vec3) {
        let up = stable_up_vector(look_dir, Vec3::ZERO);
        let target_pos = self.pivot + look_dir * self.distance;
        *transform = Transform::from_translation(target_pos).looking_at(self.pivot, up);
    }

    /// Orbit the camera around the pivot by the given rotation (arrow click).
    /// First re-centers on the pivot (corrects any prior pan), then orbits.
    fn orbit_by(&self, transform: &mut Transform, rotation: Quat) {
        // Re-center: ensure camera looks exactly at the pivot
        let up = stable_up_vector(transform.translation, self.pivot);
        transform.look_at(self.pivot, up);

        // Orbit: rotate position around pivot, then look at it again
        let offset = transform.translation - self.pivot;
        let new_offset = rotation * offset;
        transform.translation = self.pivot + new_offset;
        let up = stable_up_vector(transform.translation, self.pivot);
        transform.look_at(self.pivot, up);
    }

    /// Roll the camera around its forward axis (no pivot change).
    fn roll(&self, transform: &mut Transform, angle: f32) {
        let forward = transform.forward();
        transform.rotation = Quat::from_axis_angle(*forward, angle) * transform.rotation;
    }
}

// ============================================================================
// Resources
// ============================================================================

/// Tracks camera animation state (standalone mode only).
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
            ViewCubeEvent::FaceClicked { direction, .. } => {
                let look_dir = direction.to_look_direction();
                if let Ok(transform) = camera_query.single() {
                    start_camera_animation(look_dir, transform, &mut camera_anim, &config);
                }
            }
            ViewCubeEvent::EdgeClicked { direction, .. } => {
                let look_dir = direction.to_look_direction();
                if let Ok(transform) = camera_query.single() {
                    start_camera_animation(look_dir, transform, &mut camera_anim, &config);
                }
            }
            ViewCubeEvent::CornerClicked { position, .. } => {
                let look_dir = position.to_look_direction();
                if let Ok(transform) = camera_query.single() {
                    start_camera_animation(look_dir, transform, &mut camera_anim, &config);
                }
            }
            ViewCubeEvent::ArrowClicked { arrow, .. } => {
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

/// Compute a stable up vector for `look_at` that prevents camera flipping.
/// When the camera is nearly directly above or below the pivot,
/// uses a horizontal axis instead of Vec3::Y to avoid gimbal lock and 180-degree flips.
fn stable_up_vector(camera_pos: Vec3, pivot: Vec3) -> Vec3 {
    let dir = (camera_pos - pivot).normalize();
    if dir.y.abs() > 0.95 {
        // Near the poles: use Z as up to avoid flip
        if dir.y > 0.0 { Vec3::NEG_Z } else { Vec3::Z }
    } else {
        Vec3::Y
    }
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
// Editor Mode - Single system handling all ViewCube events
// ============================================================================

/// Handle ALL ViewCube events in editor mode (faces, edges, corners, arrows).
/// Must be a single system because Bevy Messages are consumed by the first reader.
/// Uses `OrbitState` for all camera geometry calculations.
pub fn handle_view_cube_editor(
    mut events: MessageReader<ViewCubeEvent>,
    view_cube_query: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    mut camera_query: Query<
        (&mut Transform, &GlobalTransform, Option<&EditorCam>),
        With<ViewCubeTargetCamera>,
    >,
    config: Res<ViewCubeConfig>,
) {
    for event in events.read() {
        // Find the main camera entity (source-based lookup with fallback)
        let cam_entity = match event {
            ViewCubeEvent::FaceClicked { source, .. }
            | ViewCubeEvent::EdgeClicked { source, .. }
            | ViewCubeEvent::CornerClicked { source, .. } => view_cube_query
                .get(*source)
                .or_else(|_| view_cube_query.iter().next().ok_or(()))
                .ok()
                .map(|l| l.main_camera),
            ViewCubeEvent::ArrowClicked { .. } => {
                view_cube_query.iter().next().map(|l| l.main_camera)
            }
        };

        let Some(cam) = cam_entity else {
            continue;
        };
        let Ok((mut transform, global_transform, editor_cam)) = camera_query.get_mut(cam) else {
            continue;
        };

        let orbit = OrbitState::from_camera(&transform, global_transform, editor_cam);

        match event {
            ViewCubeEvent::FaceClicked { direction, .. } => {
                orbit.snap_to_direction(&mut transform, direction.to_look_direction());
            }
            ViewCubeEvent::EdgeClicked { direction, .. } => {
                orbit.snap_to_direction(&mut transform, direction.to_look_direction());
            }
            ViewCubeEvent::CornerClicked { position, .. } => {
                orbit.snap_to_direction(&mut transform, position.to_look_direction());
            }
            ViewCubeEvent::ArrowClicked { arrow, .. } => {
                // Fixed angle (15° default). Subject stays at same position and depth.
                let angle = config.rotation_increment;
                let rotation = match arrow {
                    RotationArrow::Left => Quat::from_rotation_y(angle),
                    RotationArrow::Right => Quat::from_rotation_y(-angle),
                    RotationArrow::Up => {
                        let right = transform.right();
                        Quat::from_axis_angle(*right, angle)
                    }
                    RotationArrow::Down => {
                        let right = transform.right();
                        Quat::from_axis_angle(*right, -angle)
                    }
                    RotationArrow::RollLeft => {
                        orbit.roll(&mut transform, angle);
                        continue;
                    }
                    RotationArrow::RollRight => {
                        orbit.roll(&mut transform, -angle);
                        continue;
                    }
                };
                orbit.orbit_by(&mut transform, rotation);
            }
        }
    }
}
