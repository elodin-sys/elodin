//! Camera control systems for the ViewCube plugin.
//!
//! Two modes:
//! - **Standalone** (`auto_rotate`): direct camera animation via `CameraAnimation`.
//! - **Editor** (`use_look_to_trigger`): sends `LookToTrigger` messages to bevy_editor_cam,
//!   same approach as navigation_gizmo. This ensures proper integration with the editor's
//!   camera system instead of fighting it with direct Transform modifications.

use bevy::camera::Viewport;
use bevy::camera::visibility::RenderLayers;
use bevy::ecs::system::SystemParam;
use bevy::log::{debug, info, warn};
use bevy::math::Dir3;
use bevy::prelude::*;
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::controller::motion::CurrentMotion;
use bevy_editor_cam::extensions::look_to::LookToTrigger;
use impeller2_bevy::EntityMap;
use impeller2_wkt::ComponentValue;
use std::collections::HashMap;

use super::components::{
    FaceDirection, RotationArrow, ViewCubeCamera, ViewCubeLink, ViewCubeRenderLayer, ViewCubeRoot,
};
use super::config::ViewCubeConfig;
use super::events::ViewCubeEvent;
use crate::WorldPosExt;
use crate::object_3d::ComponentArrayExt;

// ============================================================================
// Components
// ============================================================================

/// Marker component for the camera that should be controlled by the ViewCube.
#[derive(Component)]
pub struct ViewCubeTargetCamera;

/// Marker added to new cameras to trigger a one-shot face-on snap via LookToTrigger.
/// Removed after the snap is sent.
#[derive(Component)]
pub struct NeedsInitialSnap;

/// Sends a LookToTrigger on newly spawned cameras to ensure they start face-on (looking along -Z).
/// This is the only reliable way to set the initial orientation because multiple systems
/// (setup_cell, sync_pos, EditorCam, big_space) all modify the camera transform at startup.
pub fn snap_initial_camera(
    mut commands: Commands,
    cameras: Query<(Entity, &Transform, &EditorCam), With<NeedsInitialSnap>>,
    mut look_to: MessageWriter<LookToTrigger>,
) {
    for (entity, transform, editor_cam) in cameras.iter() {
        // Snap to -Z (North face visible, face-on)
        if let Ok(direction) = Dir3::new(Vec3::NEG_Z) {
            look_to.write(LookToTrigger::auto_snap_up_direction(
                direction, entity, transform, editor_cam,
            ));
        }
        commands.entity(entity).remove::<NeedsInitialSnap>();
    }
}

// ============================================================================
// Resources
// ============================================================================

/// Tracks camera animation state.
#[derive(Resource, Default)]
pub struct CameraAnimation {
    pub animating: bool,
    pub progress: f32,
    pub target_entity: Option<Entity>,
    pub start_position: Vec3,
    pub start_rotation: Quat,
    pub target_position: Vec3,
    pub target_rotation: Quat,
}

#[derive(Clone, Copy, Debug)]
struct ArrowTargetState {
    target_rotation: Quat,
    valid_until_secs: f64,
}

/// Caches target camera orientation for incremental arrow clicks.
///
/// This prevents drift when users click arrows rapidly while a previous
/// LookTo animation is still in flight.
#[derive(Resource, Default)]
pub struct ViewCubeArrowTargetCache {
    entries: HashMap<Entity, ArrowTargetState>,
}

impl ViewCubeArrowTargetCache {
    const TTL_SECS: f64 = 0.55;

    fn prune(&mut self, now_secs: f64) {
        self.entries
            .retain(|_, state| state.valid_until_secs + 0.25 >= now_secs);
    }

    fn get_valid_target(&self, camera: Entity, now_secs: f64) -> Option<Quat> {
        self.entries
            .get(&camera)
            .filter(|state| state.valid_until_secs >= now_secs)
            .map(|state| state.target_rotation)
    }

    fn set_target(&mut self, camera: Entity, target_rotation: Quat, now_secs: f64) {
        self.entries.insert(
            camera,
            ArrowTargetState {
                target_rotation,
                valid_until_secs: now_secs + Self::TTL_SECS,
            },
        );
    }
}

// ============================================================================
// Systems
// ============================================================================

/// Handle ViewCube events and rotate the camera accordingly.
/// Uses source-based camera lookup to support multiple viewports.
pub fn handle_view_cube_camera(
    mut events: MessageReader<ViewCubeEvent>,
    view_cube_query: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    mut camera_query: Query<&mut Transform, With<ViewCubeTargetCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    config: Res<ViewCubeConfig>,
) {
    for event in events.read() {
        let Some(cam) = main_camera_for_event(event, &view_cube_query) else {
            continue;
        };

        if let Some(look_dir) = target_look_direction(event) {
            if let Ok(transform) = camera_query.get(cam) {
                start_camera_animation(look_dir, transform, &mut camera_anim, &config, cam);
            }
            continue;
        }

        if let ViewCubeEvent::ArrowClicked { arrow, .. } = event
            && let Ok(mut transform) = camera_query.get_mut(cam)
        {
            apply_arrow_rotation(*arrow, &config, &mut transform);
        }
    }
}

/// Animate the camera smoothly to its target.
/// Applies to the camera that was targeted by the last animation start.
pub fn animate_camera(
    mut camera_query: Query<&mut Transform, With<ViewCubeTargetCamera>>,
    mut camera_anim: ResMut<CameraAnimation>,
    time: Res<Time>,
) {
    if !camera_anim.animating {
        return;
    }

    camera_anim.progress += time.delta_secs() * 3.0; // Animation speed

    let Some(target_entity) = camera_anim.target_entity else {
        return;
    };
    let Ok(mut transform) = camera_query.get_mut(target_entity) else {
        return;
    };

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
    camera_entity: Entity,
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

    camera_anim.target_entity = Some(camera_entity);
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

fn main_camera_for_event(
    event: &ViewCubeEvent,
    view_cube_query: &Query<&ViewCubeLink, With<ViewCubeRoot>>,
) -> Option<Entity> {
    let source = match event {
        ViewCubeEvent::FaceClicked { source, .. }
        | ViewCubeEvent::EdgeClicked { source, .. }
        | ViewCubeEvent::CornerClicked { source, .. }
        | ViewCubeEvent::ArrowClicked { source, .. } => *source,
    };

    view_cube_query
        .get(source)
        .or_else(|_| view_cube_query.iter().next().ok_or(()))
        .ok()
        .map(|link| link.main_camera)
}

fn target_look_direction(event: &ViewCubeEvent) -> Option<Vec3> {
    match event {
        ViewCubeEvent::FaceClicked { direction, .. } => Some(direction.to_look_direction()),
        ViewCubeEvent::EdgeClicked { direction, .. } => Some(direction.to_look_direction()),
        ViewCubeEvent::CornerClicked { position, .. } => Some(position.to_look_direction()),
        ViewCubeEvent::ArrowClicked { .. } => None,
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
///
/// Uses `LookToTrigger` from bevy_editor_cam (same approach as navigation_gizmo)
/// instead of modifying the Transform directly. Direct Transform changes are
/// overridden by bevy_editor_cam's internal state management.
#[derive(SystemParam)]
pub(super) struct ViewCubeEditorLookup<'w, 's> {
    viewports: Query<
        'w,
        's,
        &'static crate::ui::inspector::viewport::Viewport,
        With<ViewCubeTargetCamera>,
    >,
    entity_map: Res<'w, EntityMap>,
    values: Query<'w, 's, &'static ComponentValue>,
    time: Res<'w, Time>,
    arrow_cache: ResMut<'w, ViewCubeArrowTargetCache>,
}

pub fn handle_view_cube_editor(
    mut events: MessageReader<ViewCubeEvent>,
    view_cube_query: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    mut camera_query: Query<
        (Entity, &Transform, &GlobalTransform, &mut EditorCam),
        With<ViewCubeTargetCamera>,
    >,
    mut lookup: ViewCubeEditorLookup,
    config: Res<ViewCubeConfig>,
    mut look_to: MessageWriter<LookToTrigger>,
) {
    for event in events.read() {
        let now_secs = lookup.time.elapsed_secs_f64();
        lookup.arrow_cache.prune(now_secs);

        let Some(cam) = main_camera_for_event(event, &view_cube_query) else {
            continue;
        };
        let Ok((entity, transform, global_transform, mut editor_cam)) = camera_query.get_mut(cam)
        else {
            continue;
        };

        update_anchor_depth_for_view_cube(
            entity,
            transform,
            global_transform,
            &mut editor_cam,
            &lookup.viewports,
            lookup.entity_map.as_ref(),
            &lookup.values,
        );

        debug!(
            event = ?event,
            camera = %entity,
            "view cube: received event"
        );

        // Cancel any EditorCam motion started by this click.
        // Left-click on the ViewCube overlay also triggers EditorCam's PanZoom
        // (default_camera_inputs sees MouseButton::Left), so we must stop it
        // before sending our LookToTrigger.
        let motion_before = editor_cam.current_motion.clone();
        let anchor_depth_before = editor_cam.last_anchor_depth;
        editor_cam.end_move();
        editor_cam.current_motion = CurrentMotion::Stationary;
        debug!(
            camera = %entity,
            motion_before = ?motion_before,
            anchor_depth_before = anchor_depth_before,
            "view cube: cleared editor motion before snap"
        );

        // Camera local rotation is relative to its parent. ViewCube directions are expressed in
        // world axes, so convert world targets into camera-local directions before snapping.
        let (_, global_rotation, _) = global_transform.to_scale_rotation_translation();
        let parent_rotation = global_rotation * transform.rotation.inverse();
        let camera_dir_local = -(*transform.forward());
        let camera_dir_global = parent_rotation * camera_dir_local;

        if let ViewCubeEvent::FaceClicked { direction, .. } = event {
            let raw_look_dir_world = direction.to_look_direction();
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let (
                    chosen_up,
                    chosen_up_source,
                    chosen_up_angle,
                    chosen_up_runner_up_angle,
                    chosen_up_margin,
                    up_candidates_considered,
                ) = choose_min_rotation_up(transform, parent_rotation, facing_local);
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local,
                    target_up_direction: chosen_up,
                    camera: entity,
                };
                let rotation_angle = angle_to_trigger(transform, &trigger);
                debug!(
                    target_kind = "face",
                    selection_policy = "face_world_to_local_min_total_rotation",
                    direction = ?direction,
                    camera_dir_local = ?camera_dir_local,
                    camera_dir_global = ?camera_dir_global,
                    raw_look_dir_world = ?raw_look_dir_world,
                    facing_world = ?facing_world,
                    facing_local = ?facing_local_vec,
                    chosen_up_source = chosen_up_source,
                    chosen_up_angle = chosen_up_angle,
                    chosen_up_runner_up_angle = chosen_up_runner_up_angle,
                    chosen_up_margin = chosen_up_margin,
                    up_candidates_considered = up_candidates_considered,
                    chosen_facing = ?*trigger.target_facing_direction,
                    chosen_up = ?*trigger.target_up_direction,
                    rotation_angle = rotation_angle,
                    "view cube: face snap"
                );
                lookup
                    .arrow_cache
                    .set_target(entity, trigger_rotation(&trigger), now_secs);
                look_to.write(trigger);
            } else {
                warn!(
                    direction = ?direction,
                    raw_look_dir_world = ?raw_look_dir_world,
                    facing_world = ?facing_world,
                    facing_local = ?facing_local_vec,
                    "view cube: invalid face snap directions"
                );
            }
            continue;
        }

        if let ViewCubeEvent::CornerClicked { position, .. } = event {
            let raw_look_dir_world = position.to_look_direction();
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let (
                    chosen_up,
                    chosen_up_source,
                    chosen_up_angle,
                    chosen_up_runner_up_angle,
                    chosen_up_margin,
                    up_candidates_considered,
                ) = choose_min_rotation_up(transform, parent_rotation, facing_local);
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local,
                    target_up_direction: chosen_up,
                    camera: entity,
                };
                let rotation_angle = angle_to_trigger(transform, &trigger);
                debug!(
                    target_kind = "corner",
                    selection_policy = "corner_world_to_local_min_total_rotation",
                    position = ?position,
                    camera_dir_local = ?camera_dir_local,
                    camera_dir_global = ?camera_dir_global,
                    raw_look_dir_world = ?raw_look_dir_world,
                    facing_world = ?facing_world,
                    facing_local = ?facing_local_vec,
                    chosen_up_source = chosen_up_source,
                    chosen_up_angle = chosen_up_angle,
                    chosen_up_runner_up_angle = chosen_up_runner_up_angle,
                    chosen_up_margin = chosen_up_margin,
                    up_candidates_considered = up_candidates_considered,
                    chosen_facing = ?*trigger.target_facing_direction,
                    chosen_up = ?*trigger.target_up_direction,
                    rotation_angle = rotation_angle,
                    "view cube: corner snap"
                );
                lookup
                    .arrow_cache
                    .set_target(entity, trigger_rotation(&trigger), now_secs);
                look_to.write(trigger);
            } else {
                warn!(
                    position = ?position,
                    raw_look_dir_world = ?raw_look_dir_world,
                    facing_world = ?facing_world,
                    facing_local = ?facing_local_vec,
                    "view cube: invalid corner snap directions"
                );
            }
            continue;
        }

        // Edges (the frame around a face): determine which of the two adjacent faces is
        // currently "in front" relative to the camera, then snap to its opposite face.
        if let ViewCubeEvent::EdgeClicked { direction, .. } = event {
            let (face_a, face_b) = direction.adjacent_faces();
            let dot_a = face_a.to_look_direction().dot(camera_dir_global);
            let dot_b = face_b.to_look_direction().dot(camera_dir_global);

            let candidate_a =
                build_edge_snap_candidate(transform, parent_rotation, face_a, dot_a, face_b, dot_b);
            let candidate_b =
                build_edge_snap_candidate(transform, parent_rotation, face_b, dot_b, face_a, dot_a);

            let chosen = match (&candidate_a, &candidate_b) {
                (Some(a), Some(b)) => {
                    if a.rotation_angle <= b.rotation_angle + 1.0e-6 {
                        Some(a)
                    } else {
                        Some(b)
                    }
                }
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            if let Some(chosen) = chosen {
                let trigger = LookToTrigger {
                    target_facing_direction: chosen.facing_local,
                    target_up_direction: chosen.chosen_up,
                    camera: entity,
                };
                debug!(
                    target_kind = "edge",
                    selection_policy = "edge_to_opposite_face_min_total_rotation",
                    edge_direction = ?direction,
                    camera_dir_local = ?camera_dir_local,
                    camera_dir_global = ?camera_dir_global,
                    candidate_a_frame_face = ?candidate_a.as_ref().map(|c| c.frame_face),
                    candidate_a_target_face = ?candidate_a.as_ref().map(|c| c.target_face),
                    candidate_a_rotation_angle = ?candidate_a.as_ref().map(|c| c.rotation_angle),
                    candidate_b_frame_face = ?candidate_b.as_ref().map(|c| c.frame_face),
                    candidate_b_target_face = ?candidate_b.as_ref().map(|c| c.target_face),
                    candidate_b_rotation_angle = ?candidate_b.as_ref().map(|c| c.rotation_angle),
                    frame_face = ?chosen.frame_face,
                    frame_face_dot = chosen.frame_face_dot,
                    secondary_face = ?chosen.secondary_face,
                    secondary_face_dot = chosen.secondary_face_dot,
                    target_face = ?chosen.target_face,
                    raw_look_dir_world = ?chosen.raw_look_dir_world,
                    facing_world = ?chosen.facing_world,
                    facing_local = ?chosen.facing_local_vec,
                    chosen_up_source = chosen.chosen_up_source,
                    chosen_up_angle = chosen.chosen_up_angle,
                    chosen_up_runner_up_angle = chosen.chosen_up_runner_up_angle,
                    chosen_up_margin = chosen.chosen_up_margin,
                    up_candidates_considered = chosen.up_candidates_considered,
                    chosen_facing = ?*trigger.target_facing_direction,
                    chosen_up = ?*trigger.target_up_direction,
                    rotation_angle = chosen.rotation_angle,
                    "view cube: edge snap"
                );
                lookup
                    .arrow_cache
                    .set_target(entity, trigger_rotation(&trigger), now_secs);
                look_to.write(trigger);
            } else {
                warn!(
                    edge_direction = ?direction,
                    camera_dir_global = ?camera_dir_global,
                    face_a = ?face_a,
                    face_b = ?face_b,
                    dot_a = dot_a,
                    dot_b = dot_b,
                    "view cube: invalid edge snap candidates"
                );
            }
            continue;
        }

        // Arrows: compute the new facing direction after rotating by the increment,
        // then send as LookToTrigger so bevy_editor_cam handles it properly.
        if let ViewCubeEvent::ArrowClicked { arrow, .. } = event {
            let angle = config.rotation_increment;
            let (base_rotation, base_rotation_source) = lookup
                .arrow_cache
                .get_valid_target(entity, now_secs)
                .map(|cached| (cached, "cached_target"))
                .unwrap_or((transform.rotation, "current_transform"));
            let base_forward_local = base_rotation * Vec3::NEG_Z;
            let base_up_local = base_rotation * Vec3::Y;
            let base_right_local = base_rotation * Vec3::X;
            let base_forward_world = parent_rotation * base_forward_local;
            let base_up_world = parent_rotation * base_up_local;
            let base_right_world = parent_rotation * base_right_local;

            let (step_axis_world, signed_angle, step_axis_source) =
                arrow_world_axis_angle(*arrow, angle);
            let step_rotation_world = Quat::from_axis_angle(*step_axis_world, signed_angle);
            let new_forward_world = step_rotation_world * base_forward_world;
            let new_up_world = step_rotation_world * base_up_world;
            let new_forward_local = parent_rotation.inverse() * new_forward_world;
            let new_up_local = parent_rotation.inverse() * new_up_world;

            let axis_dot_right = step_axis_world.dot(base_right_world);
            let axis_dot_up = step_axis_world.dot(base_up_world);
            let axis_dot_forward = step_axis_world.dot(base_forward_world);
            info!(
                arrow = ?arrow,
                camera = %entity,
                now_secs = now_secs,
                base_rotation_source = base_rotation_source,
                step_axis_source = step_axis_source,
                step_axis_world = ?*step_axis_world,
                step_angle_deg = signed_angle.to_degrees(),
                base_forward_world = ?base_forward_world,
                base_up_world = ?base_up_world,
                base_right_world = ?base_right_world,
                base_forward_world_major = dominant_world_axis_label(base_forward_world),
                base_up_world_major = dominant_world_axis_label(base_up_world),
                base_right_world_major = dominant_world_axis_label(base_right_world),
                axis_dot_right = axis_dot_right,
                axis_dot_up = axis_dot_up,
                axis_dot_forward = axis_dot_forward,
                "view cube: arrow frame probe"
            );

            if let Ok(facing) = Dir3::new(new_forward_local)
                && let Ok(up_dir) = Dir3::new(new_up_local)
            {
                let trigger = LookToTrigger {
                    target_facing_direction: facing,
                    target_up_direction: up_dir,
                    camera: entity,
                };
                let target_rotation = trigger_rotation(&trigger);
                debug!(
                    arrow = ?arrow,
                    base_rotation_source = base_rotation_source,
                    base_rotation = ?base_rotation,
                    parent_rotation = ?parent_rotation,
                    base_forward_local = ?base_forward_local,
                    base_up_local = ?base_up_local,
                    base_right_local = ?base_right_local,
                    base_forward_world = ?base_forward_world,
                    base_up_world = ?base_up_world,
                    base_right_world = ?base_right_world,
                    step_axis_source = step_axis_source,
                    step_axis_world = ?*step_axis_world,
                    step_angle = signed_angle,
                    step_rotation_world = ?step_rotation_world,
                    new_forward_world = ?new_forward_world,
                    new_up_world = ?new_up_world,
                    new_forward_local = ?new_forward_local,
                    new_up_local = ?new_up_local,
                    rotation_angle = angle_to_trigger(transform, &trigger),
                    "view cube: arrow snap"
                );
                lookup
                    .arrow_cache
                    .set_target(entity, target_rotation, now_secs);
                look_to.write(trigger);
                info!(
                    arrow = ?arrow,
                    camera = %entity,
                    new_forward_world = ?new_forward_world,
                    new_up_world = ?new_up_world,
                    new_forward_world_major = dominant_world_axis_label(new_forward_world),
                    new_up_world_major = dominant_world_axis_label(new_up_world),
                    "view cube: arrow frame result"
                );
            } else {
                warn!(
                    arrow = ?arrow,
                    new_forward_local = ?new_forward_local,
                    new_up_local = ?new_up_local,
                    "view cube: invalid arrow directions"
                );
            }
        }
    }
}

fn angle_to_trigger(transform: &Transform, trigger: &LookToTrigger) -> f32 {
    angle_to_target_rotation(
        transform,
        trigger.target_facing_direction,
        trigger.target_up_direction,
    )
}

fn trigger_rotation(trigger: &LookToTrigger) -> Quat {
    Transform::default()
        .looking_to(
            *trigger.target_facing_direction,
            *trigger.target_up_direction,
        )
        .rotation
}

fn dominant_world_axis_label(vec: Vec3) -> &'static str {
    let abs = vec.abs();
    if abs.x >= abs.y && abs.x >= abs.z {
        if vec.x >= 0.0 { "+X" } else { "-X" }
    } else if abs.y >= abs.x && abs.y >= abs.z {
        if vec.y >= 0.0 { "+Y" } else { "-Y" }
    } else if vec.z >= 0.0 {
        "+Z"
    } else {
        "-Z"
    }
}

fn arrow_world_axis_angle(arrow: RotationArrow, angle: f32) -> (Dir3, f32, &'static str) {
    match arrow {
        RotationArrow::Left => (Dir3::new_unchecked(Vec3::Y), angle, "world_pos_y"),
        RotationArrow::Right => (Dir3::new_unchecked(Vec3::Y), -angle, "world_pos_y"),
        RotationArrow::Up => (Dir3::new_unchecked(Vec3::X), angle, "world_pos_x"),
        RotationArrow::Down => (Dir3::new_unchecked(Vec3::X), -angle, "world_pos_x"),
        RotationArrow::RollLeft => (Dir3::new_unchecked(Vec3::Z), angle, "world_pos_z"),
        RotationArrow::RollRight => (Dir3::new_unchecked(Vec3::Z), -angle, "world_pos_z"),
    }
}

fn angle_to_target_rotation(transform: &Transform, facing: Dir3, up: Dir3) -> f32 {
    let target_rotation = Transform::default().looking_to(*facing, *up).rotation;
    transform.rotation.angle_between(target_rotation).abs()
}

#[derive(Debug)]
struct EdgeSnapCandidate {
    frame_face: FaceDirection,
    frame_face_dot: f32,
    secondary_face: FaceDirection,
    secondary_face_dot: f32,
    target_face: FaceDirection,
    raw_look_dir_world: Vec3,
    facing_world: Vec3,
    facing_local_vec: Vec3,
    facing_local: Dir3,
    chosen_up: Dir3,
    chosen_up_source: &'static str,
    chosen_up_angle: f32,
    chosen_up_runner_up_angle: Option<f32>,
    chosen_up_margin: Option<f32>,
    up_candidates_considered: usize,
    rotation_angle: f32,
}

fn build_edge_snap_candidate(
    transform: &Transform,
    parent_rotation: Quat,
    frame_face: FaceDirection,
    frame_face_dot: f32,
    secondary_face: FaceDirection,
    secondary_face_dot: f32,
) -> Option<EdgeSnapCandidate> {
    let target_face = frame_face.opposite();
    let raw_look_dir_world = target_face.to_look_direction();
    let facing_world = -raw_look_dir_world;
    let facing_local_vec = parent_rotation.inverse() * facing_world;
    let facing_local = Dir3::new(facing_local_vec).ok()?;
    let (
        chosen_up,
        chosen_up_source,
        chosen_up_angle,
        chosen_up_runner_up_angle,
        chosen_up_margin,
        up_candidates_considered,
    ) = choose_min_rotation_up(transform, parent_rotation, facing_local);
    let rotation_angle = angle_to_target_rotation(transform, facing_local, chosen_up);

    Some(EdgeSnapCandidate {
        frame_face,
        frame_face_dot,
        secondary_face,
        secondary_face_dot,
        target_face,
        raw_look_dir_world,
        facing_world,
        facing_local_vec,
        facing_local,
        chosen_up,
        chosen_up_source,
        chosen_up_angle,
        chosen_up_runner_up_angle,
        chosen_up_margin,
        up_candidates_considered,
        rotation_angle,
    })
}

fn choose_min_rotation_up(
    transform: &Transform,
    parent_rotation: Quat,
    facing_local: Dir3,
) -> (Dir3, &'static str, f32, Option<f32>, Option<f32>, usize) {
    let parent_inverse = parent_rotation.inverse();
    let mut best: Option<(Dir3, &'static str, f32)> = None;
    let mut runner_up_angle: Option<f32> = None;
    let mut candidates_considered = 0usize;

    for (label, up_world) in [
        ("world_pos_x", Vec3::X),
        ("world_neg_x", Vec3::NEG_X),
        ("world_pos_y", Vec3::Y),
        ("world_neg_y", Vec3::NEG_Y),
        ("world_pos_z", Vec3::Z),
        ("world_neg_z", Vec3::NEG_Z),
    ] {
        let up_local_vec = parent_inverse * up_world;
        let Ok(up_local) = Dir3::new(up_local_vec) else {
            continue;
        };
        let alignment = facing_local.dot(*up_local).abs();
        if alignment > 0.99 {
            continue;
        }
        candidates_considered += 1;
        let angle = angle_to_target_rotation(transform, facing_local, up_local);
        match best {
            Some((_, _, best_angle)) => {
                if angle + 1.0e-6 < best_angle {
                    runner_up_angle = Some(best_angle);
                    best = Some((up_local, label, angle));
                } else {
                    let should_update_runner = match runner_up_angle {
                        Some(prev) => angle + 1.0e-6 < prev,
                        None => true,
                    };
                    if should_update_runner {
                        runner_up_angle = Some(angle);
                    }
                }
            }
            None => {
                best = Some((up_local, label, angle));
            }
        }
    }

    if let Some((best_up, best_label, best_angle)) = best {
        let margin = runner_up_angle.map(|runner| runner - best_angle);
        return (
            best_up,
            best_label,
            best_angle,
            runner_up_angle,
            margin,
            candidates_considered,
        );
    }

    // Fallback: build an up vector orthogonal to facing in local space.
    let facing = *facing_local;
    let basis = if facing.y.abs() < 0.95 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let orthogonal = (basis - facing * basis.dot(facing)).normalize_or_zero();
    let fallback = if orthogonal.length_squared() > 1.0e-6 {
        orthogonal
    } else {
        Vec3::Z
    };
    let fallback_up = Dir3::new(fallback).unwrap_or(Dir3::new_unchecked(Vec3::Y));
    let fallback_angle = angle_to_target_rotation(transform, facing_local, fallback_up);
    (
        fallback_up,
        "fallback_local_orthogonal",
        fallback_angle,
        None,
        None,
        candidates_considered,
    )
}

fn update_anchor_depth_for_view_cube(
    camera: Entity,
    transform: &Transform,
    global_transform: &GlobalTransform,
    editor_cam: &mut EditorCam,
    viewports: &Query<&crate::ui::inspector::viewport::Viewport, With<ViewCubeTargetCamera>>,
    entity_map: &EntityMap,
    values: &Query<&'static ComponentValue>,
) {
    let Some(orbit_target) = view_cube_orbit_target(camera, viewports, entity_map, values) else {
        debug!(
            camera = %camera,
            previous_anchor_depth = editor_cam.last_anchor_depth,
            "view cube: orbit target unavailable; keeping previous anchor depth"
        );
        return;
    };
    let target_source = "viewport.look_at";

    let world_translation = global_transform.translation();
    let world_rotation = global_transform.rotation();
    let to_target = orbit_target - world_translation;
    let forward = world_rotation * Vec3::NEG_Z;
    // Measure depth along camera forward axis (what EditorCam expects).
    // We only trust projected depth when look_at is close to the camera centerline.
    // Off-axis projected depths collapse toward zero and create unstable pivots.
    let projected_distance = to_target.dot(forward);
    let measured_distance = to_target.length();
    let previous_distance = editor_cam.last_anchor_depth.abs() as f32;
    let alignment_ratio = if measured_distance.is_finite() && measured_distance > 1.0e-3 {
        (projected_distance / measured_distance).clamp(-1.0, 1.0)
    } else {
        f32::NAN
    };

    const MIN_ALIGNMENT_FOR_PROJECTED: f32 = 0.65;
    const MIN_ORBIT_DISTANCE: f32 = 0.25;

    let (mut distance, depth_strategy) = if projected_distance.is_finite()
        && measured_distance.is_finite()
        && projected_distance > 1.0e-3
        && measured_distance > 1.0e-3
        && alignment_ratio >= MIN_ALIGNMENT_FOR_PROJECTED
    {
        (projected_distance, "projected_forward_aligned")
    } else if previous_distance > 1.0e-3 {
        (previous_distance, "fallback_previous_off_axis")
    } else if measured_distance.is_finite() && measured_distance > 1.0e-3 {
        (measured_distance, "fallback_measured")
    } else {
        (1.0, "fallback_default")
    };

    let min_clamped = if distance < MIN_ORBIT_DISTANCE {
        distance = MIN_ORBIT_DISTANCE;
        true
    } else {
        false
    };

    let new_depth = -(distance as f64);

    let old_depth = editor_cam.last_anchor_depth;
    editor_cam.last_anchor_depth = new_depth;
    debug!(
        camera = %camera,
        orbit_target = ?orbit_target,
        target_source = target_source,
        camera_local_translation = ?transform.translation,
        camera_world_translation = ?world_translation,
        camera_world_forward = ?forward,
        projected_distance = projected_distance,
        measured_distance = measured_distance,
        alignment_ratio = alignment_ratio,
        previous_distance = previous_distance,
        depth_strategy = depth_strategy,
        min_clamped = min_clamped,
        orbit_distance = distance,
        old_anchor_depth = old_depth,
        new_anchor_depth = new_depth,
        "view cube: updated orbit anchor depth"
    );
}

fn view_cube_orbit_target(
    camera: Entity,
    viewports: &Query<&crate::ui::inspector::viewport::Viewport, With<ViewCubeTargetCamera>>,
    entity_map: &EntityMap,
    values: &Query<&'static ComponentValue>,
) -> Option<Vec3> {
    let viewport = viewports.get(camera).ok()?;
    let compiled_expr = viewport.look_at.compiled_expr.as_ref()?;
    let val = compiled_expr.execute(entity_map, values).ok()?;
    let world_pos = val.as_world_pos()?;
    Some(world_pos.bevy_pos().as_vec3())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_to_target_rotation_default_is_zero() {
        let transform = Transform::default();
        let facing = Dir3::new(Vec3::NEG_Z).expect("unit vector");
        let up = Dir3::new(Vec3::Y).expect("unit vector");
        let angle = angle_to_target_rotation(&transform, facing, up);
        assert!(
            angle.abs() < 1.0e-6,
            "expected ~0 for identity orientation, got {}",
            angle
        );
    }

    #[test]
    fn angle_to_target_rotation_opposite_forward_is_pi() {
        let transform = Transform::default();
        let facing = Dir3::new(Vec3::Z).expect("unit vector");
        let up = Dir3::new(Vec3::Y).expect("unit vector");
        let angle = angle_to_target_rotation(&transform, facing, up);
        assert!(
            (angle - std::f32::consts::PI).abs() < 1.0e-5,
            "expected PI for opposite forward, got {}",
            angle
        );
    }

    #[test]
    fn choose_min_rotation_up_keeps_up_non_parallel_to_facing() {
        let transform = Transform::default();
        let facing = Dir3::new(Vec3::Y).expect("unit vector");
        let (up, _, _, _, _, candidates) =
            choose_min_rotation_up(&transform, Quat::IDENTITY, facing);
        assert!(candidates > 0, "expected at least one valid up candidate");
        assert!(
            facing.dot(*up).abs() < 0.99,
            "up must not be parallel to facing (dot={})",
            facing.dot(*up)
        );
    }

    #[test]
    fn arrow_target_cache_is_valid_within_ttl() {
        let mut cache = ViewCubeArrowTargetCache::default();
        let entity = Entity::from_bits(42);
        let target = Quat::from_rotation_y(0.4);
        cache.set_target(entity, target, 10.0);
        let cached = cache.get_valid_target(entity, 10.3);
        assert_eq!(cached, Some(target));
    }

    #[test]
    fn arrow_target_cache_expires_after_ttl() {
        let mut cache = ViewCubeArrowTargetCache::default();
        let entity = Entity::from_bits(7);
        let target = Quat::from_rotation_x(0.2);
        cache.set_target(entity, target, 1.0);
        let cached = cache.get_valid_target(entity, 2.0);
        assert_eq!(cached, None);
    }

    #[test]
    fn arrow_world_axis_angle_maps_each_pair_to_fixed_axis() {
        let angle = 0.25;
        let (axis, signed_angle, source) = arrow_world_axis_angle(RotationArrow::Left, angle);
        assert_eq!(*axis, Vec3::Y);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "world_pos_y");

        let (axis, signed_angle, source) = arrow_world_axis_angle(RotationArrow::Right, angle);
        assert_eq!(*axis, Vec3::Y);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "world_pos_y");

        let (axis, signed_angle, source) = arrow_world_axis_angle(RotationArrow::Up, angle);
        assert_eq!(*axis, Vec3::X);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "world_pos_x");

        let (axis, signed_angle, source) = arrow_world_axis_angle(RotationArrow::Down, angle);
        assert_eq!(*axis, Vec3::X);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "world_pos_x");

        let (axis, signed_angle, source) = arrow_world_axis_angle(RotationArrow::RollLeft, angle);
        assert_eq!(*axis, Vec3::Z);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "world_pos_z");

        let (axis, signed_angle, source) = arrow_world_axis_angle(RotationArrow::RollRight, angle);
        assert_eq!(*axis, Vec3::Z);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "world_pos_z");
    }
}
