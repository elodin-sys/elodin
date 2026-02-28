//! Camera systems for the ViewCube plugin.

use bevy::camera::visibility::RenderLayers;
use bevy::ecs::hierarchy::ChildOf;
use bevy::ecs::system::SystemParam;
use bevy::log::{debug, warn};
use bevy::math::Dir3;
use bevy::prelude::*;
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::controller::motion::CurrentMotion;
use bevy_editor_cam::extensions::look_to::LookToTrigger;
use big_space::{FloatingOrigin, FloatingOriginSettings, GridCell};
use impeller2_bevy::EntityMap;
use impeller2_wkt::ComponentValue;
use std::collections::HashMap;

use super::components::{
    AxisLabelBillboard, RotationArrow, ViewCubeCamera, ViewCubeLink, ViewCubeRenderLayer,
    ViewCubeRoot, ViewportActionButton,
};
use super::config::ViewCubeConfig;
use super::events::ViewCubeEvent;
use crate::WorldPosExt;
use crate::object_3d::ComponentArrayExt;

const FACE_IN_SCREEN_PLANE_DOT_THRESHOLD: f32 = 0.999;
const CORNER_IN_SCREEN_AXIS_DOT_THRESHOLD: f32 = 0.998;
const ARROW_CACHE_MAX_DRIFT_RADIANS: f32 = 6.0_f32.to_radians();
const VIEWPORT_RESET_ANCHOR_DEPTH: f64 = -2.0;
const VIEWPORT_ZOOM_OUT_MULTIPLIER: f32 = 2.2;
const VIEWPORT_ZOOM_IN_MULTIPLIER: f32 = 1.2;

#[derive(Component)]
pub struct ViewCubeTargetCamera;

#[derive(Component)]
pub struct NeedsInitialSnap;

pub fn snap_initial_camera(
    mut commands: Commands,
    cameras: Query<(Entity, &Transform, &EditorCam), With<NeedsInitialSnap>>,
    mut look_to: MessageWriter<LookToTrigger>,
) {
    for (entity, transform, editor_cam) in cameras.iter() {
        if let Ok(direction) = Dir3::new(Vec3::NEG_Z) {
            look_to.write(LookToTrigger::auto_snap_up_direction(
                direction, entity, transform, editor_cam,
            ));
        }
        commands.entity(entity).remove::<NeedsInitialSnap>();
    }
}

#[derive(Clone, Copy, Debug)]
struct ArrowTargetState {
    target_rotation: Quat,
    valid_until_secs: f64,
    source: ArrowTargetSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArrowTargetSource {
    ArrowStep,
    ViewSnap,
}

#[derive(Resource, Default)]
pub struct ViewCubeArrowTargetCache {
    entries: HashMap<Entity, ArrowTargetState>,
}

impl ViewCubeArrowTargetCache {
    const TTL_SECS: f64 = 1.5;

    fn prune(&mut self, now_secs: f64) {
        self.entries
            .retain(|_, state| state.valid_until_secs + 0.25 >= now_secs);
    }

    fn get_valid_target(&self, camera: Entity, now_secs: f64) -> Option<ArrowTargetState> {
        self.entries
            .get(&camera)
            .filter(|state| state.valid_until_secs >= now_secs)
            .copied()
    }

    fn set_target(
        &mut self,
        camera: Entity,
        target_rotation: Quat,
        now_secs: f64,
        source: ArrowTargetSource,
    ) {
        self.entries.insert(
            camera,
            ArrowTargetState {
                target_rotation,
                valid_until_secs: now_secs + Self::TTL_SECS,
                source,
            },
        );
    }

    fn clear(&mut self, camera: Entity) {
        self.entries.remove(&camera);
    }
}

fn main_camera_for_event(
    event: &ViewCubeEvent,
    view_cube_query: &Query<&ViewCubeLink, With<ViewCubeRoot>>,
) -> Option<Entity> {
    let source = event_source(event);
    view_cube_query
        .get(source)
        .ok()
        .map(|link| link.main_camera)
}

fn event_source(event: &ViewCubeEvent) -> Entity {
    match event {
        ViewCubeEvent::FaceClicked { source, .. }
        | ViewCubeEvent::EdgeClicked { source, .. }
        | ViewCubeEvent::CornerClicked { source, .. }
        | ViewCubeEvent::ArrowClicked { source, .. }
        | ViewCubeEvent::ViewportActionClicked { source, .. } => *source,
    }
}

pub fn sync_view_cube_rotation(
    config: Res<ViewCubeConfig>,
    main_camera_query: Query<&GlobalTransform, Without<ViewCubeRoot>>,
    mut view_cube_query: Query<(&ViewCubeLink, &mut Transform), With<ViewCubeRoot>>,
) {
    for (link, mut cube_transform) in view_cube_query.iter_mut() {
        let Ok(main_camera_transform) = main_camera_query.get(link.main_camera) else {
            continue;
        };

        let (_, rotation, _) = main_camera_transform.to_scale_rotation_translation();
        cube_transform.rotation = rotation.conjugate() * config.effective_axis_correction();
    }
}

pub fn orient_axis_labels_to_screen_plane(
    mut labels: Query<(&ChildOf, &AxisLabelBillboard, &mut Transform)>,
    cubes: Query<(&ViewCubeLink, &GlobalTransform), With<ViewCubeRoot>>,
    cube_cameras: Query<(&ViewCubeLink, &GlobalTransform), With<ViewCubeCamera>>,
) {
    const AXIS_LABEL_SCREEN_GAP: f32 = 0.035;

    if labels.is_empty() {
        return;
    }

    let mut camera_rotation_by_main = HashMap::new();
    for (link, camera_global) in cube_cameras.iter() {
        camera_rotation_by_main.insert(link.main_camera, camera_global.rotation());
    }

    for (parent, label_meta, mut label_transform) in labels.iter_mut() {
        let Ok((cube_link, cube_global)) = cubes.get(parent.0) else {
            continue;
        };
        let Some(camera_rotation) = camera_rotation_by_main.get(&cube_link.main_camera) else {
            continue;
        };

        let cube_rotation = cube_global.rotation();
        let camera_up_world = *camera_rotation * Vec3::Y;
        let camera_up_local = cube_rotation.inverse() * camera_up_world;
        let axis_dir = label_meta.axis_direction.normalize_or_zero();
        let projected_up = camera_up_local - axis_dir * camera_up_local.dot(axis_dir);
        let mut gap_dir_local = projected_up.normalize_or_zero();
        if gap_dir_local.length_squared() <= 1.0e-6 {
            let camera_right_world = *camera_rotation * Vec3::X;
            let camera_right_local = cube_rotation.inverse() * camera_right_world;
            let projected_right = camera_right_local - axis_dir * camera_right_local.dot(axis_dir);
            gap_dir_local = projected_right.normalize_or_zero();
        }

        label_transform.translation =
            label_meta.base_position + gap_dir_local * AXIS_LABEL_SCREEN_GAP;
        // Cancel the cube's local rotation so labels remain parallel to the screen.
        label_transform.rotation = cube_rotation.inverse() * *camera_rotation;
    }
}

pub fn apply_render_layers_to_scene(
    view_cube_query: Query<(Entity, &ViewCubeRenderLayer), With<ViewCubeRoot>>,
    children_query: Query<&Children>,
    entities_without_layer: Query<Entity, (Without<RenderLayers>, Without<ViewCubeCamera>)>,
    mut commands: Commands,
) {
    for (cube_root, layer) in view_cube_query.iter() {
        let render_layers = RenderLayers::layer(layer.0);

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
    if entities_without_layer.get(entity).is_ok() {
        commands.entity(entity).insert(render_layers.clone());
    }

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

type FloatingOriginQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Transform, &'static GridCell<i128>),
    (With<FloatingOrigin>, Without<ViewCubeTargetCamera>),
>;

type ViewCubeCameraQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut Transform,
        &'static GlobalTransform,
        &'static mut EditorCam,
    ),
    (With<ViewCubeTargetCamera>, Without<FloatingOrigin>),
>;

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
    floating_origin: FloatingOriginQuery<'w, 's>,
    floating_origin_settings: Res<'w, FloatingOriginSettings>,
}

pub fn handle_view_cube_editor(
    mut events: MessageReader<ViewCubeEvent>,
    view_cube_query: Query<&ViewCubeLink, With<ViewCubeRoot>>,
    cube_root_query: Query<&GlobalTransform, With<ViewCubeRoot>>,
    mut camera_query: ViewCubeCameraQuery,
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
        let Ok((entity, mut transform, global_transform, mut editor_cam)) =
            camera_query.get_mut(cam)
        else {
            continue;
        };

        let origin_world = lookup
            .floating_origin
            .iter()
            .next()
            .map(|(t, c)| {
                lookup
                    .floating_origin_settings
                    .grid_position_double::<i128>(c, t)
                    .as_vec3()
            })
            .unwrap_or(Vec3::ZERO);

        if !matches!(event, ViewCubeEvent::ArrowClicked { .. }) {
            update_anchor_depth_for_view_cube(
                entity,
                transform.as_ref(),
                global_transform,
                &mut editor_cam,
                &lookup.viewports,
                lookup.entity_map.as_ref(),
                &lookup.values,
                origin_world,
            );
        }

        editor_cam.end_move();
        editor_cam.current_motion = CurrentMotion::Stationary;

        let (_, global_rotation, _) = global_transform.to_scale_rotation_translation();
        let parent_rotation = global_rotation * transform.rotation.inverse();
        let cube_global = cube_root_query
            .get(event_source(event))
            .ok()
            .map(GlobalTransform::rotation);
        let cube_rotation = if config.sync_with_camera {
            global_rotation.conjugate() * config.effective_axis_correction()
        } else {
            cube_global.unwrap_or(Quat::IDENTITY)
        };
        let camera_dir_cube = cube_rotation.inverse() * Vec3::Z;

        if let ViewCubeEvent::FaceClicked { direction, .. } = event {
            let clicked_face_dot = direction.to_look_direction().dot(camera_dir_cube);
            if clicked_face_dot >= FACE_IN_SCREEN_PLANE_DOT_THRESHOLD {
                continue;
            }

            let raw_look_dir_world = face_target_camera_dir_world(*direction, &config);
            if raw_look_dir_world.length_squared() <= 1.0e-6 {
                continue;
            }
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let chosen_up = choose_face_upright_up(*direction, parent_rotation, facing_local)
                    .or_else(|| choose_continuous_up(transform.as_ref(), facing_local))
                    .unwrap_or_else(|| {
                        choose_min_rotation_up(transform.as_ref(), parent_rotation, facing_local).0
                    });
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local,
                    target_up_direction: chosen_up,
                    camera: entity,
                };
                let target_rotation = trigger_rotation(&trigger);
                lookup.arrow_cache.set_target(
                    entity,
                    target_rotation,
                    now_secs,
                    ArrowTargetSource::ViewSnap,
                );
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

        if let ViewCubeEvent::CornerClicked {
            position,
            local_direction,
            ..
        } = event
        {
            let clicked_corner_dot = local_direction.dot(camera_dir_cube);
            if clicked_corner_dot >= CORNER_IN_SCREEN_AXIS_DOT_THRESHOLD {
                continue;
            }
            let raw_look_dir_world = direction_target_camera_dir_world(*local_direction, &config);
            if raw_look_dir_world.length_squared() <= 1.0e-6 {
                continue;
            }
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let (chosen_up, _, _, _, _, _) =
                    choose_min_rotation_up(transform.as_ref(), parent_rotation, facing_local);
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local,
                    target_up_direction: chosen_up,
                    camera: entity,
                };
                let target_rotation = trigger_rotation(&trigger);
                lookup.arrow_cache.set_target(
                    entity,
                    target_rotation,
                    now_secs,
                    ArrowTargetSource::ViewSnap,
                );
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

        if let ViewCubeEvent::EdgeClicked {
            direction,
            target_face,
            ..
        } = event
        {
            let raw_look_dir_world = face_target_camera_dir_world(*target_face, &config);
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let chosen_up = choose_face_upright_up(*target_face, parent_rotation, facing_local)
                    .or_else(|| choose_continuous_up(transform.as_ref(), facing_local))
                    .unwrap_or_else(|| {
                        choose_min_rotation_up(transform.as_ref(), parent_rotation, facing_local).0
                    });
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local,
                    target_up_direction: chosen_up,
                    camera: entity,
                };
                lookup.arrow_cache.set_target(
                    entity,
                    trigger_rotation(&trigger),
                    now_secs,
                    ArrowTargetSource::ViewSnap,
                );
                look_to.write(trigger);
            } else {
                warn!(
                    edge_direction = ?direction,
                    target_face = ?target_face,
                    raw_look_dir_world = ?raw_look_dir_world,
                    facing_world = ?facing_world,
                    facing_local = ?facing_local_vec,
                    "view cube: invalid edge snap directions"
                );
            }
            continue;
        }

        if let ViewCubeEvent::ArrowClicked { arrow, .. } = event {
            if let Some((previous_depth, refreshed_depth)) = refresh_anchor_depth_for_arrow(
                entity,
                global_transform,
                &mut editor_cam,
                &lookup.viewports,
                lookup.entity_map.as_ref(),
                &lookup.values,
                origin_world,
            ) {
                debug!(
                    target: "view_cube::arrow",
                    camera = ?entity,
                    arrow = ?arrow,
                    previous_depth,
                    refreshed_depth,
                    "refreshed anchor depth from orbit target before arrow step"
                );
            } else {
                debug!(
                    target: "view_cube::arrow",
                    camera = ?entity,
                    arrow = ?arrow,
                    current_depth = editor_cam.last_anchor_depth,
                    "anchor depth refresh unavailable; keeping current depth"
                );
            }

            let angle = config.rotation_increment;
            let (base_rotation, base_source) =
                if let Some(cached) = lookup.arrow_cache.get_valid_target(entity, now_secs) {
                    let drift = cached
                        .target_rotation
                        .angle_between(transform.rotation)
                        .abs();
                    if cached.source != ArrowTargetSource::ArrowStep {
                        debug!(
                            target: "view_cube::arrow",
                            camera = ?entity,
                            arrow = ?arrow,
                            cached_source = ?cached.source,
                            drift_deg = drift.to_degrees(),
                            "ignoring non-arrow cached target"
                        );
                        (transform.rotation, "current_rotation")
                    } else if drift <= ARROW_CACHE_MAX_DRIFT_RADIANS {
                        debug!(
                            target: "view_cube::arrow",
                            camera = ?entity,
                            arrow = ?arrow,
                            drift_deg = drift.to_degrees(),
                            "using cached arrow target as rotation base"
                        );
                        (cached.target_rotation, "cached_arrow_target")
                    } else {
                        debug!(
                            target: "view_cube::arrow",
                            camera = ?entity,
                            arrow = ?arrow,
                            drift_deg = drift.to_degrees(),
                            drift_threshold_deg = ARROW_CACHE_MAX_DRIFT_RADIANS.to_degrees(),
                            "dropping stale cached arrow target"
                        );
                        lookup.arrow_cache.clear(entity);
                        (transform.rotation, "current_rotation")
                    }
                } else {
                    (transform.rotation, "current_rotation")
                };
            let base_forward_local = base_rotation * Vec3::NEG_Z;
            let base_up_local = base_rotation * Vec3::Y;
            let base_right_local = base_rotation * Vec3::X;
            let base_forward_world = parent_rotation * base_forward_local;
            let base_up_world = parent_rotation * base_up_local;
            let base_right_world = parent_rotation * base_right_local;

            let (step_axis_world, signed_angle, _) = arrow_camera_axis_angle(
                *arrow,
                angle,
                base_right_world,
                base_up_world,
                base_forward_world,
            );
            let step_rotation_world = Quat::from_axis_angle(*step_axis_world, signed_angle);
            let new_forward_world = step_rotation_world * base_forward_world;
            let new_up_world = step_rotation_world * base_up_world;
            let new_forward_local = parent_rotation.inverse() * new_forward_world;
            let new_up_local = parent_rotation.inverse() * new_up_world;

            if let Ok(facing) = Dir3::new(new_forward_local)
                && let Ok(up_dir) = Dir3::new(new_up_local)
            {
                let trigger = LookToTrigger {
                    target_facing_direction: facing,
                    target_up_direction: up_dir,
                    camera: entity,
                };
                let target_rotation = trigger_rotation(&trigger);
                let target_delta_deg = transform
                    .rotation
                    .angle_between(target_rotation)
                    .to_degrees();
                debug!(
                    target: "view_cube::arrow",
                    camera = ?entity,
                    arrow = ?arrow,
                    step_deg = angle.to_degrees(),
                    base_source,
                    target_delta_deg,
                    "arrow click resolved target rotation"
                );
                lookup.arrow_cache.set_target(
                    entity,
                    target_rotation,
                    now_secs,
                    ArrowTargetSource::ArrowStep,
                );
                look_to.write(trigger);
            } else {
                warn!(
                    arrow = ?arrow,
                    new_forward_local = ?new_forward_local,
                    new_up_local = ?new_up_local,
                    "view cube: invalid arrow directions"
                );
            }
        }

        if let ViewCubeEvent::ViewportActionClicked { action, .. } = event {
            match action {
                ViewportActionButton::Reset => {
                    apply_viewport_reset(transform.as_mut(), &mut editor_cam);
                }
                ViewportActionButton::ZoomOut => {
                    apply_viewport_zoom(true, transform.as_mut(), &mut editor_cam);
                }
                ViewportActionButton::ZoomIn => {
                    apply_viewport_zoom(false, transform.as_mut(), &mut editor_cam);
                }
            }
            lookup.arrow_cache.clear(entity);
        }
    }
}

fn trigger_rotation(trigger: &LookToTrigger) -> Quat {
    Transform::default()
        .looking_to(
            *trigger.target_facing_direction,
            *trigger.target_up_direction,
        )
        .rotation
}

fn apply_viewport_reset(transform: &mut Transform, editor_cam: &mut EditorCam) {
    *transform = Transform::IDENTITY;
    editor_cam.current_motion = CurrentMotion::Stationary;
    editor_cam.last_anchor_depth = VIEWPORT_RESET_ANCHOR_DEPTH;
}

fn apply_viewport_zoom(out: bool, transform: &mut Transform, editor_cam: &mut EditorCam) {
    let current_depth = (editor_cam.last_anchor_depth.abs() as f32).max(0.25);
    let target_depth = if out {
        (current_depth * VIEWPORT_ZOOM_OUT_MULTIPLIER).max(0.5)
    } else {
        (current_depth / VIEWPORT_ZOOM_IN_MULTIPLIER).max(0.5)
    };
    let depth_delta = target_depth - current_depth;
    if depth_delta.abs() <= 1.0e-6 {
        return;
    }

    // Move camera backwards in its local view direction to increase orbit distance.
    transform.translation += (transform.rotation * Vec3::Z) * depth_delta;
    editor_cam.last_anchor_depth = -(target_depth as f64);
    editor_cam.current_motion = CurrentMotion::Stationary;
}

fn face_target_camera_dir_world(
    direction: super::components::FaceDirection,
    config: &ViewCubeConfig,
) -> Vec3 {
    let local_dir = direction.to_look_direction();
    direction_target_camera_dir_world(local_dir, config)
}

#[cfg(test)]
fn corner_target_camera_dir_world(
    position: super::components::CornerPosition,
    config: &ViewCubeConfig,
) -> Vec3 {
    let local_dir = position.to_look_direction();
    direction_target_camera_dir_world(local_dir, config)
}

fn direction_target_camera_dir_world(local_dir: Vec3, config: &ViewCubeConfig) -> Vec3 {
    if config.sync_with_camera {
        (config.effective_axis_correction() * local_dir).normalize_or_zero()
    } else {
        local_dir.normalize_or_zero()
    }
}

fn arrow_camera_axis_angle(
    arrow: RotationArrow,
    angle: f32,
    camera_right_world: Vec3,
    camera_up_world: Vec3,
    camera_forward_world: Vec3,
) -> (Dir3, f32, &'static str) {
    match arrow {
        RotationArrow::Left => (
            Dir3::new(camera_up_world).unwrap_or(Dir3::new_unchecked(Vec3::Y)),
            angle,
            "camera_up",
        ),
        RotationArrow::Right => (
            Dir3::new(camera_up_world).unwrap_or(Dir3::new_unchecked(Vec3::Y)),
            -angle,
            "camera_up",
        ),
        RotationArrow::Up => (
            Dir3::new(camera_right_world).unwrap_or(Dir3::new_unchecked(Vec3::X)),
            angle,
            "camera_right",
        ),
        RotationArrow::Down => (
            Dir3::new(camera_right_world).unwrap_or(Dir3::new_unchecked(Vec3::X)),
            -angle,
            "camera_right",
        ),
        RotationArrow::RollLeft => (
            Dir3::new(camera_forward_world).unwrap_or(Dir3::new_unchecked(Vec3::NEG_Z)),
            angle,
            "camera_forward",
        ),
        RotationArrow::RollRight => (
            Dir3::new(camera_forward_world).unwrap_or(Dir3::new_unchecked(Vec3::NEG_Z)),
            -angle,
            "camera_forward",
        ),
    }
}

fn angle_to_target_rotation(transform: &Transform, facing: Dir3, up: Dir3) -> f32 {
    let target_rotation = Transform::default().looking_to(*facing, *up).rotation;
    transform.rotation.angle_between(target_rotation).abs()
}

fn choose_continuous_up(transform: &Transform, facing_local: Dir3) -> Option<Dir3> {
    let facing = *facing_local;
    for candidate in [
        transform.rotation * Vec3::Y,
        transform.rotation * Vec3::NEG_Z,
        transform.rotation * Vec3::X,
    ] {
        let projected = candidate - facing * candidate.dot(facing);
        if projected.length_squared() <= 1.0e-6 {
            continue;
        }
        if let Ok(up) = Dir3::new(projected) {
            return Some(up);
        }
    }
    None
}

fn choose_face_upright_up(
    target_face: super::components::FaceDirection,
    parent_rotation: Quat,
    facing_local: Dir3,
) -> Option<Dir3> {
    let parent_inverse = parent_rotation.inverse();
    let world_candidates: &[Vec3] = match target_face {
        super::components::FaceDirection::East
        | super::components::FaceDirection::West
        | super::components::FaceDirection::North
        | super::components::FaceDirection::South => &[Vec3::Y, Vec3::Z, Vec3::X],
        super::components::FaceDirection::Up => &[Vec3::Z, Vec3::X, Vec3::Y],
        super::components::FaceDirection::Down => &[Vec3::NEG_Z, Vec3::X, Vec3::Y],
    };

    let facing = *facing_local;
    for world_up in world_candidates.iter().copied() {
        let local_up_candidate = parent_inverse * world_up;
        let projected = local_up_candidate - facing * local_up_candidate.dot(facing);
        if projected.length_squared() <= 1.0e-6 {
            continue;
        }
        if let Ok(up) = Dir3::new(projected) {
            return Some(up);
        }
    }

    None
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

#[allow(clippy::too_many_arguments)]
fn update_anchor_depth_for_view_cube(
    camera: Entity,
    _transform: &Transform,
    global_transform: &GlobalTransform,
    editor_cam: &mut EditorCam,
    viewports: &Query<&crate::ui::inspector::viewport::Viewport, With<ViewCubeTargetCamera>>,
    entity_map: &EntityMap,
    values: &Query<&'static ComponentValue>,
    origin_world: Vec3,
) {
    let Some(orbit_target_world) = view_cube_orbit_target(camera, viewports, entity_map, values)
    else {
        return;
    };
    let orbit_target = orbit_target_world - origin_world;

    let world_translation = global_transform.translation();
    let world_rotation = global_transform.rotation();
    let to_target = orbit_target - world_translation;
    let forward = world_rotation * Vec3::NEG_Z;
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

    let mut distance = if projected_distance.is_finite()
        && measured_distance.is_finite()
        && projected_distance > 1.0e-3
        && measured_distance > 1.0e-3
        && alignment_ratio >= MIN_ALIGNMENT_FOR_PROJECTED
    {
        projected_distance
    } else if previous_distance > 1.0e-3 {
        previous_distance
    } else if measured_distance.is_finite() && measured_distance > 1.0e-3 {
        measured_distance
    } else {
        1.0
    };

    if distance < MIN_ORBIT_DISTANCE {
        distance = MIN_ORBIT_DISTANCE;
    }

    let new_depth = -(distance as f64);
    editor_cam.last_anchor_depth = new_depth;
}

#[allow(clippy::too_many_arguments)]
fn refresh_anchor_depth_for_arrow(
    camera: Entity,
    global_transform: &GlobalTransform,
    editor_cam: &mut EditorCam,
    viewports: &Query<&crate::ui::inspector::viewport::Viewport, With<ViewCubeTargetCamera>>,
    entity_map: &EntityMap,
    values: &Query<&'static ComponentValue>,
    origin_world: Vec3,
) -> Option<(f32, f32)> {
    let orbit_target_world = view_cube_orbit_target(camera, viewports, entity_map, values)?;
    let orbit_target = orbit_target_world - origin_world;
    let world_translation = global_transform.translation();
    let measured_distance = (orbit_target - world_translation).length();
    if !measured_distance.is_finite() || measured_distance <= 1.0e-3 {
        return None;
    }

    const MIN_ORBIT_DISTANCE: f32 = 0.25;
    let previous_distance = editor_cam.last_anchor_depth.abs() as f32;
    let refreshed_distance = measured_distance.max(MIN_ORBIT_DISTANCE);
    editor_cam.last_anchor_depth = -(refreshed_distance as f64);
    Some((previous_distance, refreshed_distance))
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
    fn choose_continuous_up_keeps_visible_face_orientation_on_opposite_snap() {
        let transform = Transform::default();
        let facing = Dir3::new(Vec3::Z).expect("unit vector");
        let up = choose_continuous_up(&transform, facing).expect("continuous up");
        assert!(
            up.dot(Vec3::Y) > 0.99,
            "expected up close to +Y, got {:?}",
            *up
        );
    }

    #[test]
    fn choose_continuous_up_falls_back_to_forward_when_up_is_parallel() {
        let transform = Transform::default();
        let facing = Dir3::new(Vec3::Y).expect("unit vector");
        let up = choose_continuous_up(&transform, facing).expect("continuous up");
        assert!(
            up.dot(Vec3::NEG_Z).abs() > 0.99,
            "expected up close to +/-Z, got {:?}",
            *up
        );
    }

    #[test]
    fn choose_face_upright_up_keeps_east_west_consistent() {
        let parent = Quat::IDENTITY;
        let east_facing = Dir3::new(Vec3::NEG_X).expect("unit vector");
        let west_facing = Dir3::new(Vec3::X).expect("unit vector");
        let east_up = choose_face_upright_up(
            crate::plugins::view_cube::FaceDirection::East,
            parent,
            east_facing,
        )
        .expect("upright up for east");
        let west_up = choose_face_upright_up(
            crate::plugins::view_cube::FaceDirection::West,
            parent,
            west_facing,
        )
        .expect("upright up for west");
        assert!(east_up.dot(Vec3::Y) > 0.99, "east up should be +Y");
        assert!(west_up.dot(Vec3::Y) > 0.99, "west up should be +Y");
    }

    #[test]
    fn choose_face_upright_up_prefers_backward_on_down_face() {
        let parent = Quat::IDENTITY;
        let down_facing = Dir3::new(Vec3::Y).expect("unit vector");
        let up = choose_face_upright_up(
            crate::plugins::view_cube::FaceDirection::Down,
            parent,
            down_facing,
        )
        .expect("upright up for down");
        assert!(
            up.dot(Vec3::NEG_Z) > 0.99,
            "down-face up should align with -Z, got {:?}",
            *up
        );
    }

    #[test]
    fn choose_face_upright_up_prefers_forward_on_up_face() {
        let parent = Quat::IDENTITY;
        let up_facing = Dir3::new(Vec3::NEG_Y).expect("unit vector");
        let up = choose_face_upright_up(
            crate::plugins::view_cube::FaceDirection::Up,
            parent,
            up_facing,
        )
        .expect("upright up for up-face");
        assert!(
            up.dot(Vec3::Z) > 0.99,
            "up-face up should align with +Z, got {:?}",
            *up
        );
    }

    #[test]
    fn arrow_target_cache_is_valid_within_ttl() {
        let mut cache = ViewCubeArrowTargetCache::default();
        let entity = Entity::from_bits(42);
        let target = Quat::from_rotation_y(0.4);
        cache.set_target(entity, target, 10.0, ArrowTargetSource::ArrowStep);
        let cached = cache
            .get_valid_target(entity, 10.3)
            .expect("cached target should still be valid");
        assert_eq!(cached.target_rotation, target);
        assert_eq!(cached.source, ArrowTargetSource::ArrowStep);
    }

    #[test]
    fn arrow_target_cache_expires_after_ttl() {
        let mut cache = ViewCubeArrowTargetCache::default();
        let entity = Entity::from_bits(7);
        let target = Quat::from_rotation_x(0.2);
        cache.set_target(entity, target, 1.0, ArrowTargetSource::ViewSnap);
        let cached =
            cache.get_valid_target(entity, 1.0 + ViewCubeArrowTargetCache::TTL_SECS + 0.01);
        assert!(cached.is_none(), "cached target should have expired");
    }

    #[test]
    fn arrow_camera_axis_angle_maps_each_pair_to_camera_axis() {
        let angle = 0.25;
        let right = Vec3::Y;
        let up = Vec3::Z;
        let forward = Vec3::X;

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Left, angle, right, up, forward);
        assert_eq!(*axis, up);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "camera_up");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Right, angle, right, up, forward);
        assert_eq!(*axis, up);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "camera_up");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Up, angle, right, up, forward);
        assert_eq!(*axis, right);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "camera_right");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Down, angle, right, up, forward);
        assert_eq!(*axis, right);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "camera_right");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::RollLeft, angle, right, up, forward);
        assert_eq!(*axis, forward);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "camera_forward");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::RollRight, angle, right, up, forward);
        assert_eq!(*axis, forward);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "camera_forward");
    }

    #[test]
    fn face_target_camera_dir_world_applies_axis_correction() {
        let dir = crate::plugins::view_cube::FaceDirection::East;
        let config = ViewCubeConfig::default();
        let world = face_target_camera_dir_world(dir, &config);
        assert!((world - Vec3::NEG_X).length() < 1.0e-5);
    }

    #[test]
    fn corner_target_camera_dir_world_applies_axis_correction() {
        let corner = crate::plugins::view_cube::CornerPosition::TopFrontRight;
        let config = ViewCubeConfig::default();
        let world = corner_target_camera_dir_world(corner, &config);
        let expected = Vec3::new(-1.0, 1.0, -1.0).normalize();
        assert!((world - expected).length() < 1.0e-5);
    }

    #[test]
    fn viewport_reset_restores_identity_transform_and_default_depth() {
        let mut transform = Transform::from_translation(Vec3::new(1.0, -2.0, 3.0))
            .with_rotation(Quat::from_rotation_y(0.4));
        let mut editor_cam = EditorCam {
            last_anchor_depth: -9.0,
            ..Default::default()
        };

        apply_viewport_reset(&mut transform, &mut editor_cam);

        assert_eq!(transform, Transform::IDENTITY);
        assert_eq!(editor_cam.last_anchor_depth, VIEWPORT_RESET_ANCHOR_DEPTH);
        assert!(matches!(
            editor_cam.current_motion,
            CurrentMotion::Stationary
        ));
    }

    #[test]
    fn viewport_zoom_out_moves_back_along_view_and_updates_depth() {
        let mut transform = Transform::from_translation(Vec3::new(0.5, 1.0, -0.25))
            .with_rotation(Quat::from_rotation_y(0.3));
        let mut editor_cam = EditorCam {
            last_anchor_depth: -2.0,
            ..Default::default()
        };

        let initial_translation = transform.translation;
        let expected_target_depth = 2.0 * VIEWPORT_ZOOM_OUT_MULTIPLIER;
        let expected_delta = expected_target_depth - 2.0;
        let expected_translation =
            initial_translation + (transform.rotation * Vec3::Z) * expected_delta;

        apply_viewport_zoom(true, &mut transform, &mut editor_cam);

        assert!((transform.translation - expected_translation).length() < 1.0e-5);
        assert!((editor_cam.last_anchor_depth + expected_target_depth as f64).abs() < 1.0e-8);
        assert!(matches!(
            editor_cam.current_motion,
            CurrentMotion::Stationary
        ));
    }
}
