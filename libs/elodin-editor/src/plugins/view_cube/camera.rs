//! Camera systems for the ViewCube plugin.

use crate::spatial::WithoutFloatingOrigin;
#[cfg(feature = "big_space")]
use crate::spatial::{FloatingOriginSettings, GridCell, WithFloatingOrigin};
use bevy::camera::visibility::RenderLayers;
use bevy::ecs::hierarchy::ChildOf;
use bevy::ecs::system::SystemParam;
use bevy::log::warn;
use bevy::math::{DVec3, Dir3};
use bevy::prelude::*;
use bevy::world_serialization::{WorldInstance, WorldInstanceSpawner};
use bevy_editor_cam::controller::component::EditorCam;
use bevy_editor_cam::controller::motion::CurrentMotion;
use bevy_editor_cam::extensions::look_to::LookToTrigger;
use impeller2_bevy::EntityMap;
use impeller2_wkt::ComponentValue;
use std::collections::HashMap;
use std::f32::consts::PI;

use super::components::{
    AxisLabelBillboard, FaceLabel, KeepsRenderLayers, RotationArrow, ViewCubeCamera, ViewCubeFrame,
    ViewCubeFrameRef, ViewCubeLink, ViewCubeRoot, ViewportActionButton,
};
use super::config::ViewCubeConfig;
use super::events::ViewCubeEvent;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};

const FACE_IN_SCREEN_PLANE_DOT_THRESHOLD: f32 = 0.999;
const CORNER_IN_SCREEN_AXIS_DOT_THRESHOLD: f32 = 0.998;
const ARROW_CACHE_MAX_DRIFT_RADIANS: f32 = 6.0_f32.to_radians();
const VIEWPORT_RESET_ANCHOR_DEPTH: f64 = -2.0;
/// Orbit distance ratio applied per zoom button click. A single factor for both
/// directions keeps the two steps mutual inverses, so a zoom-in click undoes a
/// zoom-out click exactly.
const VIEWPORT_ZOOM_STEP: f32 = 2.2;

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
                direction.as_dvec3(),
                entity,
                &transform.rotation.as_dquat(),
                editor_cam,
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

#[derive(Resource, Default)]
pub struct ViewCubeOrbitTargetCache {
    entries: HashMap<Entity, DVec3>,
}

impl ViewCubeOrbitTargetCache {
    fn remember(&mut self, camera: Entity, target: DVec3) -> DVec3 {
        self.entries.insert(camera, target);
        target
    }

    fn last(&self, camera: Entity) -> Option<DVec3> {
        self.entries.get(&camera).copied()
    }

    fn forget(&mut self, camera: Entity) {
        self.entries.remove(&camera);
    }
}

/// Cube-local axis → Bevy world. Same `to_bevy` the mesh uses.
pub fn frame_dir_to_bevy(frame: GeoFrame, local_dir: Vec3, geo: &GeoContext) -> Vec3 {
    let q = GeoRotation::absolute(frame, bevy::math::DQuat::IDENTITY).to_bevy(geo);
    (q * local_dir.as_dvec3()).as_vec3().normalize_or_zero()
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

fn overlay_for_event(
    event: &ViewCubeEvent,
    overlay_cameras: &Query<(&ViewCubeLink, &ViewCubeFrameRef), With<ViewCubeCamera>>,
) -> Option<(Entity, GeoFrame)> {
    overlay_cameras
        .get(event_source(event))
        .ok()
        .map(|(link, frame_ref)| (link.main_camera, frame_ref.0))
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

type ViewCubeOverlayCameraTransformQuery<'w, 's> = Query<
    'w,
    's,
    (&'static ViewCubeLink, &'static mut Transform),
    (With<ViewCubeCamera>, Without<ViewCubeRoot>),
>;

pub fn sync_view_cube_camera_orientation(
    config: Res<ViewCubeConfig>,
    main_camera_query: Query<&GlobalTransform, (Without<ViewCubeRoot>, Without<ViewCubeCamera>)>,
    mut overlay_camera_query: ViewCubeOverlayCameraTransformQuery<'_, '_>,
) {
    for (link, mut camera_transform) in overlay_camera_query.iter_mut() {
        let Ok(main_camera_transform) = main_camera_query.get(link.main_camera) else {
            continue;
        };

        let (_, rotation, _) = main_camera_transform.to_scale_rotation_translation();
        // Mirror the camera's global rotation.
        camera_transform.rotation = rotation;
        // Keep it at its given distance.
        camera_transform.translation =
            camera_transform.rotation * Vec3::new(0.0, 0.0, config.camera_distance);
    }
}

pub fn orient_axis_labels_to_screen_plane(
    mut labels: Query<(&ChildOf, &AxisLabelBillboard, &mut Transform)>,
    cubes: Query<(&ViewCubeFrame, &GlobalTransform), With<ViewCubeRoot>>,
    cube_cameras: Query<(&ViewCubeFrameRef, &GlobalTransform), With<ViewCubeCamera>>,
) {
    const AXIS_LABEL_SCREEN_GAP: f32 = 0.035;

    if labels.is_empty() {
        return;
    }

    let mut camera_rotation_by_frame = HashMap::new();
    for (frame_ref, camera_global) in cube_cameras.iter() {
        camera_rotation_by_frame
            .entry(frame_ref.0)
            .or_insert(camera_global.rotation());
    }

    for (parent, label_meta, mut label_transform) in labels.iter_mut() {
        let Ok((cube_frame, cube_global)) = cubes.get(parent.0) else {
            continue;
        };
        let Some(camera_rotation) = camera_rotation_by_frame.get(&cube_frame.0) else {
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
        label_transform.rotation = cube_rotation.inverse() * *camera_rotation;
    }
}

const FACE_LABEL_EPS: f32 = 1.0e-5;
/// Angle change below which rewriting the label transform is not worth the
/// change-detection and transform propagation it would trigger.
const FACE_LABEL_ANGLE_EPS: f32 = 1.0e-4;

/// Screen-space (x right, y up) baseline and glyph-up axes of a label spun by
/// `angle` inside its face plane.
#[cfg(test)]
fn face_label_screen_axes(
    base_rotation: Quat,
    cube_rotation: Quat,
    camera_rotation: Quat,
    angle: f32,
) -> (Vec2, Vec2) {
    let to_camera = camera_rotation.inverse() * cube_rotation * base_rotation;
    let (c1, c2) = (to_camera * Vec3::X, to_camera * Vec3::Y);
    let (sin, cos) = angle.sin_cos();
    let right = cos * c1 + sin * c2;
    let up = cos * c2 - sin * c1;
    (Vec2::new(right.x, right.y), Vec2::new(up.x, up.y))
}

/// In-plane spin (radians) that lays a face label's baseline flat on the screen
/// horizontal.
///
/// Quantizing to 90° cannot do this: on an obliquely viewed face both in-plane
/// axes can sit ~60° off horizontal, so every quarter turn reads sideways.
/// Solving `cos·c1.y + sin·c2.y = 0` is exact for any pose instead, where `c1`
/// and `c2` are the face plane axes in camera space.
///
/// Returns `None` for an edge-on face, whose whole plane projects to a line.
pub(super) fn face_label_in_plane_angle(
    base_rotation: Quat,
    cube_rotation: Quat,
    camera_rotation: Quat,
) -> Option<f32> {
    let to_camera = camera_rotation.inverse() * cube_rotation * base_rotation;
    let c1 = to_camera * Vec3::X;
    let c2 = to_camera * Vec3::Y;

    // Both axes already project horizontally: nothing to solve.
    let mut angle = if c1.y.abs() <= FACE_LABEL_EPS && c2.y.abs() <= FACE_LABEL_EPS {
        0.0
    } else {
        (-c1.y).atan2(c2.y)
    };

    let (sin, cos) = angle.sin_cos();
    let mut baseline = Vec2::new(cos * c1.x + sin * c2.x, cos * c1.y + sin * c2.y);
    // Both halves of the solution are horizontal; take the one running left to
    // right. Adding π only negates the baseline, so no need to solve again.
    if baseline.x < 0.0 {
        angle += PI;
        baseline = -baseline;
    }
    (baseline.length_squared() > FACE_LABEL_EPS * FACE_LABEL_EPS).then_some(angle)
}

/// Spins each face label so its word reads horizontally in the viewport that
/// owns it.
///
/// Every label knows the overlay camera it belongs to, so two viewports on the
/// same frame — which share a single cube — each orient their own copy. Reading
/// through `Mut` does not flag a change, so a still viewport costs one
/// comparison per label and triggers no transform propagation.
pub fn orient_face_labels_to_view(
    mut labels: Query<(&ChildOf, &mut FaceLabel, &mut Transform)>,
    cubes: Query<&GlobalTransform, With<ViewCubeRoot>>,
    cube_cameras: Query<&GlobalTransform, With<ViewCubeCamera>>,
) {
    for (parent, mut face_label, mut label_transform) in labels.iter_mut() {
        let Ok(cube_global) = cubes.get(parent.0) else {
            continue;
        };
        let Ok(camera_global) = cube_cameras.get(face_label.camera) else {
            continue;
        };

        let view = (cube_global.rotation(), camera_global.rotation());
        if face_label.last_view == Some(view) {
            continue;
        }
        face_label.last_view = Some(view);

        let base_rotation = face_label.base_rotation;
        let angle = face_label_in_plane_angle(base_rotation, view.0, view.1)
            .unwrap_or(face_label.last_angle);
        if (angle - face_label.last_angle).abs() <= FACE_LABEL_ANGLE_EPS {
            continue;
        }
        face_label.last_angle = angle;
        label_transform.rotation = base_rotation * Quat::from_rotation_z(angle);
    }
}

pub fn apply_render_layers_to_scene(
    view_cube_query: Query<(Entity, &RenderLayers, &Visibility), With<ViewCubeRoot>>,
    children_query: Query<&Children>,
    scene_instances: Query<&WorldInstance>,
    scene_spawner: Res<WorldInstanceSpawner>,
    view_cube_entities: Query<Entity, Without<ViewCubeCamera>>,
    own_layers: Query<&RenderLayers, With<KeepsRenderLayers>>,
    mut commands: Commands,
) {
    for (cube_root, render_layers, visibility) in view_cube_query.iter() {
        apply_layers_recursive(
            cube_root,
            &children_query,
            &view_cube_entities,
            &own_layers,
            render_layers,
            &mut commands,
        );

        let scene_ready = scene_instances
            .get(cube_root)
            .is_ok_and(|instance| scene_spawner.instance_is_ready(**instance));
        if scene_ready && *visibility == Visibility::Hidden {
            commands.entity(cube_root).insert(Visibility::Inherited);
        }
    }
}

fn apply_layers_recursive(
    entity: Entity,
    children_query: &Query<&Children>,
    view_cube_entities: &Query<Entity, Without<ViewCubeCamera>>,
    own_layers: &Query<&RenderLayers, With<KeepsRenderLayers>>,
    render_layers: &RenderLayers,
    commands: &mut Commands,
) {
    // A per-viewport subtree keeps its own layer and hands it down, otherwise
    // the shared cube's layer would leak in and every viewport would see every
    // copy of the face labels.
    let render_layers = match own_layers.get(entity) {
        Ok(own) => own,
        Err(_) => {
            if view_cube_entities.get(entity).is_ok() {
                commands.entity(entity).insert(render_layers.clone());
            }
            render_layers
        }
    };

    if let Ok(children) = children_query.get(entity) {
        for child in children.iter() {
            apply_layers_recursive(
                child,
                children_query,
                view_cube_entities,
                own_layers,
                render_layers,
                commands,
            );
        }
    }
}
#[cfg(feature = "big_space")]
type FloatingOriginQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Transform, &'static GridCell),
    (WithFloatingOrigin, Without<ViewCubeTargetCamera>),
>;

type ViewCubeCameraQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut Transform,
        Option<&'static ChildOf>,
        &'static mut EditorCam,
    ),
    (With<ViewCubeTargetCamera>, WithoutFloatingOrigin),
>;

#[cfg(feature = "big_space")]
type CameraParentQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Transform, &'static GridCell),
    (Without<ViewCubeTargetCamera>, WithoutFloatingOrigin),
>;

#[cfg(not(feature = "big_space"))]
type CameraParentQuery<'w, 's> =
    Query<'w, 's, &'static Transform, (Without<ViewCubeTargetCamera>, WithoutFloatingOrigin)>;

#[derive(Clone, Copy)]
struct CurrentCameraPose {
    translation: Vec3,
    rotation: Quat,
    parent_rotation: Quat,
}

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
    geo_context: Res<'w, GeoContext>,
    time: Res<'w, Time>,
    arrow_cache: ResMut<'w, ViewCubeArrowTargetCache>,
    orbit_cache: ResMut<'w, ViewCubeOrbitTargetCache>,
    camera_parents: CameraParentQuery<'w, 's>,
    #[cfg(feature = "big_space")]
    floating_origin: FloatingOriginQuery<'w, 's>,
    #[cfg(feature = "big_space")]
    floating_origin_settings: Res<'w, FloatingOriginSettings>,
}

impl<'w, 's> ViewCubeEditorLookup<'w, 's> {
    #[cfg(feature = "big_space")]
    fn origin(&self) -> DVec3 {
        self.floating_origin
            .iter()
            .next()
            .map(|(t, c)| self.floating_origin_settings.grid_position_double(c, t))
            .unwrap_or(DVec3::ZERO)
    }

    #[cfg(not(feature = "big_space"))]
    fn origin(&self) -> DVec3 {
        DVec3::ZERO
    }

    /// Compose the active camera's world pose by hand instead of trusting
    /// `GlobalTransform`, which `bevy`+`big_space` propagate in `PostUpdate`
    /// (one frame later than this `Update` system runs).
    ///
    /// big_space 0.12 propagates `parent.rotation * camera.local` onto the
    /// camera's world rotation, so the viewport's look-at quaternion shows up
    /// in `camera_pose.rotation`. `parent_rotation` lets the snap math
    /// convert between world and parent frames when emitting the
    /// `LookToTrigger`.
    #[cfg(feature = "big_space")]
    fn current_camera_pose(
        &self,
        transform: &Transform,
        parent: Option<&ChildOf>,
        origin_world: DVec3,
    ) -> CurrentCameraPose {
        let Some((parent_transform, parent_cell)) =
            parent.and_then(|parent| self.camera_parents.get(parent.parent()).ok())
        else {
            return CurrentCameraPose {
                translation: (transform.translation.as_dvec3() - origin_world).as_vec3(),
                rotation: transform.rotation,
                parent_rotation: Quat::IDENTITY,
            };
        };

        // Translation: parent.translation + parent.rotation * camera.translation,
        // promoted to high-precision via `grid_position_double` to avoid
        // accumulating f32 error when the floating origin is far from zero.
        let absolute_transform = parent_transform.mul_transform(*transform);
        let absolute_translation = self
            .floating_origin_settings
            .grid_position_double(parent_cell, &absolute_transform);
        CurrentCameraPose {
            translation: (absolute_translation - origin_world).as_vec3(),
            rotation: parent_transform.rotation * transform.rotation,
            parent_rotation: parent_transform.rotation,
        }
    }

    #[cfg(not(feature = "big_space"))]
    fn current_camera_pose(
        &self,
        transform: &Transform,
        parent: Option<&ChildOf>,
        _origin_world: DVec3,
    ) -> CurrentCameraPose {
        let Some(parent_transform) =
            parent.and_then(|parent| self.camera_parents.get(parent.parent()).ok())
        else {
            return CurrentCameraPose {
                translation: transform.translation,
                rotation: transform.rotation,
                parent_rotation: Quat::IDENTITY,
            };
        };

        let absolute_transform = parent_transform.mul_transform(*transform);
        CurrentCameraPose {
            translation: absolute_transform.translation,
            rotation: absolute_transform.rotation,
            parent_rotation: parent_transform.rotation,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn handle_view_cube_editor(
    mut events: MessageReader<ViewCubeEvent>,
    overlay_cameras: Query<(&ViewCubeLink, &ViewCubeFrameRef), With<ViewCubeCamera>>,
    cubes: Query<(&ViewCubeFrame, &GlobalTransform), With<ViewCubeRoot>>,
    mut camera_query: ViewCubeCameraQuery,
    mut lookup: ViewCubeEditorLookup,
    config: Res<ViewCubeConfig>,
    mut look_to: MessageWriter<LookToTrigger>,
) {
    for event in events.read() {
        let now_secs = lookup.time.elapsed_secs_f64();
        lookup.arrow_cache.prune(now_secs);

        let Some((cam, cube_frame)) = overlay_for_event(event, &overlay_cameras) else {
            continue;
        };
        let Ok((entity, mut transform, parent, mut editor_cam)) = camera_query.get_mut(cam) else {
            continue;
        };
        let cube_world_rotation = cubes
            .iter()
            .find(|(frame, _)| frame.0 == cube_frame)
            .map(|(_, global)| global.rotation())
            .unwrap_or(Quat::IDENTITY);

        let origin_world = lookup.origin();
        let camera_pose = lookup.current_camera_pose(transform.as_ref(), parent, origin_world);

        if !matches!(event, ViewCubeEvent::ArrowClicked { .. }) {
            update_anchor_depth_for_view_cube(
                entity,
                camera_pose.translation,
                camera_pose.rotation,
                &mut editor_cam,
                &lookup.viewports,
                lookup.entity_map.as_ref(),
                &lookup.values,
                &lookup.geo_context,
                &mut lookup.orbit_cache,
                origin_world,
            );
        }

        editor_cam.end_move();
        editor_cam.current_motion = CurrentMotion::Stationary;

        let global_rotation = camera_pose.rotation;
        let parent_rotation = camera_pose.parent_rotation;
        let camera_dir_cube = camera_dir_in_cube_local(cube_world_rotation, global_rotation);

        if let ViewCubeEvent::FaceClicked { direction, .. } = event {
            let clicked_face_dot = direction.to_look_direction().dot(camera_dir_cube);
            if clicked_face_dot >= FACE_IN_SCREEN_PLANE_DOT_THRESHOLD {
                continue;
            }

            let raw_look_dir_world =
                face_target_camera_dir_world(*direction, cube_frame, &lookup.geo_context, &config);
            if raw_look_dir_world.length_squared() <= 1.0e-6 {
                continue;
            }
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let chosen_up = choose_face_upright_up(
                    parent_rotation,
                    facing_local,
                    cube_frame,
                    &lookup.geo_context,
                )
                .or_else(|| choose_continuous_up(transform.as_ref(), facing_local))
                .unwrap_or_else(|| {
                    choose_min_rotation_up(transform.as_ref(), parent_rotation, facing_local).0
                });
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local.as_dvec3(),
                    target_up_direction: chosen_up.as_dvec3(),
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
            let raw_look_dir_world = direction_target_camera_dir_world(
                *local_direction,
                cube_frame,
                &lookup.geo_context,
                &config,
            );
            if raw_look_dir_world.length_squared() <= 1.0e-6 {
                continue;
            }
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let (chosen_up, _, _, _, _, _) =
                    choose_min_rotation_up(transform.as_ref(), parent_rotation, facing_local);
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local.as_dvec3(),
                    target_up_direction: chosen_up.as_dvec3(),
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
            let raw_look_dir_world = face_target_camera_dir_world(
                *target_face,
                cube_frame,
                &lookup.geo_context,
                &config,
            );
            let facing_world = -raw_look_dir_world;
            let facing_local_vec = parent_rotation.inverse() * facing_world;

            if let Ok(facing_local) = Dir3::new(facing_local_vec) {
                let chosen_up = choose_face_upright_up(
                    parent_rotation,
                    facing_local,
                    cube_frame,
                    &lookup.geo_context,
                )
                .or_else(|| choose_continuous_up(transform.as_ref(), facing_local))
                .unwrap_or_else(|| {
                    choose_min_rotation_up(transform.as_ref(), parent_rotation, facing_local).0
                });
                let trigger = LookToTrigger {
                    target_facing_direction: facing_local.as_dvec3(),
                    target_up_direction: chosen_up.as_dvec3(),
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
            refresh_anchor_depth_for_arrow(
                entity,
                camera_pose.translation,
                &mut editor_cam,
                &lookup.viewports,
                lookup.entity_map.as_ref(),
                &lookup.values,
                &lookup.geo_context,
                &mut lookup.orbit_cache,
                origin_world,
            );

            let angle = config.rotation_increment;
            // Chain rapid repeats from the last applied arrow target so a held
            // button steps cleanly; otherwise base off the live pose.
            let base_rotation =
                if let Some(cached) = lookup.arrow_cache.get_valid_target(entity, now_secs) {
                    let drift = cached
                        .target_rotation
                        .angle_between(transform.rotation)
                        .abs();
                    if cached.source == ArrowTargetSource::ArrowStep
                        && drift <= ARROW_CACHE_MAX_DRIFT_RADIANS
                    {
                        cached.target_rotation
                    } else {
                        lookup.arrow_cache.clear(entity);
                        transform.rotation
                    }
                } else {
                    transform.rotation
                };
            let base_forward_local = base_rotation * Vec3::NEG_Z;
            let base_up_local = base_rotation * Vec3::Y;
            let base_right_local = base_rotation * Vec3::X;
            let base_forward_world = parent_rotation * base_forward_local;
            let base_up_world = parent_rotation * base_up_local;
            let base_right_world = parent_rotation * base_right_local;

            // Screen-space pairs: Left/Right yaw around camera up, Up/Down
            // pitch around camera right. World-up yaw collapsed onto pitch
            // after a side-face snap (NED E/W): camera right became Bevy Y.
            let (step_axis_world, signed_angle, _) = arrow_camera_axis_angle(
                *arrow,
                angle,
                base_right_world,
                base_forward_world,
                base_up_world,
            );
            let step_rotation_world = Quat::from_axis_angle(*step_axis_world, signed_angle);

            // Orbit the live camera around the focus object (the viewport
            // `look_at`) rather than the screen-center view axis: rotating about
            // the latter pushed an off-center module out of frame on Left/Right.
            // With no resolvable focus, pivot around the on-axis anchor so the
            // behavior degrades to the previous in-place spin.
            let parent_inv = parent_rotation.inverse();
            let new_world_rotation = step_rotation_world * (parent_rotation * base_rotation);
            let new_rotation_local = parent_inv * new_world_rotation;

            let focus_world = view_cube_orbit_target(
                entity,
                &lookup.viewports,
                lookup.entity_map.as_ref(),
                &lookup.values,
                &lookup.geo_context,
                &mut lookup.orbit_cache,
            )
            .map(|target| (target - origin_world).as_vec3())
            .unwrap_or_else(|| {
                camera_pose.translation
                    + base_forward_world * (editor_cam.last_anchor_depth.abs() as f32)
            });

            // Yaw/pitch orbit the focus object so it stays framed; roll is about
            // the view axis and must spin in place (pivot on the camera itself),
            // otherwise an off-center focus would drift the camera on roll.
            let camera_world = camera_pose.translation;
            let pivot_world = match *arrow {
                RotationArrow::RollLeft | RotationArrow::RollRight => camera_world,
                _ => focus_world,
            };
            let new_camera_world = pivot_world + step_rotation_world * (camera_world - pivot_world);

            transform.translation += parent_inv * (new_camera_world - camera_world);
            transform.rotation = new_rotation_local;

            // Record the step for chaining, and cancel any in-flight snap
            // animation by re-targeting the orientation we just applied (a no-op
            // move for LookTo on the next frame).
            lookup.arrow_cache.set_target(
                entity,
                new_rotation_local,
                now_secs,
                ArrowTargetSource::ArrowStep,
            );
            let facing_local = new_rotation_local * Vec3::NEG_Z;
            let up_local = new_rotation_local * Vec3::Y;
            if let (Ok(facing), Ok(up_dir)) = (Dir3::new(facing_local), Dir3::new(up_local)) {
                look_to.write(LookToTrigger {
                    target_facing_direction: facing.as_dvec3(),
                    target_up_direction: up_dir.as_dvec3(),
                    camera: entity,
                });
            }
        }

        if let ViewCubeEvent::ViewportActionClicked { action, .. } = event {
            match action {
                ViewportActionButton::Reset => {
                    apply_viewport_reset(transform.as_mut(), &mut editor_cam);
                    // Recompute the orbit depth from the viewport's look_at so a
                    // stale km-scale terrain pick depth can't survive the reset,
                    // then issue a LookTo to cancel any in-flight snap animation.
                    let reset_pose =
                        lookup.current_camera_pose(transform.as_ref(), parent, origin_world);
                    update_anchor_depth_for_view_cube(
                        entity,
                        reset_pose.translation,
                        reset_pose.rotation,
                        &mut editor_cam,
                        &lookup.viewports,
                        lookup.entity_map.as_ref(),
                        &lookup.values,
                        &lookup.geo_context,
                        &mut lookup.orbit_cache,
                        origin_world,
                    );
                    if let Ok(facing) = Dir3::new(Vec3::NEG_Z) {
                        look_to.write(LookToTrigger::auto_snap_up_direction(
                            facing.as_dvec3(),
                            entity,
                            &transform.rotation.as_dquat(),
                            &editor_cam,
                        ));
                    }
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
            trigger.target_facing_direction.as_vec3(),
            trigger.target_up_direction.as_vec3(),
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
        (current_depth * VIEWPORT_ZOOM_STEP).max(0.5)
    } else {
        (current_depth / VIEWPORT_ZOOM_STEP).max(0.5)
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
    frame: GeoFrame,
    geo: &GeoContext,
    config: &ViewCubeConfig,
) -> Vec3 {
    let local_dir = direction.to_look_direction();
    direction_target_camera_dir_world(local_dir, frame, geo, config)
}

#[cfg(test)]
fn corner_target_camera_dir_world(
    position: super::components::CornerPosition,
    frame: GeoFrame,
    geo: &GeoContext,
    config: &ViewCubeConfig,
) -> Vec3 {
    let local_dir = position.to_look_direction();
    direction_target_camera_dir_world(local_dir, frame, geo, config)
}

fn direction_target_camera_dir_world(
    local_dir: Vec3,
    frame: GeoFrame,
    geo: &GeoContext,
    config: &ViewCubeConfig,
) -> Vec3 {
    let local = if config.sync_with_camera {
        config.axis_correction * local_dir
    } else {
        local_dir
    };
    frame_dir_to_bevy(frame, local, geo)
}

pub(super) fn camera_dir_in_cube_local(
    cube_world_rotation: Quat,
    camera_world_rotation: Quat,
) -> Vec3 {
    cube_world_rotation.inverse() * (camera_world_rotation * Vec3::Z)
}

fn arrow_camera_axis_angle(
    arrow: RotationArrow,
    angle: f32,
    camera_right_world: Vec3,
    camera_forward_world: Vec3,
    camera_up_world: Vec3,
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

/// Frame-local "sky": ENU/ECEF +Z, NED −Z (up). Mapped through the same `to_bevy`
/// the look direction uses, so ECEF snaps stay on the equatorial plane.
fn frame_camera_up_bevy(frame: GeoFrame, geo: &GeoContext) -> Vec3 {
    let local = match frame {
        GeoFrame::NED => Vec3::NEG_Z,
        GeoFrame::ENU | GeoFrame::ECEF => Vec3::Z,
    };
    frame_dir_to_bevy(frame, local, geo)
}

fn choose_face_upright_up(
    parent_rotation: Quat,
    facing_local: Dir3,
    frame: GeoFrame,
    geo: &GeoContext,
) -> Option<Dir3> {
    let parent_inverse = parent_rotation.inverse();
    // Frame up first (ECEF Z, not Bevy Y / local vertical). Then Bevy axes
    // when looking along that up. FaceDirection names are Bevy-era.
    let facing = *facing_local;
    let frame_up = frame_camera_up_bevy(frame, geo);
    for world_up in [frame_up, Vec3::Y, Vec3::Z, Vec3::X] {
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

/// Orbit radius for view-cube snaps. Prefer the camera-to-`look_at` distance so
/// stale mesh-pick depths (e.g. km-scale terrain under the Apollo landing tile)
/// cannot hijack `LookTo` rotation.
fn resolve_view_cube_orbit_distance(
    to_target: Vec3,
    camera_rotation: Quat,
    previous_distance: f32,
) -> f32 {
    const MIN_ALIGNMENT_FOR_PROJECTED: f32 = 0.65;
    const MIN_ORBIT_DISTANCE: f32 = 0.25;
    const STALE_DEPTH_RATIO: f32 = 4.0;

    let forward = camera_rotation * Vec3::NEG_Z;
    let projected_distance = to_target.dot(forward);
    let measured_distance = to_target.length();
    let alignment_ratio = if measured_distance.is_finite() && measured_distance > 1.0e-3 {
        (projected_distance / measured_distance).clamp(-1.0, 1.0)
    } else {
        f32::NAN
    };

    let distance = if measured_distance.is_finite() && measured_distance > MIN_ORBIT_DISTANCE {
        measured_distance
    } else if projected_distance.is_finite()
        && measured_distance.is_finite()
        && projected_distance > 1.0e-3
        && measured_distance > 1.0e-3
        && alignment_ratio >= MIN_ALIGNMENT_FOR_PROJECTED
    {
        projected_distance
    } else if previous_distance > MIN_ORBIT_DISTANCE
        && (!measured_distance.is_finite()
            || measured_distance <= MIN_ORBIT_DISTANCE
            || previous_distance <= measured_distance * STALE_DEPTH_RATIO)
    {
        previous_distance
    } else if measured_distance.is_finite() && measured_distance > 1.0e-3 {
        measured_distance
    } else {
        1.0
    };

    distance.max(MIN_ORBIT_DISTANCE)
}

#[allow(clippy::too_many_arguments)]
fn update_anchor_depth_for_view_cube(
    camera: Entity,
    camera_translation: Vec3,
    camera_rotation: Quat,
    editor_cam: &mut EditorCam,
    viewports: &Query<&crate::ui::inspector::viewport::Viewport, With<ViewCubeTargetCamera>>,
    entity_map: &EntityMap,
    values: &Query<&'static ComponentValue>,
    geo_context: &GeoContext,
    orbit_cache: &mut ViewCubeOrbitTargetCache,
    origin_world: DVec3,
) {
    let Some(orbit_target_world) = view_cube_orbit_target(
        camera,
        viewports,
        entity_map,
        values,
        geo_context,
        orbit_cache,
    ) else {
        return;
    };
    let orbit_target = (orbit_target_world - origin_world).as_vec3();
    let to_target = orbit_target - camera_translation;
    let previous_distance = editor_cam.last_anchor_depth.abs() as f32;
    let distance = resolve_view_cube_orbit_distance(to_target, camera_rotation, previous_distance);
    editor_cam.last_anchor_depth = -(distance as f64);
}

#[allow(clippy::too_many_arguments)]
fn refresh_anchor_depth_for_arrow(
    camera: Entity,
    camera_translation: Vec3,
    editor_cam: &mut EditorCam,
    viewports: &Query<&crate::ui::inspector::viewport::Viewport, With<ViewCubeTargetCamera>>,
    entity_map: &EntityMap,
    values: &Query<&'static ComponentValue>,
    geo_context: &GeoContext,
    orbit_cache: &mut ViewCubeOrbitTargetCache,
    origin_world: DVec3,
) -> Option<(f32, f32)> {
    let orbit_target_world = view_cube_orbit_target(
        camera,
        viewports,
        entity_map,
        values,
        geo_context,
        orbit_cache,
    )?;
    let orbit_target = (orbit_target_world - origin_world).as_vec3();
    let measured_distance = (orbit_target - camera_translation).length();
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
    geo_context: &GeoContext,
    orbit_cache: &mut ViewCubeOrbitTargetCache,
) -> Option<DVec3> {
    let viewport = viewports.get(camera).ok()?;
    let frame = viewport.frame.unwrap_or_default();
    // Same 3-vector / WorldPos rule as viewport follow (`POS_ECEF` is 3 elems).
    // Missing compiled expr = cleared (don't aim). Execute error = sample gap.
    let Some(compiled_expr) = viewport.look_at.compiled_expr.as_ref() else {
        return apply_orbit_look_at(orbit_cache, camera, OrbitLookAt::Cleared);
    };
    match compiled_expr.execute(entity_map, values) {
        Err(_) => apply_orbit_look_at(orbit_cache, camera, OrbitLookAt::Gap),
        Ok(val) => match crate::ui::gauges::component_value_to_position(&val) {
            Some(pos) => {
                let target = GeoPosition(frame, pos).to_bevy(geo_context);
                apply_orbit_look_at(orbit_cache, camera, OrbitLookAt::Target(target))
            }
            None => apply_orbit_look_at(orbit_cache, camera, OrbitLookAt::NotAPosition),
        },
    }
}

enum OrbitLookAt {
    Cleared,
    Gap,
    NotAPosition,
    Target(DVec3),
}

fn apply_orbit_look_at(
    cache: &mut ViewCubeOrbitTargetCache,
    camera: Entity,
    look_at: OrbitLookAt,
) -> Option<DVec3> {
    match look_at {
        OrbitLookAt::Cleared | OrbitLookAt::NotAPosition => {
            cache.forget(camera);
            None
        }
        OrbitLookAt::Gap => cache.last(camera),
        OrbitLookAt::Target(target) => Some(cache.remember(camera, target)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::render_layer_alloc::view_cube_render_layers;
    use crate::plugins::view_cube::config::CoordinateSystem;
    use bevy::asset::AssetPlugin;
    use bevy::world_serialization::{WorldAsset, WorldAssetRoot, WorldSerializationPlugin};
    use bevy_geo_frames::Present;

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

    fn upright_up(facing: Dir3, frame: GeoFrame, geo: &GeoContext) -> Dir3 {
        choose_face_upright_up(Quat::IDENTITY, facing, frame, geo).expect("upright up")
    }

    #[test]
    fn choose_face_upright_up_keeps_east_west_consistent() {
        let geo = GeoContext::default();
        let east_facing = Dir3::new(Vec3::NEG_X).expect("unit vector");
        let west_facing = Dir3::new(Vec3::X).expect("unit vector");
        let east_up = upright_up(east_facing, GeoFrame::ENU, &geo);
        let west_up = upright_up(west_facing, GeoFrame::ENU, &geo);
        assert!(east_up.dot(Vec3::Y) > 0.99, "east up should be +Y");
        assert!(west_up.dot(Vec3::Y) > 0.99, "west up should be +Y");
    }

    #[test]
    fn choose_face_upright_up_uses_north_when_looking_along_vertical() {
        let geo = GeoContext::default();
        let looking_up = upright_up(
            Dir3::new(Vec3::Y).expect("unit vector"),
            GeoFrame::ENU,
            &geo,
        );
        let looking_down = upright_up(
            Dir3::new(Vec3::NEG_Y).expect("unit vector"),
            GeoFrame::ENU,
            &geo,
        );
        assert!(
            looking_up.dot(Vec3::Z) > 0.99,
            "looking up should use +Z, got {:?}",
            *looking_up
        );
        assert!(
            looking_down.dot(Vec3::Z) > 0.99,
            "looking down should use +Z, got {:?}",
            *looking_down
        );
    }

    #[test]
    fn choose_face_upright_up_keeps_world_y_on_ned_east_snap() {
        let geo = mojave_geo(Present::Plane);
        let config = ViewCubeConfig::default();
        let look = face_target_camera_dir_world(
            crate::plugins::view_cube::FaceDirection::Up,
            GeoFrame::NED,
            &geo,
            &config,
        );
        let facing = Dir3::new(-look).expect("ned +Y facing");
        let up = upright_up(facing, GeoFrame::NED, &geo);
        assert!(
            up.dot(Vec3::Y) > 0.9,
            "NED E (cube +Y) must keep world Y up, got {:?}",
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
        let forward = Vec3::X;
        let camera_up = Vec3::Z;

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Left, angle, right, forward, camera_up);
        assert_eq!(*axis, camera_up);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "camera_up");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Right, angle, right, forward, camera_up);
        assert_eq!(*axis, camera_up);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "camera_up");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Up, angle, right, forward, camera_up);
        assert_eq!(*axis, right);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "camera_right");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::Down, angle, right, forward, camera_up);
        assert_eq!(*axis, right);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "camera_right");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::RollLeft, angle, right, forward, camera_up);
        assert_eq!(*axis, forward);
        assert_eq!(signed_angle, angle);
        assert_eq!(source, "camera_forward");

        let (axis, signed_angle, source) =
            arrow_camera_axis_angle(RotationArrow::RollRight, angle, right, forward, camera_up);
        assert_eq!(*axis, forward);
        assert_eq!(signed_angle, -angle);
        assert_eq!(source, "camera_forward");
    }

    #[test]
    fn side_face_snap_keeps_yaw_orthogonal_to_pitch() {
        let geo = mojave_geo(Present::Plane);
        let config = ViewCubeConfig::default();
        let face = crate::plugins::view_cube::FaceDirection::Up;
        let look = face_target_camera_dir_world(face, GeoFrame::NED, &geo, &config);
        let facing = Dir3::new(-look).expect("ned +Y facing");
        let up = upright_up(facing, GeoFrame::NED, &geo);
        let rotation = Transform::default().looking_to(*facing, *up).rotation;
        let right = rotation * Vec3::X;
        let camera_up = rotation * Vec3::Y;
        let forward = rotation * Vec3::NEG_Z;
        assert!(
            camera_up.dot(Vec3::Y) > 0.9,
            "NED E-face snap should keep camera up on world Y, got {camera_up:?}"
        );
        assert!(
            Vec3::Y.dot(right).abs() < 0.15,
            "NED E-face snap must not bank camera right onto world up, got {}",
            Vec3::Y.dot(right)
        );

        let (yaw, _, yaw_src) =
            arrow_camera_axis_angle(RotationArrow::Left, 0.2, right, forward, camera_up);
        let (pitch, _, pitch_src) =
            arrow_camera_axis_angle(RotationArrow::Up, 0.2, right, forward, camera_up);
        assert_eq!(yaw_src, "camera_up");
        assert_eq!(pitch_src, "camera_right");
        assert!(
            yaw.dot(*pitch).abs() < 0.15,
            "Left/Right must stay off the Up/Down axis after an E-face snap, got {}",
            yaw.dot(*pitch)
        );
    }

    #[test]
    fn corner_target_camera_dir_world_applies_axis_correction() {
        let corner = crate::plugins::view_cube::CornerPosition::TopFrontRight;
        let config = ViewCubeConfig::default();
        let geo = GeoContext::default();
        let world = corner_target_camera_dir_world(corner, GeoFrame::ENU, &geo, &config);
        let expected = frame_dir_to_bevy(GeoFrame::ENU, Vec3::new(1.0, 1.0, 1.0), &geo);
        assert!((world - expected).length() < 1.0e-5);
    }

    fn mojave_geo(present: Present) -> GeoContext {
        GeoContext::from(bevy_geo_frames::GeoOrigin::new_from_degrees(
            35.3506640, -117.80902, 589.2740,
        ))
        .with_present(present)
    }

    fn assert_click_matches_to_bevy(frame: GeoFrame, local: Vec3, geo: &GeoContext) {
        let config = ViewCubeConfig::default();
        let world = direction_target_camera_dir_world(local, frame, geo, &config);
        let expected = frame_dir_to_bevy(frame, local, geo);
        assert!(
            (world - expected).length() < 1.0e-5,
            "{frame:?} click {local:?} = {world:?}, expected {expected:?}"
        );
        assert!(
            world.length() > 0.5,
            "{frame:?} click produced a near-zero direction"
        );
    }

    #[test]
    fn face_click_plus_x_matches_to_bevy_in_every_frame_plane() {
        let geo = mojave_geo(Present::Plane);
        let local = Vec3::X;
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            assert_click_matches_to_bevy(frame, local, &geo);
        }
        let ecef = frame_dir_to_bevy(GeoFrame::ECEF, local, &geo);
        assert!(
            (ecef - local).length() > 0.25,
            "Mojave ECEF +X must not be raw Bevy +X, got {ecef:?}"
        );
        let enu = frame_dir_to_bevy(GeoFrame::ENU, local, &geo);
        assert!(
            (enu - Vec3::X).length() < 1.0e-5,
            "ENU +X (East) stays Bevy +X, got {enu:?}"
        );
    }

    #[test]
    fn face_click_plus_x_matches_to_bevy_in_every_frame_sphere() {
        let geo = mojave_geo(Present::Sphere);
        let local = Vec3::X;
        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            assert_click_matches_to_bevy(frame, local, &geo);
        }
    }

    #[test]
    fn ecef_plus_x_snap_keeps_equator_level_in_plane() {
        let origin = bevy_geo_frames::GeoOrigin::new_from_degrees(34.72, -86.64, 180.5);
        let geo = GeoContext::from(origin).with_present(Present::Plane);
        let config = ViewCubeConfig::default();
        let look = face_target_camera_dir_world(
            crate::plugins::view_cube::FaceDirection::East,
            GeoFrame::ECEF,
            &geo,
            &config,
        );
        let facing = Dir3::new(-look).expect("ecef +X facing");
        let up = upright_up(facing, GeoFrame::ECEF, &geo);
        let rotation = Transform::default().looking_to(*facing, *up).rotation;
        let right = rotation * Vec3::X;
        let ecef_z = frame_dir_to_bevy(GeoFrame::ECEF, Vec3::Z, &geo);
        assert!(
            right.dot(ecef_z).abs() < 0.05,
            "ECEF +X snap must keep ECEF Z in the screen vertical, got right·ecefZ={}",
            right.dot(ecef_z)
        );
        assert!(
            (rotation * Vec3::Y).dot(ecef_z) > 0.9,
            "ECEF +X snap must use ECEF Z as camera up, got {:?}",
            rotation * Vec3::Y
        );
    }

    #[test]
    fn orbit_target_uses_viewport_frame_not_default_enu() {
        let geo = mojave_geo(Present::Plane);
        let ecef = DVec3::new(1.0, 2.0, 3.0);
        let as_ecef = GeoPosition(GeoFrame::ECEF, ecef).to_bevy(&geo);
        let as_enu = GeoPosition(GeoFrame::ENU, ecef).to_bevy(&geo);
        assert!(
            (as_ecef - as_enu).length() > 1.0e3,
            "ECEF vs ENU interpretation of the same metres must diverge at Mojave"
        );
    }

    #[test]
    fn orbit_cache_keeps_last_target_on_gap() {
        let mut cache = ViewCubeOrbitTargetCache::default();
        let camera = Entity::from_bits(9);
        let target = DVec3::new(10.0, 20.0, 30.0);
        apply_orbit_look_at(&mut cache, camera, OrbitLookAt::Target(target));
        assert_eq!(
            apply_orbit_look_at(&mut cache, camera, OrbitLookAt::Gap),
            Some(target)
        );
        assert_eq!(cache.last(Entity::from_bits(10)), None);
    }

    #[test]
    fn orbit_cache_forgets_when_look_at_cleared() {
        let mut cache = ViewCubeOrbitTargetCache::default();
        let camera = Entity::from_bits(9);
        apply_orbit_look_at(
            &mut cache,
            camera,
            OrbitLookAt::Target(DVec3::new(10.0, 20.0, 30.0)),
        );
        assert_eq!(
            apply_orbit_look_at(&mut cache, camera, OrbitLookAt::Cleared),
            None
        );
        assert_eq!(cache.last(camera), None);
        assert_eq!(
            apply_orbit_look_at(&mut cache, camera, OrbitLookAt::NotAPosition),
            None
        );
    }

    #[test]
    fn view_cube_scene_descendants_are_layered_before_scene_is_revealed() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<WorldInstanceSpawner>();
        app.add_systems(Update, apply_render_layers_to_scene);

        let expected_layers = view_cube_render_layers(bevy_geo_frames::GeoFrame::ENU);
        let default_layers = RenderLayers::layer(0);

        let root = app
            .world_mut()
            .spawn((ViewCubeRoot, Visibility::Hidden, expected_layers.clone()))
            .id();
        let child = app
            .world_mut()
            .spawn((ChildOf(root), default_layers.clone()))
            .id();
        let grandchild = app
            .world_mut()
            .spawn((ChildOf(child), default_layers.clone()))
            .id();

        app.update();

        let child_layers = app
            .world()
            .get::<RenderLayers>(child)
            .expect("child layers");
        let grandchild_layers = app
            .world()
            .get::<RenderLayers>(grandchild)
            .expect("grandchild layers");
        assert_eq!(child_layers, &expected_layers);
        assert_eq!(grandchild_layers, &expected_layers);
        assert!(
            !child_layers.intersects(&RenderLayers::layer(0)),
            "child should no longer render on the default main-camera layer",
        );
        assert_eq!(
            app.world().get::<Visibility>(root),
            Some(&Visibility::Hidden),
            "manual children are not enough to reveal the GLB root before the scene is ready",
        );
    }

    /// Per-viewport labels hang under the shared cube, so the cube's layer must
    /// not be forced onto them — otherwise every viewport draws every copy.
    #[test]
    fn per_viewport_face_label_subtrees_keep_their_own_render_layers() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.init_resource::<WorldInstanceSpawner>();
        app.add_systems(Update, apply_render_layers_to_scene);

        let frame_layers = view_cube_render_layers(GeoFrame::ECEF);
        let viewport_a = RenderLayers::layer(24);
        let viewport_b = RenderLayers::layer(25);

        let root = app
            .world_mut()
            .spawn((ViewCubeRoot, Visibility::Hidden, frame_layers.clone()))
            .id();
        let label_a = app
            .world_mut()
            .spawn((ChildOf(root), KeepsRenderLayers, viewport_a.clone()))
            .id();
        let glyph_a = app
            .world_mut()
            .spawn((ChildOf(label_a), RenderLayers::layer(0)))
            .id();
        let label_b = app
            .world_mut()
            .spawn((ChildOf(root), KeepsRenderLayers, viewport_b.clone()))
            .id();
        let glyph_b = app
            .world_mut()
            .spawn((ChildOf(label_b), RenderLayers::layer(0)))
            .id();
        let shared_axis = app
            .world_mut()
            .spawn((ChildOf(root), RenderLayers::layer(0)))
            .id();

        app.update();

        assert_eq!(app.world().get::<RenderLayers>(label_a), Some(&viewport_a));
        assert_eq!(app.world().get::<RenderLayers>(label_b), Some(&viewport_b));
        assert_eq!(
            app.world().get::<RenderLayers>(glyph_a),
            Some(&viewport_a),
            "glyphs follow their own label, not the shared cube",
        );
        assert_eq!(app.world().get::<RenderLayers>(glyph_b), Some(&viewport_b));
        assert_eq!(
            app.world().get::<RenderLayers>(shared_axis),
            Some(&frame_layers),
            "cube parts that are not per-viewport still take the frame layer",
        );
    }

    /// Two viewports on the same frame share one cube, so each has to own its
    /// copy of the labels and spin it for its own camera.
    #[test]
    fn two_viewports_sharing_a_cube_orient_their_own_labels() {
        let geo = geo_frames_geo();
        let cube = ecef_cube_rotation(&geo);
        let label = ecef_face_label("+X");
        let equator = camera_looking(Vec3::new(0.0, -8.0, 0.0), Vec3::Z);
        let oblique = camera_looking(Vec3::new(4.6, -4.6, 4.6), Vec3::Z);

        let mut app = App::new();
        app.add_systems(Update, orient_face_labels_to_view);

        let cube_root = app
            .world_mut()
            .spawn((
                ViewCubeRoot,
                GlobalTransform::from(Transform::from_rotation(cube)),
            ))
            .id();

        let mut spawn_viewport_copy = |camera_rotation: Quat| {
            let camera = app
                .world_mut()
                .spawn((
                    ViewCubeCamera,
                    GlobalTransform::from(Transform::from_rotation(camera_rotation)),
                ))
                .id();
            app.world_mut()
                .spawn((
                    ChildOf(cube_root),
                    FaceLabel {
                        base_rotation: label.rotation,
                        camera,
                        last_angle: 0.0,
                        last_view: None,
                    },
                    Transform::from_rotation(label.rotation),
                ))
                .id()
        };
        let label_equator = spawn_viewport_copy(equator);
        let label_oblique = spawn_viewport_copy(oblique);

        app.update();

        for (entity, camera, name) in [
            (label_equator, equator, "equator"),
            (label_oblique, oblique, "oblique"),
        ] {
            let expected = face_label_in_plane_angle(label.rotation, cube, camera)
                .unwrap_or_else(|| panic!("+X should be visible from the {name} camera"));
            let solved = app.world().get::<FaceLabel>(entity).expect("label");
            assert!(
                (solved.last_angle - expected).abs() <= FACE_LABEL_ANGLE_EPS,
                "{name} copy should be solved for its own camera, \
                 got {} expected {expected}",
                solved.last_angle,
            );
            let applied = app.world().get::<Transform>(entity).expect("transform");
            let wanted = label.rotation * Quat::from_rotation_z(expected);
            assert!(
                applied.rotation.abs_diff_eq(wanted, 1.0e-5),
                "{name} copy should carry its spin: applied={:?} wanted={wanted:?}",
                applied.rotation,
            );
        }

        let angle_equator = app
            .world()
            .get::<FaceLabel>(label_equator)
            .expect("label")
            .last_angle;
        let angle_oblique = app
            .world()
            .get::<FaceLabel>(label_oblique)
            .expect("label")
            .last_angle;
        assert!(
            (angle_equator - angle_oblique).abs() > 1.0e-3,
            "the two viewports look from different angles, so their labels \
             must not end up sharing one orientation",
        );
    }

    #[test]
    fn view_cube_scene_root_is_revealed_after_scene_instance_is_ready() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            WorldSerializationPlugin,
        ));
        app.add_systems(Update, apply_render_layers_to_scene);

        let render_layers = view_cube_render_layers(bevy_geo_frames::GeoFrame::ENU);
        let default_layers = RenderLayers::layer(0);
        let scene_handle = app
            .world_mut()
            .resource_mut::<Assets<WorldAsset>>()
            .add(WorldAsset::new(World::new()));

        let root = app
            .world_mut()
            .spawn((
                WorldAssetRoot(scene_handle),
                ViewCubeRoot,
                Visibility::Hidden,
                render_layers,
            ))
            .id();
        let child = app
            .world_mut()
            .spawn((ChildOf(root), default_layers.clone()))
            .id();

        app.update();
        app.update();

        let child_layers = app
            .world()
            .get::<RenderLayers>(child)
            .expect("child layers");
        assert!(
            !child_layers.intersects(&RenderLayers::layer(0)),
            "child should no longer render on the default main-camera layer",
        );
        assert_eq!(
            app.world().get::<Visibility>(root),
            Some(&Visibility::Inherited),
            "root should become visible once the scene instance is ready",
        );
    }

    #[test]
    fn resolve_view_cube_orbit_distance_prefers_look_at_over_stale_terrain_pick() {
        let to_target = Vec3::new(-10.0, -4.0, 10.0);
        let camera_rotation = Quat::from_rotation_y(1.2);
        let distance = resolve_view_cube_orbit_distance(to_target, camera_rotation, 12_000.0);
        let expected = to_target.length();
        assert!(
            (distance - expected).abs() < 1.0e-4,
            "expected orbit radius {expected}, got {distance}"
        );
        assert!(distance < 100.0, "stale km-scale pick depth leaked through");
    }

    #[test]
    fn resolve_view_cube_orbit_distance_uses_previous_when_target_not_measurable() {
        let to_target = Vec3::ZERO;
        let camera_rotation = Quat::IDENTITY;
        let distance = resolve_view_cube_orbit_distance(to_target, camera_rotation, 5.0);
        assert!((distance - 5.0).abs() < 1.0e-6);
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
        let expected_target_depth = 2.0 * VIEWPORT_ZOOM_STEP;
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

    /// The `+` and `−` clicks have to cancel each other, otherwise zoom-in reads
    /// as broken next to a much coarser zoom-out.
    #[test]
    fn viewport_zoom_in_undoes_zoom_out() {
        let start = Transform::from_translation(Vec3::new(0.5, 1.0, -0.25))
            .with_rotation(Quat::from_rotation_y(0.3));
        let mut transform = start;
        let mut editor_cam = EditorCam {
            last_anchor_depth: -4.0,
            ..Default::default()
        };

        apply_viewport_zoom(true, &mut transform, &mut editor_cam);
        apply_viewport_zoom(false, &mut transform, &mut editor_cam);

        assert!((transform.translation - start.translation).length() < 1.0e-5);
        assert!((editor_cam.last_anchor_depth + 4.0).abs() < 1.0e-5);
    }

    fn ecef_face_label(text: &str) -> crate::plugins::view_cube::config::FaceLabelConfig {
        CoordinateSystem(GeoFrame::ECEF)
            .get_face_labels(1.0)
            .into_iter()
            .find(|label| label.text == text)
            .unwrap_or_else(|| panic!("missing ECEF face label {text}"))
    }

    fn camera_looking(from: Vec3, up: Vec3) -> Quat {
        Transform::from_translation(from)
            .looking_at(Vec3::ZERO, up)
            .rotation
    }

    fn geo_frames_geo() -> GeoContext {
        GeoContext::from(bevy_geo_frames::GeoOrigin::new_from_degrees(
            34.72, -86.64, 180.5,
        ))
        .with_present(Present::Plane)
    }

    /// Real ECEF cube pose: the root carries `GeoRotation::absolute(ECEF, I)`.
    /// Testing against `Quat::IDENTITY` instead hides the oblique faces entirely.
    fn ecef_cube_rotation(geo: &GeoContext) -> Quat {
        GeoRotation::absolute(GeoFrame::ECEF, bevy::math::DQuat::IDENTITY)
            .to_bevy(geo)
            .as_quat()
    }

    fn assert_reads_horizontally(text: &str, cube: Quat, camera: Quat) {
        let label = ecef_face_label(text);
        let angle = face_label_in_plane_angle(label.rotation, cube, camera)
            .unwrap_or_else(|| panic!("{text} should not be edge-on for this pose"));
        let (baseline, _) = face_label_screen_axes(label.rotation, cube, camera, angle);
        assert!(
            baseline.y.abs() <= 1.0e-4 * baseline.length().max(1.0e-6) + 1.0e-5,
            "{text} should read horizontally, screen baseline={baseline:?}"
        );
        assert!(
            baseline.x > 0.0,
            "{text} should run left to right, screen baseline={baseline:?}"
        );
    }

    /// The ECEF cube carries `bevy_R_ecef`, so its faces are viewed obliquely and
    /// their in-plane axes never line up with the screen. Every visible face has
    /// to read horizontally anyway.
    #[test]
    fn ecef_visible_face_labels_read_horizontally_over_a_camera_sweep() {
        let geo = geo_frames_geo();
        let cube = ecef_cube_rotation(&geo);
        let labels = CoordinateSystem(GeoFrame::ECEF).get_face_labels(1.0);

        for yaw_deg in (0..360).step_by(5) {
            for pitch_deg in (-85..=85).step_by(5) {
                let camera = Quat::from_euler(
                    EulerRot::YXZ,
                    (yaw_deg as f32).to_radians(),
                    (pitch_deg as f32).to_radians(),
                    0.0,
                );
                for label in &labels {
                    let normal_cam = camera.inverse() * cube * label.position.normalize_or_zero();
                    // Only faces actually turned toward the viewer must be readable.
                    if normal_cam.z < 0.2 {
                        continue;
                    }
                    let angle = face_label_in_plane_angle(label.rotation, cube, camera)
                        .unwrap_or_else(|| {
                            panic!(
                                "{} is visible (n.z={:.3}) at yaw={yaw_deg} pitch={pitch_deg} but was treated as edge-on",
                                label.text, normal_cam.z
                            )
                        });
                    let (baseline, _) = face_label_screen_axes(label.rotation, cube, camera, angle);
                    assert!(
                        baseline.y.abs() <= 1.0e-4 * baseline.length() + 1.0e-5,
                        "{} reads sideways at yaw={yaw_deg} pitch={pitch_deg}: baseline={baseline:?}",
                        label.text
                    );
                    assert!(
                        baseline.x > 0.0,
                        "{} runs right to left at yaw={yaw_deg} pitch={pitch_deg}: baseline={baseline:?}",
                        label.text
                    );
                }
            }
        }
    }

    /// Regression: a 90°-quantized roll leaves this oblique `+X` face ~63° off
    /// horizontal, because none of its four quarter turns is horizontal.
    #[test]
    fn ecef_oblique_side_face_is_horizontal_where_quantized_rolls_fail() {
        let geo = geo_frames_geo();
        let cube = ecef_cube_rotation(&geo);
        let camera = Quat::from_euler(
            EulerRot::YXZ,
            150.0_f32.to_radians(),
            30.0_f32.to_radians(),
            0.0,
        );
        assert_reads_horizontally("+X", cube, camera);
    }

    #[test]
    fn ecef_plus_z_label_is_untouched_when_face_on() {
        let camera = camera_looking(Vec3::Z, Vec3::Y);
        let label = ecef_face_label("+Z");
        let angle = face_label_in_plane_angle(label.rotation, Quat::IDENTITY, camera)
            .expect("+Z is face-on, not edge-on");
        assert!(
            angle.abs() < 1.0e-5,
            "a face-on label is already horizontal and must not spin, got {angle}"
        );
    }

    #[test]
    fn ecef_plus_z_label_flips_when_camera_is_upside_down() {
        let camera = camera_looking(Vec3::Z, Vec3::NEG_Y);
        let label = ecef_face_label("+Z");
        let angle = face_label_in_plane_angle(label.rotation, Quat::IDENTITY, camera)
            .expect("+Z stays face-on when the camera rolls");
        assert!(
            (angle.abs() - PI).abs() < 1.0e-5,
            "upside-down view should spin the label 180°, got {angle}"
        );
        assert_reads_horizontally("+Z", Quat::IDENTITY, camera);
    }

    #[test]
    fn face_label_has_no_angle_when_face_is_edge_on() {
        // Camera in the +X face plane, screen-up along the face: the whole face
        // projects to a vertical line, so no angle can be horizontal.
        let label = ecef_face_label("+X");
        let camera = camera_looking(Vec3::Y, Vec3::Z);
        assert!(
            face_label_in_plane_angle(label.rotation, Quat::IDENTITY, camera).is_none(),
            "an edge-on face has no meaningful in-plane angle"
        );
    }
}
