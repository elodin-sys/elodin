use bevy::camera::RenderTarget;
use bevy::camera::visibility::RenderLayers;
use bevy::picking::prelude::Pickable;
use bevy::ui::{Node, PositionType, UiTargetCamera, Val, ZIndex};
use bevy::window::WindowRef;
use bevy::{
    app::{App, Plugin, PostUpdate, Startup, Update},
    ecs::system::{Query, Res, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore, GizmoLineJoint},
        gizmos::Gizmos,
    },
    log::warn,
    math::DVec3,
    prelude::*,
    text::{TextColor, TextFont},
    transform::components::Transform,
};
use bevy_geo_frames::prelude::*;
use bevy_render::alpha::AlphaMode;
use impeller2::types::ComponentId;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{
    BodyAxes, Color as WktColor, ComponentValue as WktComponentValue, LabelPosition, VectorArrow3d,
    WorldPos,
};
use std::collections::{HashMap, HashSet};

use crate::plugins::render_layer_alloc::RenderLayerLease;
#[cfg(feature = "big_space")]
use crate::spatial::GridCell;
use crate::{
    MainCamera, WorldPosExt,
    object_3d::ComponentArrayExt,
    ui::tiles::ViewportConfig,
    ui::window::window_entity_from_target,
    vector_arrow::{
        ArrowEndpoint, ArrowEndpoints, ArrowLabelScope, ArrowVisual, CachedArrowPose,
        VectorArrowState, ViewportArrow, component_value_tail_to_vec3,
    },
};

type ArrowLabelCameraItem<'w> = (
    Entity,
    &'w Camera,
    &'w RenderTarget,
    &'w GlobalTransform,
    Option<&'w ViewportConfig>,
);

type MainCameraQueryItem<'w> = (
    Entity,
    &'w Camera,
    &'w Projection,
    &'w GlobalTransform,
    Option<&'w ViewportConfig>,
    Option<&'w RenderLayerLease>,
);

/// Marker for UI cameras spawned specifically for arrow labels per window.
#[derive(Component)]
pub struct ArrowLabelUiCamera;

/// Marker component for Bevy UI arrow labels
#[derive(Component)]
pub struct ArrowLabelUI;

pub const GIZMO_RENDER_LAYER: usize = 30;
pub(crate) const MIN_ARROW_LENGTH_SQUARED: f64 = 1.0e-6;
/// Camera order for arrow label UI cameras. Must be higher than all viewport cameras
/// (SECONDARY_GRAPH_ORDER_BASE=1000, stride=50 per window) to avoid order collisions.
const ARROW_LABEL_UI_CAMERA_ORDER: isize = 100_000;
const HEAD_LENGTH_FRAC: f32 = 0.32;
const HEAD_RADIUS_FACTOR: f32 = 2.2;
const MAX_HEAD_PORTION: f32 = 0.50;
const MIN_HEAD_LENGTH_PX: f32 = 14.0;
const MIN_SHAFT_LENGTH_PX: f32 = 4.0;
const DRAW_RAW_ARROW_MESHES: bool = true;
const TARGET_DIAMETER_PX: f32 = 7.0;
const MIN_VISIBLE_DIAMETER_PX: f32 = 2.0;
const MAX_RADIUS_FRAC: f32 = 0.05;

pub struct GizmoPlugin;

impl Plugin for GizmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (gizmo_setup, arrow_mesh_setup));
        // Evaluate arrow expressions into endpoint `WorldPos` before the
        // canonical position pipeline converts them to Transforms.
        app.add_systems(
            bevy::app::PreUpdate,
            evaluate_vector_arrows
                .after(impeller2_bevy::apply_cached_data)
                .before(crate::sync_pos),
        );
        app.add_systems(
            PostUpdate,
            render_vector_arrow.after(bevy::transform::TransformSystems::Propagate),
        );
        app.add_systems(bevy::app::PostUpdate, cleanup_removed_arrows);
        app.add_systems(Update, render_body_axis);
        // Bevy UI labels for arrows - runs in PostUpdate after transforms are finalized
        app.add_systems(
            PostUpdate,
            update_arrow_label_ui.after(bevy::transform::TransformSystems::Propagate),
        );
    }
}

fn arrow_head_and_shaft_lengths(draw_length: f32, world_per_px: f32) -> (f32, f32) {
    let mut head_length = (draw_length * HEAD_LENGTH_FRAC)
        .max(world_per_px * MIN_HEAD_LENGTH_PX)
        .min(draw_length * MAX_HEAD_PORTION)
        .min(draw_length);
    let min_shaft_length = world_per_px * MIN_SHAFT_LENGTH_PX;
    if draw_length - head_length < min_shaft_length {
        head_length = (draw_length - min_shaft_length).max(0.0);
    }
    let shaft_length = (draw_length - head_length).max(0.0);
    (head_length, shaft_length)
}

/// Pixel-based shaft radius with a length-fraction cap when zoomed in. When zoomed
/// out, the length-fraction cap can fall below the screen-minimum diameter — in that
/// case screen visibility wins so arrows stay visible at planetary scale.
fn compute_shaft_radius(draw_length: f32, world_per_px: f32, thickness: f32) -> f32 {
    let desired_radius = TARGET_DIAMETER_PX * 0.5 * world_per_px * thickness;
    let screen_min_radius = MIN_VISIBLE_DIAMETER_PX * 0.5 * world_per_px;
    let max_radius = draw_length * MAX_RADIUS_FRAC;

    if max_radius < screen_min_radius {
        screen_min_radius
    } else {
        desired_radius.clamp(screen_min_radius, max_radius)
    }
}

fn world_units_per_pixel(
    camera: &Camera,
    projection: &Projection,
    camera_transform: &GlobalTransform,
    world_pos: Vec3,
) -> f32 {
    let viewport = camera
        .physical_viewport_size()
        .unwrap_or_else(|| UVec2::new(1920, 1080));
    let px_height = viewport.y.max(1) as f32;

    match projection {
        Projection::Perspective(persp) => {
            let view = camera_transform.to_matrix().inverse();
            let cam_space = view.transform_point3(world_pos);
            let depth = (-cam_space.z).max(0.001);
            let focal_px = px_height / (2.0 * (persp.fov * 0.5).tan());
            (depth / focal_px).max(0.001)
        }
        Projection::Orthographic(ortho) => {
            // In ortho, world units map linearly to pixels via scale.
            ((2.0 * ortho.scale) / px_height).max(0.001)
        }
        Projection::Custom(_) => {
            // Fallback: assume roughly perspective-like scaling using near plane as depth proxy.
            0.001
        }
    }
}

fn gizmo_setup(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.line.width = 5.0;
    config.line.joints = GizmoLineJoint::Round(12);
    config.enabled = true;
    config.render_layers = RenderLayers::layer(GIZMO_RENDER_LAYER);
}

#[derive(Resource, Clone)]
struct ArrowMeshes {
    shaft: Handle<Mesh>,
    head: Handle<Mesh>,
}

#[derive(Component)]
struct ArrowVisualOwner {
    owner: Entity,
}

fn arrow_mesh_setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let shaft = meshes.add(Mesh::from(Cylinder::new(1.0, 1.0)));
    let head = meshes.add(Mesh::from(Cone::new(1.0, 1.0)));
    commands.insert_resource(ArrowMeshes { shaft, head });
}

/// Evaluate a vector arrow's expressions entirely in simulation coordinates,
/// returning the start and end poses as `WorldPos` (same frame as the arrow).
fn evaluate_arrow_endpoints(
    arrow: &VectorArrow3d,
    state: &VectorArrowState,
    entity_map: &EntityMap,
    component_values: &Query<'_, '_, &'static WktComponentValue>,
) -> Option<(WorldPos, WorldPos)> {
    let vector_expr = state.vector_expr.as_ref()?;
    let vector_value = vector_expr.execute(entity_map, component_values).ok()?;

    let mut direction = component_value_tail_to_vec3(&vector_value)?;

    if direction.length_squared() <= MIN_ARROW_LENGTH_SQUARED {
        return None;
    }

    if arrow.normalize {
        direction = direction.normalize();
    }

    direction *= arrow.scale;
    if direction.length_squared() <= MIN_ARROW_LENGTH_SQUARED {
        return None;
    }

    let mut start = WorldPos::default();
    if let Some(origin_expr) = &state.origin_expr {
        let origin_value = origin_expr.execute(entity_map, component_values).ok()?;
        if let Some(world_pos) = origin_value.as_world_pos() {
            start = world_pos;
        } else if let Some(origin) = component_value_tail_to_vec3(&origin_value) {
            start.pos = nox::Vector3::new(origin.x, origin.y, origin.z);
        }
    }

    Some((start, arrow_end_pos(&start, direction, arrow.body_frame)))
}

/// Compute the arrow's end pose in simulation coordinates: body-frame vectors
/// are rotated by the body's attitude, world-frame vectors are added as-is.
fn arrow_end_pos(start: &WorldPos, mut direction: DVec3, body_frame: bool) -> WorldPos {
    if body_frame {
        direction = start.att() * direction;
    }
    let end_pos = start.pos() + direction;
    WorldPos {
        att: start.att,
        pos: nox::Vector3::new(end_pos.x, end_pos.y, end_pos.z),
    }
}

fn spawn_arrow_endpoint(
    commands: &mut Commands,
    owner: Entity,
    world_pos: WorldPos,
    frame: Option<GeoFrame>,
    name: &'static str,
) -> Entity {
    let mut endpoint = commands.spawn((
        world_pos,
        Transform::default(),
        #[cfg(feature = "big_space")]
        GridCell::default(),
        ArrowEndpoint { owner },
        Name::new(name),
    ));
    if let Some(frame) = frame {
        endpoint.insert((
            GeoPosition(frame, DVec3::ZERO),
            GeoRotation::new(frame, bevy::math::DQuat::IDENTITY),
        ));
    }
    endpoint.id()
}

/// Evaluate arrow expressions and write the results into the arrow's two
/// endpoint entities as `WorldPos`. The canonical position pipeline
/// (`sync_pos` -> Geo* -> Transform/GridCell) then places the endpoints in
/// Bevy space; `render_vector_arrow` only reads the resulting poses.
pub fn evaluate_vector_arrows(
    mut commands: Commands,
    entity_map: Res<EntityMap>,
    component_values: Query<&'static WktComponentValue>,
    mut arrows: Query<(
        Entity,
        &VectorArrow3d,
        &mut VectorArrowState,
        Option<&ArrowEndpoints>,
    )>,
    mut world_pos: Query<&mut WorldPos>,
    mut logged_missing: Local<HashSet<Entity>>,
) {
    for (entity, arrow, mut state, endpoints) in arrows.iter_mut() {
        let Some((start, end)) =
            evaluate_arrow_endpoints(arrow, &state, &entity_map, &component_values)
        else {
            if logged_missing.insert(entity) {
                info!(
                    ?entity,
                    name = ?arrow.name,
                    "vector_arrow: evaluation failed (missing data or zero-length)"
                );
            }
            state.valid = false;
            continue;
        };
        state.valid = true;

        if let Some(endpoints) = endpoints {
            if let Ok(mut pos) = world_pos.get_mut(endpoints.start) {
                pos.set_if_neq(start);
            }
            if let Ok(mut pos) = world_pos.get_mut(endpoints.end) {
                pos.set_if_neq(end);
            }
        } else {
            // Use ENU if no frame is specified.
            let frame = arrow.frame.or_default();
            let start = spawn_arrow_endpoint(&mut commands, entity, start, frame, "arrow_start");
            let end = spawn_arrow_endpoint(&mut commands, entity, end, frame, "arrow_end");
            commands
                .entity(entity)
                .insert(ArrowEndpoints { start, end });
        }
    }
}

fn resolve_arrow_pose(
    endpoints: &ArrowEndpoints,
    endpoint_gt: &Query<&GlobalTransform, With<ArrowEndpoint>>,
) -> Option<CachedArrowPose> {
    let Ok([start_gt, end_gt]) = endpoint_gt.get_many([endpoints.start, endpoints.end]) else {
        return None;
    };
    let direction_world = end_gt.translation() - start_gt.translation();
    if direction_world.length_squared() <= MIN_ARROW_LENGTH_SQUARED as f32 {
        return None;
    }
    let dir_local = start_gt
        .affine()
        .inverse()
        .transform_vector3(direction_world);
    let local_rotation = rotation_y_to(dir_local);
    Some(CachedArrowPose {
        direction_world,
        local_rotation,
    })
}

fn rotation_y_to(direction: Vec3) -> Quat {
    let len_sq = direction.length_squared();
    if len_sq <= MIN_ARROW_LENGTH_SQUARED as f32 {
        return Quat::IDENTITY;
    }
    Quat::from_rotation_arc(Vec3::Y, direction / len_sq.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn render_vector_arrow(
    mut commands: Commands,
    mut vector_arrows: Query<(
        Entity,
        &VectorArrow3d,
        &mut VectorArrowState,
        Option<&ArrowEndpoints>,
    )>,
    endpoint_gt: Query<&GlobalTransform, With<ArrowEndpoint>>,
    arrow_meshes: Res<ArrowMeshes>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    main_cameras: Query<MainCameraQueryItem<'_>, With<crate::MainCamera>>,
    viewport_arrows: Query<&ViewportArrow>,
    mut logged_small: Local<HashSet<Entity>>,
) {
    let main_camera_data: Vec<_> = main_cameras.iter().collect();
    let mut camera_index: HashMap<Entity, usize> = HashMap::new();
    for (idx, (entity, ..)) in main_camera_data.iter().enumerate() {
        camera_index.insert(*entity, idx);
    }
    let has_show_arrows = main_camera_data
        .iter()
        .any(|(_, _, _, _, config, _)| config.as_ref().map(|c| c.show_arrows).unwrap_or(true));

    for (entity, arrow, mut state, endpoints) in vector_arrows.iter_mut() {
        if !has_show_arrows {
            for visual in state.visuals.values() {
                despawn_arrow_visual(&mut commands, visual);
            }
            state.visuals.clear();
            state.cached_pose = None;
            continue;
        }

        let Some(endpoints) = endpoints else {
            continue;
        };

        if !state.valid {
            for visual in state.visuals.values() {
                despawn_arrow_visual(&mut commands, visual);
            }
            state.visuals.clear();
            state.cached_pose = None;
            continue;
        }

        let live_pose = resolve_arrow_pose(endpoints, &endpoint_gt);
        let pose = live_pose.clone().or_else(|| state.cached_pose.clone());
        if live_pose.is_some() {
            state.cached_pose = live_pose;
        }
        let Some(pose) = pose else {
            continue;
        };

        let direction_world = pose.direction_world;
        let local_rotation = pose.local_rotation;

        if !DRAW_RAW_ARROW_MESHES {
            for visual in state.visuals.values() {
                despawn_arrow_visual(&mut commands, visual);
            }
            state.visuals.clear();
            state.cached_pose = None;
            continue;
        }

        let length = direction_world.length();
        let dir_norm = direction_world / length;
        let base_color =
            axis_color_from_name(arrow.name.as_deref(), wkt_color_to_bevy(&arrow.color));

        // Keep a minimum draw length so a near-zero vector still shows a tiny arrow.
        let draw_length = length.max(0.05);
        if draw_length <= (MIN_ARROW_LENGTH_SQUARED as f32).sqrt() && logged_small.insert(entity) {
            info!(
                ?entity,
                name = ?arrow.name,
                length,
                "vector_arrow: very small magnitude, drawing minimum-sized arrow"
            );
        }

        // Skip this frame if the start endpoint transform is unavailable. The
        // visual is parented to the start endpoint, so a transient read failure
        // must not fall back to the world origin for screen-space sizing.
        let Ok(start_world) = endpoint_gt.get(endpoints.start).map(|gt| gt.translation()) else {
            continue;
        };

        let mut seen_cameras: HashSet<Entity> = HashSet::new();

        let mut render_for_camera = |idx: usize| {
            let (cam_entity, cam, proj, cam_tf, viewport_config, render_layer_lease) =
                main_camera_data[idx];
            seen_cameras.insert(cam_entity);

            let show_arrows = viewport_config
                .map(|config| config.show_arrows)
                .unwrap_or(true);
            if !show_arrows {
                if let Some(visual) = state.visuals.remove(&cam_entity) {
                    despawn_arrow_visual(&mut commands, &visual);
                }
                state.label_offset = None;
                state.label_name = None;
                state.label_color = None;
                if let Some(label_entity) = state.label.take() {
                    hide_label(&mut commands, Some(label_entity));
                }
                return;
            }

            // Arrows must stay isolated to the viewport-specific lease layer.
            // Using the camera's full RenderLayers mask would also copy shared
            // layers like 0 / gizmo / grid, causing cross-render between
            // otherwise independent viewports.
            let Some(arrow_layers) = render_layer_lease.map(RenderLayerLease::render_layers) else {
                if let Some(visual) = state.visuals.remove(&cam_entity) {
                    despawn_arrow_visual(&mut commands, &visual);
                }
                return;
            };

            let world_per_px = world_units_per_pixel(cam, proj, cam_tf, start_world);
            let (head_length, shaft_length) =
                arrow_head_and_shaft_lengths(draw_length, world_per_px);
            let thickness = arrow.thickness.value();
            let shaft_radius = compute_shaft_radius(draw_length, world_per_px, thickness);
            let head_radius = (shaft_radius * HEAD_RADIUS_FACTOR).min(draw_length * 0.75);

            let visual = state.visuals.entry(cam_entity).or_insert_with(|| {
                let visual = spawn_arrow_visual(
                    &mut commands,
                    &arrow_meshes,
                    &mut materials,
                    base_color,
                    entity,
                    arrow_layers.clone(),
                );
                commands
                    .entity(visual.root)
                    .insert(ChildOf(endpoints.start));
                visual
            });

            commands.entity(visual.root).insert((
                Transform::from_rotation(local_rotation),
                Visibility::Visible,
                arrow_layers.clone(),
            ));

            commands.entity(visual.shaft).insert(Transform {
                translation: Vec3::new(0.0, shaft_length * 0.5, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(shaft_radius, shaft_length, shaft_radius),
            });

            commands.entity(visual.head).insert(Transform {
                translation: Vec3::new(0.0, shaft_length + head_length * 0.5, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(head_radius, head_length, head_radius),
            });
        };

        let processed = if let Ok(viewport_arrow) = viewport_arrows.get(entity) {
            if let Some(&idx) = camera_index.get(&viewport_arrow.camera) {
                render_for_camera(idx);
                true
            } else {
                for visual in state.visuals.values() {
                    despawn_arrow_visual(&mut commands, visual);
                }
                state.visuals.clear();
                false
            }
        } else {
            for idx in 0..main_camera_data.len() {
                render_for_camera(idx);
            }
            true
        };

        let arrow_scope = if viewport_arrows.get(entity).is_ok() {
            ArrowLabelScope::Viewport
        } else {
            ArrowLabelScope::Global
        };

        if !processed {
            continue;
        }

        // Hide visuals for cameras that disappeared.
        let mut to_remove = Vec::new();
        for cam_entity in state.visuals.keys() {
            if !seen_cameras.contains(cam_entity) {
                to_remove.push(*cam_entity);
            }
        }
        for cam_entity in to_remove {
            if let Some(visual) = state.visuals.remove(&cam_entity) {
                despawn_arrow_visual(&mut commands, &visual);
            }
        }

        // Calculate and cache label offset from arrow root for the UI system.
        // Store as offset from arrow start so UI can use arrow's GlobalTransform.
        if arrow.show_name && arrow.name.is_some() {
            // Use proportional separation (10% of arrow length) with min/max bounds
            // to handle both very small and very large arrows gracefully
            let separation = (length * 0.1).clamp(0.005, 0.08);
            let total_offset = match arrow.label_position {
                LabelPosition::Proportionate(label_position) => {
                    // Place the label near the arrow tip by biasing toward the end of the vector
                    let label_t = label_position.max(0.8);
                    let label_offset = direction_world * label_t;
                    // Keep a small separation to avoid overlapping the head
                    label_offset + dir_norm * separation
                }
                LabelPosition::Absolute(length) => {
                    direction_world.normalize() * length + dir_norm * separation
                }
                LabelPosition::None => {
                    // Position at the arrow tip with proportional separation
                    direction_world + dir_norm * separation
                }
            };
            // Store just the offset from the arrow root
            state.label_offset = Some(total_offset);
            state.label_name = arrow.name.clone();
            state.label_color = None;
            state.label_scope = arrow_scope;
        } else {
            state.label_offset = None;
            state.label_name = None;
            state.label_color = None;
            state.label_scope = ArrowLabelScope::Global;
        }

        // 3D labels disabled; rely on Bevy UI system instead
        if let Some(label_entity) = state.label.take() {
            hide_label(&mut commands, Some(label_entity));
        }
    }
}

fn render_body_axis(
    entity_map: Res<EntityMap>,
    query: Query<&Transform>,
    arrows: Query<&BodyAxes>,
    mut gizmos: Gizmos,
) {
    for gizmo in arrows.iter() {
        let BodyAxes { entity_id, scale } = gizmo;

        let Some(entity_id) = entity_map.get(&ComponentId(entity_id.0)) else {
            warn!("body axes entity {entity_id:?} not found in EntityMap");
            continue;
        };

        let Ok(&transform) = query.get(*entity_id) else {
            continue;
        };
        gizmos.axes(transform, *scale)
    }
}

fn wkt_color_to_bevy(color: &WktColor) -> Color {
    Color::srgba(color.r, color.g, color.b, color.a)
}

fn spawn_arrow_visual(
    commands: &mut Commands,
    meshes: &ArrowMeshes,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    color: Color,
    owner: Entity,
    render_layers: RenderLayers,
) -> ArrowVisual {
    let shaft_material = materials.add(StandardMaterial {
        base_color: color,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let head_material = materials.add(StandardMaterial {
        base_color: lighten_color(color, 1.2),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });

    // Keep arrow transforms unaffected by parent scaling, so don't parent to the owner entity.
    let root = commands
        .spawn((
            Transform::default(),
            GlobalTransform::default(),
            Visibility::Hidden,
            render_layers.clone(),
            ArrowVisualOwner { owner },
            Name::new("vector_arrow_mesh"),
        ))
        .id();

    let shaft = commands
        .spawn((
            Mesh3d(meshes.shaft.clone()),
            MeshMaterial3d(shaft_material),
            Transform::default(),
            render_layers.clone(),
            ChildOf(root),
        ))
        .id();

    let head = commands
        .spawn((
            Mesh3d(meshes.head.clone()),
            MeshMaterial3d(head_material),
            Transform::default(),
            render_layers,
            ChildOf(root),
        ))
        .id();

    ArrowVisual { root, shaft, head }
}

fn despawn_arrow_visual(commands: &mut Commands, visual: &ArrowVisual) {
    commands.entity(visual.root).despawn();
}

fn hide_label(commands: &mut Commands, label: Option<Entity>) {
    if let Some(label) = label {
        commands.entity(label).insert(Visibility::Hidden);
    }
}

// Despawn visuals and endpoints when a VectorArrow3d is removed, to avoid stray entities.
fn cleanup_removed_arrows(
    mut removed_arrows: RemovedComponents<VectorArrow3d>,
    mut removed_states: RemovedComponents<VectorArrowState>,
    mut commands: Commands,
    mut states: Query<&mut VectorArrowState>,
    visuals: Query<(Entity, &ArrowVisualOwner)>,
    endpoints: Query<(Entity, &ArrowEndpoint)>,
) {
    let mut owners: HashSet<Entity> = removed_arrows.read().collect();
    owners.extend(removed_states.read());

    // Endpoint entities belong to the arrow regardless of whether its state survives.
    for (endpoint_entity, endpoint) in endpoints.iter() {
        if owners.contains(&endpoint.owner) {
            commands.entity(endpoint_entity).despawn();
            if let Ok(mut owner) = commands.get_entity(endpoint.owner) {
                owner.remove::<ArrowEndpoints>();
            }
        }
    }

    // If the state still exists, use it to clean up the associated visuals.
    for owner in owners.clone() {
        if let Ok(mut state) = states.get_mut(owner) {
            for visual in state.visuals.values() {
                commands.entity(visual.root).despawn();
            }
            state.visuals.clear();
            if let Some(label) = state.label.take() {
                commands.entity(label).despawn();
            }
            owners.remove(&owner);
        }
    }

    // Fallback: handle orphaned visuals whose owner entity was despawned.
    for (visual_entity, visual_owner) in visuals.iter() {
        if owners.contains(&visual_owner.owner) {
            commands.entity(visual_entity).despawn();
        }
    }
}

fn axis_color_from_name(_name: Option<&str>, default: Color) -> Color {
    default
}

/// Multiply RGB elements of `color` by `factor` in linear space.
fn lighten_color(color: Color, factor: f32) -> Color {
    let linear = color.to_linear();
    let r = linear.red;
    let g = linear.green;
    let b = linear.blue;
    let a = linear.alpha;
    let scale = |c: f32| (c * factor).clamp(0.0, 1.0);
    Color::linear_rgba(scale(r), scale(g), scale(b), a)
}

/// Convert arrow color to a readable label color (ensure good contrast)
/// Bevy UI system to render arrow labels as screen-space text nodes.
/// Spawns a label for each viewport where the arrow is visible, targeting the correct window's UI camera.
#[allow(clippy::too_many_arguments)]
fn update_arrow_label_ui(
    mut commands: Commands,
    arrows: Query<(Entity, &VectorArrowState)>,
    arrow_transforms: Query<&GlobalTransform>,
    cameras: Query<ArrowLabelCameraItem<'_>, With<MainCamera>>,
    mut labels: Query<(Entity, &ArrowLabelUI, &mut Node, &mut Text, &mut TextColor)>,
    primary_window: Query<Entity, With<bevy::window::PrimaryWindow>>,
    ui_cameras: Query<(Entity, &RenderTarget), With<ArrowLabelUiCamera>>,
    // Key: (arrow_entity, camera_entity) -> label_entity
    mut label_map: Local<HashMap<(Entity, Entity), Entity>>,
) {
    let Some(primary_entity) = primary_window.iter().next() else {
        return;
    };

    let mut window_cameras: HashMap<Entity, Vec<_>> = HashMap::new();
    for (entity, cam, render_target, gt, config) in cameras.iter() {
        if !cam.is_active {
            continue;
        }
        let Some(window) = window_entity_from_target(render_target, primary_entity) else {
            continue;
        };
        window_cameras
            .entry(window)
            .or_default()
            .push((entity, cam, gt, config));
    }

    if window_cameras.is_empty() {
        for (_, label_entity) in label_map.drain() {
            commands.entity(label_entity).despawn();
        }
        for (ui_cam_entity, _) in ui_cameras.iter() {
            commands.entity(ui_cam_entity).despawn();
        }
        return;
    }

    let mut window_ui_camera: HashMap<Entity, Entity> = HashMap::new();
    for (ui_cam_entity, ui_target) in ui_cameras.iter() {
        if let Some(window) = window_entity_from_target(ui_target, primary_entity)
            && window_cameras.contains_key(&window)
        {
            window_ui_camera.insert(window, ui_cam_entity);
            continue;
        }
        commands.entity(ui_cam_entity).despawn();
    }

    for window in window_cameras.keys() {
        window_ui_camera.entry(*window).or_insert_with(|| {
            commands
                .spawn((
                    Camera2d,
                    Camera {
                        order: ARROW_LABEL_UI_CAMERA_ORDER,
                        ..default()
                    },
                    RenderTarget::Window(if *window == primary_entity {
                        WindowRef::Primary
                    } else {
                        WindowRef::Entity(*window)
                    }),
                    ArrowLabelUiCamera,
                    Name::new(format!("ArrowLabelUiCamera_{:?}", window)),
                ))
                .id()
        });
    }

    let mut seen_labels = HashSet::new();

    for (arrow_entity, arrow_state) in arrows.iter() {
        let Some(visual) = arrow_state.visuals.values().next() else {
            continue;
        };
        // GlobalTransform is propagated grid-aware by big_space, so it is
        // already relative to the floating origin like the camera's.
        let Ok(arrow_transform) = arrow_transforms.get(visual.root) else {
            continue;
        };
        let Some(ref name) = arrow_state.label_name else {
            continue;
        };

        let label_color = {
            use crate::ui::colors::ColorExt;
            crate::ui::colors::get_scheme().text_primary.into_bevy()
        };
        let label_offset = arrow_state.label_offset.unwrap_or(Vec3::ZERO);

        let label_text = name.clone();
        let label_pos = arrow_transform.translation() + label_offset;

        for (window, cameras) in &window_cameras {
            let Some(&ui_cam_entity) = window_ui_camera.get(window) else {
                continue;
            };
            for (cam_entity, cam, cam_transform, config) in cameras {
                let show_arrows = config.map(|config| config.show_arrows).unwrap_or(true);
                let key = (arrow_entity, *cam_entity);

                if !show_arrows {
                    if let Some(&label_entity) = label_map.get(&key)
                        && let Ok((_, _, mut node, _, _)) = labels.get_mut(label_entity)
                    {
                        node.display = bevy::ui::Display::None;
                        commands
                            .entity(label_entity)
                            .insert(UiTargetCamera(ui_cam_entity));
                        seen_labels.insert(key);
                    }
                    continue;
                }

                if !arrow_state.visuals.contains_key(cam_entity) {
                    continue;
                }

                let Ok(screen_pos) = cam.world_to_viewport(cam_transform, label_pos) else {
                    if let Some(&label_entity) = label_map.get(&key)
                        && let Ok((_, _, mut node, _, _)) = labels.get_mut(label_entity)
                    {
                        node.display = bevy::ui::Display::None;
                        commands
                            .entity(label_entity)
                            .insert(UiTargetCamera(ui_cam_entity));
                        seen_labels.insert(key);
                    }
                    continue;
                };

                if let Some(rect) = cam.logical_viewport_rect()
                    && (screen_pos.x < rect.min.x
                        || screen_pos.x > rect.max.x
                        || screen_pos.y < rect.min.y
                        || screen_pos.y > rect.max.y)
                {
                    if let Some(&label_entity) = label_map.get(&key)
                        && let Ok((_, _, mut node, _, _)) = labels.get_mut(label_entity)
                    {
                        node.display = bevy::ui::Display::None;
                        commands
                            .entity(label_entity)
                            .insert(UiTargetCamera(ui_cam_entity));
                        seen_labels.insert(key);
                    }
                    continue;
                }

                let screen_x = screen_pos.x.round();
                let screen_y = (screen_pos.y - 8.0).round();

                if let Some(&label_entity) = label_map.get(&key)
                    && let Ok((_, _, mut node, mut text, mut text_color)) =
                        labels.get_mut(label_entity)
                {
                    node.left = Val::Px(screen_x);
                    node.top = Val::Px(screen_y);
                    node.display = bevy::ui::Display::Flex;
                    *text = Text::new(label_text.clone());
                    *text_color = TextColor(label_color);
                    commands
                        .entity(label_entity)
                        .insert(UiTargetCamera(ui_cam_entity));
                } else {
                    let label_entity = commands
                        .spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                left: Val::Px(screen_x),
                                top: Val::Px(screen_y),
                                ..default()
                            },
                            Text::new(label_text.clone()),
                            TextFont {
                                font_size: 14.0,
                                ..default()
                            },
                            TextColor(label_color),
                            ZIndex(1000), // Render above 3D content
                            Pickable::IGNORE,
                            ArrowLabelUI,
                            UiTargetCamera(ui_cam_entity),
                            Name::new(format!(
                                "arrow_label_{}_{:?}_{:?}",
                                label_text, cam_entity, window
                            )),
                        ))
                        .id();
                    label_map.insert(key, label_entity);
                }

                seen_labels.insert(key);
            }
        }
    }

    let labels_to_remove: Vec<_> = label_map
        .keys()
        .filter(|key| !seen_labels.contains(key))
        .copied()
        .collect();

    for key in labels_to_remove {
        if let Some(label_entity) = label_map.remove(&key) {
            commands.entity(label_entity).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::DQuat;
    use bevy_geo_frames::{GeoContext, GeoRotation};

    fn world_pos(pos: DVec3, att: DQuat) -> WorldPos {
        WorldPos {
            att: nox::Quaternion::new(att.w, att.x, att.y, att.z),
            pos: nox::Vector3::new(pos.x, pos.y, pos.z),
        }
    }

    fn bevy_delta(start: &WorldPos, end: &WorldPos, frame: GeoFrame, ctx: &GeoContext) -> DVec3 {
        GeoPosition(frame, end.pos()).to_bevy(ctx) - GeoPosition(frame, start.pos()).to_bevy(ctx)
    }

    #[test]
    fn compute_shaft_radius_visible_when_zoomed_out() {
        let draw_length = 1_000_000.0;
        let world_per_px = 50_000.0;
        let thickness = 2500.0;
        let radius = compute_shaft_radius(draw_length, world_per_px, thickness);
        let diameter_px = radius * 2.0 / world_per_px;
        assert!(diameter_px >= MIN_VISIBLE_DIAMETER_PX);
        assert_eq!(radius, MIN_VISIBLE_DIAMETER_PX * 0.5 * world_per_px);
    }

    #[test]
    fn compute_shaft_radius_caps_when_zoomed_in() {
        let draw_length = 1_000_000.0;
        let world_per_px = 1.0;
        let thickness = 1.0;
        let radius = compute_shaft_radius(draw_length, world_per_px, thickness);
        assert!(radius <= draw_length * MAX_RADIUS_FRAC);
        assert!(radius >= MIN_VISIBLE_DIAMETER_PX * 0.5 * world_per_px);
    }

    #[test]
    fn arrow_head_and_shaft_lengths_keep_visible_shaft() {
        let draw_length = 100.0;
        let world_per_px = 10.0;
        let (head, shaft) = arrow_head_and_shaft_lengths(draw_length, world_per_px);
        assert!(shaft > 0.0);
        assert!(head + shaft <= draw_length);
        assert!(shaft >= world_per_px * MIN_SHAFT_LENGTH_PX);
    }

    #[test]
    fn arrow_head_and_shaft_lengths_degenerate_when_very_short() {
        let (head, shaft) = arrow_head_and_shaft_lengths(0.05, 10.0);
        assert_eq!(head, 0.0);
        assert_eq!(shaft, 0.05);
    }

    /// World-frame arrows: routing the endpoints through `GeoPosition::to_bevy`
    /// must match converting the direction directly with `bevy_R_(frame)`.
    #[test]
    fn endpoints_match_direct_world_frame_conversion() {
        let ctx = GeoContext::default();
        let start = world_pos(
            DVec3::new(10.0, -4.0, 2.5),
            DQuat::from_euler(bevy::math::EulerRot::XYZ, 0.3, -0.8, 1.1),
        );
        let direction = DVec3::new(1.0, 2.0, 3.0);

        for frame in [GeoFrame::ENU, GeoFrame::NED, GeoFrame::ECEF] {
            let end = arrow_end_pos(&start, direction, false);
            let delta = bevy_delta(&start, &end, frame, &ctx);
            let expected = GeoFrame::bevy_R_(&frame, &ctx) * direction;
            assert!(
                (delta - expected).length() < 1e-9,
                "{frame:?}: got {delta:?}, expected {expected:?}"
            );
        }
    }

    /// Body-frame arrows: rotating the direction by the body attitude in sim
    /// coordinates must match the Bevy-space rotation by
    /// `GeoRotation::absolute(frame, att).to_bevy`.
    #[test]
    fn endpoints_match_direct_body_frame_conversion() {
        let ctx = GeoContext::default();
        let att = DQuat::from_euler(bevy::math::EulerRot::XYZ, 0.5, 0.2, -0.7);
        let start = world_pos(DVec3::new(-3.0, 8.0, 1.0), att);
        let direction = DVec3::new(0.5, -1.5, 2.0);

        for frame in [GeoFrame::ENU, GeoFrame::NED] {
            let end = arrow_end_pos(&start, direction, true);
            let delta = bevy_delta(&start, &end, frame, &ctx);
            let expected = GeoRotation::absolute(frame, att).to_bevy(&ctx) * direction;
            assert!(
                (delta - expected).length() < 1e-9,
                "{frame:?}: got {delta:?}, expected {expected:?}"
            );
        }
    }
}
