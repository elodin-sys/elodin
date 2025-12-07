use bevy::render::view::RenderLayers;
use bevy::ui::{Node, PositionType, Val, ZIndex};
use bevy::{
    app::{App, Plugin, PostUpdate, Startup, Update},
    ecs::system::{Query, Res, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore, GizmoLineJoint},
        gizmos::Gizmos,
    },
    log::warn,
    math::{DQuat, DVec3},
    prelude::*,
    text::{TextColor, TextFont},
    transform::components::Transform,
};
use bevy_render::alpha::AlphaMode;
use big_space::FloatingOriginSettings;
use impeller2::types::ComponentId;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{
    ArrowThickness, BodyAxes, Color as WktColor, ComponentValue as WktComponentValue, VectorArrow3d,
};
use std::collections::{HashMap, HashSet};

use crate::{
    MainCamera, WorldPosExt,
    object_3d::ComponentArrayExt,
    vector_arrow::{ArrowVisual, VectorArrowState, component_value_tail_to_vec3},
};

/// Marker component for Bevy UI arrow labels
#[derive(Component)]
pub struct ArrowLabelUI {
    pub arrow_entity: Entity,
}

pub const GIZMO_RENDER_LAYER: usize = 30;
pub(crate) const MIN_ARROW_LENGTH_SQUARED: f64 = 1.0e-6;
const BASE_HEAD_LENGTH: f32 = 0.06;
const HEAD_RADIUS_FACTOR: f32 = 1.6;
const MAX_HEAD_PORTION: f32 = 0.5;
const DRAW_RAW_ARROW_MESHES: bool = true;
const TARGET_DIAMETER_PX: f32 = 7.0;
const MIN_RADIUS_WORLD: f32 = 0.005;
const MAX_RADIUS_WORLD: f32 = 0.25;

#[derive(Clone)]
pub struct EvaluatedVectorArrow {
    pub start: DVec3,
    pub end: DVec3,
    pub color: Color,
    pub name: Option<String>,
    pub label_position: f32,
}

pub struct GizmoPlugin;

impl Plugin for GizmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (gizmo_setup, arrow_mesh_setup));
        // This is how the `big_space` crate did it.
        // app.add_systems(PostUpdate, render_vector_arrow.after(TransformSystem::TransformPropagate));
        app.add_systems(bevy::app::PreUpdate, render_vector_arrow);
        app.add_systems(
            bevy::app::PostUpdate,
            cleanup_removed_arrows.after(render_vector_arrow),
        );
        app.add_systems(Update, render_body_axis);
        // Bevy UI labels for arrows - runs in PostUpdate after transforms are finalized
        app.add_systems(
            PostUpdate,
            update_arrow_label_ui.after(bevy::transform::TransformSystem::TransformPropagate),
        );
    }
}

fn radius_for_target_pixels(
    target_radius_px: f32,
    camera: &Camera,
    projection: &Projection,
    camera_transform: &GlobalTransform,
    world_pos: Vec3,
) -> f32 {
    let viewport = camera
        .physical_viewport_size()
        .unwrap_or_else(|| UVec2::new(1920, 1080));
    let height_px = viewport.y.max(1) as f32;

    match projection {
        Projection::Perspective(persp) => {
            let view = camera_transform.compute_matrix().inverse();
            let cam_space = view.transform_point3(world_pos);
            let depth = (-cam_space.z).max(0.001);
            let focal_px = height_px / (2.0 * (persp.fov * 0.5).tan());
            (target_radius_px * depth / focal_px).max(0.001)
        }
        Projection::Orthographic(ortho) => {
            // In ortho, world units map linearly to pixels via scale.
            let world_per_px = (2.0 * ortho.scale) / height_px;
            (target_radius_px * world_per_px).max(0.001)
        }
        Projection::Custom(_) => {
            // Fallback: assume roughly perspective-like scaling using near plane as depth proxy.
            target_radius_px * 0.001
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

/// Evaluate a vector arrow's expressions and return its world-space positions+metadata.
pub fn evaluate_vector_arrow(
    arrow: &VectorArrow3d,
    state: &VectorArrowState,
    entity_map: &EntityMap,
    component_values: &Query<'_, '_, &'static WktComponentValue>,
) -> Option<EvaluatedVectorArrow> {
    let Some(vector_expr) = &state.vector_expr else {
        return None;
    };

    let Ok(vector_value) = vector_expr.execute(entity_map, component_values) else {
        return None;
    };

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

    let mut start_world = DVec3::ZERO;
    let mut rotation = DQuat::IDENTITY;
    if let Some(origin_expr) = &state.origin_expr {
        let Ok(origin_value) = origin_expr.execute(entity_map, component_values) else {
            return None;
        };
        if let Some(world_pos) = origin_value.as_world_pos() {
            start_world = world_pos.bevy_pos();
            rotation = world_pos.bevy_att();
        } else if let Some(origin) = component_value_tail_to_vec3(&origin_value) {
            start_world = origin;
        }
    }

    if arrow.body_frame {
        direction = rotation * direction;
    }

    let end_world = start_world + direction;
    let label_position = arrow.label_position.clamp(0.0, 1.0);

    Some(EvaluatedVectorArrow {
        start: start_world,
        end: end_world,
        color: axis_color_from_name(arrow.name.as_deref(), wkt_color_to_bevy(&arrow.color)),
        name: arrow.name.clone(),
        label_position,
    })
}

#[allow(clippy::too_many_arguments)]
fn render_vector_arrow(
    mut commands: Commands,
    entity_map: Res<EntityMap>,
    mut vector_arrows: Query<(Entity, &VectorArrow3d, &mut VectorArrowState)>,
    component_values: Query<&'static WktComponentValue>,
    floating_origin: Res<FloatingOriginSettings>,
    arrow_meshes: Res<ArrowMeshes>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    main_cameras: Query<(&Camera, &Projection, &GlobalTransform), With<crate::MainCamera>>,
    mut logged_missing: Local<HashSet<Entity>>,
    mut logged_small: Local<HashSet<Entity>>,
) {
    let active_cam = main_cameras.iter().next();

    for (entity, arrow, mut state) in vector_arrows.iter_mut() {
        let Some(result) = evaluate_vector_arrow(arrow, &state, &entity_map, &component_values)
        else {
            if logged_missing.insert(entity) {
                info!(
                    ?entity,
                    name = ?arrow.name,
                    "vector_arrow: evaluation failed (missing data or zero-length)"
                );
            }
            if let Some(visual) = state.visual.take() {
                hide_arrow_visual(&mut commands, &visual);
            }
            continue;
        };

        let (start_cell, start) = floating_origin.translation_to_grid::<i128>(result.start);

        if !DRAW_RAW_ARROW_MESHES {
            if let Some(visual) = state.visual.take() {
                hide_arrow_visual(&mut commands, &visual);
            }
            continue;
        }

        let direction_world = (result.end - result.start).as_vec3();
        let length = direction_world.length();
        let dir_norm = if length > 0.0 {
            direction_world / length
        } else {
            Vec3::Y
        };
        let rotation = Quat::from_rotation_arc(Vec3::Y, dir_norm);
        let base_color = axis_color_from_name(result.name.as_deref(), result.color);

        // Keep a minimum draw length so a near-zero vector still shows a tiny arrow.
        let draw_length = length.max(0.05);
        if draw_length <= (MIN_ARROW_LENGTH_SQUARED as f32).sqrt() && logged_small.insert(entity) {
            info!(
                ?entity,
                name = ?result.name,
                length,
                "vector_arrow: very small magnitude, drawing minimum-sized arrow"
            );
        }

        let mut head_length = BASE_HEAD_LENGTH.min(draw_length * MAX_HEAD_PORTION);
        // Ensure the head never exceeds the total length.
        head_length = head_length.min(draw_length);
        let shaft_length = (draw_length - head_length).max(0.0);
        // Keep a roughly constant on-screen thickness by scaling radius from screen pixels.
        let shaft_radius = if let Some((cam, proj, cam_tf)) = active_cam {
            let radius =
                radius_for_target_pixels(TARGET_DIAMETER_PX * 0.5, cam, proj, cam_tf, start);
            radius.clamp(MIN_RADIUS_WORLD, MAX_RADIUS_WORLD)
        } else {
            0.03
        };
        let dimension_mult = match arrow.thickness {
            ArrowThickness::Small => 1.0,
            ArrowThickness::Middle => 1.5,
            ArrowThickness::Big => 2.0,
        };
        let shaft_radius = shaft_radius * dimension_mult;
        let head_radius = (shaft_radius * HEAD_RADIUS_FACTOR).min(draw_length * 0.75);

        if state.visual.is_none() {
            state.visual = Some(spawn_arrow_visual(
                &mut commands,
                &arrow_meshes,
                &mut materials,
                base_color,
                entity,
            ));
        }

        let _label_root;
        {
            let visual = state.visual.as_mut().unwrap();

            commands.entity(visual.root).insert((
                Transform::from_translation(start).with_rotation(rotation),
                start_cell,
                Visibility::Visible,
                RenderLayers::layer(GIZMO_RENDER_LAYER),
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

            _label_root = visual.root;
        }

        // Calculate and cache label offset from arrow root for the UI system.
        // Store as offset from arrow start so UI can use arrow's GlobalTransform.
        if arrow.display_name && result.name.is_some() {
            let label_t = result.label_position;
            // Offset from arrow start (which is at the root's Transform position)
            let label_offset = direction_world * label_t;
            // Additional offset along the arrow direction
            let along_offset = if label_t > 0.9 { 0.06 } else { 0.04 };
            let total_offset = label_offset + dir_norm * along_offset;

            // Store just the offset from the arrow root
            state.label_grid_pos = Some((0, 0, 0, total_offset));
            state.label_name = result.name.clone();
            state.label_color = Some(base_color);
        } else {
            state.label_grid_pos = None;
            state.label_name = None;
            state.label_color = None;
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
) -> ArrowVisual {
    let shaft_material = materials.add(StandardMaterial {
        base_color: color.with_alpha(0.65),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    });
    let head_material = materials.add(StandardMaterial {
        base_color: lighten_color(color, 1.2).with_alpha(0.65),
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
            RenderLayers::layer(GIZMO_RENDER_LAYER),
            ArrowVisualOwner { owner },
            Name::new("vector_arrow_mesh"),
        ))
        .id();

    let shaft = commands
        .spawn((
            Mesh3d(meshes.shaft.clone()),
            MeshMaterial3d(shaft_material),
            Transform::default(),
            RenderLayers::layer(GIZMO_RENDER_LAYER),
            ChildOf(root),
        ))
        .id();

    let head = commands
        .spawn((
            Mesh3d(meshes.head.clone()),
            MeshMaterial3d(head_material),
            Transform::default(),
            RenderLayers::layer(GIZMO_RENDER_LAYER),
            ChildOf(root),
        ))
        .id();

    ArrowVisual { root, shaft, head }
}

fn hide_arrow_visual(commands: &mut Commands, visual: &ArrowVisual) {
    commands.entity(visual.root).insert(Visibility::Hidden);
}

fn hide_label(commands: &mut Commands, label: Option<Entity>) {
    if let Some(label) = label {
        commands.entity(label).insert(Visibility::Hidden);
    }
}

// Despawn visuals when a VectorArrow3d is removed, to avoid stray meshes.
fn cleanup_removed_arrows(
    mut removed_arrows: RemovedComponents<VectorArrow3d>,
    mut removed_states: RemovedComponents<VectorArrowState>,
    mut commands: Commands,
    mut states: Query<&mut VectorArrowState>,
    visuals: Query<(Entity, &ArrowVisualOwner)>,
) {
    let mut owners: HashSet<Entity> = removed_arrows.read().collect();
    owners.extend(removed_states.read());

    // If the state still exists, use it to clean up the associated visuals.
    for owner in owners.clone() {
        if let Ok(mut state) = states.get_mut(owner) {
            if let Some(visual) = state.visual.take() {
                commands.entity(visual.root).despawn();
            }
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

fn lighten_color(color: Color, factor: f32) -> Color {
    let linear = color.to_linear();
    let r = (linear.red * factor).clamp(0.0, 1.0);
    let g = (linear.green * factor).clamp(0.0, 1.0);
    let b = (linear.blue * factor).clamp(0.0, 1.0);
    let a = linear.alpha;
    let scale = |c: f32| (c * factor).clamp(0.0, 1.0);
    Color::linear_rgba(scale(r), scale(g), scale(b), a)
}

/// Convert arrow color to a readable label color (ensure good contrast)
fn readable_label_color(color: Color) -> Color {
    let linear = color.to_linear();
    // Brighten the color for better visibility
    let r = (linear.red * 1.5 + 0.3).clamp(0.0, 1.0);
    let g = (linear.green * 1.5 + 0.3).clamp(0.0, 1.0);
    let b = (linear.blue * 1.5 + 0.3).clamp(0.0, 1.0);
    Color::linear_rgba(r, g, b, 1.0)
}

/// Bevy UI system to render arrow labels as screen-space text nodes.
/// Spawns a label for each viewport where the arrow is visible.
fn update_arrow_label_ui(
    mut commands: Commands,
    arrows: Query<(Entity, &VectorArrowState)>,
    arrow_transforms: Query<(&Transform, &big_space::GridCell<i128>)>,
    cameras: Query<(Entity, &Camera, &GlobalTransform, &big_space::GridCell<i128>), With<MainCamera>>,
    floating_origin: Res<FloatingOriginSettings>,
    mut labels: Query<(Entity, &ArrowLabelUI, &mut Node, &mut Text)>,
    // Key: (arrow_entity, camera_entity) -> label_entity
    mut label_map: Local<HashMap<(Entity, Entity), Entity>>,
) {
    let edge = floating_origin.grid_edge_length();

    // Collect all active cameras
    let active_cameras: Vec<_> = cameras
        .iter()
        .filter(|(_, cam, _, _)| cam.is_active)
        .collect();

    if active_cameras.is_empty() {
        return;
    }

    let mut seen_labels = HashSet::new();

    for (arrow_entity, arrow_state) in arrows.iter() {
        // Get the arrow visual's root entity
        let Some(ref visual) = arrow_state.visual else {
            continue;
        };

        // Get arrow's Transform and GridCell
        let Ok((arrow_transform, arrow_cell)) = arrow_transforms.get(visual.root) else {
            continue;
        };

        let Some(ref name) = arrow_state.label_name else {
            continue;
        };

        let label_color = arrow_state
            .label_color
            .map(readable_label_color)
            .unwrap_or(Color::WHITE);

        // Get the label offset from arrow root
        let label_offset = arrow_state
            .label_grid_pos
            .map(|(_, _, _, offset)| offset)
            .unwrap_or(Vec3::ZERO);

        // Calculate label position in arrow's local grid space
        let label_local = arrow_transform.translation + label_offset;

        // Process each active camera/viewport
        for (cam_entity, camera, camera_transform, cam_cell) in &active_cameras {
            let key = (arrow_entity, *cam_entity);

            // Convert from arrow's grid cell to camera-relative position
            let dx = (arrow_cell.x as f64 - cam_cell.x as f64) as f32 * edge;
            let dy = (arrow_cell.y as f64 - cam_cell.y as f64) as f32 * edge;
            let dz = (arrow_cell.z as f64 - cam_cell.z as f64) as f32 * edge;
            let camera_relative_pos = label_local + Vec3::new(dx, dy, dz);

            // Project to screen space
            let Ok(screen_pos) = camera.world_to_viewport(camera_transform, camera_relative_pos)
            else {
                // Off-screen for this camera - hide label if it exists
                if let Some(label_entity) = label_map.get(&key) {
                    if let Ok((_, _, mut node, _)) = labels.get_mut(*label_entity) {
                        node.display = bevy::ui::Display::None;
                    }
                }
                continue;
            };

            // Check if screen position falls within this camera's viewport
            let viewport = camera.logical_viewport_rect();
            if let Some(rect) = viewport {
                if screen_pos.x < rect.min.x
                    || screen_pos.x > rect.max.x
                    || screen_pos.y < rect.min.y
                    || screen_pos.y > rect.max.y
                {
                    // Outside this viewport - hide label if exists
                    if let Some(label_entity) = label_map.get(&key) {
                        if let Ok((_, _, mut node, _)) = labels.get_mut(*label_entity) {
                            node.display = bevy::ui::Display::None;
                        }
                    }
                    continue;
                }
            }

            seen_labels.insert(key);

            // Round to avoid sub-pixel jitter
            let screen_x = screen_pos.x.round();
            let screen_y = (screen_pos.y - 8.0).round(); // Slight vertical offset

            // Check if label already exists for this arrow+camera pair
            if let Some(&label_entity) = label_map.get(&key) {
                // Update existing label
                if let Ok((_, _, mut node, mut text)) = labels.get_mut(label_entity) {
                    node.left = Val::Px(screen_x);
                    node.top = Val::Px(screen_y);
                    node.display = bevy::ui::Display::Flex;
                    *text = Text::new(name.clone());
                }
            } else {
                // Spawn new label for this arrow+camera pair
                let label_entity = commands
                    .spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(screen_x),
                            top: Val::Px(screen_y),
                            ..default()
                        },
                        Text::new(name.clone()),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(label_color),
                        ZIndex(1000), // Render above 3D content
                        ArrowLabelUI { arrow_entity },
                        Name::new(format!("arrow_label_{}_{:?}", name, cam_entity)),
                    ))
                    .id();
                label_map.insert(key, label_entity);
            }
        }
    }

    // Clean up labels that are no longer needed
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
