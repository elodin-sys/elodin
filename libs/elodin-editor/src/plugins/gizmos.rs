use bevy::camera::RenderTarget;
use bevy::camera::visibility::RenderLayers;
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
    BodyAxes, Color as WktColor, ComponentValue as WktComponentValue, LabelPosition, VectorArrow3d,
};
use std::collections::{HashMap, HashSet};

use crate::{
    MainCamera, WorldPosExt,
    object_3d::ComponentArrayExt,
    ui::tiles::ViewportConfig,
    ui::window::window_entity_from_target,
    vector_arrow::{
        ArrowLabelScope, ArrowVisual, VectorArrowState, ViewportArrow, component_value_tail_to_vec3,
    },
};

type ArrowLabelCameraItem<'w> = (
    Entity,
    &'w Camera,
    &'w GlobalTransform,
    &'w big_space::GridCell<i128>,
    Option<&'w ViewportConfig>,
);

type MainCameraQueryItem<'w> = (
    Entity,
    &'w Camera,
    &'w Projection,
    &'w GlobalTransform,
    Option<&'w ViewportConfig>,
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
const BASE_HEAD_LENGTH: f32 = 0.06;
const HEAD_RADIUS_FACTOR: f32 = 1.6;
const MAX_HEAD_PORTION: f32 = 0.5;
const DRAW_RAW_ARROW_MESHES: bool = true;
const TARGET_DIAMETER_PX: f32 = 7.0;
const MIN_RADIUS_WORLD: f32 = 0.005;
const MAX_RADIUS_WORLD: f32 = 0.05;
const MAX_FINAL_RADIUS_WORLD: f32 = 0.1;

#[derive(Clone)]
pub struct EvaluatedVectorArrow {
    pub start: DVec3,
    pub end: DVec3,
    pub color: Color,
    pub name: Option<String>,
    pub label_position: LabelPosition,
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
            let view = camera_transform.compute_matrix().inverse();
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
    let label_position = arrow.label_position.clone();

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
    main_cameras: Query<MainCameraQueryItem<'_>, With<crate::MainCamera>>,
    viewport_arrows: Query<&ViewportArrow>,
    mut logged_missing: Local<HashSet<Entity>>,
    mut logged_small: Local<HashSet<Entity>>,
) {
    let main_camera_data: Vec<_> = main_cameras.iter().collect();
    let mut camera_index: HashMap<Entity, usize> = HashMap::new();
    for (idx, (entity, ..)) in main_camera_data.iter().enumerate() {
        camera_index.insert(*entity, idx);
    }
    let has_show_arrows = main_camera_data
        .iter()
        .any(|(_, _, _, _, config)| config.as_ref().map(|c| c.show_arrows).unwrap_or(false));

    for (entity, arrow, mut state) in vector_arrows.iter_mut() {
        if !has_show_arrows {
            for visual in state.visuals.values() {
                hide_arrow_visual(&mut commands, visual);
            }
            state.visuals.clear();
            continue;
        }

        let Some(result) = evaluate_vector_arrow(arrow, &state, &entity_map, &component_values)
        else {
            if logged_missing.insert(entity) {
                info!(
                    ?entity,
                    name = ?arrow.name,
                    "vector_arrow: evaluation failed (missing data or zero-length)"
                );
            }
            for visual in state.visuals.values() {
                hide_arrow_visual(&mut commands, visual);
            }
            state.visuals.clear();
            continue;
        };

        let (start_cell, start) = floating_origin.translation_to_grid::<i128>(result.start);

        if !DRAW_RAW_ARROW_MESHES {
            for visual in state.visuals.values() {
                hide_arrow_visual(&mut commands, visual);
            }
            state.visuals.clear();
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
        let mut seen_cameras: HashSet<Entity> = HashSet::new();

        let mut render_for_camera = |idx: usize| {
            let (cam_entity, cam, proj, cam_tf, viewport_config) = main_camera_data[idx];
            seen_cameras.insert(cam_entity);

            let show_arrows = viewport_config
                .map(|config| config.show_arrows)
                .unwrap_or(true);
            if !show_arrows {
                if let Some(visual) = state.visuals.remove(&cam_entity) {
                    hide_arrow_visual(&mut commands, &visual);
                }
                state.label_grid_pos = None;
                state.label_name = None;
                state.label_color = None;
                if let Some(label_entity) = state.label.take() {
                    hide_label(&mut commands, Some(label_entity));
                }
                return;
            }

            let Some(viewport_layer) = viewport_config.and_then(|config| config.viewport_layer)
            else {
                if let Some(visual) = state.visuals.remove(&cam_entity) {
                    hide_arrow_visual(&mut commands, &visual);
                }
                return;
            };
            let arrow_layers = RenderLayers::layer(viewport_layer);

            let world_per_px = world_units_per_pixel(cam, proj, cam_tf, start);
            let shaft_radius = (TARGET_DIAMETER_PX * 0.5 * world_per_px)
                .clamp(MIN_RADIUS_WORLD, MAX_RADIUS_WORLD)
                * arrow.thickness.value();
            let shaft_radius = shaft_radius.clamp(MIN_RADIUS_WORLD, MAX_FINAL_RADIUS_WORLD);
            let head_radius = (shaft_radius * HEAD_RADIUS_FACTOR).min(draw_length * 0.75);

            let visual = state.visuals.entry(cam_entity).or_insert_with(|| {
                spawn_arrow_visual(
                    &mut commands,
                    &arrow_meshes,
                    &mut materials,
                    base_color,
                    entity,
                    arrow_layers.clone(),
                )
            });

            commands.entity(visual.root).insert((
                Transform::from_translation(start).with_rotation(rotation),
                start_cell,
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
                    hide_arrow_visual(&mut commands, visual);
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
                hide_arrow_visual(&mut commands, &visual);
            }
        }

        // Calculate and cache label offset from arrow root for the UI system.
        // Store as offset from arrow start so UI can use arrow's GlobalTransform.
        if arrow.show_name && result.name.is_some() {
            let total_offset = match result.label_position {
                LabelPosition::Proportionate(label_position) => {
                    // Place the label near the arrow tip by biasing toward the end of the vector
                    let label_t = label_position.max(0.8);
                    let label_offset = direction_world * label_t;
                    // Keep a small separation to avoid overlapping the head
                    label_offset + dir_norm * 0.08
                }
                LabelPosition::Absolute(length) => {
                    direction_world.normalize() * length + dir_norm * 0.08
                }
                LabelPosition::None => {
                    let length = 0.1; // meters
                    direction_world.normalize() * length + dir_norm * 0.08
                }
            };
            // Store just the offset from the arrow root
            state.label_grid_pos = Some((0, 0, 0, total_offset));
            state.label_name = result.name.clone();
            state.label_color = None;
            state.label_scope = arrow_scope;
        } else {
            state.label_grid_pos = None;
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
/// Bevy UI system to render arrow labels as screen-space text nodes.
/// Spawns a label for each viewport where the arrow is visible, targeting the correct window's UI camera.
#[allow(clippy::too_many_arguments)]
fn update_arrow_label_ui(
    mut commands: Commands,
    arrows: Query<(Entity, &VectorArrowState)>,
    arrow_transforms: Query<(&Transform, &big_space::GridCell<i128>)>,
    cameras: Query<ArrowLabelCameraItem<'_>, With<MainCamera>>,
    floating_origin: Res<FloatingOriginSettings>,
    mut labels: Query<(Entity, &ArrowLabelUI, &mut Node, &mut Text, &mut TextColor)>,
    primary_window: Query<Entity, With<bevy::window::PrimaryWindow>>,
    ui_cameras: Query<(Entity, &Camera), With<ArrowLabelUiCamera>>,
    // Key: (arrow_entity, camera_entity) -> label_entity
    mut label_map: Local<HashMap<(Entity, Entity), Entity>>,
) {
    let edge = floating_origin.grid_edge_length();
    let Some(primary_entity) = primary_window.iter().next() else {
        return;
    };

    let mut window_cameras: HashMap<Entity, Vec<_>> = HashMap::new();
    for (entity, cam, gt, cell, config) in cameras.iter() {
        if !cam.is_active {
            continue;
        }
        let Some(window) = window_entity_from_target(&cam.target, primary_entity) else {
            continue;
        };
        window_cameras
            .entry(window)
            .or_default()
            .push((entity, cam, gt, cell, config));
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
    for (ui_cam_entity, ui_cam) in ui_cameras.iter() {
        if let Some(window) = window_entity_from_target(&ui_cam.target, primary_entity)
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
                        target: RenderTarget::Window(if *window == primary_entity {
                            WindowRef::Primary
                        } else {
                            WindowRef::Entity(*window)
                        }),
                        order: ARROW_LABEL_UI_CAMERA_ORDER,
                        ..default()
                    },
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
        let Ok((arrow_transform, arrow_cell)) = arrow_transforms.get(visual.root) else {
            continue;
        };
        let Some(ref name) = arrow_state.label_name else {
            continue;
        };

        let label_color = {
            use crate::ui::colors::ColorExt;
            crate::ui::colors::get_scheme().text_primary.into_bevy()
        };
        let label_offset = arrow_state
            .label_grid_pos
            .map(|(_, _, _, offset)| offset)
            .unwrap_or(Vec3::ZERO);

        let label_text = name.clone();
        let label_local = arrow_transform.translation + label_offset;

        for (window, cameras) in &window_cameras {
            let Some(&ui_cam_entity) = window_ui_camera.get(window) else {
                continue;
            };
            for (cam_entity, cam, cam_transform, cam_cell, config) in cameras {
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

                let dx = (arrow_cell.x as f64 - cam_cell.x as f64) as f32 * edge;
                let dy = (arrow_cell.y as f64 - cam_cell.y as f64) as f32 * edge;
                let dz = (arrow_cell.z as f64 - cam_cell.z as f64) as f32 * edge;
                let camera_relative_pos = label_local + Vec3::new(dx, dy, dz);

                let Ok(screen_pos) = cam.world_to_viewport(cam_transform, camera_relative_pos)
                else {
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
