use bevy::render::view::RenderLayers;
use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::system::{Query, Res, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore, GizmoLineJoint},
        gizmos::Gizmos,
    },
    log::warn,
    math::{DQuat, DVec3},
    prelude::*,
    transform::components::Transform,
};
use bevy_render::alpha::AlphaMode;
use big_space::FloatingOriginSettings;
use impeller2::types::ComponentId;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{
    BodyAxes, Color as WktColor, ComponentValue as WktComponentValue, VectorArrow3d,
};

use crate::{
    WorldPosExt,
    object_3d::ComponentArrayExt,
    vector_arrow::{ArrowVisual, VectorArrowState, component_value_tail_to_vec3},
};

pub const GIZMO_RENDER_LAYER: usize = 30;
pub(crate) const MIN_ARROW_LENGTH_SQUARED: f64 = 1.0e-6;

#[derive(Clone)]
pub struct EvaluatedVectorArrow {
    pub start: DVec3,
    pub end: DVec3,
    pub color: Color,
    pub name: Option<String>,
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

    Some(EvaluatedVectorArrow {
        start: start_world,
        end: end_world,
        color: axis_color_from_name(arrow.name.as_deref(), wkt_color_to_bevy(&arrow.color)),
        name: arrow.name.clone(),
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
    main_cameras: Query<&GlobalTransform, With<crate::MainCamera>>,
) {
    let _camera_transform = main_cameras.iter().next().copied();

    for (entity, arrow, mut state) in vector_arrows.iter_mut() {
        let Some(result) = evaluate_vector_arrow(arrow, &state, &entity_map, &component_values)
        else {
            if let Some(visual) = state.visual.take() {
                hide_arrow_visual(&mut commands, &visual);
            }
            continue;
        };

        let (start_cell, start) = floating_origin.translation_to_grid::<i128>(result.start);

        let direction_world = (result.end - result.start).as_vec3();
        let length = direction_world.length();
        let too_small = length <= (MIN_ARROW_LENGTH_SQUARED as f32).sqrt();

        let dir_norm = direction_world / length;
        let rotation = Quat::from_rotation_arc(Vec3::Y, dir_norm);
        let base_color = axis_color_from_name(result.name.as_deref(), result.color);

        let shaft_length = length * 0.8;
        let head_length = length * 0.2;
        // Keep radii mostly stable so long arrows don't become chunky
        let shaft_radius = 0.01;
        let head_radius = shaft_radius * 1.6;

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

            if too_small {
                hide_arrow_visual(&mut commands, visual);
                hide_label(&mut commands, state.label.take());
                continue;
            }

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

        // 3D labels disabled; rely on overlay to avoid duplication issues
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
    _owner: Entity,
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
    mut removed: RemovedComponents<VectorArrow3d>,
    mut commands: Commands,
    mut states: Query<&mut VectorArrowState>,
) {
    for entity in removed.read() {
        if let Ok(mut state) = states.get_mut(entity) {
            if let Some(visual) = state.visual.take() {
                commands.entity(visual.root).despawn();
            }
            if let Some(label) = state.label.take() {
                commands.entity(label).despawn();
            }
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
