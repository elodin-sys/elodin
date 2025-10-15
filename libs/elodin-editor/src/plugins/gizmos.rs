use bevy::render::view::RenderLayers;
use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::system::{Query, Res, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore, GizmoLineJoint},
        gizmos::Gizmos,
    },
    log::warn,
    math::{DVec3, Quat, Vec3},
    prelude::Color,
    transform::components::Transform,
};
use big_space::FloatingOriginSettings;
use impeller2::types::ComponentId;
use impeller2_bevy::EntityMap;
use impeller2_wkt::{
    BodyAxes, Color as WktColor, ComponentValue as WktComponentValue, VectorArrow3d,
};

use crate::{
    WorldPosExt,
    object_3d::ComponentArrayExt,
    vector_arrow::{VectorArrowState, component_value_tail_to_vec3},
};

pub const GIZMO_RENDER_LAYER: usize = 30;
const MIN_ARROW_LENGTH_SQUARED: f32 = 1.0e-6;

pub struct GizmoPlugin;

impl Plugin for GizmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, gizmo_setup);
        app.add_systems(Update, render_vector_arrow);
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

fn render_vector_arrow(
    entity_map: Res<EntityMap>,
    vector_arrows: Query<(&VectorArrow3d, &VectorArrowState)>,
    component_values: Query<&'static WktComponentValue>,
    floating_origin: Res<FloatingOriginSettings>,
    mut gizmos: Gizmos,
) {
    for (arrow, state) in vector_arrows.iter() {
        let Some(vector_expr) = &state.vector_expr else {
            continue;
        };

        let Ok(vector_value) = vector_expr.execute(&entity_map, &component_values) else {
            continue;
        };

        let Some(mut direction) = component_value_tail_to_vec3(&vector_value) else {
            continue;
        };

        if direction.length_squared() <= MIN_ARROW_LENGTH_SQUARED {
            continue;
        }

        if arrow.normalize {
            direction = direction.normalize();
        }

        direction *= arrow.scale;
        if direction.length_squared() <= MIN_ARROW_LENGTH_SQUARED {
            continue;
        }

        let mut start_world = Vec3::ZERO;
        let mut rotation = Quat::IDENTITY;
        if let Some(origin_expr) = &state.origin_expr {
            let Ok(origin_value) = origin_expr.execute(&entity_map, &component_values) else {
                continue;
            };
            if let Some(world_pos) = origin_value.as_world_pos() {
                start_world = world_pos.bevy_pos().as_vec3();
                rotation = world_pos.bevy_att().as_quat();
            } else {
                let Some(origin) = component_value_tail_to_vec3(&origin_value) else {
                    continue;
                };
                start_world = origin;
            }
        }

        if arrow.body_frame {
            direction = rotation * direction;
        }

        let start_world_d = DVec3::new(
            start_world.x as f64,
            start_world.y as f64,
            start_world.z as f64,
        );
        let (_, start) = floating_origin.translation_to_grid::<i128>(start_world_d);
        let end = start + direction;
        gizmos.arrow(start, end, wkt_color_to_bevy(&arrow.color));
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
