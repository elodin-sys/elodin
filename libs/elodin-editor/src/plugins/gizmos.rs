use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::system::{Query, Res, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore, GizmoLineJoint},
        gizmos::Gizmos,
    },
    math::Vec3,
    transform::components::Transform,
};
use impeller2_bevy::{ComponentValueMap, EntityMap};
use impeller2_wkt::{
    BodyAxes, ComponentValue as WktComponentValue, VectorArrow, VectorArrow3d, WorldPos,
};
use nox::{Quaternion, Vector3};

use crate::{
    WorldPosExt,
    vector_arrow::{VectorArrowState, component_value_tail_to_vec3},
};

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
}

fn render_vector_arrow(
    entity_map: Res<EntityMap>,
    query: Query<(Option<&Transform>, &ComponentValueMap)>,
    arrows: Query<&VectorArrow>,
    vector_arrows: Query<(&VectorArrow3d, &VectorArrowState)>,
    values: Query<&WktComponentValue>,
    mut gizmos: Gizmos,
) {
    for gizmo in arrows.iter() {
        let VectorArrow {
            id,
            color,
            range,
            attached,
            body_frame,
            scale,
            ..
        } = gizmo;

        let Some(entity_id) = entity_map.get(id) else {
            continue;
        };

        let Ok((transform, values)) = query.get(*entity_id) else {
            continue;
        };

        let Some(value) = values.0.get(id) else {
            continue;
        };
        let vec = match value {
            WktComponentValue::F32(arr) => {
                let data = arr.buf.as_buf();
                if range.start + 2 >= data.len() {
                    continue;
                }
                Vector3::new(
                    data[range.start] as f64,
                    data[range.start + 1] as f64,
                    data[range.start + 2] as f64,
                )
            }
            WktComponentValue::F64(arr) => {
                let data = arr.buf.as_buf();
                if range.start + 2 >= data.len() {
                    continue;
                }
                Vector3::new(
                    data[range.start],
                    data[range.start + 1],
                    data[range.start + 2],
                )
            }
            _ => {
                continue;
            }
        };
        let vec = WorldPos {
            att: Quaternion::identity(),
            pos: vec,
        }
        .bevy_pos()
        .as_vec3();
        let (start, end) = if *attached {
            let Some(attached_transform) = transform else {
                continue;
            };
            let end = if *body_frame {
                attached_transform.translation + attached_transform.rotation * vec * *scale
            } else {
                attached_transform.translation + vec * *scale
            };
            (attached_transform.translation, end)
        } else if *body_frame {
            let Some(attached_transform) = transform else {
                continue;
            };
            (Vec3::ZERO, attached_transform.rotation * vec * *scale)
        } else {
            (Vec3::ZERO, vec * *scale)
        };
        let color = bevy::prelude::Color::rgba(color.r, color.g, color.b, color.a);
        gizmos.arrow(start, end, color);
    }

    for (arrow, state) in vector_arrows.iter() {
        let Some(vector_expr) = &state.vector_expr else {
            continue;
        };

        let Ok(vector_value) = vector_expr.execute(&entity_map, &values) else {
            continue;
        };

        let Some(direction_value) = component_value_tail_to_vec3(&vector_value) else {
            continue;
        };

        let direction = impeller_vec3_to_bevy(direction_value) * arrow.scale;

        let mut start = Vec3::ZERO;
        if let Some(origin_expr) = &state.origin_expr {
            if let Ok(origin_value) = origin_expr.execute(&entity_map, &values) {
                if let Some(origin) = component_value_tail_to_vec3(&origin_value) {
                    start = impeller_vec3_to_bevy(origin);
                }
            }
        }

        let end = start + direction;
        let color = bevy::prelude::Color::srgb(arrow.color.r, arrow.color.g, arrow.color.b);
        gizmos.arrow(start, end, color);
    }
}

fn impeller_vec3_to_bevy(vec: Vec3) -> Vec3 {
    WorldPos {
        att: Quaternion::identity(),
        pos: Vector3::new(vec.x as f64, vec.y as f64, vec.z as f64),
    }
    .bevy_pos()
    .as_vec3()
}

fn render_body_axis(
    entity_map: Res<EntityMap>,
    query: Query<&Transform>,
    arrows: Query<&BodyAxes>,
    mut gizmos: Gizmos,
) {
    for gizmo in arrows.iter() {
        let BodyAxes { entity_id, scale } = gizmo;

        let Some(entity_id) = entity_map.get(entity_id) else {
            println!("entity not found");
            continue;
        };

        let Ok(&transform) = query.get(*entity_id) else {
            continue;
        };
        gizmos.axes(transform, *scale)
    }
}
