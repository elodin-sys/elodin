use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::system::{Query, Res, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore, GizmoLineJoint},
        gizmos::Gizmos,
    },
    log::warn,
    math::Vec3,
    prelude::Color,
    transform::components::Transform,
};
use impeller2::types::ComponentId;
use impeller2_bevy::{ComponentValueMap, EntityMap};
use impeller2_wkt::{
    BodyAxes, Color as WktColor, ComponentValue as WktComponentValue, VectorArrow, VectorArrow3d,
    WorldPos,
};
use nox::{ArrayBuf, Quaternion, Vector3};
use std::ops::Range;

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
    component_values: Query<&'static WktComponentValue>,
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

        let Ok((transform, value_map)) = query.get(*entity_id) else {
            continue;
        };

        let Some(value) = value_map.0.get(id) else {
            continue;
        };
        let Some(vec) = component_value_slice_to_bevy_vec(value, range) else {
            continue;
        };
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
        gizmos.arrow(start, end, wkt_color_to_bevy(color));
    }

    for (arrow, state) in vector_arrows.iter() {
        let Some(vector_expr) = &state.vector_expr else {
            continue;
        };

        let Ok(vector_value) = vector_expr.execute(&entity_map, &component_values) else {
            continue;
        };

        let Some(direction) = component_value_tail_to_vec3(&vector_value) else {
            continue;
        };

        let direction = direction * arrow.scale;

        let mut start = Vec3::ZERO;
        if let Some(origin_expr) = &state.origin_expr {
            if let Ok(origin_value) = origin_expr.execute(&entity_map, &component_values) {
                if let Some(origin) = component_value_tail_to_vec3(&origin_value) {
                    start = origin;
                }
            }
        }

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

fn component_value_slice_to_bevy_vec(
    value: &WktComponentValue,
    range: &Range<usize>,
) -> Option<Vec3> {
    match value {
        WktComponentValue::F32(arr) => {
            let data = arr.buf.as_buf();
            array_slice_to_bevy_vec(data, range, f64::from)
        }
        WktComponentValue::F64(arr) => {
            let data = arr.buf.as_buf();
            array_slice_to_bevy_vec(data, range, |value| value)
        }
        _ => None,
    }
}

fn array_slice_to_bevy_vec<T>(
    data: &[T],
    range: &Range<usize>,
    mut to_f64: impl FnMut(T) -> f64,
) -> Option<Vec3>
where
    T: Copy,
{
    let x_idx = range.start;
    let y_idx = x_idx.checked_add(1)?;
    let z_idx = x_idx.checked_add(2)?;

    if x_idx >= data.len() || y_idx >= data.len() || z_idx >= data.len() {
        return None;
    }

    if range.end <= z_idx {
        return None;
    }

    let world = WorldPos {
        att: Quaternion::identity(),
        pos: Vector3::new(
            to_f64(data[x_idx]),
            to_f64(data[y_idx]),
            to_f64(data[z_idx]),
        ),
    };

    Some(world.bevy_pos().as_vec3())
}

fn wkt_color_to_bevy(color: &WktColor) -> Color {
    Color::srgba(color.r, color.g, color.b, color.a)
}
