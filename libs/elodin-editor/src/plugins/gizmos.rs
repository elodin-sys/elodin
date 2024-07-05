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
use conduit::{
    bevy::{ComponentValueMap, EntityMap},
    well_known::{BodyAxes, VectorArrow, WorldPos},
};
use nalgebra::{UnitQuaternion, Vector3};

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
    config.line_width = 5.0;
    config.line_joints = GizmoLineJoint::Round(12);
    config.enabled = true;
}

fn render_vector_arrow(
    entity_map: Res<EntityMap>,
    query: Query<(Option<&Transform>, &ComponentValueMap)>,
    arrows: Query<&VectorArrow>,
    mut gizmos: Gizmos,
) {
    for gizmo in arrows.iter() {
        let VectorArrow {
            id,
            color,
            range,
            attached,
            entity_id,
            body_frame,
            scale,
        } = gizmo;

        let Some(entity_id) = entity_map.get(entity_id) else {
            continue;
        };

        let Ok((transform, values)) = query.get(*entity_id) else {
            continue;
        };

        let Some(value) = values.0.get(id) else {
            continue;
        };
        let vec = match value {
            conduit::ComponentValue::F32(arr) => Vector3::new(
                arr[range.start] as f64,
                arr[range.start + 1] as f64,
                arr[range.start + 2] as f64,
            ),
            conduit::ComponentValue::F64(arr) => {
                Vector3::new(arr[range.start], arr[range.start + 1], arr[range.start + 2])
            }
            _ => {
                continue;
            }
        };
        let vec = WorldPos {
            att: UnitQuaternion::identity(),
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
        gizmos.arrow(start, end, *color);
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
