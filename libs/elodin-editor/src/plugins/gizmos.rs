use bevy::{
    app::{App, Plugin, Startup, Update},
    ecs::system::{Query, ResMut},
    gizmos::{
        config::{DefaultGizmoConfigGroup, GizmoConfigStore},
        gizmos::Gizmos,
    },
    math::Vec3,
    transform::components::Transform,
};
use conduit::{
    bevy::ComponentValueMap,
    well_known::{Gizmo, GizmoType},
};

pub struct GizmoPlugin;

impl Plugin for GizmoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, gizmo_setup);
        app.add_systems(Update, render_gizmo);
    }
}

fn gizmo_setup(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.enabled = true;
}

fn render_gizmo(query: Query<(&Transform, &ComponentValueMap, &Gizmo)>, mut gizmos: Gizmos) {
    for (transform, values, gizmo) in query.iter() {
        let Some(value) = values.0.get(&gizmo.id) else {
            continue;
        };
        match &gizmo.ty {
            GizmoType::Vector { range, color } => {
                let vec = match value {
                    conduit::ComponentValue::F32(arr) => {
                        Vec3::new(arr[range.start], arr[range.start + 1], arr[range.start + 2])
                    }
                    conduit::ComponentValue::F64(arr) => {
                        if arr.len() <= range.end {
                            continue;
                        }
                        Vec3::new(
                            arr[range.start] as f32,
                            arr[range.start + 1] as f32,
                            arr[range.start + 2] as f32,
                        )
                    }
                    _ => {
                        continue;
                    }
                };

                let end = transform.translation + transform.rotation * (vec * 10.0);
                gizmos.arrow(transform.translation, end, (*color).into());
            }
        }
    }
}
