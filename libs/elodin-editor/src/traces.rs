use bevy::prelude::*;
use bevy_polyline::prelude::{Polyline, PolylineBundle, PolylineMaterial};
use elodin_core::TraceAnchor;
use std::collections::VecDeque;

pub struct TracesPlugin;

impl Plugin for TracesPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Update, setup_query);
        app.add_systems(Update, update_lines);
    }
}

fn setup_query(
    mut commands: Commands,
    query: Query<(Entity, &TraceAnchor), Without<TraceEntity>>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
) {
    for (entity, anchor) in query.iter() {
        commands.spawn((
            PolylineBundle {
                polyline: polylines.add(Polyline {
                    vertices: VecDeque::with_capacity(1024),
                }),
                material: polyline_materials.add(PolylineMaterial {
                    width: 3.0,
                    color: Color::hex("B8CCFF").unwrap(),
                    perspective: false,
                    ..default()
                }),
                ..default()
            },
            anchor.clone(),
            TraceEntity(entity),
        ));
        commands.entity(entity).insert(TraceEntity(entity));
    }
}

fn update_lines(
    trace: Query<(&TraceEntity, &TraceAnchor, &Handle<Polyline>)>,
    transform: Query<&Transform>,
    mut polylines: ResMut<Assets<Polyline>>,
) {
    for (entity, anchor, polyline) in &trace {
        let Ok(pos) = transform.get(entity.0) else {
            continue;
        };
        let Some(polyline) = polylines.get_mut(polyline) else {
            continue;
        };
        if polyline.vertices.len() == 1024 {
            polyline.vertices.pop_front();
        }
        let anchor_pos = Vec3::new(
            anchor.anchor.x as f32,
            anchor.anchor.y as f32,
            anchor.anchor.z as f32,
        );
        polyline
            .vertices
            .push_back(pos.translation + pos.rotation * anchor_pos);
    }
}

#[derive(Component)]
struct TraceEntity(Entity);
