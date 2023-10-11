use bevy::prelude::*;
use bevy_ecs::entity::Entity;
use bevy_polyline::prelude::{Polyline, PolylineBundle, PolylineMaterial};
use nalgebra::Vector3;

use crate::{history::HistoryStore, spatial::SpatialPos, types::Config};

pub struct TracesPlugin;

impl Plugin for TracesPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(PostStartup, setup_query);
        app.add_systems(Update, update_lines);
    }
}

fn setup_query(
    mut commands: Commands,
    query: Query<(Entity, &TraceAnchor)>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
) {
    for (entity, anchor) in &query {
        commands.spawn(TraceLine {
            line: PolylineBundle {
                polyline: polylines.add(Polyline { vertices: vec![] }),
                material: polyline_materials.add(PolylineMaterial {
                    width: 3.0,
                    color: Color::hex("B8CCFF").unwrap(),
                    perspective: false,
                    ..default()
                }),
                ..default()
            },
            entity: TraceEntity(entity),
            anchor: anchor.clone(),
        });
    }
}

fn update_lines(
    mut query: Query<(&TraceEntity, &TraceAnchor, &mut Handle<Polyline>)>,
    mut polylines: ResMut<Assets<Polyline>>,
    config: Res<Config>,
    history: Res<HistoryStore>,
) {
    for (trace, anchor, polyline) in &mut query {
        let polyline = polylines.get_mut(&polyline).unwrap();
        let Some(hist) = history.history(&trace.0) else {
            continue;
        };
        let end = history.current_index();
        let len = history
            .current_index()
            .saturating_sub(polyline.vertices.len());
        let start = end - len;
        for SpatialPos { pos, att } in &hist.pos()[start..end] {
            let pos = (att * anchor.anchor) + pos;
            let pos = Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32) * config.scale;
            polyline.vertices.push(pos);
        }
    }
}

#[derive(Bundle)]
struct TraceLine {
    line: PolylineBundle,
    entity: TraceEntity,
    anchor: TraceAnchor,
}

#[derive(Component)]
pub struct TraceEntity(Entity);

#[derive(Component, Clone)]
pub struct TraceAnchor {
    pub anchor: Vector3<f64>,
}
