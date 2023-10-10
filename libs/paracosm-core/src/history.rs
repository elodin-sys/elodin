use bevy::prelude::{App, Assets, Handle, Plugin, Quat, Transform, Update, Vec3};
use bevy_ecs::{
    entity::Entity,
    event::{Event, EventReader},
    query::WorldQuery,
    schedule::IntoSystemConfigs,
    system::{Query, Res, ResMut, Resource},
};
use bevy_polyline::prelude::Polyline;
use nalgebra::{UnitQuaternion, Vector3};
use std::collections::HashMap;

use crate::{
    plugin::{PhysicsSchedule, TickSet},
    types::{Config, Effect, EntityQuery},
};

#[derive(Default, Resource)]
pub struct HistoryStore {
    entities: HashMap<Entity, EntityHistory>,
    count: usize,
    current_index: usize,
}

impl HistoryStore {
    pub fn record<'a>(
        &mut self,
        query: impl Iterator<Item = (Entity, HistoryQueryReadOnlyItem<'a>)>,
    ) {
        self.current_index += 1;
        if self.current_index > self.count {
            self.count = self.current_index;
            for (entity, data) in query {
                let history = self.entities.entry(entity).or_default();
                history.record(data);
            }
        }
    }

    pub fn rollback(&mut self, index: usize, query: &mut Query<HistoryQuery>, scale: f32) {
        self.current_index = index;
        for (entity, history) in &self.entities {
            let Ok(entity) = query.get_mut(*entity) else {
                continue;
            };
            history.rollback(index, entity, scale);
        }
    }

    pub fn history(&self, entity: &Entity) -> Option<&EntityHistory> {
        self.entities.get(entity)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }
}

#[derive(Default)]
pub struct EntityHistory {
    pos: Vec<Vector3<f64>>,
    vel: Vec<Vector3<f64>>,

    att: Vec<UnitQuaternion<f64>>,
    ang_vel: Vec<Vector3<f64>>,

    mass: Vec<f64>,
    effects: Vec<Effect>,
}

impl EntityHistory {
    fn record(&mut self, query: HistoryQueryReadOnlyItem<'_>) {
        let data = query.entity;
        self.pos.push(data.pos.0);
        self.vel.push(data.vel.0);
        self.att.push(data.att.0);
        self.ang_vel.push(data.ang_vel.0);
        self.mass.push(data.mass.0);
        self.effects.push(*query.effect);
    }

    fn rollback(&self, index: usize, mut query: HistoryQueryItem<'_>, scale: f32) {
        let mut entity = query.entity;
        let att = self.att[index];
        let pos = self.pos[index];
        entity.pos.0 = pos;
        entity.vel.0 = self.vel[index];
        entity.att.0 = att;
        entity.ang_vel.0 = self.ang_vel[index];
        entity.mass.0 = self.mass[index];
        *query.effect = self.effects[index];
        query.transform.translation = Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32) * scale;
        query.transform.rotation =
            Quat::from_xyzw(att.i as f32, att.j as f32, att.k as f32, att.w as f32);
    }

    pub fn pos(&self) -> &[Vector3<f64>] {
        &self.pos
    }

    pub fn vel(&self) -> &[Vector3<f64>] {
        &self.vel
    }

    pub fn ang_vel(&self) -> &[Vector3<f64>] {
        &self.ang_vel
    }

    pub fn att(&self) -> &[UnitQuaternion<f64>] {
        &self.att
    }

    pub fn mass(&self) -> &[f64] {
        &self.mass
    }

    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }
}

pub fn record_system(
    mut history: ResMut<HistoryStore>,
    query: Query<(Entity, HistoryQueryReadOnly)>,
) {
    history.record(query.iter())
}

#[derive(Event)]
pub struct RollbackEvent(pub usize);

pub fn rollback_system(
    mut history: ResMut<HistoryStore>,
    mut event_reader: EventReader<RollbackEvent>,
    mut query: Query<HistoryQuery>,
    config: Res<Config>,
    mut polyline: Query<&mut Handle<Polyline>>,
    mut polylines: Option<ResMut<Assets<Polyline>>>,
) {
    for event in &mut event_reader {
        history.rollback(event.0, &mut query, config.scale);

        if let Some(ref mut polylines) = polylines {
            for polyline in &mut polyline {
                let polyline = polylines.get_mut(&polyline).unwrap();
                polyline.vertices.clear()
            }
        }
    }
}

pub struct HistoryPlugin;
impl Plugin for HistoryPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<RollbackEvent>();
        app.insert_resource(HistoryStore::default());
        app.add_systems(PhysicsSchedule, (record_system,).in_set(TickSet::SyncPos));
        app.add_systems(Update, rollback_system);
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct HistoryQuery {
    entity: EntityQuery,
    transform: &'static mut Transform,
    effect: &'static mut Effect,
}

#[cfg(test)]
mod tests {
    use bevy_ecs::{schedule::Schedule, world::World};
    use nalgebra::vector;

    use crate::{builder::EntityBuilder, Pos};

    use super::*;

    #[test]
    fn test_record() {
        let mut world = World::default();
        let a = world
            .spawn((
                EntityBuilder::default()
                    .pos(vector![1.0, 0.0, 0.0])
                    .bundle(),
                Transform::default(),
            ))
            .id();
        let b = world
            .spawn((
                EntityBuilder::default()
                    .pos(vector![0.0, 1.0, 0.0])
                    .bundle(),
                Transform::default(),
            ))
            .id();
        world.spawn((
            EntityBuilder::default()
                .pos(vector![0.0, 0.0, 1.0])
                .bundle(),
            Transform::default(),
        ));
        let mut history = HistoryStore::default();
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let mut a_entity = world.get_entity_mut(a).unwrap();
        a_entity.get_mut::<Pos>().unwrap().0 = vector![2.0, 0.0, 0.0];
        let mut b_entity = world.get_entity_mut(b).unwrap();
        b_entity.get_mut::<Pos>().unwrap().0 = vector![0.0, 2.0, 0.0];
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let a_history = history.history(&a).unwrap();
        assert_eq!(a_history.pos()[0], vector![1.0, 0.0, 0.0]);
        assert_eq!(a_history.pos()[1], vector![2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rewind() {
        let mut world = World::default();
        let a = world
            .spawn((
                EntityBuilder::default()
                    .pos(vector![1.0, 0.0, 0.0])
                    .bundle(),
                Transform::default(),
            ))
            .id();
        let b = world
            .spawn((
                EntityBuilder::default()
                    .pos(vector![0.0, 1.0, 0.0])
                    .bundle(),
                Transform::default(),
            ))
            .id();
        world.spawn((
            EntityBuilder::default()
                .pos(vector![0.0, 0.0, 1.0])
                .bundle(),
            Transform::default(),
        ));
        let mut history = HistoryStore::default();
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let mut a_entity = world.get_entity_mut(a).unwrap();
        a_entity.get_mut::<Pos>().unwrap().0 = vector![2.0, 0.0, 0.0];
        let mut b_entity = world.get_entity_mut(b).unwrap();
        b_entity.get_mut::<Pos>().unwrap().0 = vector![0.0, 2.0, 0.0];
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let mut rollback = Schedule::default();
        rollback.add_systems(move |mut query: Query<HistoryQuery>| {
            history.rollback(0, &mut query, 1.0);
        });
        rollback.run(&mut world);
        let a_entity = world.get_entity_mut(a).unwrap();
        assert_eq!(a_entity.get::<Pos>().unwrap().0, vector![1.0, 0.0, 0.0])
    }
}
