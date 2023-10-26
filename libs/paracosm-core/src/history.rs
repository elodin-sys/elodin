use bevy::prelude::{App, Assets, Handle, Plugin, Transform, Update};
use bevy_ecs::{
    entity::Entity,
    event::{Event, EventReader},
    query::WorldQuery,
    schedule::IntoSystemConfigs,
    system::{Query, Res, ResMut, Resource},
};
use bevy_polyline::prelude::Polyline;
use std::collections::HashMap;

use crate::{
    plugin::{PhysicsSchedule, TickSet},
    spatial::{GeneralizedMotion, GeneralizedPos, SpatialMotion, SpatialPos},
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
    pos: Vec<GeneralizedPos>,
    vel: Vec<GeneralizedMotion>,

    world_pos: Vec<SpatialPos>,
    world_vel: Vec<SpatialMotion>,

    mass: Vec<f64>,
    effects: Vec<Effect>,
}

impl EntityHistory {
    fn record(&mut self, query: HistoryQueryReadOnlyItem<'_>) {
        let data = query.entity;
        self.pos.push(data.pos.0);
        self.vel.push(data.vel.0);
        self.mass.push(data.mass.0);
        self.effects.push(*query.effect);
        self.world_pos.push(data.world_pos.0);
        self.world_vel.push(data.world_vel.0);
    }

    fn rollback(&self, index: usize, mut query: HistoryQueryItem<'_>, scale: f32) {
        let mut entity = query.entity;
        let pos = self.pos[index];
        entity.pos.0 = pos;
        entity.vel.0 = self.vel[index];
        entity.mass.0 = self.mass[index];
        *query.effect = self.effects[index];
        *query.transform = self.world_pos[index].bevy(scale);
    }

    pub fn pos(&self) -> &[GeneralizedPos] {
        &self.pos
    }

    pub fn world_pos(&self) -> &[SpatialPos] {
        &self.world_pos
    }

    pub fn vel(&self) -> &[GeneralizedMotion] {
        &self.vel
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

    use crate::{
        builder::{EntityBuilder, Free},
        JointPos,
    };

    use super::*;

    #[test]
    fn test_record() {
        let mut world = World::default();
        let a = world
            .spawn((
                EntityBuilder::default()
                    .joint(Free::default().pos(SpatialPos::linear(vector![1.0, 0.0, 0.0])))
                    .bundle(),
                Transform::default(),
            ))
            .id();
        let b = world
            .spawn((
                EntityBuilder::default()
                    .joint(Free::default().pos(SpatialPos::linear(vector![0.0, 1.0, 0.0])))
                    .bundle(),
                Transform::default(),
            ))
            .id();
        world.spawn((
            EntityBuilder::default()
                .joint(Free::default().pos(SpatialPos::linear(vector![0.0, 0.0, 1.0])))
                .bundle(),
            Transform::default(),
        ));
        let mut history = HistoryStore::default();
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let mut a_entity = world.get_entity_mut(a).unwrap();
        a_entity.get_mut::<JointPos>().unwrap().0.inner =
            SpatialPos::linear(vector![2.0, 0.0, 0.0]).vector();
        let mut b_entity = world.get_entity_mut(b).unwrap();
        b_entity.get_mut::<JointPos>().unwrap().0.inner =
            SpatialPos::linear(vector![0.0, 2.0, 0.0]).vector();
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let a_history = history.history(&a).unwrap();
        assert_eq!(
            a_history.pos()[0].inner.as_slice(),
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            a_history.pos()[1].inner.as_slice(),
            &[0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_rewind() {
        let mut world = World::default();
        let a = world
            .spawn((
                EntityBuilder::default()
                    .joint(Free::default().pos(SpatialPos::linear(vector![1.0, 0.0, 0.0])))
                    .bundle(),
                Transform::default(),
            ))
            .id();
        let b = world
            .spawn((
                EntityBuilder::default()
                    .joint(Free::default().pos(SpatialPos::linear(vector![0.0, 1.0, 0.0])))
                    .bundle(),
                Transform::default(),
            ))
            .id();
        world.spawn((
            EntityBuilder::default()
                .joint(Free::default().pos(SpatialPos::linear(vector![0.0, 0.0, 1.0])))
                .bundle(),
            Transform::default(),
        ));
        let mut history = HistoryStore::default();
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let mut a_entity = world.get_entity_mut(a).unwrap();
        a_entity.get_mut::<JointPos>().unwrap().0.inner =
            SpatialPos::linear(vector![2.0, 0.0, 0.0]).vector();
        let mut b_entity = world.get_entity_mut(b).unwrap();
        b_entity.get_mut::<JointPos>().unwrap().0.inner =
            SpatialPos::linear(vector![0.0, 2.0, 0.0]).vector();
        history.record(world.query::<(Entity, HistoryQueryReadOnly)>().iter(&world));
        let mut rollback = Schedule::default();
        rollback.add_systems(move |mut query: Query<HistoryQuery>| {
            history.rollback(0, &mut query, 1.0);
        });
        rollback.run(&mut world);
        let a_entity = world.get_entity_mut(a).unwrap();
        assert_eq!(
            a_entity.get::<JointPos>().unwrap().0.inner.as_slice(),
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        )
    }
}
