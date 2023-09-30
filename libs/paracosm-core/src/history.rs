use bevy_ecs::{entity::Entity, system::Query};
use nalgebra::{UnitQuaternion, Vector3};
use std::collections::HashMap;

use crate::xpbd::components::{Effect, EntityQuery, EntityQueryItem, EntityQueryReadOnlyItem};

#[derive(Default)]
pub struct HistoryStore {
    entities: HashMap<Entity, EntityHistory>,
}

impl HistoryStore {
    pub fn record<'a>(
        &mut self,
        query: impl Iterator<Item = (Entity, EntityQueryReadOnlyItem<'a>)>,
    ) {
        for (entity, data) in query {
            let history = self.entities.entry(entity).or_default();
            history.record(data);
        }
    }

    pub fn rollback(&mut self, index: usize, mut query: Query<EntityQuery>) {
        for (entity, history) in &self.entities {
            let Ok(entity) = query.get_mut(*entity) else {
                continue;
            };
            history.rollback(index, entity);
        }
    }

    pub fn history(&self, entity: &Entity) -> Option<&EntityHistory> {
        self.entities.get(entity)
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
    fn record(&mut self, data: EntityQueryReadOnlyItem<'_>) {
        self.pos.push(data.pos.0);
        self.vel.push(data.vel.0);
        self.att.push(data.att.0);
        self.ang_vel.push(data.ang_vel.0);
        self.mass.push(data.mass.0);
        self.effects.push(*data.effect);
    }

    fn rollback(&self, index: usize, mut entity: EntityQueryItem<'_>) {
        entity.pos.0 = self.pos[index];
        entity.vel.0 = self.vel[index];
        entity.att.0 = self.att[index];
        entity.ang_vel.0 = self.ang_vel[index];
        entity.mass.0 = self.mass[index];
        *entity.effect = self.effects[index];
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

#[cfg(test)]
mod tests {
    use bevy_ecs::{schedule::Schedule, world::World};
    use nalgebra::vector;

    use crate::{
        xpbd::{builder::EntityBuilder, components::EntityQueryReadOnly},
        Pos,
    };

    use super::*;

    #[test]
    fn test_record() {
        let mut world = World::default();
        let a = world
            .spawn(
                EntityBuilder::default()
                    .pos(vector![1.0, 0.0, 0.0])
                    .bundle(),
            )
            .id();
        let b = world
            .spawn(
                EntityBuilder::default()
                    .pos(vector![0.0, 1.0, 0.0])
                    .bundle(),
            )
            .id();
        world.spawn(
            EntityBuilder::default()
                .pos(vector![0.0, 0.0, 1.0])
                .bundle(),
        );
        let mut history = HistoryStore::default();
        history.record(world.query::<(Entity, EntityQueryReadOnly)>().iter(&world));
        let mut a_entity = world.get_entity_mut(a).unwrap();
        a_entity.get_mut::<Pos>().unwrap().0 = vector![2.0, 0.0, 0.0];
        let mut b_entity = world.get_entity_mut(b).unwrap();
        b_entity.get_mut::<Pos>().unwrap().0 = vector![0.0, 2.0, 0.0];
        history.record(world.query::<(Entity, EntityQueryReadOnly)>().iter(&world));
        let a_history = history.history(&a).unwrap();
        assert_eq!(a_history.pos()[0], vector![1.0, 0.0, 0.0]);
        assert_eq!(a_history.pos()[1], vector![2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rewind() {
        let mut world = World::default();
        let a = world
            .spawn(
                EntityBuilder::default()
                    .pos(vector![1.0, 0.0, 0.0])
                    .bundle(),
            )
            .id();
        let b = world
            .spawn(
                EntityBuilder::default()
                    .pos(vector![0.0, 1.0, 0.0])
                    .bundle(),
            )
            .id();
        world.spawn(
            EntityBuilder::default()
                .pos(vector![0.0, 0.0, 1.0])
                .bundle(),
        );
        let mut history = HistoryStore::default();
        history.record(world.query::<(Entity, EntityQueryReadOnly)>().iter(&world));
        let mut a_entity = world.get_entity_mut(a).unwrap();
        a_entity.get_mut::<Pos>().unwrap().0 = vector![2.0, 0.0, 0.0];
        let mut b_entity = world.get_entity_mut(b).unwrap();
        b_entity.get_mut::<Pos>().unwrap().0 = vector![0.0, 2.0, 0.0];
        history.record(world.query::<(Entity, EntityQueryReadOnly)>().iter(&world));
        let mut rollback = Schedule::default();
        rollback.add_systems(move |query: Query<EntityQuery>| {
            history.rollback(0, query);
        });
        rollback.run(&mut world);
        let a_entity = world.get_entity_mut(a).unwrap();
        assert_eq!(a_entity.get::<Pos>().unwrap().0, vector![1.0, 0.0, 0.0])
    }
}
