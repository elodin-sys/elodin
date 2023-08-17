use bevy_ecs::world::{EntityMut, World};

use crate::Time;

use self::{builder::EntityBuilder, components::Config, systems::SubstepSchedule};

pub mod body;
pub mod builder;
pub mod components;
pub mod systems;

pub struct Xpbd {
    world: bevy_ecs::world::World,
}

impl Default for Xpbd {
    fn default() -> Self {
        let mut world = World::new();
        world.insert_resource(Time(0.0));
        world.insert_resource(Config { dt: 0.001 });
        world.add_schedule(systems::schedule(), SubstepSchedule);
        Self { world }
    }
}

impl Xpbd {
    pub fn entity(&mut self, entity: EntityBuilder) -> EntityMut<'_> {
        let bundle = entity.bundle();
        self.world.spawn(bundle)
    }

    pub fn tick(&mut self) {
        self.world.run_schedule(SubstepSchedule)
    }
}
