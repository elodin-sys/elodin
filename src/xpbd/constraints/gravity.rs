use bevy_ecs::{component::Component, entity::Entity, system::Query};

use crate::xpbd::components::{Effect, EntityQuery};

#[derive(Component)]
pub struct GravityConstriant {
    entity_a: Entity,
    entity_b: Entity,
}

const G: f64 = 6.649e-11;

impl GravityConstriant {
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        Self { entity_a, entity_b }
    }
}

pub fn gravity_system(
    query: Query<&mut GravityConstriant>,
    mut bodies: Query<(EntityQuery, &mut Effect)>,
) {
    for constraint in &query {
        let Ok([(entity_a, mut effect_a), (entity_b, mut effect_b)]) =
            bodies.get_many_mut([constraint.entity_a, constraint.entity_b])
        else {
            return;
        };
        let r = entity_a.pos.0 - entity_b.pos.0;
        let mu = (r / r.norm().powi(3) * G) * entity_a.mass.0 * entity_b.mass.0;
        effect_a.force.0 -= mu;
        effect_b.force.0 += mu;
    }
}
