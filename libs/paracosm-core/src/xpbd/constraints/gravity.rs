use bevy_ecs::{component::Component, entity::Entity, system::Query};

use crate::xpbd::components::{Effect, EntityQuery};

#[derive(Component)]
pub struct GravityConstraint {
    entity_a: Entity,
    entity_b: Entity,
    g_constant: f64,
}

const G: f64 = 6.649e-11;

impl GravityConstraint {
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        Self {
            entity_a,
            entity_b,
            g_constant: G,
        }
    }

    pub fn constant(mut self, constant: f64) -> Self {
        self.g_constant = constant;
        self
    }
}

pub fn gravity_system(
    query: Query<&mut GravityConstraint>,
    mut bodies: Query<(EntityQuery, &mut Effect)>,
) {
    for constraint in &query {
        let Ok([(entity_a, mut effect_a), (entity_b, mut effect_b)]) =
            bodies.get_many_mut([constraint.entity_a, constraint.entity_b])
        else {
            return;
        };
        let r = entity_a.pos.0 - entity_b.pos.0;
        let mu = (r / r.norm().powi(3) * constraint.g_constant) * entity_a.mass.0 * entity_b.mass.0;
        effect_a.force.0 -= mu;
        effect_b.force.0 += mu;
    }
}
