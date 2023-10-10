use bevy_ecs::{
    prelude::{Component, Entity},
    system::{Query, Res},
};
use nalgebra::{UnitVector3, Vector3};

use crate::{
    types::{Config, EntityQuery},
    Pos,
};

use super::{apply_distance_constraint, pos_generalized_inverse_mass};

#[derive(Component, Debug, Clone)]
pub struct DistanceConstraint {
    pub entity_a: Entity,
    pub entity_b: Entity,
    pub anchor_a: Pos,
    pub anchor_b: Pos,
    pub distance_target: f64,
    pub compliance: f64,
    pub lagrange_multiplier: f64,
}

impl DistanceConstraint {
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        DistanceConstraint {
            entity_a,
            entity_b,
            anchor_a: Pos(Vector3::zeros()),
            anchor_b: Pos(Vector3::zeros()),
            distance_target: 1.0,
            compliance: 0.001,
            lagrange_multiplier: 0.0,
        }
    }

    pub fn anchor_a(mut self, pos: impl Into<Pos>) -> Self {
        self.anchor_a = pos.into();
        self
    }

    pub fn anchor_b(mut self, pos: impl Into<Pos>) -> Self {
        self.anchor_b = pos.into();
        self
    }

    pub fn distance_target(mut self, distance: f64) -> Self {
        self.distance_target = distance;
        self
    }
    pub fn compliance(mut self, compliance: f64) -> Self {
        self.compliance = compliance;
        self
    }
}

pub fn clear_distance_lagrange(mut query: Query<&mut DistanceConstraint>) {
    query.par_iter_mut().for_each_mut(|mut c| {
        c.lagrange_multiplier = 0.0;
    });
}

pub fn distance_system(
    mut query: Query<&mut DistanceConstraint>,
    mut bodies: Query<EntityQuery>,
    config: Res<Config>,
) {
    query.for_each_mut(|mut constraint| {
        let Ok([mut entity_a, mut entity_b]) =
            bodies.get_many_mut([constraint.entity_a, constraint.entity_b])
        else {
            return;
        };
        let world_anchor_a = constraint.anchor_a.to_world_basis(&entity_a);
        let world_anchor_b = constraint.anchor_b.to_world_basis(&entity_b);
        let dist = (world_anchor_a.0 + entity_a.pos.0) - (world_anchor_b.0 + entity_b.pos.0);
        let c = dist.norm() - constraint.distance_target;
        let n = UnitVector3::new_normalize(dist);

        let inverse_mass_a = pos_generalized_inverse_mass(
            entity_a.mass.0,
            entity_a.inverse_inertia.to_world(&entity_a).0,
            world_anchor_a.0,
            n,
        );
        let inverse_mass_b = pos_generalized_inverse_mass(
            entity_a.mass.0,
            entity_b.inverse_inertia.to_world(&entity_b).0,
            world_anchor_b.0,
            n,
        );

        let compliance = constraint.compliance;
        apply_distance_constraint(
            &mut entity_a,
            &mut entity_b,
            c,
            n,
            inverse_mass_a,
            inverse_mass_b,
            &mut constraint.lagrange_multiplier,
            compliance,
            config.sub_dt,
            world_anchor_a,
            world_anchor_b,
        );
    });
}
