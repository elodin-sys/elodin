use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Query, Res},
};
use nalgebra::{UnitQuaternion, UnitVector3, Vector3};

use crate::types::{Config, EntityQuery};

use super::{apply_distance_constraint, apply_rot_constraint, pos_generalized_inverse_mass};

#[derive(Component)]
pub struct FixedJoint {
    pub entity_a: Entity,
    pub entity_b: Entity,
    pub anchor_a: Vector3<f64>,
    pub anchor_b: Vector3<f64>,
    pub compliance: f64,

    pub pos_lagrange: f64,
    pub angle_lagrange: f64,

    pub pos_damping: f64,
    pub ang_damping: f64,
}

impl FixedJoint {
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        FixedJoint {
            entity_a,
            entity_b,
            anchor_a: Vector3::default(),
            anchor_b: Vector3::default(),
            compliance: 0.0,
            pos_lagrange: 0.0,
            angle_lagrange: 0.0,
            pos_damping: 1.0,
            ang_damping: 1.0,
        }
    }

    pub fn anchor_a(mut self, pos: Vector3<f64>) -> Self {
        self.anchor_a = pos;
        self
    }

    pub fn anchor_b(mut self, pos: Vector3<f64>) -> Self {
        self.anchor_b = pos;
        self
    }

    pub fn compliance(mut self, compliance: f64) -> Self {
        self.compliance = compliance;
        self
    }

    pub fn ang_damping(mut self, ang_damping: f64) -> Self {
        self.ang_damping = ang_damping;
        self
    }

    pub fn pos_damping(mut self, pos_damping: f64) -> Self {
        self.pos_damping = pos_damping;
        self
    }
}

pub fn clear_fixed_lagrange(mut query: Query<&mut FixedJoint>) {
    query.par_iter_mut().for_each_mut(|mut c| {
        c.angle_lagrange = 0.0;
        c.pos_lagrange = 0.0;
    });
}

pub fn fixed_joint_system(
    mut query: Query<&mut FixedJoint>,
    mut bodies: Query<EntityQuery>,
    config: Res<Config>,
) {
    query.for_each_mut(|mut constraint| {
        let Ok([mut entity_a, mut entity_b]) =
            bodies.get_many_mut([constraint.entity_a, constraint.entity_b])
        else {
            return;
        };

        let world_anchor_a = entity_a.world_pos.0.att * constraint.anchor_a;
        let world_anchor_b = entity_b.world_pos.0.att * constraint.anchor_b;
        let dist = (world_anchor_a + entity_a.world_pos.0.pos)
            - (world_anchor_b + entity_b.world_pos.0.pos);
        let n = UnitVector3::new_normalize(dist);
        let c = dist.norm();
        let compliance = constraint.compliance;
        let delta_q = delta_q(entity_a.world_pos.0.att, entity_b.world_pos.0.att);
        apply_rot_constraint(
            &mut entity_a,
            &mut entity_b,
            delta_q,
            &mut constraint.angle_lagrange,
            compliance,
            config.sub_dt,
        );

        let inverse_mass_a = pos_generalized_inverse_mass(
            entity_a.mass.0,
            entity_b.world_pos.0.transform() * entity_a.inverse_inertia.0,
            world_anchor_a,
            n,
        );
        let inverse_mass_b = pos_generalized_inverse_mass(
            entity_b.mass.0,
            entity_b.world_pos.0.transform() * entity_a.inverse_inertia.0,
            world_anchor_b,
            n,
        );

        apply_distance_constraint(
            &mut entity_a,
            &mut entity_b,
            c,
            n,
            inverse_mass_a,
            inverse_mass_b,
            &mut constraint.pos_lagrange,
            compliance,
            config.sub_dt,
            world_anchor_a,
            world_anchor_b,
        );
    })
}

pub fn fixed_damping(
    query: Query<&FixedJoint>,
    mut bodies: Query<EntityQuery>,
    config: Res<Config>,
) {
    for constraint in &query {
        let Ok([mut entity_a, mut entity_b]) =
            bodies.get_many_mut([constraint.entity_a, constraint.entity_b])
        else {
            return;
        };

        let delta_v = (entity_b.world_vel.0.vel - entity_a.world_vel.0.vel)
            * (constraint.pos_damping * config.sub_dt).min(1.0);

        let delta_omega = (entity_b.world_vel.0.ang_vel - entity_a.world_vel.0.ang_vel)
            * (constraint.ang_damping * config.sub_dt).min(1.0);

        if !entity_a.fixed.0 {
            entity_a.world_vel.0.ang_vel += delta_omega;
        }

        if !entity_b.fixed.0 {
            entity_b.world_vel.0.ang_vel -= delta_omega;
        }

        let w_a = if entity_a.fixed.0 {
            0.0
        } else {
            1.0 / entity_a.mass.0
        };
        let w_b = if entity_b.fixed.0 {
            0.0
        } else {
            1.0 / entity_b.mass.0
        };

        let w_sum = w_a + w_b;
        if w_sum <= f64::EPSILON {
            continue;
        }
        let p = delta_v / w_sum;
        if !entity_a.fixed.0 {
            entity_a.world_vel.0.vel += w_a * p;
        }
        if !entity_b.fixed.0 {
            entity_b.world_vel.0.vel -= w_b * p;
        }
    }
}

fn delta_q(att_a: UnitQuaternion<f64>, att_b: UnitQuaternion<f64>) -> Vector3<f64> {
    2.0 * (att_a * att_b.inverse()).vector()
}
