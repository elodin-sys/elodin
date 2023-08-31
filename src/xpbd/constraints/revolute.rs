use std::ops::Range;

use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Query, Res},
};
use nalgebra::{UnitQuaternion, UnitVector3, Vector3};

use crate::{
    xpbd::components::{Config, EntityQuery},
    Pos,
};

use super::{
    apply_distance_constraint, apply_rot_constraint, pos_generalized_inverse_mass,
    rot_generalized_inverse_mass,
};

#[derive(Component, Debug, Clone)]
pub struct RevoluteJoint {
    pub entity_a: Entity,
    pub entity_b: Entity,
    pub anchor_a: Pos,
    pub anchor_b: Pos,
    pub joint_axis: UnitVector3<f64>,
    pub angle_limits: Option<Range<f64>>,
    pub compliance: f64,

    pub angle_limit_lagrange: f64,
    pub pos_lagrange: f64,
    pub angle_lagrange: f64,

    pub pos_damping: f64,
    pub ang_damping: f64,
}

impl RevoluteJoint {
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        RevoluteJoint {
            entity_a,
            entity_b,
            anchor_a: Pos(Vector3::default()),
            anchor_b: Pos(Vector3::default()),
            joint_axis: Vector3::x_axis(),
            angle_limits: None,
            compliance: 1.0 / 100.0,
            angle_limit_lagrange: 0.0,
            pos_lagrange: 0.0,
            angle_lagrange: 0.0,
            pos_damping: 1.0,
            ang_damping: 1.0,
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

    pub fn join_axis(mut self, axis: UnitVector3<f64>) -> Self {
        self.joint_axis = axis;
        self
    }

    pub fn angle_limits(mut self, limits: impl Into<Option<Range<f64>>>) -> Self {
        self.angle_limits = limits.into();
        self
    }

    pub fn compliance(mut self, compliance: f64) -> Self {
        self.compliance = compliance;
        self
    }
}

pub fn clear_revolute_lagrange(mut query: Query<&mut RevoluteJoint>) {
    query.par_iter_mut().for_each_mut(|mut c| {
        c.angle_lagrange = 0.0;
        c.angle_limit_lagrange = 0.0;
        c.pos_lagrange = 0.0;
    });
}

pub fn revolute_system(
    mut query: Query<&mut RevoluteJoint>,
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
        let n = UnitVector3::new_normalize(dist);
        let c = dist.norm();

        let compliance = constraint.compliance;
        let delta_q = delta_q(entity_a.att.0, entity_b.att.0, constraint.joint_axis);
        let inverse_inertia_a = entity_a.inverse_inertia.to_world(&entity_a);
        let inverse_inertia_b = entity_b.inverse_inertia.to_world(&entity_b);
        let inverse_mass_a = rot_generalized_inverse_mass(inverse_inertia_a.0, n);
        let inverse_mass_b = rot_generalized_inverse_mass(inverse_inertia_b.0, n);

        apply_rot_constraint(
            &mut entity_a,
            &mut entity_b,
            delta_q,
            inverse_mass_a,
            inverse_mass_b,
            &mut constraint.angle_lagrange,
            compliance,
            config.sub_dt,
        );
        let inverse_mass_a = pos_generalized_inverse_mass(
            entity_a.mass.0,
            entity_a.inverse_inertia.to_world(&entity_a).0,
            world_anchor_a.0,
            n,
        );
        let inverse_mass_b = pos_generalized_inverse_mass(
            entity_b.mass.0,
            entity_b.inverse_inertia.to_world(&entity_b).0,
            world_anchor_b.0,
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

pub fn revolute_damping(
    query: Query<&RevoluteJoint>,
    mut bodies: Query<EntityQuery>,
    config: Res<Config>,
) {
    for constraint in &query {
        let Ok([mut entity_a, mut entity_b]) =
            bodies.get_many_mut([constraint.entity_a, constraint.entity_b])
        else {
            return;
        };

        let delta_v =
            (entity_b.vel.0 - entity_a.vel.0) * (constraint.pos_damping * config.sub_dt).min(1.0);

        let delta_omega = (entity_b.ang_vel.0 - entity_a.ang_vel.0)
            * (constraint.ang_damping * config.sub_dt).min(1.0);

        if !entity_a.fixed.0 {
            entity_a.ang_vel.0 += delta_omega;
        }

        if !entity_b.fixed.0 {
            entity_b.ang_vel.0 -= delta_omega;
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
            entity_a.vel.0 += w_a * p;
        }
        if !entity_b.fixed.0 {
            entity_b.vel.0 -= w_b * p;
        }
    }
}

pub fn delta_q(
    att_a: UnitQuaternion<f64>,
    att_b: UnitQuaternion<f64>,
    axis: UnitVector3<f64>,
) -> Vector3<f64> {
    let axis_a = att_a * axis;
    let axis_b = att_b * axis;
    axis_a.into_inner().cross(&*axis_b)
}
