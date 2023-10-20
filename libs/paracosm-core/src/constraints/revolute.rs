use std::{f64::consts::PI, fmt::Debug, ops::Range};

use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Query, Res},
};
use nalgebra::{UnitQuaternion, UnitVector3, Vector3};

use crate::{
    effector::{concrete_effector, Effector},
    types::{Config, EntityQuery},
    FromState, Time,
};

use super::{apply_distance_constraint, apply_rot_constraint, pos_generalized_inverse_mass};

#[derive(Component)]
pub struct RevoluteJoint {
    pub entity_a: Entity,
    pub entity_b: Entity,
    pub anchor_a: Vector3<f64>,
    pub anchor_b: Vector3<f64>,
    pub joint_axis: UnitVector3<f64>,
    pub compliance: f64,

    pub angle_limits: Option<AngleLimits>,

    pub angle_limit_lagrange: f64,
    pub pos_lagrange: f64,
    pub angle_lagrange: f64,

    pub pos_damping: f64,
    pub ang_damping: f64,

    pub effector: Option<Box<dyn RevoluteEffector + Send + Sync>>,
}

impl RevoluteJoint {
    pub fn new(entity_a: Entity, entity_b: Entity) -> Self {
        RevoluteJoint {
            entity_a,
            entity_b,
            anchor_a: Vector3::default(),
            anchor_b: Vector3::default(),
            joint_axis: Vector3::x_axis(),
            angle_limits: None,
            compliance: 1.0 / 100.0,
            angle_limit_lagrange: 0.0,
            pos_lagrange: 0.0,
            angle_lagrange: 0.0,
            pos_damping: 1.0,
            ang_damping: 1.0,
            effector: None,
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

    pub fn join_axis(mut self, axis: UnitVector3<f64>) -> Self {
        self.joint_axis = axis;
        self
    }

    pub fn angle_limits(mut self, limits: impl Into<Option<Range<f64>>>) -> Self {
        self.angle_limits = limits.into().map(|range| AngleLimits { range });
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

    pub fn effector<T, E, EF>(mut self, effector: E) -> Self
    where
        T: 'static + Send + Sync,
        E: for<'a> Effector<T, &'a RevoluteJoint, Effect = EF> + Send + Sync + 'static,
        EF: Into<JointSetPoint> + Send + Sync,
    {
        let unified: ConcereteRevoluteEffector<E, T> = ConcereteRevoluteEffector::new(effector);
        self.effector = Some(Box::new(unified));
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
    time: Res<Time>,
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
        let delta_q = delta_q(
            entity_a.world_pos.0.att,
            entity_b.world_pos.0.att,
            constraint.joint_axis,
        );
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
            entity_a.world_pos.0.transform() * entity_a.inverse_inertia.0,
            world_anchor_a,
            n,
        );
        let inverse_mass_b = pos_generalized_inverse_mass(
            entity_b.mass.0,
            entity_b.world_pos.0.transform() * entity_b.inverse_inertia.0,
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

        if let Some(ref angle_limit) = constraint.angle_limits {
            // NOTE(sphw): This is sourced from `revolute.rs` in bevy_xpbd, I'm not sure exavtly what the algo is doing, and I can't source it
            // so if something is broken with angle limits look here

            let limit_axis = Vector3::new(
                constraint.joint_axis.z,
                constraint.joint_axis.x,
                constraint.joint_axis.y,
            );
            let b1 = entity_a.world_pos.0.att * limit_axis;
            let b2 = entity_b.world_pos.0.att * limit_axis;
            let n = b1.cross(&b2).normalize();
            if let Some(delta_q) = angle_limit.delta_q(&n, b1, b2) {
                apply_rot_constraint(
                    &mut entity_a,
                    &mut entity_b,
                    delta_q,
                    &mut constraint.angle_limit_lagrange,
                    compliance,
                    config.sub_dt,
                );
            }
        }

        if let Some(ref effector) = constraint.effector {
            if let Some(angle) = effector.effect(*time, &constraint).theta {
                let perp_axis = Vector3::new(
                    constraint.joint_axis.z,
                    constraint.joint_axis.x,
                    constraint.joint_axis.y,
                );
                let b1 = entity_a.world_pos.0.att * perp_axis;
                let b2 = entity_b.world_pos.0.att * perp_axis;
                let a1 = entity_a.world_pos.0.att * constraint.joint_axis;
                let b_target = UnitQuaternion::from_axis_angle(&a1, angle) * b1;
                let delta_q_target = b_target.cross(&b2);

                apply_rot_constraint(
                    &mut entity_a,
                    &mut entity_b,
                    delta_q_target,
                    &mut constraint.angle_lagrange,
                    compliance,
                    config.sub_dt,
                );
            }
        }
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

pub fn delta_q(
    att_a: UnitQuaternion<f64>,
    att_b: UnitQuaternion<f64>,
    axis: UnitVector3<f64>,
) -> Vector3<f64> {
    let axis_a = att_a * axis;
    let axis_b = att_b * axis;
    axis_a.into_inner().cross(&*axis_b)
}

#[derive(Clone, Debug)]
pub struct AngleLimits {
    range: Range<f64>,
}

impl AngleLimits {
    fn delta_q(
        &self,
        n: &Vector3<f64>,
        n1: Vector3<f64>,
        n2: Vector3<f64>,
    ) -> Option<Vector3<f64>> {
        let mut phi = n1.cross(&n2).dot(n).asin();
        if n1.dot(&n2) < 0.0 {
            phi = PI - phi
        };
        if phi > PI {
            phi -= 2.0 * PI;
        }
        if phi < -PI {
            phi += 2.0 * PI;
        }

        if phi < self.range.start || phi > self.range.end {
            phi = phi.clamp(self.range.start, self.range.end);
            let n1 = UnitQuaternion::from_axis_angle(&UnitVector3::new_normalize(*n), phi) * n1;
            let mut dq = n1.cross(&n2);
            let norm = dq.norm();
            if norm > PI {
                dq *= PI / norm;
            }
            return Some(dq);
        }
        None
    }
}

concrete_effector!(
    ConcereteRevoluteEffector,
    RevoluteEffector,
    &'s RevoluteJoint,
    JointSetPoint
);

#[derive(Default)]
pub struct JointSetPoint {
    pub theta: Option<f64>,
    pub omega: Option<f64>,
}

pub struct Angle(pub f64);
pub struct AngleVel(pub f64);
impl From<Angle> for JointSetPoint {
    fn from(val: Angle) -> Self {
        JointSetPoint {
            theta: Some(val.0),
            omega: None,
        }
    }
}

impl From<AngleVel> for JointSetPoint {
    fn from(val: AngleVel) -> Self {
        JointSetPoint {
            theta: None,
            omega: Some(val.0),
        }
    }
}

impl<T> From<Option<T>> for JointSetPoint
where
    JointSetPoint: From<T>,
{
    fn from(val: Option<T>) -> Self {
        val.map(JointSetPoint::from).unwrap_or_default()
    }
}

impl<'a> FromState<&'a RevoluteJoint> for Time {
    fn from_state(time: Time, _state: &&'a RevoluteJoint) -> Self {
        time
    }
}
