use crate::{
    hierarchy::{Link, TopologicalSort},
    types::{BiasForce, Effect, JointForce, WorldAccel},
    AngVel, Att, Inertia, Mass, Pos, Vel, WorldAtt, WorldVel,
};
use bevy::prelude::{Children, Parent};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without, WorldQuery},
    system::{Query, ResMut},
};
use nalgebra::{UnitQuaternion, UnitVector3, Vector3};
use std::ops::{Add, AddAssign, Mul, Sub};

pub fn pos_tree_step(parent: &SpatialPos, child: &SpatialPos, joint: &Joint) -> WorldPos {
    match joint.joint_type {
        JointType::Free => WorldPos(child.clone()),
        JointType::Revolute { .. } | JointType::Sphere | JointType::Fixed => {
            let att = parent.att * child.att;
            let pos = parent.pos + parent.att * joint.pos + att * child.pos;
            WorldPos(SpatialPos { pos, att })
        }
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct KinematicQuery {
    world_pos: &'static mut crate::types::WorldPos,
    world_vel: &'static mut crate::types::WorldVel,
    pos: &'static mut Pos,
    vel: &'static mut Vel,

    world_att: &'static mut WorldAtt,
    att: &'static mut Att,
    ang_vel: &'static mut AngVel,

    joint: &'static mut Joint,
}

pub fn kinematic_system(
    mut root_query: Query<(Entity, Option<&Children>, KinematicQuery), Without<Parent>>,
    mut query: Query<(Entity, KinematicQuery), With<Parent>>,
    children_query: Query<Option<&Children>, With<Parent>>,
) {
    fn recurisve_step(
        parent_pos: &SpatialPos,
        children: &Children,
        query: &mut Query<(Entity, KinematicQuery), With<Parent>>,
        children_query: &Query<Option<&Children>, With<Parent>>,
    ) {
        for child in children {
            let Ok((_, mut kinematic)) = query.get_mut(*child) else {
                continue;
            };
            let child_pos = SpatialPos {
                pos: kinematic.pos.0,
                att: kinematic.att.0,
            };
            let world_pos = pos_tree_step(parent_pos, &child_pos, &kinematic.joint);
            kinematic.world_pos.0 = world_pos.0.pos;
            kinematic.world_att.0 = world_pos.0.att;
            let children = children_query.get(*child);
            let Ok(Some(children)) = children else {
                continue;
            };
            recurisve_step(&world_pos.0, children, query, children_query);
        }
    }

    for (_, children, mut parent_kinematics) in root_query.iter_mut() {
        let parent_pos = SpatialPos {
            pos: parent_kinematics.pos.0,
            att: parent_kinematics.att.0,
        };

        parent_kinematics.world_pos.0 = parent_kinematics.pos.0;
        parent_kinematics.world_att.0 = parent_kinematics.att.0;

        let Some(children) = children else { continue };

        recurisve_step(&parent_pos, children, &mut query, &children_query)
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct RNEChildQuery {
    pos: &'static Pos,
    vel: &'static Vel,

    att: &'static Att,
    ang_vel: &'static AngVel,

    inertia: &'static Inertia,
    mass: &'static Mass,

    world_accel: &'static mut WorldAccel,
    world_vel: &'static mut WorldVel,
    bias_force: &'static mut BiasForce,

    joint_force: &'static mut JointForce,

    effect: &'static mut Effect,

    joint: &'static Joint,
}

pub fn rne_system(
    mut parent_query: Query<(&WorldAccel, &WorldVel, &mut BiasForce)>,
    mut child_query: Query<RNEChildQuery>,
    sort: ResMut<TopologicalSort>,
) {
    for Link { parent, child } in &sort.0 {
        let Ok(mut child) = child_query.get_mut(*child) else {
            continue;
        };
        let (vel, accel, force) = if let Some(parent) = parent {
            let Ok((WorldAccel(parent_accel), WorldVel(parent_vel), _)) = parent_query.get(*parent)
            else {
                continue;
            };
            forward_rne_step(
                child.joint,
                &parent_vel,
                &parent_accel,
                &SpatialPos {
                    pos: child.pos.0,
                    att: child.att.0,
                },
                &SpatialMotion {
                    vel: child.vel.0,
                    ang_vel: child.ang_vel.0,
                },
                &SpatialInertia {
                    inertia: *child.inertia,
                    momentum: child.vel.0 * child.mass.0, // TODO: this should maybe be world
                    mass: child.mass.0,
                },
                SpatialForce {
                    force: child.effect.force.0,
                    torque: child.effect.torque.0,
                },
            )
        } else {
            (
                SpatialMotion {
                    vel: child.vel.0,
                    ang_vel: child.ang_vel.0,
                },
                SpatialMotion::default(),
                SpatialForce {
                    force: child.effect.force.0,
                    torque: child.effect.torque.0,
                },
            )
        };
        child.world_vel.0 = vel;
        child.world_accel.0 = accel;
        child.bias_force.0 = force;
    }

    for Link { parent, child } in sort.0.iter().rev() {
        let Ok(mut child) = child_query.get_mut(*child) else {
            continue;
        };
        child.joint_force.0 = child.joint.apply_force_subspace(&child.bias_force.0);
        if let Some(parent) = parent {
            let Ok((_, _, mut bias_force)) = parent_query.get_mut(*parent) else {
                continue;
            };
            bias_force.0 += child.joint.apply_transpose_force(
                &SpatialPos {
                    pos: child.pos.0,
                    att: child.att.0,
                },
                &child.bias_force.0,
            );
        }
    }
}

fn forward_rne_step(
    joint: &Joint,
    parent_vel: &SpatialMotion,
    parent_accel: &SpatialMotion,
    child_pos: &SpatialPos,
    child_vel: &SpatialMotion,
    child_inertia: &SpatialInertia,
    force_ext: SpatialForce,
) -> (SpatialMotion, SpatialMotion, SpatialForce) {
    let joint_vel = joint.apply_motion_subspace(child_vel);
    let vel = joint.apply_transform_motion(child_pos, parent_vel) + joint_vel;
    let accel = joint.apply_transform_motion(child_pos, parent_accel) + vel.cross(&joint_vel);
    // NOTE: S_i * ddot(q_i)  is not included, because accel is set to zero
    let force = child_inertia * accel + vel.cross_dual(&(child_inertia * vel)) - force_ext;
    (vel, accel, force)
}

pub struct SpatialInertia {
    inertia: Inertia,
    momentum: Vector3<f64>,
    mass: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct SpatialMotion {
    vel: Vector3<f64>,
    ang_vel: Vector3<f64>,
}

#[derive(Clone, Copy)]
pub struct SpatialPos {
    pos: Vector3<f64>,
    att: UnitQuaternion<f64>,
}

pub struct WorldPos(SpatialPos);

#[derive(Component, Debug)]
pub struct Joint {
    pub pos: Vector3<f64>,
    pub joint_type: JointType,
}

#[derive(Debug)]
pub enum JointType {
    Free,
    Revolute { axis: UnitVector3<f64> },
    Sphere,
    Fixed,
}

impl Joint {
    fn apply_force_subspace(&self, force: &SpatialForce) -> SpatialForce {
        match self.joint_type {
            JointType::Free => force.clone(),
            JointType::Revolute { axis } => SpatialForce {
                force: Vector3::zeros(),
                torque: unit_project(force.torque, axis),
            },
            JointType::Sphere => SpatialForce {
                force: Vector3::zeros(),
                torque: force.torque,
            },
            JointType::Fixed => SpatialForce {
                force: Vector3::zeros(),
                torque: Vector3::zeros(),
            },
        }
    }

    fn apply_motion_subspace(&self, vel: &SpatialMotion) -> SpatialMotion {
        match self.joint_type {
            JointType::Free => vel.clone(),
            JointType::Revolute { axis } => SpatialMotion {
                vel: Vector3::zeros(),
                ang_vel: unit_project(vel.ang_vel, axis),
            },
            JointType::Sphere => SpatialMotion {
                vel: Vector3::zeros(),
                ang_vel: vel.ang_vel,
            },
            JointType::Fixed => SpatialMotion {
                vel: Vector3::zeros(),
                ang_vel: Vector3::zeros(),
            },
        }
    }

    fn apply_transform_motion(&self, child: &SpatialPos, motion: &SpatialMotion) -> SpatialMotion {
        match self.joint_type {
            JointType::Free => motion.clone(),
            JointType::Revolute { .. } | JointType::Sphere | JointType::Fixed => {
                motion.offset(&SpatialPos {
                    pos: child.att * self.pos + child.pos,
                    att: child.att,
                })
            }
        }
    }

    fn apply_transpose_force(&self, child: &SpatialPos, force: &SpatialForce) -> SpatialForce {
        match self.joint_type {
            JointType::Free => SpatialForce {
                force: Vector3::zeros(),
                torque: Vector3::zeros(),
            },
            JointType::Revolute { .. } | JointType::Sphere | JointType::Fixed => {
                let r = child.att * self.pos + child.pos;
                let f = child.att.inverse() * force.force;
                let torque = child.att.inverse() * force.torque + r.cross(&f);
                SpatialForce { force: f, torque }
            }
        }
    }
}

impl Sub for SpatialForce {
    type Output = SpatialForce;

    fn sub(self, rhs: SpatialForce) -> Self::Output {
        SpatialForce {
            force: self.force - rhs.force,
            torque: self.torque - rhs.torque,
        }
    }
}

impl Add for SpatialForce {
    type Output = SpatialForce;

    fn add(self, rhs: SpatialForce) -> Self::Output {
        SpatialForce {
            force: self.force + rhs.force,
            torque: self.torque + rhs.torque,
        }
    }
}

impl AddAssign for SpatialForce {
    fn add_assign(&mut self, rhs: Self) {
        self.force += rhs.force;
        self.torque += rhs.torque;
    }
}

impl Add for SpatialMotion {
    type Output = SpatialMotion;

    fn add(self, rhs: SpatialMotion) -> Self::Output {
        SpatialMotion {
            vel: self.vel + rhs.vel,
            ang_vel: self.ang_vel + rhs.ang_vel,
        }
    }
}

impl<'a> Mul<SpatialMotion> for &'a SpatialInertia {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialMotion) -> Self::Output {
        SpatialForce {
            force: self.mass * rhs.vel - self.momentum.cross(&rhs.ang_vel),
            torque: self.inertia.0 * rhs.ang_vel + self.momentum.cross(&rhs.vel),
        }
    }
}

impl Mul<SpatialMotion> for SpatialInertia {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialMotion) -> Self::Output {
        (&self).mul(rhs)
    }
}

impl SpatialMotion {
    fn offset(&self, pos: &SpatialPos) -> SpatialMotion {
        let ang_vel = pos.att * self.ang_vel;
        let vel = pos.att * self.vel + ang_vel.cross(&pos.pos);
        SpatialMotion { vel, ang_vel }
    }

    fn cross(&self, other: &SpatialMotion) -> SpatialMotion {
        let ang_vel = self.ang_vel.cross(&other.ang_vel);
        let vel = self.ang_vel.cross(&other.vel) + self.vel.cross(&other.ang_vel);
        SpatialMotion { vel, ang_vel }
    }

    fn cross_dual(&self, other: &SpatialForce) -> SpatialForce {
        SpatialForce {
            force: self.ang_vel.cross(&other.torque) + self.vel.cross(&other.force),
            torque: self.ang_vel.cross(&other.force),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialForce {
    force: Vector3<f64>,
    torque: Vector3<f64>,
}

fn unit_project(a: Vector3<f64>, b: UnitVector3<f64>) -> Vector3<f64> {
    a.dot(&b) * *b
}
