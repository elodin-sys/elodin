use std::f64::consts::PI;

use bevy::prelude::{Children, Parent};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without, WorldQuery},
    system::Query,
};
use nalgebra::{Quaternion, UnitQuaternion, UnitVector3, Vector3};

use crate::{AngVel, Att, Pos, Vel, WorldAngVel, WorldAtt};

pub fn pos_tree_step(parent: &SpatialPos, child: &SpatialPos, joint: &Joint) -> WorldPos {
    match joint.joint_type {
        JointType::Free => WorldPos(child.clone()),
        JointType::Revolute { axis } => {
            let (_, twist) = swing_twist_decomp(child.att, axis);
            // NOTE: The above step is done purley for safety, it shouldn't be neccesary given child.att should only ever
            // accumulate rotations around the
            let att = parent.att * twist;
            let pos = parent.pos + parent.att * joint.pos + att * child.pos;
            WorldPos(SpatialPos { pos, att })
        }
        JointType::Sphere | JointType::Fixed => {
            let att = parent.att * child.att;
            let pos = parent.pos + parent.att * joint.pos + att * child.pos;
            WorldPos(SpatialPos { pos, att })
        }
    }
}

pub fn vel_tree_step(
    parent: &SpatialVel,
    child_pos: &SpatialPos,
    child_vel: SpatialVel,
    joint: &Joint,
) -> WorldVel {
    match joint.joint_type {
        JointType::Free => WorldVel(child_vel),
        JointType::Revolute { axis } => {
            let child_ang_vel = unit_project(child_vel.ang_vel, axis);
            let parent_ang_vel = parent.ang_vel - unit_project(parent.ang_vel, axis);
            let ang_vel = child_ang_vel + parent_ang_vel;
            let vel = parent.vel + ang_vel.cross(&child_pos.pos);
            WorldVel(SpatialVel { vel, ang_vel })
        }
        JointType::Sphere => WorldVel(SpatialVel {
            vel: parent.vel,
            ang_vel: child_vel.ang_vel,
        }),
        JointType::Fixed => WorldVel(parent.clone()),
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
    world_ang_vel: &'static mut WorldAngVel,
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
        parent_vel: &SpatialVel,
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
            let child_vel = SpatialVel {
                vel: kinematic.vel.0,
                ang_vel: kinematic.ang_vel.0,
            };
            let world_pos = pos_tree_step(parent_pos, &child_pos, &kinematic.joint);
            let world_vel = vel_tree_step(parent_vel, &child_pos, child_vel, &kinematic.joint);
            kinematic.world_pos.0 = world_pos.0.pos;
            kinematic.world_att.0 = world_pos.0.att;
            let children = children_query.get(*child);
            let Ok(Some(children)) = children else {
                continue;
            };
            recurisve_step(&world_pos.0, &world_vel.0, children, query, children_query);
        }
    }

    for (_, children, mut parent_kinematics) in root_query.iter_mut() {
        let parent_pos = SpatialPos {
            pos: parent_kinematics.pos.0,
            att: parent_kinematics.att.0,
        };
        let parent_vel = SpatialVel {
            vel: parent_kinematics.vel.0,
            ang_vel: parent_kinematics.ang_vel.0,
        };

        parent_kinematics.world_pos.0 = parent_kinematics.pos.0;
        parent_kinematics.world_att.0 = parent_kinematics.att.0;
        parent_kinematics.world_vel.0 = parent_kinematics.vel.0;
        parent_kinematics.world_ang_vel.0 = parent_kinematics.ang_vel.0;

        let Some(children) = children else { continue };

        recurisve_step(
            &parent_pos,
            &parent_vel,
            children,
            &mut query,
            &children_query,
        )
    }
}

#[derive(Clone)]
pub struct SpatialVel {
    vel: Vector3<f64>,
    ang_vel: Vector3<f64>,
}

pub struct WorldVel(SpatialVel);

#[derive(Clone)]
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

fn swing_twist_decomp(
    rot: UnitQuaternion<f64>,
    axis: UnitVector3<f64>,
) -> (UnitQuaternion<f64>, UnitQuaternion<f64>) {
    let rot_norm = rot.vector().norm();
    if rot_norm < f64::EPSILON {
        let twist = UnitQuaternion::new_normalize(Quaternion::from_parts(
            rot.scalar(),
            unit_project(rot.vector().into(), axis),
        ));
        let swing = rot / twist;
        (swing, twist)
    } else {
        let twist = UnitQuaternion::from_axis_angle(&axis, PI);
        let rot_twist = rot * axis;
        let swing_axis = axis.cross(&rot_twist);
        let swing = if let Some(swing_axis) = UnitVector3::try_new(swing_axis, f64::EPSILON) {
            let angle = axis.angle(&swing_axis);
            UnitQuaternion::from_axis_angle(&swing_axis, angle)
        } else {
            UnitQuaternion::identity()
        };
        (swing, twist)
    }
}

fn unit_project(a: Vector3<f64>, b: UnitVector3<f64>) -> Vector3<f64> {
    a.dot(&b) * *b
}
