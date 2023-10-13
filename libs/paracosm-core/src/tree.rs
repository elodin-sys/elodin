use crate::{
    hierarchy::{Link, TopologicalSort},
    spatial::{
        SpatialForce, SpatialInertia, SpatialMotion, SpatialPos, SpatialSubspace, SpatialTransform,
    },
    types::{BiasForce, Effect, JointForce, WorldAccel},
    BodyPos, BodyVel, Inertia, JointAccel, Mass, SubtreeInertia, TreeIndex, TreeMassMatrix,
    WorldVel,
};
use bevy::prelude::{Children, Parent};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without, WorldQuery},
    system::{Query, Res, ResMut},
};
use nalgebra::{vector, DMatrix, Matrix6, MatrixXx1, UnitQuaternion, UnitVector3, Vector3};

pub fn pos_tree_step(parent: &SpatialPos, child: &SpatialPos, joint: &Joint) -> WorldPos {
    match joint.joint_type {
        JointType::Free => WorldPos(*child),
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
    pos: &'static mut BodyPos,
    vel: &'static mut BodyVel,

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
            let child_pos = kinematic.pos.0;
            let world_pos = pos_tree_step(parent_pos, &child_pos, &kinematic.joint);
            kinematic.world_pos.0 = world_pos.0;
            let children = children_query.get(*child);
            let Ok(Some(children)) = children else {
                continue;
            };
            recurisve_step(&world_pos.0, children, query, children_query);
        }
    }

    for (_, children, mut parent_kinematics) in root_query.iter_mut() {
        parent_kinematics.world_pos.0 = parent_kinematics.pos.0;

        let Some(children) = children else { continue };

        recurisve_step(
            &parent_kinematics.pos.0,
            children,
            &mut query,
            &children_query,
        )
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct RNEChildQuery {
    pos: &'static BodyPos,
    vel: &'static BodyVel,

    inertia: &'static Inertia,
    mass: &'static Mass,

    world_accel: &'static mut WorldAccel,
    world_vel: &'static mut WorldVel,
    bias_force: &'static mut BiasForce,

    joint_force: &'static mut JointForce,

    effect: &'static mut Effect,

    joint: &'static Joint,
}

pub fn rne_system(mut child_query: Query<RNEChildQuery>, sort: ResMut<TopologicalSort>) {
    for Link { parent, child } in &sort.0 {
        if let Some(parent) = parent {
            let Ok([parent, mut child]) = child_query.get_many_mut([*parent, *child]) else {
                continue;
            };
            let (vel, accel, force) = forward_rne_step(
                child.joint,
                &parent.world_vel.0,
                &parent.world_accel.0,
                &child.pos.0,
                &child.vel.0,
                &SpatialInertia {
                    inertia: child.inertia.0,
                    momentum: Vector3::zeros(),
                    // momentum: child.pos.0.pos * child.mass.0, // TODO: this should maybe be world
                    mass: child.mass.0,
                },
                SpatialForce {
                    force: child.effect.force.0,
                    torque: child.effect.torque.0,
                },
            );

            child.world_vel.0 = vel;
            child.world_accel.0 = accel;
            child.bias_force.0 = force;
        } else {
            let Ok(mut child) = child_query.get_mut(*child) else {
                continue;
            };

            child.world_vel.0 = child.vel.0;
            child.world_accel.0 = SpatialMotion::default();
            child.bias_force.0 = SpatialForce {
                force: child.effect.force.0,
                torque: child.effect.torque.0,
            };
        }
    }

    for Link { parent, child } in sort.0.iter().rev() {
        {
            let Ok(mut child) = child_query.get_mut(*child) else {
                continue;
            };
            let dual = SpatialTransform {
                linear: child.joint.pos,
                angular: UnitQuaternion::identity(),
            }
            .dual_mul(&child.bias_force.0);

            child.joint_force.0 = child.joint.apply_force_subspace(&dual);
        }
        if let Some(parent) = parent {
            let Ok([mut parent, child]) = child_query.get_many_mut([*parent, *child]) else {
                continue;
            };
            parent.bias_force.0 += child
                .joint
                .apply_transpose_force(&child.pos.0, &child.bias_force.0);
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
    let force = child_inertia * accel + vel.cross_dual(&(child_inertia * vel)) + force_ext;
    (vel, accel, force)
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
            JointType::Free => *force,
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
            JointType::Free => *vel,
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
            JointType::Free => *motion,
            JointType::Revolute { .. } | JointType::Sphere | JointType::Fixed => {
                motion.offset(&SpatialPos {
                    pos: child.att * self.pos + child.pos,
                    att: child.att,
                })
            }
        }
    }

    fn transform(&self, child: &SpatialPos) -> SpatialTransform {
        match self.joint_type {
            JointType::Free => SpatialTransform::identity(),
            JointType::Revolute { .. } | JointType::Sphere | JointType::Fixed => SpatialTransform {
                linear: child.att * self.pos + child.pos,
                angular: child.att,
            },
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

    fn subspace(&self) -> SpatialSubspace {
        SpatialSubspace(match self.joint_type {
            JointType::Free => Matrix6::identity(),
            JointType::Revolute { axis } => {
                let mut subspace = Matrix6::zeros();
                let axis = axis.into_inner();
                subspace
                    .fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(&(axis * axis.transpose()));
                subspace
            }
            JointType::Sphere => {
                let mut subspace = Matrix6::zeros();
                subspace.set_diagonal(&vector![1., 1., 1., 0., 0., 0.]);
                subspace
            }
            JointType::Fixed => Matrix6::zeros(),
        })
    }
}

fn unit_project(vec: Vector3<f64>, axis: UnitVector3<f64>) -> Vector3<f64> {
    let axis = *axis;
    (axis * axis.transpose()) * vec
}

pub fn cri_system(
    mut query: Query<CRIQuery>,
    sort: ResMut<TopologicalSort>,
    mut mass_matrix: ResMut<TreeMassMatrix>,
) {
    // // NOTE: There is probably a faster way to do this, in particular
    // // we do not need to zero the whole thing
    mass_matrix.0 = DMatrix::zeros(6 * sort.0.len(), 6 * sort.0.len());
    // for x in mass_matrix.0.iter_mut() {
    //     *x = 0.0;
    // }
    let mass_matrix = &mut mass_matrix.0;

    for mut x in &mut query {
        x.subtree_inertia.0 = SpatialInertia {
            inertia: x.inertia.0,
            momentum: Vector3::zeros(),
            //momentum: x.mass.0 * x.body_pos.0.pos, // TODO world or nah?
            mass: x.mass.0,
        }
    }
    for (i, Link { parent, child }) in sort.0.iter().enumerate().rev() {
        if let Some(parent) = parent {
            let Ok([mut parent, child]) = query.get_many_mut([*parent, *child]) else {
                continue;
            };

            parent.subtree_inertia.0 += child.joint.transform(&child.body_pos.0).transpose()
                * child.subtree_inertia.0.clone();
        }

        let Ok(mut child) = query.get(*child) else {
            continue;
        };
        let subspace = child.joint.subspace();
        let mut f = Matrix6::zeros();
        for (i, c) in subspace.0.column_iter().enumerate() {
            let ang_vel = c.fixed_view::<3, 1>(0, 0).into_owned();
            let vel = c.fixed_view::<3, 1>(3, 0).into_owned();
            let col = (&child.subtree_inertia.0 * SpatialMotion { vel, ang_vel }).vector();
            f.column_mut(i).copy_from(&col);
        }
        let subspace_force = subspace.0.transpose() * f;
        mass_matrix
            .view_mut((6 * i, 6 * i), (6, 6))
            .copy_from(&subspace_force);

        let i = child.tree_index.0;
        while let Some(p) = child.parent {
            for mut c in f.column_iter_mut() {
                let torque = c.fixed_view::<3, 1>(0, 0).into_owned();
                let force = c.fixed_view::<3, 1>(3, 0).into_owned();

                c.copy_from(
                    &child
                        .joint
                        .apply_transpose_force(&child.body_pos.0, &SpatialForce { force, torque }) // TODO allow borrowed vec for force
                        .vector(),
                )
            }
            let Ok(q) = query.get(**p) else { break };
            child = q;
            let j = child.tree_index.0;
            let h_block = f.transpose() * child.joint.subspace().0;
            mass_matrix
                .view_mut((6 * i, 6 * j), (6, 6))
                .copy_from(&h_block);
            mass_matrix
                .view_mut((6 * j, 6 * i), (6, 6))
                .copy_from(&h_block.transpose());
        }
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct CRIQuery {
    mass: &'static Mass,
    inertia: &'static Inertia,
    subtree_inertia: &'static mut SubtreeInertia,

    joint: &'static Joint,
    body_pos: &'static BodyPos,
    body_vel: &'static BodyVel,

    parent: Option<&'static Parent>,
    tree_index: &'static TreeIndex,
}

pub fn forward_dynamics(
    query: Query<&JointForce>,
    mut accel_query: Query<&mut JointAccel>,
    sort: Res<TopologicalSort>,
    mut mass_matrix: ResMut<TreeMassMatrix>,
) {
    let mut bias_forces = MatrixXx1::zeros(6 * sort.0.len());
    for (i, Link { child, .. }) in sort.0.iter().enumerate() {
        let Ok(bias_force) = query.get(*child) else {
            return;
        };

        bias_forces
            .fixed_view_mut::<6, 1>(i * 6, 0)
            .copy_from(&bias_force.0.vector());
    }
    let mass_matrix = std::mem::replace(&mut mass_matrix.0, DMatrix::zeros(0, 0));
    let Ok(inv_mass_matrix) = mass_matrix.pseudo_inverse(f64::EPSILON) else {
        return;
    };
    let accel = inv_mass_matrix * bias_forces;

    for (i, link) in sort.0.iter().enumerate() {
        let ang: Vector3<f64> = accel.fixed_view::<3, 1>(i * 6, 0).into_owned();
        let lin: Vector3<f64> = accel.fixed_view::<3, 1>(i * 6 + 3, 0).into_owned();
        let Ok(mut accel) = accel_query.get_mut(link.child) else {
            continue;
        };
        accel.0 = SpatialMotion {
            vel: lin,
            ang_vel: ang,
        };
    }
}
