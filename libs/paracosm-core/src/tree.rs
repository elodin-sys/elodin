use crate::{
    hierarchy::{Link, TopologicalSort},
    spatial::{
        GeneralizedMotion, GeneralizedPos, SpatialForce, SpatialInertia, SpatialMotion, SpatialPos,
        SpatialSubspace, SpatialTransform,
    },
    types::{BiasForce, Effect, JointForce, WorldAccel},
    BodyPos, Inertia, JointAccel, JointPos, JointVel, Mass, SubtreeCoM, SubtreeCoMSum,
    SubtreeInertia, SubtreeMass, TreeIndex, TreeMassMatrix, WorldAnchorPos, WorldPos, WorldVel,
};
use bevy::prelude::{Children, Parent};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without, WorldQuery},
    system::{Query, Res, ResMut},
};
use nalgebra::{vector, DMatrix, Matrix6, MatrixXx1, UnitVector3, Vector3, Vector6};

pub fn pos_tree_step(
    parent: &SpatialPos,
    child: &GeneralizedPos,
    body_pos: &SpatialPos,
    joint: &Joint,
) -> (WorldPos, WorldAnchorPos) {
    match joint.joint_type {
        JointType::Free => (
            WorldPos(child.to_spatial(joint)),
            WorldAnchorPos(SpatialTransform::identity()),
        ),
        JointType::Revolute { .. } | JointType::Sphere | JointType::Fixed => {
            let anchor_pos = child.to_spatial(joint);
            let world_anchor_pos = parent.transform() * anchor_pos.transform();
            let pos = world_anchor_pos * body_pos.transform();
            (
                WorldPos(SpatialPos {
                    pos: pos.linear,
                    att: pos.angular,
                }),
                WorldAnchorPos(world_anchor_pos),
            )
        }
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct KinematicQuery {
    world_pos: &'static mut crate::types::WorldPos,
    world_anchor_pos: &'static mut crate::types::WorldAnchorPos,
    world_vel: &'static mut crate::types::WorldVel,

    pos: &'static mut JointPos,
    vel: &'static mut JointVel,

    body_pos: &'static mut BodyPos,

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
            let (world_pos, world_anchor_pos) = pos_tree_step(
                parent_pos,
                &child_pos,
                &kinematic.body_pos.0,
                &kinematic.joint,
            );
            *kinematic.world_pos = world_pos;
            *kinematic.world_anchor_pos = world_anchor_pos;

            let children = children_query.get(*child);
            let Ok(Some(children)) = children else {
                continue;
            };
            recurisve_step(&world_pos.0, children, query, children_query);
        }
    }

    for (_, children, mut parent_kinematics) in root_query.iter_mut() {
        let (world_pos, _) = pos_tree_step(
            &SpatialPos::default(),
            &parent_kinematics.pos.0,
            &parent_kinematics.body_pos.0,
            &parent_kinematics.joint,
        );
        *parent_kinematics.world_pos = world_pos;

        let Some(children) = children else { continue };

        recurisve_step(
            &parent_kinematics.world_pos.0,
            children,
            &mut query,
            &children_query,
        )
    }
}

pub fn com_system(
    query: Query<(&WorldPos, &Mass)>,
    mut root_query: Query<(&mut SubtreeMass, &mut SubtreeCoMSum)>,
    mut com_query: Query<&mut SubtreeCoM>,
    sort: ResMut<TopologicalSort>,
) {
    for Link {
        child,
        root,
        parent,
    } in &sort.0
    {
        let Ok((pos, mass)) = query.get(*child) else {
            continue;
        };
        let Ok((mut mass_sum, mut com_sum)) = root_query.get_mut(*root) else {
            continue;
        };

        if parent.is_none() {
            **com_sum = Vector3::zeros();
            **mass_sum = 0.0;
        }

        **com_sum += mass.0 * pos.0.pos;
        **mass_sum += mass.0;
    }

    for Link {
        parent,
        child,
        root,
    } in &sort.0
    {
        if parent.is_none() {
            let Ok(mut com) = com_query.get_mut(*child) else {
                continue;
            };
            let Ok((mass_sum, com_sum)) = root_query.get(*child) else {
                continue;
            };
            **com = **com_sum / **mass_sum;
        } else {
            let Ok([mut child_com, root_com]) = com_query.get_many_mut([*child, *root]) else {
                continue;
            };
            *child_com = *root_com;
        }
    }
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct RNEChildQuery {
    pos: &'static JointPos,
    vel: &'static JointVel,
    world_pos: &'static mut WorldPos,
    world_anchor_pos: &'static mut WorldAnchorPos,
    subtree_com: &'static SubtreeCoM,

    inertia: &'static Inertia,
    mass: &'static Mass,

    world_accel: &'static mut WorldAccel,
    world_vel: &'static mut WorldVel,
    bias_force: &'static mut BiasForce,

    joint_force: &'static mut JointForce,

    effect: &'static mut Effect,

    joint: &'static Joint,

    body_pos: &'static BodyPos,
}

pub fn rne_system(mut child_query: Query<RNEChildQuery>, sort: ResMut<TopologicalSort>) {
    for Link { parent, child, .. } in &sort.0 {
        if let Some(parent) = parent {
            let Ok([parent, mut child]) = child_query.get_many_mut([*parent, *child]) else {
                continue;
            };

            let (vel, accel, force) = forward_rne_step(
                child.joint,
                &parent.world_vel.0,
                &parent.world_accel.0,
                child.world_anchor_pos.0.linear - **child.subtree_com,
                &child.vel.0,
                &SpatialInertia::from_body_inertia(
                    child.mass.0,
                    &child.inertia.0,
                    &(child.world_pos.0.pos - **child.subtree_com),
                    &child.world_pos.0.att,
                ),
                child
                    .effect
                    .to_spatial(child.world_pos.0.pos - **child.subtree_com),
            );

            child.world_vel.0 = vel;
            child.world_accel.0 = accel;
            child.bias_force.0 = force;
        } else {
            let Ok(mut child) = child_query.get_mut(*child) else {
                continue;
            };

            // child.world_vel.0 = child.vel.0; // FIXME
            child.world_accel.0 = SpatialMotion::linear(vector![0.0, 9.81, 0.0]);
            child.bias_force.0 = child
                .effect
                .to_spatial(child.world_pos.0.pos - **child.subtree_com);
        }
    }

    for Link { parent, child, .. } in sort.0.iter().rev() {
        {
            let Ok(mut child) = child_query.get_mut(*child) else {
                continue;
            };
            child.joint_force.0 = child
                .joint
                .subspace(&(child.world_anchor_pos.0.linear - **child.subtree_com))
                .transpose()
                * child.bias_force.0;
        }
        if let Some(parent) = parent {
            let Ok([mut parent, child]) = child_query.get_many_mut([*parent, *child]) else {
                continue;
            };
            parent.bias_force.0 += child.bias_force.0;
        }
    }
}

fn forward_rne_step(
    joint: &Joint,
    parent_vel: &SpatialMotion,
    parent_accel: &SpatialMotion,
    anchor_pos: Vector3<f64>,
    child_vel: &GeneralizedMotion,
    child_inertia: &SpatialInertia,
    force_ext: SpatialForce,
) -> (SpatialMotion, SpatialMotion, SpatialForce) {
    let joint_vel = joint.subspace(&anchor_pos) * child_vel;
    let vel = *parent_vel + joint_vel;
    let accel = *parent_accel + vel.cross(&joint_vel);
    // NOTE: S_i * ddot(q_i)  is not included, because accel is set to zero
    let force = child_inertia * accel + vel.cross_dual(&(child_inertia * vel)) - force_ext;
    (vel, accel, force)
}

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
    pub fn free() -> Joint {
        Joint {
            pos: Vector3::zeros(),
            joint_type: JointType::Free,
        }
    }

    pub fn fixed() -> Joint {
        Joint {
            pos: Vector3::zeros(),
            joint_type: JointType::Fixed,
        }
    }

    fn subspace(&self, anchor_pos: &Vector3<f64>) -> SpatialSubspace {
        match self.joint_type {
            JointType::Free => SpatialSubspace {
                cols: 6,
                inner: Matrix6::identity(),
            },
            JointType::Revolute { axis } => {
                let mut inner = Matrix6::zeros();
                let offset = anchor_pos.cross(&axis);
                inner
                    .fixed_view_mut::<3, 1>(0, 0)
                    .copy_from_slice(axis.as_slice());
                inner
                    .fixed_view_mut::<3, 1>(3, 0)
                    .copy_from_slice(offset.as_slice());
                SpatialSubspace { cols: 1, inner }
            }
            JointType::Sphere => {
                let mut inner = Matrix6::zeros();
                inner.m11 = 1.0;
                inner.m22 = 1.0;
                inner.m33 = 1.0;
                SpatialSubspace { cols: 3, inner }
            }
            JointType::Fixed => SpatialSubspace {
                cols: 0,
                inner: Matrix6::zeros(),
            },
        }
    }

    pub fn dof(&self) -> usize {
        match self.joint_type {
            JointType::Free => 6,
            JointType::Revolute { .. } => 1,
            JointType::Sphere => 3,
            JointType::Fixed => 0,
        }
    }
}

pub fn cri_system(
    mut query: Query<CRIQuery>,
    sort: ResMut<TopologicalSort>,
    mut mass_matrix: ResMut<TreeMassMatrix>,
) {
    let dof_count = query.iter().map(|q| q.joint.dof()).sum();
    // // NOTE: There is probably a faster way to do this, in particular
    // // we do not need to zero the whole thing
    mass_matrix.0 = DMatrix::zeros(dof_count, dof_count);
    let mass_matrix = &mut mass_matrix.0;

    for mut x in &mut query {
        x.subtree_inertia.0 = SpatialInertia::from_body_inertia(
            x.mass.0,
            &x.inertia.0,
            &(x.world_pos.0.pos - **x.subtree_com),
            &x.world_pos.0.att,
        );
    }
    for Link { parent, child, .. } in sort.0.iter().rev() {
        if let Some(parent) = parent {
            let Ok([mut parent, child]) = query.get_many_mut([*parent, *child]) else {
                continue;
            };
            parent.subtree_inertia.0 += child.subtree_inertia.0.clone();
        }

        let Ok(mut child) = query.get(*child) else {
            continue;
        };
        let i = child.tree_index.0;
        let subspace = child.joint.subspace(&child.anchor_pos.0.linear);
        let mut f = DMatrix::zeros(6, child.joint.dof());
        for (i, c) in subspace.matrix().column_iter().enumerate() {
            let ang_vel = c.fixed_view::<3, 1>(0, 0).into_owned();
            let vel = c.fixed_view::<3, 1>(3, 0).into_owned();
            let col = (&child.subtree_inertia.0 * SpatialMotion { vel, ang_vel }).vector();
            f.column_mut(i).copy_from(&col);
        }
        let h_block = subspace.matrix().transpose() * f.clone();
        mass_matrix
            .view_mut((i, i), h_block.shape())
            .copy_from(&h_block);
        let child_dof = child.joint.dof();
        while let Some(parent) = child.parent {
            let Ok(parent) = query.get(**parent) else {
                break;
            };
            let j = parent.tree_index.0;
            let h_block =
                f.transpose() * parent.joint.subspace(&parent.anchor_pos.0.linear).matrix();
            mass_matrix
                .view_mut((i, j), (child_dof, parent.joint.dof()))
                .copy_from(&h_block);
            mass_matrix
                .view_mut((j, i), (parent.joint.dof(), child_dof))
                .copy_from(&h_block.transpose());
            child = parent;
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
    joint_pos: &'static JointPos,
    body_vel: &'static JointVel,

    body_pos: &'static BodyPos,
    world_pos: &'static WorldPos,
    subtree_com: &'static SubtreeCoM,
    anchor_pos: &'static WorldAnchorPos,

    parent: Option<&'static Parent>,
    tree_index: &'static TreeIndex,
}

pub fn forward_dynamics(
    query: Query<&JointForce>,
    mut accel_query: Query<&mut JointAccel>,
    sort: Res<TopologicalSort>,
    mut mass_matrix: ResMut<TreeMassMatrix>,
) {
    let mut dof_count = 0;
    for Link { child, .. } in sort.0.iter() {
        let Ok(joint_force) = query.get(*child) else {
            return;
        };
        dof_count += joint_force.0.dof as usize;
    }
    let mut joint_forces = MatrixXx1::zeros(dof_count);

    let mut i = 0;
    for Link { child, .. } in sort.0.iter() {
        let Ok(joint_force) = query.get(*child) else {
            return;
        };
        let dof = joint_force.0.dof as usize;
        joint_forces
            .view_mut((i, 0), (dof, 1))
            .copy_from(&joint_force.0.vector());
        i += dof;
    }
    let mass_matrix = std::mem::replace(&mut mass_matrix.0, DMatrix::zeros(0, 0));
    let Some(inv_mass_matrix) = mass_matrix.try_inverse() else {
        return;
    };
    let accel = -1.0 * inv_mass_matrix * joint_forces;
    let mut i = 0;
    for link in sort.0.iter() {
        let Ok(bias_force) = query.get(link.child) else {
            return;
        };
        let Ok(mut joint_accel) = accel_query.get_mut(link.child) else {
            continue;
        };
        let dof = bias_force.0.dof as usize;
        let out = accel.view((i, 0), (dof, 1));
        i += dof;
        let mut inner = Vector6::zeros();
        for (row, o) in out.row_iter().zip(inner.iter_mut()) {
            *o = row[0];
        }

        joint_accel.0 = GeneralizedMotion {
            dof: dof as u8,
            inner,
        }
    }
}
