use crate::{tree::Joint, FromState};
use bevy::{ecs::query::WorldQuery, prelude::Bundle};

pub use super::*;

#[derive(Bundle)]
pub struct EntityBundle {
    pub fixed: FixedBody,

    // pos
    pub world_pos: WorldPos,
    pub world_anchor_pos: WorldAnchorPos,
    pub world_vel: WorldVel,

    // mass
    pub mass: Mass,
    pub inertia: Inertia,
    pub inverse_inertia: InverseInertia,

    pub effect: Effect,

    pub effectors: Effectors,
    pub sensors: Sensors,

    pub picked: Picked,

    pub joint: JointBundle,

    pub bias_force: BiasForce,
    pub world_accel: WorldAccel,

    pub tree_index: TreeIndex,
    pub subtree_inertia: SubtreeInertia,

    pub body_pos: BodyPos,

    pub subtree_com: SubtreeCoM,
    pub subtree_com_sum: SubtreeCoMSum,
    pub subtree_mass: SubtreeMass,

    pub synced: Synced,
}

#[derive(Bundle, Debug)]
pub struct JointBundle {
    pub joint: Joint,
    pub pos: JointPos,
    pub vel: JointVel,
    pub joint_accel: JointAccel,
    pub joint_force: JointForce,
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct EntityQuery {
    pub fixed: &'static FixedBody,
    pub pos: &'static mut JointPos,
    pub vel: &'static mut JointVel,

    pub world_pos: &'static mut WorldPos,
    pub world_vel: &'static mut WorldVel,

    pub mass: &'static mut Mass,
    pub inertia: &'static mut Inertia,
    pub inverse_inertia: &'static mut InverseInertia,
}

pub struct EntityStateRef<'a> {
    pub fixed: &'a FixedBody,
    pub pos: &'a JointPos,
    pub vel: &'a JointVel,

    pub world_pos: &'a WorldPos,
    pub world_vel: &'a WorldVel,

    pub mass: &'a Mass,
    pub inertia: &'a Inertia,
    pub inverse_inertia: &'a InverseInertia,
}

impl<'a> EntityStateRef<'a> {
    pub fn from_query(value: &EntityQueryReadOnlyItem<'a>) -> Self {
        Self {
            pos: value.pos,
            vel: value.vel,
            mass: value.mass,
            inertia: value.inertia,
            inverse_inertia: value.inverse_inertia,
            fixed: value.fixed,
            world_pos: value.world_pos,
            world_vel: value.world_vel,
        }
    }
}

impl<'a> EntityQueryReadOnlyItem<'a> {
    pub fn state_ref(&self) -> EntityStateRef<'_> {
        EntityStateRef::from_query(self)
    }
}

macro_rules! impl_from_state {
    ($state: ty, $component: ty, $field: ident) => {
        impl<'a> FromState<$state> for $component {
            fn from_state(_time: super::Time, state: &$state) -> Self {
                *state.$field
            }
        }
    };
}

macro_rules! impl_entity_state {
    ($component: ty, $field: ident) => {
        impl_from_state!(EntityStateRef<'a>, $component, $field);
        impl_from_state!(EntityQueryItem<'a>, $component, $field);
        impl_from_state!(EntityQueryReadOnlyItem<'a>, $component, $field);
    };
}

impl_entity_state!(Mass, mass);
impl_entity_state!(JointPos, pos);
impl_entity_state!(WorldPos, world_pos);
impl_entity_state!(JointVel, vel);
impl_entity_state!(Inertia, inertia);
impl_entity_state!(InverseInertia, inverse_inertia);

impl<'a> FromState<EntityStateRef<'a>> for Time {
    fn from_state(time: Time, _state: &EntityStateRef<'a>) -> Self {
        time
    }
}
