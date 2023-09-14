use std::{
    ops::AddAssign,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use bevy_ecs::{
    prelude::{Bundle, Component},
    query::WorldQuery,
    system::Resource,
};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

pub use crate::{AngVel, Att, Force, Inertia, Mass, Pos, Vel};
use crate::{FromState, Time, Torque};

use super::builder::{XpbdEffector, XpbdSensor};

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct PrevPos(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct PrevAtt(pub UnitQuaternion<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct InverseInertia(pub Matrix3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct Fixed(pub bool);

#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct Config {
    pub dt: f64,
    pub sub_dt: f64,
    pub substep_count: usize,
}

impl Default for Config {
    fn default() -> Self {
        let dt = 1.0 / 60.0;
        let substep_count = 24;
        Self {
            dt,
            sub_dt: dt / substep_count as f64,
            substep_count,
        }
    }
}

#[derive(Component, Default)]
pub struct Effectors(pub Vec<Box<dyn XpbdEffector + Send + Sync>>);

#[derive(Component, Default)]
pub struct Sensors(pub Vec<Box<dyn XpbdSensor + Send + Sync>>);

#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct Effect {
    pub force: Force,
    pub torque: Torque,
}

impl AddAssign for Effect {
    fn add_assign(&mut self, rhs: Self) {
        self.force.0 += rhs.force.0;
        self.torque.0 += rhs.torque.0;
    }
}

#[derive(Bundle)]
pub struct EntityBundle {
    pub fixed: Fixed,

    // pos
    pub prev_pos: PrevPos,
    pub pos: Pos,
    pub vel: Vel,

    // attitude
    pub prev_att: PrevAtt,
    pub att: Att,
    pub ang_vel: AngVel,

    // mass
    pub mass: Mass,
    pub inertia: Inertia,
    pub inverse_inertia: InverseInertia,

    pub effect: Effect,

    pub effectors: Effectors,
    pub sensors: Sensors,
}

#[derive(WorldQuery, Debug)]
#[world_query(mutable, derive(Debug))]
pub struct EntityQuery {
    pub fixed: &'static Fixed,
    pub pos: &'static mut Pos,
    pub vel: &'static mut Vel,

    pub att: &'static mut Att,
    pub ang_vel: &'static mut AngVel,

    pub mass: &'static mut Mass,
    pub inertia: &'static mut Inertia,
    pub inverse_inertia: &'static mut InverseInertia,
}

pub struct EntityStateRef<'a> {
    pub fixed: &'a Fixed,
    pub pos: &'a Pos,
    pub vel: &'a Vel,

    pub att: &'a Att,
    pub ang_vel: &'a AngVel,

    pub mass: &'a Mass,
    pub inertia: &'a Inertia,
    pub inverse_inertia: &'a InverseInertia,
}

impl<'a> EntityStateRef<'a> {
    pub fn from_query(value: &EntityQueryReadOnlyItem<'a>) -> Self {
        Self {
            pos: value.pos,
            vel: value.vel,
            att: value.att,
            ang_vel: value.ang_vel,
            mass: value.mass,
            inertia: value.inertia,
            inverse_inertia: value.inverse_inertia,
            fixed: value.fixed,
        }
    }
}

impl Pos {
    pub fn to_world<S>(&self, state: &S) -> Self
    where
        Pos: FromState<S>,
        Att: FromState<S>,
    {
        Pos(Att::from_state(Time(0.0), state).0 * self.0 + Pos::from_state(Time(0.0), state).0)
    }

    pub fn to_world_basis<S>(&self, state: &S) -> Self
    where
        Pos: FromState<S>,
        Att: FromState<S>,
    {
        Pos(Att::from_state(Time(0.0), state).0 * self.0)
    }
}

impl InverseInertia {
    pub fn to_world<S>(&self, state: &S) -> Self
    where
        Att: FromState<S>,
    {
        let att = Att::from_state(Time(0.0), state);
        InverseInertia(att.0.to_rotation_matrix() * self.0)
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
            fn from_state(_time: crate::Time, state: &$state) -> Self {
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
impl_entity_state!(Pos, pos);
impl_entity_state!(Vel, vel);
impl_entity_state!(Att, att);
impl_entity_state!(AngVel, ang_vel);
impl_entity_state!(Inertia, inertia);
impl_entity_state!(InverseInertia, inverse_inertia);

impl<'a> FromState<EntityStateRef<'a>> for Time {
    fn from_state(time: Time, _state: &EntityStateRef<'a>) -> Self {
        time
    }
}

impl From<Force> for Effect {
    fn from(val: Force) -> Self {
        Effect {
            force: val,
            torque: Torque(Vector3::zeros()),
        }
    }
}

impl From<Torque> for Effect {
    fn from(val: Torque) -> Self {
        Effect {
            torque: val,
            force: Force(Vector3::zeros()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LockStepSignal(Arc<AtomicBool>);

impl LockStepSignal {
    pub fn signal(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn can_continue(&self) -> bool {
        self.0.swap(false, Ordering::Acquire)
    }
}
