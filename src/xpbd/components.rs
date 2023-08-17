use std::ops::AddAssign;

use bevy_ecs::{
    prelude::{Bundle, Component},
    query::WorldQuery,
    system::Resource,
};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

pub use crate::{AngVel, Att, Force, Inertia, Mass, Pos, Vel};
use crate::{FromState, Torque};

use super::builder::{XpbdEffector, XpbdSensor};

#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct PrevPos(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct PrevAtt(pub UnitQuaternion<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component)]
pub struct InverseInertia(pub Matrix3<f64>);

#[derive(Debug, Clone, Copy, PartialEq, Resource)]
pub struct Config {
    pub dt: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self { dt: 0.01 }
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

#[derive(WorldQuery)]
pub struct EntityQuery {
    pub pos: &'static Pos,
    pub vel: &'static Vel,

    pub att: &'static Att,
    pub ang_vel: &'static AngVel,

    pub mass: &'static Mass,
    pub inertia: &'static Inertia,
    pub inverse_inertia: &'static InverseInertia,
}

pub struct EntityStateRef<'a> {
    pub pos: &'a Pos,
    pub vel: &'a Vel,

    pub att: &'a Att,
    pub ang_vel: &'a AngVel,

    pub mass: &'a Mass,
    pub inertia: &'a Inertia,
    pub inverse_inertia: &'a InverseInertia,
}

impl<'a> EntityStateRef<'a> {
    pub fn from_query(value: &EntityQueryItem<'a>) -> Self {
        Self {
            pos: value.pos,
            vel: value.vel,
            att: value.att,
            ang_vel: value.ang_vel,
            mass: value.mass,
            inertia: value.inertia,
            inverse_inertia: value.inverse_inertia,
        }
    }
}

impl<'a> FromState<EntityStateRef<'a>> for Mass {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.mass
    }
}

impl<'a> FromState<EntityStateRef<'a>> for Pos {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.pos
    }
}

impl<'a> FromState<EntityStateRef<'a>> for Vel {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.vel
    }
}

impl<'a> FromState<EntityStateRef<'a>> for Att {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.att
    }
}

impl<'a> FromState<EntityStateRef<'a>> for AngVel {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.ang_vel
    }
}

impl<'a> FromState<EntityStateRef<'a>> for Inertia {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.inertia
    }
}

impl<'a> FromState<EntityStateRef<'a>> for InverseInertia {
    fn from_state(_time: crate::Time, state: &EntityStateRef<'a>) -> Self {
        *state.inverse_inertia
    }
}

impl<'a> FromState<EntityStateRef<'a>> for crate::Time {
    fn from_state(time: crate::Time, _state: &EntityStateRef<'a>) -> Self {
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
