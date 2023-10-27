use std::ops::AddAssign;

use crate::spatial::SpatialForce;
use bevy::prelude::*;
use nalgebra::{UnitQuaternion, Vector3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Default)]
pub struct Force(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Component, Default)]
pub struct Torque(pub Vector3<f64>);

#[derive(Debug, Clone, Copy, PartialEq, Component, Default)]
pub struct Effect {
    pub force: Force,
    pub torque: Torque,
}

impl Effect {
    pub fn force_at_point(
        r: Vector3<f64>,
        force: Vector3<f64>,
        att: UnitQuaternion<f64>,
    ) -> Effect {
        Effect {
            torque: Torque(att * r.cross(&force)),
            force: Force(att * force),
        }
    }

    pub fn to_spatial(&self, world_pos: Vector3<f64>) -> SpatialForce {
        SpatialForce {
            force: -self.force.0 + world_pos.cross(&self.torque.0),
            torque: -self.torque.0 - world_pos.cross(&self.force.0),
        }
    }
}

impl AddAssign for Effect {
    fn add_assign(&mut self, rhs: Self) {
        self.force.0 += rhs.force.0;
        self.torque.0 += rhs.torque.0;
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
