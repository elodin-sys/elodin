use std::ops::{Add, AddAssign, Sub};

use elodin_conduit::{cid, ComponentType, ComponentValue};
use nalgebra::{Vector3, Vector6};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpatialForce {
    pub force: Vector3<f64>,
    pub torque: Vector3<f64>,
}

impl SpatialForce {
    pub fn vector(&self) -> Vector6<f64> {
        Vector6::from_iterator(self.torque.into_iter().chain(&self.force).copied())
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

impl elodin_conduit::Component for SpatialForce {
    fn component_id() -> elodin_conduit::ComponentId {
        cid!(31;spatial_force)
    }

    fn component_type() -> elodin_conduit::ComponentType {
        ComponentType::SpatialPosF64
    }

    fn component_value<'a>(&self) -> elodin_conduit::ComponentValue<'a> {
        elodin_conduit::ComponentValue::SpatialMotionF64((self.torque, self.force))
    }

    fn from_component_value(value: elodin_conduit::ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized,
    {
        let ComponentValue::SpatialMotionF64((torque, force)) = value else {
            return None;
        };
        Some(Self { force, torque })
    }
}
