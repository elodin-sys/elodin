use std::ops::{Add, AddAssign, Sub};

use nalgebra::{Vector3, Vector6};

#[derive(Debug, Clone, Copy, PartialEq)]
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
