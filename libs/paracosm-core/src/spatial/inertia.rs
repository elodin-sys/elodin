use std::ops::Mul;

use nalgebra::{Matrix3, Vector3};

use super::{SpatialForce, SpatialMotion};

pub struct SpatialInertia {
    pub inertia: Matrix3<f64>,
    pub momentum: Vector3<f64>,
    pub mass: f64,
}

impl<'a> Mul<SpatialMotion> for &'a SpatialInertia {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialMotion) -> Self::Output {
        SpatialForce {
            force: self.mass * rhs.vel - self.momentum.cross(&rhs.ang_vel),
            torque: self.inertia * rhs.ang_vel + self.momentum.cross(&rhs.vel),
        }
    }
}

impl Mul<SpatialMotion> for SpatialInertia {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialMotion) -> Self::Output {
        (&self).mul(rhs)
    }
}
