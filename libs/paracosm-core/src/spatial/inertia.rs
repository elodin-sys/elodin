use std::ops::{AddAssign, Mul};

use nalgebra::{Matrix3, Vector3};

use super::{SpatialForce, SpatialMotion};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpatialInertia {
    pub inertia: Matrix3<f64>,
    // mass * COM
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

impl AddAssign for SpatialInertia {
    fn add_assign(&mut self, rhs: Self) {
        self.inertia += rhs.inertia;
        self.mass += rhs.mass;
        self.momentum += rhs.momentum;
    }
}
