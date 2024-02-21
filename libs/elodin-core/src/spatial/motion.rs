use std::ops::Add;

use nalgebra::{Vector3, Vector6};

use super::{SpatialForce, SpatialPos};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct SpatialMotion {
    pub vel: Vector3<f64>,
    pub ang_vel: Vector3<f64>,
}

impl SpatialMotion {
    pub fn linear(vel: Vector3<f64>) -> Self {
        SpatialMotion {
            vel,
            ang_vel: Vector3::zeros(),
        }
    }
    pub fn new(vel: Vector3<f64>, ang_vel: Vector3<f64>) -> Self {
        Self { vel, ang_vel }
    }

    pub fn vector(self) -> Vector6<f64> {
        Vector6::from_iterator(self.ang_vel.into_iter().chain(&self.vel).cloned())
    }

    pub fn offset(&self, pos: &SpatialPos) -> SpatialMotion {
        let ang_vel = pos.att * self.ang_vel;
        let vel = pos.att * self.vel + ang_vel.cross(&pos.pos);
        SpatialMotion { vel, ang_vel }
    }

    pub fn cross(&self, other: &SpatialMotion) -> SpatialMotion {
        let ang_vel = self.ang_vel.cross(&other.ang_vel);
        let vel = self.ang_vel.cross(&other.vel) + self.vel.cross(&other.ang_vel);
        SpatialMotion { vel, ang_vel }
    }

    pub fn cross_dual(&self, other: &SpatialForce) -> SpatialForce {
        SpatialForce {
            force: self.ang_vel.cross(&other.torque) + self.vel.cross(&other.force),
            torque: self.ang_vel.cross(&other.force),
        }
    }
}

impl Add for SpatialMotion {
    type Output = SpatialMotion;

    fn add(self, rhs: SpatialMotion) -> Self::Output {
        SpatialMotion {
            vel: self.vel + rhs.vel,
            ang_vel: self.ang_vel + rhs.ang_vel,
        }
    }
}
