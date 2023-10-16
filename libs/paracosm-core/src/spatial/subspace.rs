use nalgebra::{Matrix6, Vector3};
use std::ops::Mul;

use super::{SpatialForce, SpatialMotion, Transpose};

#[derive(Debug)]
pub struct SpatialSubspace(pub Matrix6<f64>);

impl SpatialSubspace {
    fn transpose(self) -> Transpose<Self> {
        Transpose(self)
    }
}

impl Mul<SpatialForce> for Transpose<SpatialSubspace> {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialForce) -> Self::Output {
        let out = self.0 .0.transpose() * rhs.vector();
        let torque = Vector3::new(out[0], out[1], out[2]);
        let force = Vector3::new(out[3], out[4], out[5]);
        SpatialForce { force, torque }
    }
}

impl Mul<SpatialMotion> for SpatialSubspace {
    type Output = SpatialMotion;

    fn mul(self, rhs: SpatialMotion) -> Self::Output {
        self * &rhs
    }
}

impl<'a> Mul<&'a SpatialMotion> for SpatialSubspace {
    type Output = SpatialMotion;

    fn mul(self, rhs: &'a SpatialMotion) -> Self::Output {
        let out = self.0 * rhs.vector();
        let vel = Vector3::new(out[0], out[1], out[2]);
        let ang_vel = Vector3::new(out[3], out[4], out[5]);
        SpatialMotion { vel, ang_vel }
    }
}
