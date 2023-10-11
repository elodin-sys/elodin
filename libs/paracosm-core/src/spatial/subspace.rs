use nalgebra::{Matrix6, Vector3};
use std::ops::Mul;

use super::SpatialForce;

pub struct SpatialSubspace(pub Matrix6<f64>);

impl Mul<SpatialForce> for SpatialSubspace {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialForce) -> Self::Output {
        let out = self.0 * rhs.vector();
        let torque = Vector3::new(out[0], out[1], out[2]);
        let force = Vector3::new(out[4], out[5], out[6]);
        SpatialForce { force, torque }
    }
}
