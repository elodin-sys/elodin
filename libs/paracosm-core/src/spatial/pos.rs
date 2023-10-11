use nalgebra::{UnitQuaternion, Vector3};

use super::SpatialTransform;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpatialPos {
    pub pos: Vector3<f64>,
    pub att: UnitQuaternion<f64>,
}

impl SpatialPos {
    #[inline]
    pub fn transform(self) -> SpatialTransform {
        SpatialTransform {
            linear: self.pos,
            angular: self.att,
        }
    }
}
