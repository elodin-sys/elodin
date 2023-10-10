use nalgebra::{UnitQuaternion, Vector3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialPos {
    pub pos: Vector3<f64>,
    pub att: UnitQuaternion<f64>,
}
