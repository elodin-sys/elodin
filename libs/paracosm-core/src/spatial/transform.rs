use std::ops::Mul;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

pub struct SpatialTransform {
    pub linear: Vector3<f64>,
    pub angular: UnitQuaternion<f64>,
}

impl<'a> Mul<SpatialTransform> for &'a SpatialTransform {
    type Output = SpatialTransform;

    fn mul(self, rhs: SpatialTransform) -> Self::Output {
        SpatialTransform {
            linear: self.linear + self.angular * rhs.linear,
            angular: self.angular * rhs.angular,
        }
    }
}

impl Mul<SpatialTransform> for SpatialTransform {
    type Output = SpatialTransform;

    fn mul(self, rhs: SpatialTransform) -> Self::Output {
        SpatialTransform {
            linear: self.linear + self.angular * rhs.linear,
            angular: self.angular * rhs.angular,
        }
    }
}

impl Mul<Vector3<f64>> for SpatialTransform {
    type Output = Vector3<f64>;

    fn mul(self, rhs: Vector3<f64>) -> Self::Output {
        self.linear + self.angular * rhs
    }
}

impl Mul<Matrix3<f64>> for SpatialTransform {
    type Output = Matrix3<f64>;

    fn mul(self, rhs: Matrix3<f64>) -> Self::Output {
        let rot = self.angular.to_rotation_matrix();
        rot * rhs * rot.transpose()
    }
}
