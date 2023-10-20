use std::ops::Mul;

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::{SpatialForce, SpatialInertia, SpatialMotion};

pub struct Transpose<T>(pub T);

#[derive(Clone, Copy)]
pub struct SpatialTransform {
    pub linear: Vector3<f64>,
    pub angular: UnitQuaternion<f64>,
}

impl SpatialTransform {
    pub fn identity() -> SpatialTransform {
        SpatialTransform {
            linear: Vector3::zeros(),
            angular: UnitQuaternion::identity(),
        }
    }

    pub fn transpose(self) -> Transpose<Self> {
        Transpose(self)
    }

    pub fn inverse(&self) -> Self {
        Self {
            linear: -1.0 * (self.angular * self.linear),
            angular: self.angular.inverse(),
        }
    }

    pub fn dual_mul(&self, other: &SpatialForce) -> SpatialForce {
        let force = self.angular * other.force;
        let torque = self.angular * (other.torque - self.linear.cross(&other.force));
        SpatialForce { force, torque }
    }
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

impl Mul<SpatialMotion> for SpatialTransform {
    type Output = SpatialMotion;

    fn mul(self, rhs: SpatialMotion) -> Self::Output {
        self * &rhs
    }
}

impl<'a> Mul<&'a SpatialMotion> for SpatialTransform {
    type Output = SpatialMotion;

    fn mul(self, rhs: &'a SpatialMotion) -> Self::Output {
        let ang_vel = self.angular * rhs.ang_vel;
        SpatialMotion {
            vel: self.angular * (rhs.vel + rhs.ang_vel.cross(&self.linear)),
            ang_vel,
        }
    }
}

impl Mul<Vector3<f64>> for SpatialTransform {
    type Output = Vector3<f64>;

    fn mul(self, rhs: Vector3<f64>) -> Self::Output {
        self.linear + self.angular * rhs
    }
}

impl Mul<SpatialInertia> for Transpose<SpatialTransform> {
    type Output = SpatialInertia;

    fn mul(self, rhs: SpatialInertia) -> Self::Output {
        // let ang_inverse = self.0.angular.inverse();
        // let rot = self.0.angular.to_rotation_matrix();
        // let rot_trans = rot.transpose();
        // let momentum = ang_inverse * rhs.momentum + rhs.mass * self.0.linear;
        // let inertia = rot_trans * rhs.inertia * rot
        //     - self.0.linear.cross_matrix() * (ang_inverse * rhs.momentum).cross_matrix()
        //     - momentum.cross_matrix() * self.0.linear.cross_matrix();

        let rot = self.0.angular.inverse().to_rotation_matrix();
        let inertia = rot.transpose() * rhs.inertia * rot
            - self.0.linear.cross_matrix() * (rot.transpose() * rhs.momentum).cross_matrix()
            - (rot.transpose() * rhs.momentum + rhs.mass * self.0.linear).cross_matrix()
                * self.0.linear.cross_matrix();
        let momentum = rot.transpose() * rhs.momentum + rhs.mass * self.0.linear;

        SpatialInertia {
            momentum,
            mass: rhs.mass,
            inertia,
        }
    }
}

impl Mul<SpatialForce> for Transpose<SpatialTransform> {
    type Output = SpatialForce;

    fn mul(self, rhs: SpatialForce) -> Self::Output {
        let force = self.0.angular.inverse() * rhs.force;
        let torque = self.0.angular.inverse() * rhs.torque + self.0.linear.cross(&force);
        SpatialForce { force, torque }
    }
}

impl Mul<Matrix3<f64>> for SpatialTransform {
    type Output = Matrix3<f64>;

    fn mul(self, rhs: Matrix3<f64>) -> Self::Output {
        let rot = self.angular.to_rotation_matrix();
        rot * rhs * rot.transpose()
    }
}
