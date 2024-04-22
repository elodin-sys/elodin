use crate::Field;
use crate::FixedSliceExt;
use crate::Tensor;
use crate::TensorItem;
use crate::{Quaternion, Scalar, Vector};
use nalgebra::Const;
use nox_ecs_macros::{FromBuilder, FromOp, IntoOp};
use std::ops::Div;
use std::ops::{Add, Mul};
use xla::ArrayElement;
use xla::NativeType;

/// A spatial transform is a 7D vector that represents a rigid body transformation in 3D space.
#[derive(FromBuilder, IntoOp, Clone, Debug, FromOp)]
pub struct SpatialTransform<T: TensorItem> {
    pub inner: Vector<T, 7>,
}

impl<T: TensorItem + Field> SpatialTransform<T> {
    pub fn new(angular: impl Into<Quaternion<T>>, linear: impl Into<Vector<T, 3>>) -> Self {
        let angular = angular.into();
        let linear = linear.into();
        let inner = angular.0.concat(linear);
        SpatialTransform { inner }
    }

    /// Create a spatial transform from a quaternion
    pub fn from_angular(angular: impl Into<Quaternion<T>>) -> Self {
        let zero = T::zero().broadcast::<Const<3>>();
        SpatialTransform::new(angular, zero)
    }

    /// Create a spatial transform from a linear vector
    pub fn from_linear(linear: impl Into<Vector<T, 3>>) -> Self {
        SpatialTransform::new(Quaternion::identity(), linear)
    }

    /// Create a spatial transform from an axis and angle
    pub fn from_axis_angle(axis: impl Into<Vector<T, 3>>, angle: impl Into<Scalar<T>>) -> Self {
        Self::from_angular(Quaternion::from_axis_angle(axis, angle))
    }

    /// Get the angular part of the spatial transform as a quaternion
    pub fn angular(&self) -> Quaternion<T> {
        Quaternion(self.inner.fixed_slice(&[0]))
    }

    /// Get the linear part of the spatial transform as a vector with shape (3,)
    pub fn linear(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[4])
    }

    pub fn zero() -> Self {
        SpatialTransform {
            inner: Tensor::zeros(),
        }
    }
}

impl<T: TensorItem + ArrayElement + NativeType + Field> Mul for SpatialTransform<T> {
    type Output = SpatialTransform<T>;

    fn mul(self, rhs: SpatialTransform<T>) -> Self::Output {
        let angular = self.angular() * rhs.angular();
        let linear = self.linear() + self.angular() * rhs.linear();
        SpatialTransform::new(angular, linear)
    }
}

/// A spatial force is a 6D vector that represents the linear force and torque applied to a rigid body in 3D space.
#[derive(FromBuilder, IntoOp, Clone, Debug, FromOp)]
pub struct SpatialForce<T: TensorItem> {
    pub inner: Vector<T, 6>,
}

impl<T: Field> SpatialForce<T> {
    pub fn new(torque: impl Into<Vector<T, 3>>, force: impl Into<Vector<T, 3>>) -> Self {
        let torque = torque.into();
        let force = force.into();
        let inner = torque.concat(force);
        SpatialForce { inner }
    }

    /// Create a spatial force from a linear force vector
    pub fn from_linear(force: impl Into<Vector<T, 3>>) -> Self {
        let force = force.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = zero.concat(force);
        SpatialForce { inner }
    }

    /// Create a spatial force from a torque vector
    pub fn from_torque(torque: impl Into<Vector<T, 3>>) -> Self {
        let torque = torque.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = torque.concat(zero);
        SpatialForce { inner }
    }

    /// Get the torque part of the spatial force as a vector with shape (3,)
    pub fn torque(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[0])
    }

    /// Get the linear force part of the spatial force as a vector with shape (3,)
    pub fn force(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[3])
    }

    pub fn zero() -> Self {
        SpatialForce {
            inner: Tensor::zeros(),
        }
    }
}

impl<T: Field> Add for SpatialForce<T> {
    type Output = SpatialForce<T>;

    fn add(self, rhs: SpatialForce<T>) -> Self::Output {
        SpatialForce {
            inner: self.inner + rhs.inner,
        }
    }
}

#[derive(FromBuilder, IntoOp, Clone, Debug, FromOp)]
pub struct SpatialInertia<T: TensorItem> {
    pub inner: Vector<T, 7>,
}

impl<T: TensorItem + Field + NativeType + ArrayElement> SpatialInertia<T> {
    pub fn new(
        inertia: impl Into<Vector<T, 3>>,
        momentum: impl Into<Vector<T, 3>>,
        mass: impl Into<Scalar<T>>,
    ) -> Self {
        let inertia = inertia.into();
        let momentum = momentum.into();
        let mass = mass.into().reshape::<Const<1>>();
        let inner = inertia.concat(momentum).concat(mass);
        SpatialInertia { inner }
    }

    pub fn from_mass(mass: impl Into<Scalar<T>>) -> Self {
        let mass = mass.into();
        SpatialInertia::new(
            T::one().broadcast::<Const<3>>() * mass.clone(),
            Vector::zeros(),
            mass,
        )
    }

    pub fn inertia_diag(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[0])
    }
    pub fn momentum(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[3])
    }
    pub fn mass(&self) -> Scalar<T> {
        self.inner.fixed_slice::<Const<1>>(&[6]).reshape()
    }
}

impl<T: TensorItem + Field + NativeType + ArrayElement> Div<SpatialInertia<T>> for SpatialForce<T> {
    type Output = SpatialMotion<T>;

    fn div(self, rhs: SpatialInertia<T>) -> Self::Output {
        let accel = self.force() / rhs.mass();
        let ang_accel = self.torque() / rhs.inertia_diag();
        SpatialMotion::new(ang_accel, accel)
    }
}

impl<T: TensorItem + ArrayElement + NativeType + Field> Mul<SpatialMotion<T>>
    for SpatialInertia<T>
{
    type Output = SpatialForce<T>;

    fn mul(self, rhs: SpatialMotion<T>) -> Self::Output {
        let force: Vector<T, 3> =
            self.mass() * rhs.linear() - self.momentum().cross(&rhs.angular());
        let torque = self.inertia_diag() * rhs.angular() + self.momentum().cross(&rhs.linear());
        SpatialForce::new(torque, force)
    }
}

/// A spatial motion is a 6D vector that represents the velocity of a rigid body in 3D space.
#[derive(FromBuilder, IntoOp, Clone, Debug, FromOp)]
pub struct SpatialMotion<T: TensorItem> {
    pub inner: Vector<T, 6>,
}

impl<T: Field> SpatialMotion<T> {
    pub fn new(angular: impl Into<Vector<T, 3>>, linear: impl Into<Vector<T, 3>>) -> Self {
        let angular = angular.into();
        let linear = linear.into();
        let inner = angular.concat(linear);
        SpatialMotion { inner }
    }

    /// Create a spatial motion from a linear vector
    pub fn from_linear(linear: impl Into<Vector<T, 3>>) -> Self {
        let linear = linear.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = zero.concat(linear);
        SpatialMotion { inner }
    }

    /// Create a spatial motion from an angular vector
    pub fn from_angular(angular: impl Into<Vector<T, 3>>) -> Self {
        let angular = angular.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = angular.concat(zero);
        SpatialMotion { inner }
    }

    /// Get the angular part of the spatial motion as a vector with shape (3,)
    pub fn angular(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[0])
    }

    /// Get the linear part of the spatial motion as a vector with shape (3,)
    pub fn linear(&self) -> Vector<T, 3> {
        self.inner.fixed_slice(&[3])
    }

    pub fn offset(&self, pos: SpatialTransform<T>) -> Self {
        let ang_vel = pos.angular() * self.angular();
        let vel = pos.angular() * self.linear() + ang_vel.cross(&pos.linear());
        SpatialMotion::new(ang_vel, vel)
    }

    pub fn cross(&self, other: &Self) -> Self {
        let ang_vel = self.angular().cross(&other.angular());
        let vel = self.angular().cross(&other.linear()) + self.linear().cross(&other.angular());
        SpatialMotion::new(ang_vel, vel)
    }

    pub fn cross_dual(&self, other: &SpatialForce<T>) -> SpatialForce<T> {
        let force = self.angular().cross(&other.torque()) + self.linear().cross(&other.force());
        let torque = self.angular().cross(&other.force());
        SpatialForce::new(torque, force)
    }

    pub fn zero() -> Self {
        SpatialMotion {
            inner: Tensor::zeros(),
        }
    }
}

impl Mul<SpatialMotion<f64>> for f64 {
    type Output = SpatialMotion<f64>;
    fn mul(self, rhs: SpatialMotion<f64>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl Mul<SpatialMotion<f32>> for f32 {
    type Output = SpatialMotion<f32>;
    fn mul(self, rhs: SpatialMotion<f32>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl<T> Add<SpatialMotion<T>> for SpatialTransform<T>
where
    T: ArrayElement + NativeType + Field,
    Quaternion<T>: Add<Quaternion<T>, Output = Quaternion<T>>,
    Vector<T, 3>: Add<Vector<T, 3>, Output = Vector<T, 3>>,
{
    type Output = SpatialTransform<T>;

    fn add(self, rhs: SpatialMotion<T>) -> Self::Output {
        let omega: Vector<T, 3> = rhs.angular() / T::two();
        let zero = T::zero().reshape::<Const<1>>();
        let omega = Quaternion(omega.concat(zero));
        let q = self.angular();
        let angular = q.clone() + omega * q;
        let angular = angular.normalize();
        let linear = self.linear() + rhs.linear();
        SpatialTransform::new(angular, linear)
    }
}

impl<T: Field> Add<SpatialMotion<T>> for SpatialMotion<T> {
    type Output = SpatialMotion<T>;

    fn add(self, rhs: SpatialMotion<T>) -> Self::Output {
        SpatialMotion {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: Field> Add<SpatialTransform<T>> for SpatialTransform<T> {
    type Output = SpatialTransform<T>;

    fn add(self, rhs: SpatialTransform<T>) -> Self::Output {
        SpatialTransform {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: Field> Mul<SpatialMotion<T>> for Quaternion<T> {
    type Output = SpatialMotion<T>;

    fn mul(self, rhs: SpatialMotion<T>) -> Self::Output {
        SpatialMotion::new(self.clone() * rhs.angular(), self * rhs.linear())
    }
}

impl<T: Field> Mul<SpatialTransform<T>> for Quaternion<T> {
    type Output = SpatialTransform<T>;

    fn mul(self, rhs: SpatialTransform<T>) -> Self::Output {
        SpatialTransform::new(self.clone() * rhs.angular(), self * rhs.linear())
    }
}

impl<T: Field> Mul<SpatialForce<T>> for Quaternion<T> {
    type Output = SpatialForce<T>;

    fn mul(self, rhs: SpatialForce<T>) -> Self::Output {
        SpatialForce::new(self.clone() * rhs.torque(), self * rhs.force())
    }
}

#[cfg(test)]
mod tests {
    use crate::{CompFn, ToHost};
    use nalgebra::{vector, Vector3};

    use super::*;

    #[test]
    fn test_spatial_transform_mul() {
        let f = || -> Vector<f64, 7> {
            let a = SpatialTransform::new(
                nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 45f64.to_radians())
                    .into_inner(),
                nalgebra::Vector3::new(1.0, 0.0, 0.0),
            );
            let b = SpatialTransform::new(
                nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -45f64.to_radians())
                    .into_inner(),
                nalgebra::Vector3::new(0.0, 2.0, 0.0),
            );
            (a * b).inner
        };
        let client = crate::Client::cpu().unwrap();
        let comp = f.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let res = exec.run(&client).unwrap().to_host();
        assert_eq!(
            res,
            vector![
                0.0,
                0.0,
                0.0,
                1.0,
                -0.41421356237309515,
                1.414213562373095,
                0.0
            ]
        )
    }

    #[test]
    fn test_spatial_transform_add() {
        let f = || -> Vector<f64, 7> {
            let a = SpatialTransform::new(
                nalgebra::UnitQuaternion::identity().into_inner(),
                nalgebra::Vector3::new(0.0, 0.0, 0.0),
            );
            let b = SpatialMotion::new(
                nalgebra::Vector3::new(0.0, 0.0, 1.0),
                nalgebra::Vector3::new(0.0, 0.0, 0.0),
            );
            (a + b).inner
        };
        let client = crate::Client::cpu().unwrap();
        let comp = f.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let res = exec.run(&client).unwrap().to_host();
        assert_eq!(
            res,
            vector![
                0.0,
                0.0,
                0.7071067811865475,
                0.7071067811865475,
                0.0,
                0.0,
                0.0
            ]
        )
    }
}
