//! Provides abstractions for rigid body dynamics in 3D space.
//! Uses Featherstone’s spatial vector algebra notation for rigid-body dynamics as it is a compact way of representing the state of a rigid body with six degrees of freedom.
//! You can read a short into [here](https://homes.cs.washington.edu/~todorov/courses/amath533/FeatherstoneSlides.pdf) or in [Rigid Body Dynamics Algorithms (Featherstone - 2008)](https://link.springer.com/book/10.1007/978-1-4899-7560-7).
use crate::ArrayRepr;
use crate::DefaultRepr;
use crate::Field;
use crate::RealField;
use crate::Repr;
use crate::Tensor;
use crate::TensorItem;
use crate::MRP;
use crate::{Quaternion, Scalar, Vector};
use nalgebra::Const;
use std::ops::Div;
use std::ops::{Add, Mul};

/// A spatial transform is a 7D vector that represents a rigid body transformation in 3D space.
pub struct SpatialTransform<T: TensorItem, R: Repr = DefaultRepr> {
    pub inner: Vector<T, 7, R>,
}

impl<T: TensorItem> Default for SpatialTransform<T, ArrayRepr>
where
    T::Elem: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T: Field> Clone for SpatialTransform<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Field, R: Repr> std::fmt::Debug for SpatialTransform<T, R>
where
    R::Inner<T, Const<7>>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SpatialTransform")
            .field(&self.inner)
            .finish()
    }
}

impl<T: TensorItem + RealField, R: Repr> SpatialTransform<T, R> {
    /// Constructs a new spatial transform from an angular component (Quaternion) and a linear component (Vector).
    pub fn new(angular: impl Into<Quaternion<T, R>>, linear: impl Into<Vector<T, 3, R>>) -> Self {
        let angular = angular.into();
        let linear = linear.into();
        let inner = angular.0.concat(linear);
        Self { inner }
    }

    /// Creates a spatial transform from a quaternion.
    pub fn from_angular(angular: impl Into<Quaternion<T, R>>) -> Self {
        SpatialTransform::new(angular, Tensor::zeros())
    }

    /// Creates a spatial transform from a linear vector.
    pub fn from_linear(linear: impl Into<Vector<T, 3, R>>) -> Self {
        Self::new(Quaternion::identity(), linear)
    }

    /// Creates a spatial transform from an axis and angle.
    pub fn from_axis_angle(
        axis: impl Into<Vector<T, 3, R>>,
        angle: impl Into<Scalar<T, R>>,
    ) -> Self {
        Self::from_angular(Quaternion::from_axis_angle(axis, angle))
    }

    /// Creates a zero spatial transform.
    pub fn zero() -> Self {
        SpatialTransform::from_linear(Tensor::zeros())
    }

    /// Gets the angular part of the spatial transform as a quaternion.
    pub fn angular(&self) -> Quaternion<T, R> {
        Quaternion(self.inner.fixed_slice(&[0]))
    }

    /// Gets the angular part of the spatial transform as a quaternion.
    pub fn mrp(&self) -> MRP<T, R> {
        MRP::from(self.angular())
    }

    /// Gets the linear part of the spatial transform as a vector with shape (3,).
    pub fn linear(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[4])
    }
}

impl<T: TensorItem + RealField, R: Repr> Mul for SpatialTransform<T, R> {
    type Output = SpatialTransform<T, R>;

    fn mul(self, rhs: SpatialTransform<T, R>) -> Self::Output {
        let angular = self.angular() * rhs.angular();
        let linear = self.linear() + self.angular() * rhs.linear();
        SpatialTransform::new(angular, linear)
    }
}

/// A spatial force is a 6D vector that represents the linear force and torque applied to a rigid body in 3D space.
pub struct SpatialForce<T: TensorItem, R: Repr = DefaultRepr> {
    pub inner: Vector<T, 6, R>,
}

impl<T: Field> Clone for SpatialForce<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: TensorItem> Default for SpatialForce<T, ArrayRepr>
where
    T::Elem: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T: Field, R: Repr> std::fmt::Debug for SpatialForce<T, R>
where
    R::Inner<T, Const<6>>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SpatialForce").field(&self.inner).finish()
    }
}

impl<T: RealField, R: Repr> SpatialForce<T, R> {
    /// Constructs a new spatial force from a torque component (Vector) and a force component (Vector).
    pub fn new(torque: impl Into<Vector<T, 3, R>>, force: impl Into<Vector<T, 3, R>>) -> Self {
        let torque = torque.into();
        let force = force.into();
        let inner = torque.concat(force);
        SpatialForce { inner }
    }

    /// Creates a spatial force from a linear force vector.
    pub fn from_linear(force: impl Into<Vector<T, 3, R>>) -> Self {
        let force = force.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = zero.concat(force);
        SpatialForce { inner }
    }

    /// Creates a spatial force from a torque vector.
    pub fn from_torque(torque: impl Into<Vector<T, 3, R>>) -> Self {
        let torque = torque.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = torque.concat(zero);
        SpatialForce { inner }
    }

    /// Gets the torque part of the spatial force as a vector with shape (3,).
    pub fn torque(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[0])
    }

    /// Gets the linear force part of the spatial force as a vector with shape (3,).
    pub fn force(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[3])
    }

    /// Creates a zero spatial force.
    pub fn zero() -> Self {
        SpatialForce {
            inner: Tensor::zeros(),
        }
    }
}

impl<T: RealField, R: Repr> Add for SpatialForce<T, R> {
    type Output = SpatialForce<T, R>;

    fn add(self, rhs: SpatialForce<T, R>) -> Self::Output {
        SpatialForce {
            inner: self.inner + rhs.inner,
        }
    }
}

/// A spatial inertia is a 7D vector that represents the mass, moment of inertia, and momentum of a rigid body in 3D space.
/// The inertia matrix is assumed to be symmetric and represented in its diagonalized form.
pub struct SpatialInertia<T: TensorItem, R: Repr = DefaultRepr> {
    pub inner: Vector<T, 7, R>,
}

impl<T: Field> Clone for SpatialInertia<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Field, R: Repr> std::fmt::Debug for SpatialInertia<T, R>
where
    R::Inner<T, Const<7>>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SpatialInertia").field(&self.inner).finish()
    }
}

impl<T: TensorItem + RealField, R: Repr> SpatialInertia<T, R> {
    /// Constructs a new spatial inertia, in diagonalized form, from inertia, momentum, and mass components.
    pub fn new(
        inertia: impl Into<Vector<T, 3, R>>,
        momentum: impl Into<Vector<T, 3, R>>,
        mass: impl Into<Scalar<T, R>>,
    ) -> Self {
        let inertia = inertia.into();
        let momentum = momentum.into();
        let mass = mass.into().broadcast::<Const<1>>();
        let inner = inertia.concat(momentum).concat(mass);
        SpatialInertia { inner }
    }

    /// Constructs spatial inertia from a mass, assuming momentum is 0 and the inertia is the same value as the mass along all axes.
    pub fn from_mass(mass: impl Into<Scalar<T, R>>) -> Self {
        let mass = mass.into();
        SpatialInertia::new(
            T::one().broadcast::<Const<3>>() * &mass,
            Vector::zeros(),
            mass,
        )
    }

    /// Returns the diagonal inertia as a diagonalized vector.
    pub fn inertia_diag(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[0])
    }

    /// Returns the momentum as a vector.
    pub fn momentum(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[3])
    }

    /// Returns the mass as a scalar.
    pub fn mass(&self) -> Scalar<T, R> {
        self.inner.fixed_slice::<Const<1>>(&[6]).reshape()
    }
}

impl<T: TensorItem + RealField, R: Repr> Div<SpatialInertia<T, R>> for SpatialForce<T, R> {
    type Output = SpatialMotion<T, R>;

    fn div(self, rhs: SpatialInertia<T, R>) -> Self::Output {
        let accel = self.force() / rhs.mass();
        let ang_accel = self.torque() / rhs.inertia_diag();
        SpatialMotion::new(ang_accel, accel)
    }
}

impl<T: TensorItem + RealField, R: Repr> Mul<SpatialMotion<T, R>> for SpatialInertia<T, R> {
    type Output = SpatialForce<T, R>;

    fn mul(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        let force: Vector<T, 3, R> =
            self.mass() * rhs.linear() - self.momentum().cross(&rhs.angular());
        let torque = self.inertia_diag() * rhs.angular() + self.momentum().cross(&rhs.linear());
        SpatialForce::new(torque, force)
    }
}

impl<T: TensorItem> Default for SpatialMotion<T, ArrayRepr>
where
    T::Elem: Default,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

/// A spatial motion is a 6D vector that represents the velocity of a rigid body in 3D space.
pub struct SpatialMotion<T: TensorItem, R: Repr = DefaultRepr> {
    pub inner: Vector<T, 6, R>,
}

impl<T: Field, R: Repr> Clone for SpatialMotion<T, R>
where
    R::Inner<T::Elem, Const<6>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: RealField, R: Repr> SpatialMotion<T, R> {
    /// Constructs a new spatial motion from angular and linear components.
    pub fn new(angular: impl Into<Vector<T, 3, R>>, linear: impl Into<Vector<T, 3, R>>) -> Self {
        let angular = angular.into();
        let linear = linear.into();
        let inner = angular.concat(linear);
        SpatialMotion { inner }
    }

    /// Creates a spatial motion from a linear vector.
    pub fn from_linear(linear: impl Into<Vector<T, 3, R>>) -> Self {
        let linear = linear.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = zero.concat(linear);
        SpatialMotion { inner }
    }

    /// Creates a spatial motion from an angular vector.
    pub fn from_angular(angular: impl Into<Vector<T, 3, R>>) -> Self {
        let angular = angular.into();
        let zero = T::zero().broadcast::<Const<3>>();
        let inner = angular.concat(zero);
        SpatialMotion { inner }
    }

    /// Gets the angular part of the spatial motion as a vector with shape (3,).
    pub fn angular(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[0])
    }

    /// Gets the linear part of the spatial motion as a vector with shape (3,).
    pub fn linear(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[3])
    }

    /// Adjusts spatial motion based on the given spatial transform.
    pub fn offset(&self, pos: SpatialTransform<T, R>) -> Self {
        let ang_vel = pos.angular() * self.angular();
        let vel = pos.angular() * self.linear() + ang_vel.cross(&pos.linear());
        SpatialMotion::new(ang_vel, vel)
    }

    /// Computes the cross product of two spatial motions.
    pub fn cross(&self, other: &Self) -> Self {
        let ang_vel = self.angular().cross(&other.angular());
        let vel = self.angular().cross(&other.linear()) + self.linear().cross(&other.angular());
        SpatialMotion::new(ang_vel, vel)
    }

    /// Computes the dual cross product of spatial motion and spatial force.
    pub fn cross_dual(&self, other: &SpatialForce<T, R>) -> SpatialForce<T, R> {
        let force = self.angular().cross(&other.torque()) + self.linear().cross(&other.force());
        let torque = self.angular().cross(&other.force());
        SpatialForce::new(torque, force)
    }

    /// Creates a zero spatial motion.
    pub fn zero() -> Self {
        SpatialMotion {
            inner: Tensor::zeros(),
        }
    }
}

impl<R: Repr> Mul<SpatialMotion<f64, R>> for f64 {
    type Output = SpatialMotion<f64, R>;
    fn mul(self, rhs: SpatialMotion<f64, R>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl<R: Repr> Mul<SpatialMotion<f32, R>> for f32 {
    type Output = SpatialMotion<f32, R>;
    fn mul(self, rhs: SpatialMotion<f32, R>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl<T, R> Add<SpatialMotion<T, R>> for SpatialTransform<T, R>
where
    R: Repr,
    T: RealField,
    Quaternion<T, R>: Add<Quaternion<T, R>, Output = Quaternion<T, R>>,
    Vector<T, 3, R>: Add<Vector<T, 3, R>, Output = Vector<T, 3, R>>,
{
    type Output = SpatialTransform<T, R>;

    fn add(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        let half_omega: Vector<T, 3, R> = rhs.angular() / T::two();
        let zero = T::zero().broadcast::<Const<1>>();
        let half_omega = Quaternion(half_omega.concat(zero));
        let q = self.angular();
        let angular = &q + half_omega * &q;
        let angular = angular.normalize();
        let linear = self.linear() + rhs.linear();
        SpatialTransform::new(angular, linear)
    }
}

impl<T: RealField, R: Repr> Add<SpatialMotion<T, R>> for SpatialMotion<T, R> {
    type Output = SpatialMotion<T, R>;

    fn add(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        SpatialMotion {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: RealField, R: Repr> Add<SpatialTransform<T, R>> for SpatialTransform<T, R> {
    type Output = SpatialTransform<T, R>;

    fn add(self, rhs: SpatialTransform<T, R>) -> Self::Output {
        SpatialTransform {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: RealField, R: Repr> Mul<SpatialMotion<T, R>> for Quaternion<T, R> {
    type Output = SpatialMotion<T, R>;

    fn mul(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        SpatialMotion::new(&self * rhs.angular(), &self * rhs.linear())
    }
}

impl<T: RealField, R: Repr> Mul<SpatialTransform<T, R>> for Quaternion<T, R> {
    type Output = SpatialTransform<T, R>;

    fn mul(self, rhs: SpatialTransform<T, R>) -> Self::Output {
        SpatialTransform::new(&self * rhs.angular(), &self * rhs.linear())
    }
}

impl<T: RealField, R: Repr> Mul<SpatialForce<T, R>> for Quaternion<T, R> {
    type Output = SpatialForce<T, R>;

    fn mul(self, rhs: SpatialForce<T, R>) -> Self::Output {
        SpatialForce::new(&self * rhs.torque(), &self * rhs.force())
    }
}

#[cfg(test)]
mod tests {
    use crate::{tensor, CompFn, ToHost};
    use approx::assert_relative_eq;
    use nalgebra::vector;

    use super::*;

    #[test]
    fn test_spatial_transform_mul() {
        let f = || -> Vector<f64, 7> {
            let a = SpatialTransform::new(
                Quaternion::<_, ArrayRepr>::from_axis_angle(
                    tensor![0.0, 0.0, 1.0],
                    45f64.to_radians(),
                ),
                tensor![1.0, 0.0, 0.0],
            );
            let b = SpatialTransform::new(
                Quaternion::<_, ArrayRepr>::from_axis_angle(
                    tensor![0.0, 0.0, 1.0],
                    -45f64.to_radians(),
                ),
                tensor![0.0, 2.0, 0.0],
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
                Quaternion::<_, ArrayRepr>::identity(),
                tensor![0.0, 0.0, 0.0],
            );
            let b = SpatialMotion::new(tensor![0.0, 0.0, 1.0], tensor![0.0, 0.0, 0.0]);
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
                0.4472135954999579,
                0.8944271909999159,
                0.0,
                0.0,
                0.0
            ]
        )
    }

    #[test]
    fn test_spatial_transform_integrate() {
        let f = || -> Vector<f64, 7> {
            let a = SpatialTransform::new(
                Quaternion::<_, ArrayRepr>::identity(),
                tensor![0.0, 0.0, 0.0],
            );
            (0..20)
                .fold(a, |acc, _| {
                    acc + SpatialMotion::new(tensor![0.0, 0.0, 0.25 / 20.0], tensor![0.0, 0.0, 0.0])
                })
                .inner
        };
        let client = crate::Client::cpu().unwrap();
        let comp = f.build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let res = exec.run(&client).unwrap().to_host();
        assert_relative_eq!(
            res,
            vector![
                0.0,
                0.0,
                0.12467473338522769,
                0.992197667229329,
                0.0,
                0.0,
                0.0
            ], // roation of 0.25 around axis angle
            epsilon = 1e-5
        )
    }
}
