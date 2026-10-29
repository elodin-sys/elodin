//! Provides abstractions for rigid body dynamics in 3D space.
//! Uses Featherstone’s spatial vector algebra notation for rigid-body dynamics as it is a compact way of representing the state of a rigid body with six degrees of freedom.
//! You can read a short into [here](https://homes.cs.washington.edu/~todorov/courses/amath533/FeatherstoneSlides.pdf) or in [Rigid Body Dynamics Algorithms (Featherstone - 2008)](https://link.springer.com/book/10.1007/978-1-4899-7560-7).
use crate::{
    ArrayRepr, Const, DefaultRepr, Field, MRP, Matrix, Matrix3, OwnedRepr, Quaternion, RealField,
    ReprMonad, Scalar, Tensor, TensorItem, Vector,
};
use core::ops::Div;
use core::ops::{Add, Mul};

/// A spatial transform is a 7D vector that represents a rigid body transformation in 3D space.
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(transparent)]
pub struct SpatialTransform<T: TensorItem, R: OwnedRepr = DefaultRepr> {
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

impl<T: Field, R: OwnedRepr> Clone for SpatialTransform<T, R>
where
    Vector<T, 7, R>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Field, R: OwnedRepr> core::fmt::Debug for SpatialTransform<T, R>
where
    R::Inner<T, Const<7>>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SpatialTransform")
            .field(&self.inner)
            .finish()
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> ReprMonad<R> for SpatialTransform<T, R> {
    type Elem = T;
    type Dim = Const<7>;
    type Map<N: OwnedRepr> = SpatialTransform<T, N>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        SpatialTransform {
            inner: Tensor::from_inner(func(self.inner.inner)),
        }
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.inner.inner
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        &self.inner.inner
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        SpatialTransform {
            inner: Tensor::from_inner(inner),
        }
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> SpatialTransform<T, R> {
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

impl<T: TensorItem + RealField, R: OwnedRepr> Mul for SpatialTransform<T, R> {
    type Output = SpatialTransform<T, R>;

    fn mul(self, rhs: SpatialTransform<T, R>) -> Self::Output {
        let angular = self.angular() * rhs.angular();
        let linear = self.linear() + self.angular() * rhs.linear();
        SpatialTransform::new(angular, linear)
    }
}

/// A spatial force is a 6D vector that represents the linear force and torque applied to a rigid body in 3D space.
#[derive(zerocopy::IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
#[repr(transparent)]
pub struct SpatialForce<T: TensorItem, R: OwnedRepr = DefaultRepr> {
    pub inner: Vector<T, 6, R>,
}

impl<T: Field, R: OwnedRepr> Clone for SpatialForce<T, R>
where
    Vector<T, 6, R>: Clone,
{
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

impl<T: Field, R: OwnedRepr> core::fmt::Debug for SpatialForce<T, R>
where
    R::Inner<T, Const<6>>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SpatialForce").field(&self.inner).finish()
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> ReprMonad<R> for SpatialForce<T, R> {
    type Elem = T;
    type Dim = Const<6>;
    type Map<N: OwnedRepr> = SpatialForce<T, N>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        SpatialForce {
            inner: Tensor::from_inner(func(self.inner.inner)),
        }
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.inner.inner
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        &self.inner.inner
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        SpatialForce {
            inner: Tensor::from_inner(inner),
        }
    }
}

impl<T: RealField, R: OwnedRepr> SpatialForce<T, R> {
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

impl<T: RealField, R: OwnedRepr> Add for SpatialForce<T, R> {
    type Output = SpatialForce<T, R>;

    fn add(self, rhs: SpatialForce<T, R>) -> Self::Output {
        SpatialForce {
            inner: self.inner + rhs.inner,
        }
    }
}

/// A spatial inertia is a 7D vector that represents the symmetric 3x3 moment of inertia tensor and mass of a rigid body in 3D space.
/// The layout is `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz, mass]`.
///
/// Sign convention: slots `3..6` store the signed off-diagonal entries of the
/// inertia tensor matrix itself, i.e. `I_xy` in
/// `[[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]`, where
/// `I_xy = -∫ x*y dm`. If source data (e.g. a CAD workbook) gives products of
/// inertia `P_xy = +∫ x*y dm`, negate them before storing: `I_xy = -P_xy`.
pub struct SpatialInertia<T: TensorItem, R: OwnedRepr = DefaultRepr> {
    pub inner: Vector<T, 7, R>,
}

impl<T: TensorItem + RealField, R: OwnedRepr> ReprMonad<R> for SpatialInertia<T, R> {
    type Elem = T;
    type Dim = Const<7>;
    type Map<N: OwnedRepr> = SpatialInertia<T, N>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        SpatialInertia {
            inner: Tensor::from_inner(func(self.inner.inner)),
        }
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.inner.inner
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        &self.inner.inner
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        SpatialInertia {
            inner: Tensor::from_inner(inner),
        }
    }
}

impl<T: Field, R: OwnedRepr> Clone for SpatialInertia<T, R>
where
    Vector<T, 7, R>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Field, R: OwnedRepr> core::fmt::Debug for SpatialInertia<T, R>
where
    R::Inner<T, Const<7>>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("SpatialInertia").field(&self.inner).finish()
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> SpatialInertia<T, R> {
    /// Constructs a new spatial inertia from diagonal inertia `[Ixx, Iyy, Izz]`, off-diagonal inertia `[Ixy, Ixz, Iyz]`, and scalar mass.
    pub fn new(
        inertia: impl Into<Vector<T, 3, R>>,
        inertia_off_diag: impl Into<Vector<T, 3, R>>,
        mass: impl Into<Scalar<T, R>>,
    ) -> Self {
        let inertia = inertia.into();
        let inertia_off_diag = inertia_off_diag.into();
        let mass = mass.into().broadcast::<Const<1>>();
        let inner = inertia.concat(inertia_off_diag).concat(mass);
        SpatialInertia { inner }
    }

    /// Constructs spatial inertia from a mass, assuming off-diagonal inertia is 0 and the diagonal inertia equals the mass along all axes.
    pub fn from_mass(mass: impl Into<Scalar<T, R>>) -> Self {
        let mass = mass.into();
        SpatialInertia::new(
            T::one().broadcast::<Const<3>>() * &mass,
            Vector::zeros(),
            mass,
        )
    }

    /// Returns the diagonal inertia `[Ixx, Iyy, Izz]` as a 3-vector.
    pub fn inertia_diag(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[0])
    }

    /// Returns the off-diagonal inertia tensor entries `[Ixy, Ixz, Iyz]` as a 3-vector.
    /// These are the signed matrix entries (`I_xy = -∫ x*y dm`), not products of inertia.
    pub fn inertia_off_diag(&self) -> Vector<T, 3, R> {
        self.inner.fixed_slice(&[3])
    }

    /// Reconstructs the symmetric 3x3 inertia tensor from `inertia_diag` and `inertia_off_diag`.
    pub fn inertia_matrix(&self) -> Matrix3<T, R> {
        let [ixx, iyy, izz] = self.inertia_diag().parts();
        let [ixy, ixz, iyz] = self.inertia_off_diag().parts();
        Matrix::from_scalars([
            ixx,
            ixy.clone(),
            ixz.clone(),
            ixy,
            iyy,
            iyz.clone(),
            ixz,
            iyz,
            izz,
        ])
    }

    /// Multiplies the 3x3 inertia tensor by an angular velocity vector (`I * omega`).
    pub fn apply_inertia(&self, omega: &Vector<T, 3, R>) -> Vector<T, 3, R> {
        let [ixx, iyy, izz] = self.inertia_diag().parts();
        let [ixy, ixz, iyz] = self.inertia_off_diag().parts();
        let [wx, wy, wz] = omega.parts();
        let hx = ixx * &wx + &ixy * &wy + &ixz * &wz;
        let hy = ixy * &wx + iyy * &wy + &iyz * &wz;
        let hz = ixz * wx + iyz * wy + izz * wz;
        Vector::from_arr([hx, hy, hz])
    }

    /// Solves `I * alpha = torque` for angular acceleration `alpha` using the symmetric 3x3 cofactor inverse.
    pub fn solve_inertia(&self, torque: &Vector<T, 3, R>) -> Vector<T, 3, R> {
        let [ixx, iyy, izz] = self.inertia_diag().parts();
        let [ixy, ixz, iyz] = self.inertia_off_diag().parts();
        let [tx, ty, tz] = torque.parts();

        let c_xx = &iyy * &izz - &iyz * &iyz;
        let c_xy = &ixz * &iyz - &ixy * &izz;
        let c_xz = &ixy * &iyz - &ixz * &iyy;
        let c_yy = &ixx * &izz - &ixz * &ixz;
        let c_yz = &ixy * &ixz - &ixx * &iyz;
        let c_zz = &ixx * &iyy - &ixy * &ixy;

        let det = &ixx * &c_xx + &ixy * &c_xy + &ixz * &c_xz;

        let ax = (&c_xx * &tx + &c_xy * &ty + &c_xz * &tz) / &det;
        let ay = (&c_xy * &tx + &c_yy * &ty + &c_yz * &tz) / &det;
        let az = (c_xz * tx + c_yz * ty + c_zz * tz) / det;
        Vector::from_arr([ax, ay, az])
    }

    /// Returns the mass as a scalar.
    pub fn mass(&self) -> Scalar<T, R> {
        self.inner.get(6)
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> Div<SpatialInertia<T, R>> for SpatialForce<T, R> {
    type Output = SpatialMotion<T, R>;

    fn div(self, rhs: SpatialInertia<T, R>) -> Self::Output {
        let accel = self.force() / rhs.mass();
        let ang_accel = rhs.solve_inertia(&self.torque());
        SpatialMotion::new(ang_accel, accel)
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> Mul<SpatialMotion<T, R>> for SpatialInertia<T, R> {
    type Output = SpatialForce<T, R>;

    fn mul(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        let force: Vector<T, 3, R> = self.mass() * rhs.linear();
        let torque = self.apply_inertia(&rhs.angular());
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
pub struct SpatialMotion<T: TensorItem, R: OwnedRepr = DefaultRepr> {
    pub inner: Vector<T, 6, R>,
}

impl<T: Field, R: OwnedRepr> Clone for SpatialMotion<T, R>
where
    R::Inner<T::Elem, Const<6>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: TensorItem + RealField, R: OwnedRepr> ReprMonad<R> for SpatialMotion<T, R> {
    type Elem = T;
    type Dim = Const<6>;
    type Map<N: OwnedRepr> = SpatialMotion<T, N>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        SpatialMotion {
            inner: Tensor::from_inner(func(self.inner.inner)),
        }
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.inner.inner
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        &self.inner.inner
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        SpatialMotion {
            inner: Tensor::from_inner(inner),
        }
    }
}

impl<T: RealField, R: OwnedRepr> SpatialMotion<T, R> {
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

impl<R: OwnedRepr> Mul<SpatialMotion<f64, R>> for f64 {
    type Output = SpatialMotion<f64, R>;
    fn mul(self, rhs: SpatialMotion<f64, R>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl<R: OwnedRepr> Mul<SpatialMotion<f32, R>> for f32 {
    type Output = SpatialMotion<f32, R>;
    fn mul(self, rhs: SpatialMotion<f32, R>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl<T: RealField> Mul<&Scalar<T>> for SpatialMotion<T> {
    type Output = SpatialMotion<T>;
    fn mul(self, rhs: &Scalar<T>) -> Self::Output {
        SpatialMotion {
            inner: self.inner * rhs,
        }
    }
}

impl<T: RealField> Mul<SpatialMotion<T>> for &Scalar<T> {
    type Output = SpatialMotion<T>;
    fn mul(self, rhs: SpatialMotion<T>) -> Self::Output {
        SpatialMotion {
            inner: self * rhs.inner,
        }
    }
}

impl<T, R> Add<SpatialMotion<T, R>> for SpatialTransform<T, R>
where
    R: OwnedRepr,
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

impl<T: RealField, R: OwnedRepr> Add<SpatialMotion<T, R>> for SpatialMotion<T, R> {
    type Output = SpatialMotion<T, R>;

    fn add(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        SpatialMotion {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: RealField, R: OwnedRepr> Add<SpatialTransform<T, R>> for SpatialTransform<T, R> {
    type Output = SpatialTransform<T, R>;

    fn add(self, rhs: SpatialTransform<T, R>) -> Self::Output {
        SpatialTransform {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<T: RealField, R: OwnedRepr> Mul<SpatialMotion<T, R>> for Quaternion<T, R> {
    type Output = SpatialMotion<T, R>;

    fn mul(self, rhs: SpatialMotion<T, R>) -> Self::Output {
        SpatialMotion::new(&self * rhs.angular(), &self * rhs.linear())
    }
}

impl<T: RealField, R: OwnedRepr> Mul<SpatialTransform<T, R>> for Quaternion<T, R> {
    type Output = SpatialTransform<T, R>;

    fn mul(self, rhs: SpatialTransform<T, R>) -> Self::Output {
        SpatialTransform::new(&self * rhs.angular(), &self * rhs.linear())
    }
}

impl<T: RealField, R: OwnedRepr> Mul<SpatialForce<T, R>> for Quaternion<T, R> {
    type Output = SpatialForce<T, R>;

    fn mul(self, rhs: SpatialForce<T, R>) -> Self::Output {
        SpatialForce::new(&self * rhs.torque(), &self * rhs.force())
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor;
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_spatial_transform_mul() {
        let a = SpatialTransform::new(
            Quaternion::<_, ArrayRepr>::from_axis_angle(tensor![0.0, 0.0, 1.0], 45f64.to_radians()),
            tensor![1.0, 0.0, 0.0],
        );
        let b = SpatialTransform::new(
            Quaternion::<_, ArrayRepr>::from_axis_angle(
                tensor![0.0, 0.0, 1.0],
                -45f64.to_radians(),
            ),
            tensor![0.0, 2.0, 0.0],
        );
        assert_eq!(
            (a * b).inner,
            tensor![
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
        let a = SpatialTransform::new(
            Quaternion::<_, ArrayRepr>::identity(),
            tensor![0.0, 0.0, 0.0],
        );
        let b = SpatialMotion::new(tensor![0.0, 0.0, 1.0], tensor![0.0, 0.0, 0.0]);

        assert_eq!(
            (a + b).inner,
            tensor![
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
        let a = SpatialTransform::new(
            Quaternion::<_, ArrayRepr>::identity(),
            tensor![0.0, 0.0, 0.0],
        );
        let out = (0..20)
            .fold(a, |acc, _| {
                acc + SpatialMotion::new(tensor![0.0, 0.0, 0.25 / 20.0], tensor![0.0, 0.0, 0.0])
            })
            .inner;
        assert_relative_eq!(
            out,
            tensor![
                0.0,
                0.0,
                0.12467473338522769,
                0.992197667229329,
                0.0,
                0.0,
                0.0
            ], // rotation of 0.25 around axis angle
            epsilon = 1e-5
        )
    }

    #[test]
    fn test_spatial_inertia_diagonal_solve_matches_elementwise() {
        let diag = tensor![2.0, 4.0, 5.0];
        let off_diag = tensor![0.0, 0.0, 0.0];
        let inertia: SpatialInertia<f64, ArrayRepr> = SpatialInertia::new(diag, off_diag, 10.0);
        let torque = tensor![6.0, -8.0, 15.0];

        assert_eq!(inertia.inertia_diag(), diag);
        assert_eq!(inertia.inertia_off_diag(), off_diag);
        assert_eq!(
            inertia.inertia_matrix(),
            tensor![[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]
        );
        assert_relative_eq!(
            inertia.apply_inertia(&tensor![1.0, 2.0, 3.0]),
            tensor![2.0, 8.0, 15.0],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            inertia.solve_inertia(&torque),
            torque / diag,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_spatial_inertia_dense_symmetric_solve() {
        let diag = tensor![4.0, 5.0, 6.0];
        let off_diag = tensor![-0.5, 0.25, -0.75];
        let inertia: SpatialInertia<f64, ArrayRepr> = SpatialInertia::new(diag, off_diag, 3.0);

        let i_mat = inertia.inertia_matrix();
        assert_eq!(
            i_mat,
            tensor![[4.0, -0.5, 0.25], [-0.5, 5.0, -0.75], [0.25, -0.75, 6.0]]
        );

        let alpha_expected = tensor![1.2, -0.8, 2.5];
        let torque = inertia.apply_inertia(&alpha_expected);
        let alpha_solved = inertia.solve_inertia(&torque);
        assert_relative_eq!(alpha_solved, alpha_expected, epsilon = 1e-12);

        let force = SpatialForce::new(torque, tensor![9.0, -6.0, 3.0]);
        let motion = force / inertia;
        assert_relative_eq!(motion.angular(), alpha_expected, epsilon = 1e-12);
        assert_relative_eq!(
            motion.linear(),
            tensor![3.0, -2.0, 1.0],
            epsilon = 1e-12
        );
    }
}
