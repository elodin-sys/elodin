//! Provides functionality to handle quaternions, which are constructs used to represent and manipulate spatial orientations and rotations in 3D space.
use crate::ArrayRepr;
use crate::Const;
use crate::DefaultRepr;
use crate::Elem;
use crate::Matrix3;
use crate::ReprMonad;
use core::ops::{Add, Mul};

use crate::{Field, OwnedRepr, RealField, Scalar, TensorItem, Vector, MRP};

/// Represents a quaternion for spatial orientation or rotation in 3D space.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Quaternion<T: TensorItem, P: OwnedRepr = DefaultRepr>(
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            deserialize = "Vector<T, 4, P>: serde::Deserialize<'de>",
            serialize = "Vector<T, 4, P>: serde::Serialize"
        ))
    )]
    pub Vector<T, 4, P>,
);
impl<T: Field, R: OwnedRepr> Clone for Quaternion<T, R>
where
    R::Inner<T::Elem, Const<4>>: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: TensorItem, R: OwnedRepr> ReprMonad<R> for Quaternion<T, R> {
    type Elem = T::Elem;

    type Dim = Const<4>;

    type Map<N: OwnedRepr> = Quaternion<T, N>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Quaternion(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        Quaternion(Vector::from_inner(inner))
    }
}

impl<T: Field, R: OwnedRepr> Copy for Quaternion<T, R> where R::Inner<T::Elem, Const<4>>: Copy {}

impl<T: RealField, R: OwnedRepr> Default for Quaternion<T, R> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Elem + PartialEq> PartialEq for Quaternion<T, ArrayRepr> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Field, R: OwnedRepr> core::fmt::Debug for Quaternion<T, R>
where
    R::Inner<T, Const<4>>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Quaternion").field(&self.0).finish()
    }
}

impl<T: RealField, R: OwnedRepr> Quaternion<T, R> {
    /// Constructs a new quaternion from individual scalar components.
    pub fn new(
        w: impl Into<Scalar<T, R>>,
        x: impl Into<Scalar<T, R>>,
        y: impl Into<Scalar<T, R>>,
        z: impl Into<Scalar<T, R>>,
    ) -> Self {
        let w = w.into();
        let x = x.into();
        let y = y.into();
        let z = z.into();
        let inner = Vector::from_arr([x, y, z, w]);
        Quaternion(inner)
    }

    // Constructs a new quaternion from euler angles
    pub fn from_euler(
        angles: Vector<T, 3, R>,
        // roll: impl Into<Scalar<T, R>>,
        // pitch: impl Into<Scalar<T, R>>,
        // yaw: impl Into<Scalar<T, R>>,
    ) -> Self {
        let [roll, pitch, yaw] = angles.parts();
        let cr = &(&roll / T::two()).cos();
        let sr = &(&roll / T::two()).sin();
        let cp = &(&pitch / T::two()).cos();
        let sp = &(&pitch / T::two()).sin();
        let cy = &(&yaw / T::two()).cos();
        let sy = &(&yaw / T::two()).sin();
        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;
        let inner = Vector::from_arr([x, y, z, w]);
        Quaternion(inner)
    }

    /// Creates a unit quaternion with no rotation.
    pub fn identity() -> Self {
        let inner = T::zero::<R>()
            .broadcast::<Const<3>>()
            .concat(T::one().broadcast::<Const<1>>());
        Quaternion(inner)
    }

    /// Returns the four parts (components) of the quaternion as scalars, in the order [x,y,z, w];
    pub fn parts(&self) -> [Scalar<T, R>; 4] {
        let Quaternion(v) = self;
        v.parts()
    }

    /// Returns the conjugate of the quaternion.
    pub fn conjugate(&self) -> Self {
        let [i, j, k, w] = self.parts();
        Quaternion(Vector::from_arr([-i, -j, -k, w]))
    }

    /// Normalizes to a unit quaternion.
    pub fn normalize(&self) -> Self {
        Quaternion(&self.0 / self.0.norm())
    }

    /// Computes the inverse of the quaternion.
    pub fn inverse(&self) -> Self {
        // TODO: Check for division by zero
        Quaternion(self.conjugate().0 / self.0.norm_squared())
    }

    /// Creates a quaternion from an axis and an angle.
    pub fn from_axis_angle(
        axis: impl Into<Vector<T, 3, R>>,
        angle: impl Into<Scalar<T, R>>,
    ) -> Self {
        let axis = axis.into();
        let axis = axis.normalize();
        let angle = angle.into();
        let half_angle = angle / (T::two::<R>());
        let sin = half_angle.sin();
        let cos = half_angle.cos();
        let inner = (axis * sin).concat(cos.broadcast::<Const<1>>());
        Quaternion(inner)
    }

    pub fn mrp(&self) -> MRP<T, R> {
        MRP::from(self)
    }

    pub fn integrate_body(&self, body_delta: Vector<T, 3, R>) -> Self {
        let half_omega: Vector<T, 3, R> = body_delta / T::two::<R>();
        let zero = T::zero().broadcast::<Const<1>>();
        let half_omega = Quaternion(half_omega.concat(zero));
        let q = self + self * half_omega;
        q.normalize()
    }
}

impl<T: RealField> Quaternion<T, ArrayRepr> {
    pub fn from_rot_mat(mat: Matrix3<T, ArrayRepr>) -> Self {
        let m00 = mat.get([0, 0]).into_buf();
        let m01 = mat.get([0, 1]).into_buf();
        let m02 = mat.get([0, 2]).into_buf();

        let m10 = mat.get([1, 0]).into_buf();
        let m11 = mat.get([1, 1]).into_buf();
        let m12 = mat.get([1, 2]).into_buf();

        let m20 = mat.get([2, 0]).into_buf();
        let m21 = mat.get([2, 1]).into_buf();
        let m22 = mat.get([2, 2]).into_buf();
        let w = (T::one_prim() + m00 + m11 + m22).max(T::zero_prim()).sqrt() / T::two_prim();
        let x = (T::one_prim() + m00 - m11 - m22).max(T::zero_prim()).sqrt() / T::two_prim();
        let y = (T::one_prim() - m00 + m11 - m22).max(T::zero_prim()).sqrt() / T::two_prim();
        let z = (T::one_prim() - m00 - m11 + m22).max(T::zero_prim()).sqrt() / T::two_prim();
        Quaternion::new(
            w,
            x.copysign(m21 - m12),
            y.copysign(m02 - m20),
            z.copysign(m10 - m01),
        )
    }

    pub fn look_at_rh(
        dir: impl Into<Vector<T, 3, ArrayRepr>>,
        up: impl Into<Vector<T, 3, ArrayRepr>>,
    ) -> Self {
        Self::from_rot_mat(Matrix3::look_at_rh(dir, up))
    }
}

impl<'a, T: RealField, R: OwnedRepr> From<&'a MRP<T, R>> for Quaternion<T, R> {
    fn from(mrp: &'a MRP<T, R>) -> Self {
        let MRP(mrp) = mrp;
        let magsq = mrp.norm_squared();
        let [m1, m2, m3] = mrp.parts();
        let w = T::one::<R>() - &magsq;
        let inner = Vector::<T, 4, R>::from_arr([
            (m1 * T::two::<R>()),
            (m2 * T::two::<R>()),
            (m3 * T::two::<R>()),
            w,
        ]);
        Quaternion(inner / (T::one::<R>() + magsq))
    }
}

impl<T: RealField, R: OwnedRepr> Mul for Quaternion<T, R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, T: RealField, R: OwnedRepr> Mul<&'a Quaternion<T, R>> for Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn mul(self, rhs: &'a Quaternion<T, R>) -> Self::Output {
        &self * rhs
    }
}

impl<'a, T: RealField, R: OwnedRepr> Mul<Quaternion<T, R>> for &'a Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn mul(self, rhs: Quaternion<T, R>) -> Self::Output {
        self * &rhs
    }
}

impl<'a, T: RealField, R: OwnedRepr> Mul<&'a Quaternion<T, R>> for &'a Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn mul(self, rhs: &Quaternion<T, R>) -> Self::Output {
        let [l_i, l_j, l_k, l_w] = &self.parts();
        let [r_i, r_j, r_k, r_w] = &rhs.parts();
        let i = l_w * r_i + l_i * r_w + l_j * r_k - l_k * r_j;
        let j = l_w * r_j - l_i * r_k + l_j * r_w + l_k * r_i;
        let k = l_w * r_k + l_i * r_j - l_j * r_i + l_k * r_w;
        let w = l_w * r_w - l_i * r_i - l_j * r_j - l_k * r_k;

        Quaternion(Vector::from_arr([i, j, k, w]))
    }
}

impl<T: RealField, R: OwnedRepr> Mul<Vector<T, 3, R>> for Quaternion<T, R> {
    type Output = Vector<T, 3, R>;

    fn mul(self, rhs: Vector<T, 3, R>) -> Self::Output {
        let zero: Vector<T, 1, R> = T::zero().broadcast();
        let v = Quaternion(rhs.concat(zero));
        let inv = self.inverse();
        let [x, y, z, _] = (self * v * inv).0.parts();
        Vector::from_arr([x, y, z]) // TODO: use fixed slice instead
    }
}

impl<'a, T: RealField, R: OwnedRepr> Mul<Vector<T, 3, R>> for &'a Quaternion<T, R> {
    type Output = Vector<T, 3, R>;

    fn mul(self, rhs: Vector<T, 3, R>) -> Self::Output {
        let zero: Vector<T, 1, R> = T::zero().broadcast();
        let v = Quaternion(rhs.concat(zero));
        let inv = self.inverse();
        let [x, y, z, _] = (self * v * inv).0.parts();
        Vector::from_arr([x, y, z]) // TODO: use fixed slice instead
    }
}

impl<T: RealField, R: OwnedRepr> Add for Quaternion<T, R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Quaternion(self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: OwnedRepr> Add<Quaternion<T, R>> for &'a Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn add(self, rhs: Quaternion<T, R>) -> Self::Output {
        Quaternion(&self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: OwnedRepr> Add<&'a Quaternion<T, R>> for Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn add(self, rhs: &'a Quaternion<T, R>) -> Self::Output {
        Quaternion(self.0 + &rhs.0)
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;

    use crate::{tensor, ArrayRepr, Vector3};

    use super::*;

    #[test]
    fn test_quat_mult() {
        let out = Quaternion::from_axis_angle(Vector3::x_axis(), 3.0)
            * Quaternion::from_axis_angle(Vector3::x_axis(), 1.0);

        assert_eq!(
            out.0,
            tensor![0.9092974268256817, 0.0, 0.0, -0.4161468365471424]
        )
    }

    #[test]
    fn test_quat_inverse() {
        let out = Quaternion::from_axis_angle(Vector3::x_axis(), 3.0).inverse();
        assert_eq!(
            out.0,
            tensor![-0.9974949866040544, -0.0, -0.0, 0.0707372016677029]
        )
    }

    #[test]
    fn test_quat_vec_mult() {
        let out = Quaternion::from_axis_angle(Vector3::x_axis(), 3.0) * tensor![1.0, 2.0, 3.0];

        approx::assert_relative_eq!(
            out,
            tensor![1.0, -2.4033450173804924, -2.6877374736816018],
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn test_quat_convention() {
        let out: Quaternion<f64, ArrayRepr> =
            Quaternion::new(0.0, 1.0, 0.0, 0.0) * Quaternion::new(0.0, 0.0, 1.0, 0.0);
        assert_eq!(Quaternion::new(0.0, 0.0, 0.0, 1.0).0, out.0);
    }

    #[test]
    fn test_quat_mrp_conversion() {
        let input: Quaternion<f64, crate::ArrayRepr> =
            Quaternion::from_axis_angle(Vector3::z_axis(), 3.14);
        let q = Quaternion::from(&input.mrp());
        approx::assert_relative_eq!(input.0, q.0, epsilon = 1.0e-6);
    }

    #[test]
    fn test_quat_mat_conv() {
        let mat = tensor![
            [0.34202014332566877, 0.9396926207859083, 0.0],
            [-0.9396926207859083, 0.34202014332566877, 0.0],
            [0.0, 0.0, 0.9999999999999999],
        ]
        .transpose();
        let quat = Quaternion::from_rot_mat(mat);
        assert_relative_eq!(
            quat.0,
            tensor![0.0, 0.0, 0.573576436351046, 0.8191520442889918],
            epsilon = 1e-8,
        );
    }
}
