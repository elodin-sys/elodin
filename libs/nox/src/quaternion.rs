//! Provides functionality to handle quaternions, which are constructs used to represent and manipulate spatial orientations and rotations in 3D space.
use std::ops::{Add, Mul};

use nalgebra::{Const, Scalar as NalgebraScalar};
use num_traits::Zero;
use xla::{ArrayElement, NativeType};

use crate::{
    ArrayRepr, AsBuffer, Buffer, BufferArg, BufferForm, Builder, Client, Field, FromBuilder,
    FromHost, FromOp, FromPjrtBuffer, IntoOp, MaybeOwned, Noxpr, Op, RealField, Repr, Scalar,
    TensorItem, ToHost, Vector,
};

/// Represents a quaternion for spatial orientation or rotation in 3D space.
pub struct Quaternion<T: TensorItem, P: Repr = Op>(pub Vector<T, 4, P>);

impl<T: RealField, R: Repr> Default for Quaternion<T, R> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: TensorItem> Clone for Quaternion<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: TensorItem> FromPjrtBuffer for Quaternion<T, Buffer> {
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self {
        Self(Vector::from_pjrt(pjrt))
    }
}

impl<T: TensorItem> FromOp for Quaternion<T, Op> {
    fn from_op(op: Noxpr) -> Self {
        Self(Vector::from_op(op))
    }
}

impl<T: Field, R: Repr> std::fmt::Debug for Quaternion<T, R>
where
    R::Inner<T, Const<4>>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Quaternion").field(&self.0).finish()
    }
}

impl<T: RealField, R: Repr> Quaternion<T, R> {
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
        let inner = Vector::from_arr([&x, &y, &z, &w]);
        Quaternion(inner)
    }

    /// Creates a unit quaternion with no rotation.
    pub fn identity() -> Self {
        let inner = T::zero::<R>()
            .broadcast::<Const<3>>()
            .concat(T::one().broadcast::<Const<1>>());
        Quaternion(inner)
    }

    /// Returns the four parts (components) of the quaternion as scalars.
    fn parts(&self) -> [Scalar<T, R>; 4] {
        let Quaternion(v) = self;
        v.parts()
    }

    /// Returns the conjugate of the quaternion.
    pub fn conjugate(&self) -> Self {
        let [i, j, k, w] = self.parts();
        Quaternion(Vector::from_arr([&-i, &-j, &-k, &w]))
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
}

impl<T: RealField, R: Repr> Mul for Quaternion<T, R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<'a, T: RealField, R: Repr> Mul<&'a Quaternion<T, R>> for Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn mul(self, rhs: &'a Quaternion<T, R>) -> Self::Output {
        &self * rhs
    }
}

impl<'a, T: RealField, R: Repr> Mul<Quaternion<T, R>> for &'a Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn mul(self, rhs: Quaternion<T, R>) -> Self::Output {
        self * &rhs
    }
}

impl<'a, T: RealField, R: Repr> Mul<&'a Quaternion<T, R>> for &'a Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn mul(self, rhs: &Quaternion<T, R>) -> Self::Output {
        let [l_i, l_j, l_k, l_w] = &self.parts();
        let [r_i, r_j, r_k, r_w] = &rhs.parts();
        let i = l_w * r_i + l_i * r_w + l_j * r_k - l_k * r_j;
        let j = l_w * r_j - l_i * r_k + l_j * r_w + l_k * r_i;
        let k = l_w * r_k + l_i * r_j - l_j * r_i + l_k * r_w;
        let w = l_w * r_w - l_i * r_i - l_j * r_j - l_k * r_k;

        Quaternion(Vector::from_arr([&i, &j, &k, &w]))
    }
}

impl<T: RealField, R: Repr> Mul<Vector<T, 3, R>> for Quaternion<T, R> {
    type Output = Vector<T, 3, R>;

    fn mul(self, rhs: Vector<T, 3, R>) -> Self::Output {
        let zero: Vector<T, 1, R> = T::zero().broadcast();
        let v = Quaternion(rhs.concat(zero));
        let inv = self.inverse();
        let [x, y, z, _] = (self * v * inv).0.parts();
        Vector::from_arr([&x, &y, &z]) // TODO: use fixed slice instead
    }
}

impl<'a, T: RealField, R: Repr> Mul<Vector<T, 3, R>> for &'a Quaternion<T, R> {
    type Output = Vector<T, 3, R>;

    fn mul(self, rhs: Vector<T, 3, R>) -> Self::Output {
        let zero: Vector<T, 1, R> = T::zero().broadcast();
        let v = Quaternion(rhs.concat(zero));
        let inv = self.inverse();
        let [x, y, z, _] = (self * v * inv).0.parts();
        Vector::from_arr([&x, &y, &z]) // TODO: use fixed slice instead
    }
}

impl<T: RealField, R: Repr> Add for Quaternion<T, R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Quaternion(self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: Repr> Add<Quaternion<T, R>> for &'a Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn add(self, rhs: Quaternion<T, R>) -> Self::Output {
        Quaternion(&self.0 + rhs.0)
    }
}

impl<'a, T: RealField, R: Repr> Add<&'a Quaternion<T, R>> for Quaternion<T, R> {
    type Output = Quaternion<T, R>;

    fn add(self, rhs: &'a Quaternion<T, R>) -> Self::Output {
        Quaternion(self.0 + &rhs.0)
    }
}

impl<T: TensorItem> IntoOp for Quaternion<T> {
    fn into_op(self) -> Noxpr {
        self.0.into_op()
    }
}

impl<T: TensorItem> AsBuffer for Quaternion<T, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        &self.0.inner
    }
}

impl<T: xla::ArrayElement + NativeType> FromBuilder for Quaternion<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        Quaternion(Vector::from_builder(builder))
    }
}

impl<T> FromHost for Quaternion<T, Buffer>
where
    T: NativeType + RealField + NalgebraScalar + ArrayElement,
{
    type HostTy = nalgebra::Quaternion<T>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        Quaternion(Vector::<T, 4, Buffer>::from_host(client, native.coords))
    }
}

impl<T> BufferArg<Quaternion<T, Buffer>> for nalgebra::Quaternion<T>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement + RealField,
{
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let inner = client
            .copy_host_buffer(self.coords.as_slice(), &[4])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}

impl<T> ToHost for Quaternion<T, Buffer>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement + RealField + nalgebra::RealField,
{
    type HostTy = nalgebra::Quaternion<T>;

    fn to_host(&self) -> Self::HostTy {
        let literal = self.0.inner.to_literal_sync().unwrap();
        let mut out = nalgebra::Quaternion::zero();
        out.coords
            .as_mut_slice()
            .copy_from_slice(literal.typed_buf::<T>().unwrap());
        out
    }
}

impl<T: TensorItem> BufferForm for Quaternion<T, Op> {
    type BufferTy = Quaternion<T, Buffer>;
}

impl<T> FromPjrtBuffer for nalgebra::Quaternion<T>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement + RealField + nalgebra::RealField,
{
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self {
        let buf = &pjrt[0];
        let literal = buf.to_literal_sync().unwrap();
        let mut out = nalgebra::Quaternion::zero();
        out.coords
            .as_mut_slice()
            .copy_from_slice(literal.typed_buf().unwrap());
        //literal.copy_raw_to(out.coords.as_mut_slice()).unwrap();
        out
    }
}

impl<T: Field + ArrayElement + NativeType> From<Quaternion<T, ArrayRepr>> for Quaternion<T, Op> {
    fn from(q: Quaternion<T, ArrayRepr>) -> Self {
        Quaternion(q.0.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::Client;
    use crate::CompFn;
    use nalgebra::vector;
    use nalgebra::{UnitQuaternion, Vector3};

    use super::*;

    #[test]
    fn test_quat_mult() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Quaternion<f32>, b: Quaternion<f32>| -> Quaternion<f32> { a * b })
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0).into_inner(),
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 1.0).into_inner(),
            )
            .unwrap()
            .to_host();
        let correct_out = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0).into_inner()
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 1.0).into_inner();

        assert_eq!(out, correct_out)
    }

    #[test]
    fn test_quat_inverse() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Quaternion<f32>| -> Quaternion<f32> { a.inverse() })
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0).into_inner(),
            )
            .unwrap()
            .to_host();
        let correct_out = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0)
            .into_inner()
            .try_inverse()
            .unwrap();

        assert_eq!(out, correct_out)
    }

    #[test]
    fn test_quat_vec_mult() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Quaternion<f32>, b: Vector<f32, 3>| -> Vector<f32, 3> { a * b })
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0).into_inner(),
                vector![1.0, 2.0, 3.0],
            )
            .unwrap()
            .to_host();
        let correct_out =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0) * vector![1.0, 2.0, 3.0];

        approx::assert_relative_eq!(out, correct_out, epsilon = 1.0e-6);
    }

    #[test]
    fn test_quat_convention() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Quaternion<f32>, b: Quaternion<f32>| -> Quaternion<f32> { a * b })
            .build()
            .unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec
            .run(
                &client,
                nalgebra::Quaternion::new(0.0, 1.0, 0.0, 0.0),
                nalgebra::Quaternion::new(0.0, 0.0, 1.0, 0.0),
            )
            .unwrap()
            .to_host();
        assert_eq!(nalgebra::Quaternion::new(0.0, 0.0, 0.0, 1.0), out);
    }
}
