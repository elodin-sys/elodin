use std::ops::{Add, Mul};

use nalgebra::{Const, RealField, Scalar as NalgebraScalar};
use num_traits::Zero;
use xla::{ArrayElement, NativeType};

use crate::{
    AsBuffer, Buffer, BufferArg, BufferForm, Builder, Client, Field, FixedSliceExt, FromBuilder,
    FromHost, FromOp, FromPjrtBuffer, IntoOp, MaybeOwned, Noxpr, Op, Param, ToHost, Vector,
};

pub struct Quaternion<T, P: Param = Op>(pub Vector<T, 4, P>);

impl<T> Clone for Quaternion<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> FromPjrtBuffer for Quaternion<T, Buffer> {
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self {
        Self(Vector::from_pjrt(pjrt))
    }
}

impl<T> FromOp for Quaternion<T, Op> {
    fn from_op(op: Noxpr) -> Self {
        Self(Vector::from_op(op))
    }
}

impl<T: Field> Quaternion<T> {
    pub fn identity() -> Self {
        let inner = T::zero()
            .broadcast::<Const<3>>()
            .concat(T::one().reshape::<Const<1>>());
        Quaternion(inner)
    }
}

impl<T: Field> Quaternion<T> {
    fn parts(&self) -> [Vector<T, 1>; 4] {
        let Quaternion(v) = self;
        v.parts()
    }

    pub fn conjugate(&self) -> Self {
        let [i, j, k, w] = self.parts();
        Quaternion(Vector::from_arr([-i, -j, -k, w]))
    }
}

impl<T: Field> Quaternion<T> {
    pub fn inverse(&self) -> Self {
        // TODO: Check for division by zero
        Quaternion(self.conjugate().0 / self.0.norm_squared())
    }

    pub fn normalize(&self) -> Self {
        Quaternion(self.0.clone() / self.0.norm())
    }
}

impl<T: Field> Mul for Quaternion<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let [l_i, l_j, l_k, l_w] = &self.parts();
        let [r_i, r_j, r_k, r_w] = &rhs.parts();
        let i = l_w * r_i + l_i * r_w + l_j * r_k - l_k * r_j;
        let j = l_w * r_j - l_i * r_k + l_j * r_w + l_k * r_i;
        let k = l_w * r_k + l_i * r_j - l_j * r_i + l_k * r_w;
        let w = l_w * r_w - l_i * r_i - l_j * r_j - l_k * r_k;

        Quaternion(Vector::from_arr([i, j, k, w]))
    }
}

impl<T: Field> Mul<Vector<T, 3>> for Quaternion<T> {
    type Output = Vector<T, 3>;

    fn mul(self, rhs: Vector<T, 3>) -> Self::Output {
        let zero: Vector<T, 1> = T::zero().reshape();
        let v = Quaternion(rhs.concat(zero));
        let inv = self.inverse();
        (self * v * inv).0.fixed_slice([0])
    }
}

impl<T: Field> Add for Quaternion<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Quaternion(self.0 + rhs.0)
    }
}

impl<T> IntoOp for Quaternion<T> {
    fn into_op(self) -> Noxpr {
        self.0.into_op()
    }
}

impl<T> AsBuffer for Quaternion<T, Buffer> {
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
    T: NativeType + Field + NalgebraScalar + ArrayElement,
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
            .0
            .copy_host_buffer(self.coords.as_slice(), &[4])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}

impl<T> ToHost for Quaternion<T, Buffer>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement + RealField,
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

impl<T> BufferForm for Quaternion<T, Op> {
    type BufferTy = Quaternion<T, Buffer>;
}

impl<T> FromPjrtBuffer for nalgebra::Quaternion<T>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement + RealField,
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
}
