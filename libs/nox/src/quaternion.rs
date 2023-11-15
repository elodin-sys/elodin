use std::ops::Mul;

use nalgebra::{ClosedDiv, RealField, Scalar as NalgebraScalar};
use num_traits::Zero;
use simba::scalar::ClosedNeg;
use xla::{ArrayElement, NativeType, XlaOp};

use crate::{
    AsBuffer, AsOp, Buffer, BufferForm, Builder, Client, FromBuilder, FromHost, FromPjrtBuffer, Op,
    Param, ToHost, Vector,
};

pub struct Quaternion<T, P: Param = Op>(Vector<T, 4, P>);

impl<T: NalgebraScalar + ClosedNeg> Quaternion<T> {
    fn parts(&self) -> [Vector<T, 1>; 4] {
        let Quaternion(v) = self;
        [
            v.fixed_slice(0),
            v.fixed_slice(1),
            v.fixed_slice(2),
            v.fixed_slice(3),
        ]
    }

    pub fn conjugate(&self) -> Self {
        let [i, j, k, w] = self.parts();
        Quaternion(Vector::from_arr([-i, -j, -k, w]))
    }
}

impl<T: NalgebraScalar + ClosedDiv + ClosedNeg> Quaternion<T> {
    pub fn inverse(&self) -> Self {
        // TODO: Check for division by zero
        Quaternion(self.conjugate().0 / self.0.norm_squared())
    }
}

impl<T: RealField> Mul for Quaternion<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let [l_i, l_j, l_k, l_w] = self.parts();
        let [r_i, r_j, r_k, r_w] = rhs.parts();
        let i = l_w.clone() * r_i.clone() + l_i.clone() * r_w.clone() + l_j.clone() * r_k.clone()
            - l_k.clone() * r_j.clone();
        let j = l_w.clone() * r_j.clone() - l_i.clone() * r_k.clone()
            + l_j.clone() * r_w.clone()
            + l_k.clone() * r_i.clone();
        let k = l_w.clone() * r_k.clone() + l_i.clone() * r_j.clone() - l_j.clone() * r_i.clone()
            + l_k.clone() * r_w.clone();
        let w = l_w.clone() * r_w.clone()
            - l_i.clone() * r_i.clone()
            - l_j.clone() * r_j.clone()
            - l_k.clone() * r_k.clone();

        Quaternion(Vector::from_arr([i, j, k, w]))
    }
}

impl<T: NativeType + RealField> Mul<Vector<T, 3>> for Quaternion<T> {
    type Output = Vector<T, 3>;

    fn mul(self, rhs: Vector<T, 3>) -> Self::Output {
        let v = Quaternion(rhs.extend(T::zero()));
        let inv = self.inverse();
        (self * v * inv).0.fixed_slice(0)
    }
}

impl<T> AsOp for Quaternion<T> {
    fn as_op(&self) -> &XlaOp {
        self.0.as_op()
    }
}

impl<T> AsBuffer for Quaternion<T, Buffer> {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        self.0.inner.as_ref()
    }
}

impl<T: xla::ArrayElement> FromBuilder for Quaternion<T, Op> {
    type Item<'a> = Self;

    fn from_builder(builder: &Builder) -> Self::Item<'_> {
        Quaternion(Vector::from_builder(builder))
    }
}

impl<T> FromHost for Quaternion<T, Buffer>
where
    T: NativeType + NalgebraScalar + ArrayElement,
{
    type HostTy = nalgebra::Quaternion<T>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        Quaternion(Vector::<T, 4, Buffer>::from_host(client, native.coords))
    }
}

impl<T, P: Param> ToHost for Quaternion<T, P> {
    type HostTy = nalgebra::Quaternion<T>;
}

impl<T> BufferForm for Quaternion<T, Op> {
    type BufferTy = Quaternion<T, Buffer>;
}

impl<T> FromPjrtBuffer for nalgebra::Quaternion<T>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement + RealField,
{
    fn from_pjrt(pjrt: Vec<Vec<xla::PjRtBuffer>>) -> Self {
        let buf = &pjrt[0][0];
        let literal = buf.to_literal_sync().unwrap();
        let mut out = nalgebra::Quaternion::zero();
        literal.copy_raw_to(out.coords.as_mut_slice()).unwrap();
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
            .unwrap();
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
            .unwrap();
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
            .unwrap();
        let correct_out =
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 3.0) * vector![1.0, 2.0, 3.0];

        approx::assert_relative_eq!(out, correct_out, epsilon = 1.0e-6);
    }
}
