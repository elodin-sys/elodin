use crate::{Client, FromOp, FromPjrtBuffer, MaybeOwned};
use nalgebra::Scalar as NalgebraScalar;
use num_traits::Zero;

use crate::{
    ArrayRepr, AsBuffer, Buffer, BufferArg, BufferForm, Builder, Field, FromBuilder, FromHost,
    IntoOp, Noxpr, Op, Quaternion, RealField, TensorItem, ToHost, Vector,
};
use xla::{ArrayElement, NativeType};

impl<T: TensorItem> IntoOp for Quaternion<T, Op> {
    fn into_op(self) -> Noxpr {
        self.0.inner
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

impl<T: TensorItem> FromPjrtBuffer for Quaternion<T, Buffer> {
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self {
        Self(Vector::from_pjrt(pjrt))
    }
}

impl<T: Field + NativeType + ArrayElement> From<Quaternion<T, ArrayRepr>> for Quaternion<T, Op> {
    fn from(q: Quaternion<T, ArrayRepr>) -> Self {
        Quaternion(q.0.into())
    }
}

impl<T: TensorItem> FromOp for Quaternion<T, Op> {
    fn from_op(inner: Noxpr) -> Self {
        Quaternion(Vector::from_op(inner))
    }
}
