use std::{marker::PhantomData, ops::Add};

use xla::{ArrayElement, NativeType};

use crate::{Buffer, BufferArg, Field, MaybeOwned, Repr, Scalar, ToHost};

impl<T: NativeType + ArrayElement> ToHost for Scalar<T, Buffer> {
    type HostTy = T;

    fn to_host(&self) -> Self::HostTy {
        let literal = self.inner.to_literal_sync().unwrap();
        literal.typed_buf::<Self::HostTy>().unwrap()[0]
    }
}

impl<T: Field, R: Repr> Add<T> for Scalar<T, R> {
    type Output = Scalar<T, R>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = R::scalar_from_const(rhs);
        Scalar {
            inner: R::add(&self.inner, &rhs),
            phantom: PhantomData,
        }
    }
}

impl<T> BufferArg<Scalar<T, Buffer>> for T
where
    T: xla::NativeType + nalgebra::Scalar + ArrayElement + Copy,
{
    fn as_buffer(&self, client: &crate::Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let inner = client
            .copy_host_buffer(std::slice::from_ref(self), &[])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}
