//! Provides functionality for managing scalar tensors, including operations and transformations between host and client representations.
use crate::{Buffer, BufferArg, MaybeOwned, NoxprScalarExt, Op, ScalarDim, Tensor, ToHost};
use nalgebra::ClosedAdd;
use nalgebra::Scalar as NalgebraScalar;

use std::{marker::PhantomData, ops::Add};
use xla::{ArrayElement, NativeType};

/// Type alias for a scalar tensor with a specific type `T`, an underlying representation `P`.
pub type Scalar<T, P = Op> = Tensor<T, ScalarDim, P>;

impl<T: NativeType + ArrayElement> ToHost for Scalar<T, Buffer> {
    type HostTy = T;

    fn to_host(&self) -> Self::HostTy {
        let literal = self.inner.to_literal_sync().unwrap();
        literal.typed_buf::<Self::HostTy>().unwrap()[0]
    }
}

impl<T: ClosedAdd + ArrayElement + NativeType> Add<T> for Scalar<T, Op> {
    type Output = Scalar<T, Op>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = NoxprScalarExt::constant(rhs);
        Scalar {
            inner: (self.inner + rhs),
            phantom: PhantomData,
        }
    }
}

impl<T> BufferArg<Scalar<T, Buffer>> for T
where
    T: xla::NativeType + NalgebraScalar + ArrayElement + Copy,
{
    fn as_buffer(&self, client: &crate::Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let inner = client
            .copy_host_buffer(std::slice::from_ref(self), &[])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Client, CompFn};

    use super::*;

    #[test]
    fn test_sqrt_log_opt() {
        let client = Client::cpu().unwrap();
        let comp = (|a: Scalar<f32>| a.sqrt().log()).build().unwrap();
        let exec = comp.compile(&client).unwrap();
        let out = exec.run(&client, 3.141592653589793).unwrap().to_host();
        assert_eq!(out, 0.5723649);
    }
}
