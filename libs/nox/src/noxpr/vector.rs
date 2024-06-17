use std::marker::PhantomData;

use crate::ArrayTy;
use crate::Buffer;
use crate::BufferArg;
use crate::Client;
use crate::Field;
use crate::FromHost;
use crate::MaybeOwned;
use crate::Noxpr;
use crate::Op;
use crate::TensorItem;
use crate::ToHost;
use crate::Vector;
use nalgebra::ArrayStorage;
use nalgebra::Const;
use nalgebra::Scalar as NalgebraScalar;
use num_traits::Zero;
use smallvec::smallvec;
use xla::{ArrayElement, NativeType};

impl<T: NativeType + ArrayElement> Vector<T, 3, Op> {
    /// Extends a 3-dimensional vector to a 4-dimensional vector by appending a given element.
    pub fn extend(&self, elem: T) -> Vector<T, 4, Op> {
        let elem = elem.literal();
        let constant = Noxpr::constant(elem, ArrayTy::new(T::TY, smallvec![1]));
        let inner = Noxpr::concat_in_dim(vec![self.inner.clone(), constant], 0);
        Vector {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> ToHost for Vector<T, N, Buffer>
where
    T: xla::NativeType + NalgebraScalar + Zero + ArrayElement,
{
    type HostTy = nalgebra::Vector<T, Const<N>, ArrayStorage<T, N, 1>>;

    fn to_host(&self) -> Self::HostTy {
        let literal = self.inner.to_literal_sync().unwrap();
        let mut out = Self::HostTy::zeros();
        out.as_mut_slice()
            .copy_from_slice(literal.typed_buf::<T>().unwrap());
        out
    }
}

impl<T, const R: usize> FromHost for Vector<T, R, Buffer>
where
    T: NativeType + Field + ArrayElement,
{
    type HostTy = nalgebra::Vector<T, Const<R>, ArrayStorage<T, R, 1>>;

    fn from_host(client: &Client, native: Self::HostTy) -> Self {
        let inner = client
            .copy_host_buffer(native.as_slice(), &[R as i64])
            .unwrap();
        Vector {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T: TensorItem, const R: usize> BufferArg<Vector<T, R, Buffer>>
    for nalgebra::Vector<T, Const<R>, ArrayStorage<T, R, 1>>
where
    T: NativeType + Field + ArrayElement,
{
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        let inner = client
            .copy_host_buffer(self.as_slice(), &[R as i64])
            .unwrap();
        MaybeOwned::Owned(inner)
    }
}
