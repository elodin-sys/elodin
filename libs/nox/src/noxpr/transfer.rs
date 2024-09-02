//! Traits and implementations for data transfer between host and XLA devices, buffer management, and XLA client operations.
use std::marker::PhantomData;

use bytemuck::AnyBitPattern;
use smallvec::SmallVec;
use xla::{ArrayElement, PjRtBuffer, Shape};

use crate::{ArrayBuf, ArrayRepr, Client, Dim, Error, ReprMonad, Tensor, TensorItem};

pub struct TypedBuffer<T> {
    pub buffer: PjRtBuffer,
    phantom_data: PhantomData<T>,
}

impl<T> TypedBuffer<T>
where
    T: FromTypedBuffers<TypedBuffers = Self>,
{
    pub fn to_host(&self) -> T {
        T::from_typed_buffers(self).unwrap()
    }
}

pub trait AsTypedBuffer<T> {
    fn as_typed_buffer(&self, client: &Client) -> Result<impl AsRef<TypedBuffer<T>>, Error>;
}

impl<T> AsRef<TypedBuffer<T>> for TypedBuffer<T> {
    fn as_ref(&self) -> &TypedBuffer<T> {
        self
    }
}

impl<T: TensorItem, D: Dim> AsTypedBuffer<Tensor<T, D, ArrayRepr>> for Tensor<T, D, ArrayRepr>
where
    T::Elem: ArrayElement,
{
    fn as_typed_buffer(
        &self,
        client: &Client,
    ) -> Result<impl AsRef<TypedBuffer<Tensor<T, D, ArrayRepr>>>, Error> {
        let shape = D::array_shape(&self.inner.buf);
        let shape = shape
            .as_ref()
            .iter()
            .map(|&x| x as i64)
            .collect::<SmallVec<[i64; 4]>>();
        let buffer = client.copy_host_buffer(self.inner.buf.as_buf(), &shape)?;
        Ok(TypedBuffer {
            buffer,
            phantom_data: PhantomData,
        })
    }
}

pub trait FromTypedBuffers: Sized {
    type TypedBuffers;
    fn from_typed_buffers(buffers: &Self::TypedBuffers) -> Result<Self, Error>;

    fn from_pjrt_buffers(buffers: &mut Vec<PjRtBuffer>) -> Self::TypedBuffers;
}

impl<M: ReprMonad<ArrayRepr>> FromTypedBuffers for M
where
    M::Elem: ArrayElement + AnyBitPattern,
{
    type TypedBuffers = TypedBuffer<M>;

    fn from_typed_buffers(buffers: &Self::TypedBuffers) -> Result<Self, Error> {
        let literal = buffers.buffer.to_literal_sync()?;
        let buf = literal.typed_buf()?;
        let Shape::Array(shape) = literal.shape()? else {
            return Err(Error::IncompatibleDType);
        };
        let dims: SmallVec<[usize; 4]> = shape.dims().iter().map(|&x| x as usize).collect();
        let mut array = crate::Array::<M::Elem, M::Dim>::zeroed(&dims);
        array.buf.as_mut_buf().copy_from_slice(buf);
        Ok(M::from_inner(array))
    }

    fn from_pjrt_buffers(buffers: &mut Vec<PjRtBuffer>) -> Self::TypedBuffers {
        let buffer = buffers.pop().unwrap();
        TypedBuffer {
            buffer,
            phantom_data: PhantomData,
        }
    }
}
