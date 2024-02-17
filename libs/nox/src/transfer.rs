use std::{borrow::Borrow, ops::Deref};

use crate::{Buffer, Client, Noxpr, Op, Tensor, TensorDim};

pub trait FromHost {
    type HostTy;

    fn from_host(client: &Client, native: Self::HostTy) -> Self;
}

pub trait FromPjrtBuffer {
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self;
}

pub trait AsBuffer {
    fn as_buffer(&self) -> &xla::PjRtBuffer;
}

impl<'a, A: AsBuffer> AsBuffer for &'a mut A {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        A::as_buffer(*self)
    }
}

pub trait FromOp {
    fn from_op(noxpr: Noxpr) -> Self;
}

pub trait IntoOp {
    fn into_op(self) -> Noxpr;
}

impl IntoOp for () {
    fn into_op(self) -> Noxpr {
        Noxpr::tuple(vec![])
    }
}

pub trait BufferForm {
    type BufferTy;
}

pub trait BufferArg<BufferTy> {
    fn is_mut_borrowed() -> bool {
        false
    }

    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer>;

    fn replace_buffer(&mut self, _new_buffer: xla::PjRtBuffer) {}
}

impl<T, R: TensorDim> FromPjrtBuffer for Tensor<T, R, Buffer> {
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self {
        let inner = pjrt.into_iter().next().unwrap();
        Tensor {
            inner,
            phantom: std::marker::PhantomData,
        }
    }
}

impl BufferForm for () {
    type BufferTy = ();
}

impl FromPjrtBuffer for () {
    fn from_pjrt(_pjrt: Vec<xla::PjRtBuffer>) -> Self {}
}

impl<T, R: TensorDim> BufferForm for Tensor<T, R, Op> {
    type BufferTy = Tensor<T, R, Buffer>;
}

impl<'a, T, R: TensorDim> BufferForm for &'a mut Tensor<T, R, Op> {
    type BufferTy = &'a mut Tensor<T, R, Buffer>;
}

impl<'a, T, R: TensorDim> BufferArg<Self> for &'a mut Tensor<T, R, Buffer> {
    fn is_mut_borrowed() -> bool {
        true
    }
    fn as_buffer(&self, _client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer> {
        MaybeOwned::borrow(&self.inner)
    }

    fn replace_buffer(&mut self, new_buffer: xla::PjRtBuffer) {
        self.inner = new_buffer;
    }
}

// This macro allows us to implement `BufferForm` for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_buffer_form {
      ($($ty:tt),+) => {
        impl<$($ty,)*> BufferForm for ($($ty,)*)
              where $($ty: BufferForm, )*
        {
            type BufferTy = ($($ty::BufferTy,)*);
        }
      }
}

impl_buffer_form!(T1);
impl_buffer_form!(T1, T2);
impl_buffer_form!(T1, T2, T3);
impl_buffer_form!(T1, T2, T3, T4);
impl_buffer_form!(T1, T2, T3, T4, T5);
impl_buffer_form!(T1, T2, T3, T4, T5, T6);
impl_buffer_form!(T1, T2, T3, T4, T5, T6, T7);
impl_buffer_form!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_buffer_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_buffer_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_buffer_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_buffer_form!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> MaybeOwned<'a, T> {
    fn borrow(val: &'a T) -> Self {
        MaybeOwned::Borrowed(val)
    }
}

impl<'a, T> Borrow<T> for MaybeOwned<'a, T> {
    fn borrow(&self) -> &T {
        match self {
            MaybeOwned::Borrowed(b) => b,
            MaybeOwned::Owned(b) => b,
        }
    }
}

impl<'a, T> Deref for MaybeOwned<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            MaybeOwned::Owned(b) => b,
            MaybeOwned::Borrowed(b) => b,
        }
    }
}
