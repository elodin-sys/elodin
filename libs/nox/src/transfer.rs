//! Traits and implementations for data transfer between host and XLA devices, buffer management, and XLA client operations.
use std::{borrow::Borrow, ops::Deref};

use crate::{Buffer, Client, Dim, Noxpr, Op, Tensor, TensorItem};

/// Defines a trait for converting host data types into computation graph nodes.
pub trait FromHost {
    /// Type of the host data that can be converted.
    type HostTy;

    /// Converts a native host type into an instance of the implementing type, using a provided client.
    fn from_host(client: &Client, native: Self::HostTy) -> Self;
}

/// Defines a trait for constructing objects from PJRT buffers.
pub trait FromPjrtBuffer {
    /// Constructs an object from PJRT buffers.
    fn from_pjrt(pjrt: Vec<xla::PjRtBuffer>) -> Self;
}

/// Provides a mechanism to retrieve a reference to an underlying PJRT buffer.
pub trait AsBuffer {
    /// Returns a reference to an associated PJRT buffer.
    fn as_buffer(&self) -> &xla::PjRtBuffer;
}

impl<'a, A: AsBuffer> AsBuffer for &'a mut A {
    fn as_buffer(&self) -> &xla::PjRtBuffer {
        A::as_buffer(*self)
    }
}

/// A trait for converting computation graph nodes into instances of the implementing type.
pub trait FromOp {
    /// Constructs an instance of the implementing type from a computation graph node.
    fn from_op(noxpr: Noxpr) -> Self;
}

/// A trait for converting instances into computation graph nodes.
pub trait IntoOp {
    /// Converts an instance into a computation graph node.
    fn into_op(self) -> Noxpr;
}

impl IntoOp for () {
    fn into_op(self) -> Noxpr {
        Noxpr::tuple(vec![])
    }
}

/// Defines a trait to represent the buffer form of a type.
pub trait BufferForm {
    /// The type of the buffer that represents the data.
    type BufferTy;
}

/// Defines operations for managing buffer arguments, allowing for checking mutable borrow status and buffer manipulation.
pub trait BufferArg<BufferTy> {
    /// Determines if the buffer is currently mutably borrowed.
    ///
    /// By default, returns false, indicating no mutable borrow.
    fn is_mut_borrowed() -> bool {
        false
    }

    /// Retrieves the buffer associated with this object, possibly in a borrowed state.
    ///
    /// `client`: The client associated with the buffer's lifecycle.
    fn as_buffer(&self, client: &Client) -> MaybeOwned<'_, xla::PjRtBuffer>;

    /// Replaces the current buffer with a new one.
    ///
    /// `new_buffer`: The new buffer to set.
    fn replace_buffer(&mut self, _new_buffer: xla::PjRtBuffer) {}
}

impl<T: TensorItem, R: Dim> FromPjrtBuffer for Tensor<T, R, Buffer> {
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

impl<T: TensorItem, R: Dim> BufferForm for Tensor<T, R, Op> {
    type BufferTy = Tensor<T, R, Buffer>;
}

impl<'a, T: TensorItem, R: Dim> BufferForm for &'a mut Tensor<T, R, Op> {
    type BufferTy = &'a mut Tensor<T, R, Buffer>;
}

impl<'a, T: TensorItem, R: Dim> BufferArg<Self> for &'a mut Tensor<T, R, Buffer> {
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

/// A wrapper to manage ownership of resources, allowing for both owned and borrowed references.
pub enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> MaybeOwned<'a, T> {
    /// Constructs a `MaybeOwned` as a borrowed reference.
    fn borrow(val: &'a T) -> Self {
        MaybeOwned::Borrowed(val)
    }
}

impl<'a, T> Borrow<T> for MaybeOwned<'a, T> {
    /// Provides a reference to the owned or borrowed value.
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
