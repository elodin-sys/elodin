use crate::Client;

pub trait FromHost {
    type HostTy;

    fn from_host(client: &Client, native: Self::HostTy) -> Self;
}

pub trait FromPjrtBuffer {
    fn from_pjrt(pjrt: Vec<Vec<xla::PjRtBuffer>>) -> Self;
}

pub trait AsBuffer {
    fn as_buffer(&self) -> &xla::PjRtBuffer;
}

pub trait AsOp {
    fn as_op(&self) -> &xla::XlaOp;
}

pub trait BufferForm {
    type BufferTy;
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
