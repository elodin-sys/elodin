//! Provides functionality for executing operations and transferring data between host and client device.
use crate::{ArrayRepr, AsTypedBuffer, Client, FromTypedBuffers, Op, ReprMonad};
use core::marker::PhantomData;
use paste::paste;

/// Represents an executable compiled from an XLA computation.
pub struct Exec<T, R> {
    pub(crate) exec: xla::PjRtLoadedExecutable,
    pub(crate) phantom: PhantomData<(T, R)>,
}

// This macro allows us to implement the run function for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_exec {
      ($($ty:tt),*) => {
        #[allow(non_snake_case, clippy::too_many_arguments, non_camel_case_types, unused_variables, unused_mut)]
        impl<$($ty,)* R> Exec<($($ty,)*), R>
        where
            R: ReprMonad<Op>,
            R::Map<ArrayRepr>: FromTypedBuffers,
            $($ty: ReprMonad<Op>, )*
        {
            paste! {
                #[doc = "Executes the compiled XLA computation with provided arguments."]
                pub fn run(&self, client: &Client, $(mut $ty: impl AsTypedBuffer<$ty::Map<ArrayRepr>>,)*) -> Result<<R::Map<ArrayRepr> as FromTypedBuffers>::TypedBuffers, xla::Error> {
                    let mut args = xla::BufferArgsRef::default();
                    $(
                        let $ty = $ty.as_typed_buffer(client).unwrap();
                        args.push(&$ty.as_ref().buffer);
                    )*
                    let mut res = self.exec.execute_buffers(args.untuple_result(true))?;
                    Ok(<R::Map<ArrayRepr> as FromTypedBuffers>::from_pjrt_buffers(&mut res))
                }
            }
        }
    }
}

impl_exec!();
impl_exec!(T1);
impl_exec!(T1, T2);
impl_exec!(T1, T2, T3);
impl_exec!(T1, T2, T3, T4);
impl_exec!(T1, T2, T3, T4, T5);
impl_exec!(T1, T2, T3, T4, T5, T6);
impl_exec!(T1, T2, T3, T4, T5, T6, T7);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_exec!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);
