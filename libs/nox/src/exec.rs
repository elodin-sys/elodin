//! Provides functionality for executing operations and transferring data between host and client device.
use crate::{AsBuffer, BufferArg, BufferForm, Client, FromPjrtBuffer};
use paste::paste;
use std::marker::PhantomData;

/// Represents an executable compiled from an XLA computation.
pub struct Exec<T, R> {
    pub(crate) exec: xla::PjRtLoadedExecutable,
    pub(crate) phantom: PhantomData<(T, R)>,
}

/// Defines a trait for converting computation results from client device representations to host types.
pub trait ToHost {
    /// The type of data to be transferred to the host.
    type HostTy;

    /// Transfers data from the client device to the host.
    fn to_host(&self) -> Self::HostTy;
}

// This macro allows us to implement the run function for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_exec {
      ($($ty:tt),*) => {
        #[allow(non_snake_case, clippy::too_many_arguments, non_camel_case_types, unused_variables, unused_mut)]
        impl<$($ty,)* R> Exec<($($ty,)*), R>
        where
            R: BufferForm,
            R::BufferTy: FromPjrtBuffer,
            $($ty: AsBuffer, )*
        {
            paste! {
                #[doc = "Executes the compiled XLA computation with provided arguments."]
                pub fn run<$([<arg_$ty>]: BufferArg<$ty>,)*>(&self, client: &Client, $(mut $ty: [<arg_$ty>],)*) -> Result<R::BufferTy, xla::Error> {
                    let mut args = xla::BufferArgsRef::default();
                    $(
                        let [<buf_$ty>] = $ty.as_buffer(client);
                        args.push(&[<buf_$ty>]);
                    )*
                    let mut res = self.exec.execute_buffers(args.untuple_result(true))?;
                    $(
                        if [<arg_$ty>]::is_mut_borrowed() {
                            let buf = res.pop().unwrap();
                            $ty.replace_buffer(buf);
                        }
                    )*
                    Ok(R::BufferTy::from_pjrt(res))
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
