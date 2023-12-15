use crate::{AsBuffer, BufferArg, BufferForm, Client, FromPjrtBuffer};
use paste::paste;
use std::marker::PhantomData;

pub struct Exec<T, R> {
    pub(crate) exec: xla::PjRtLoadedExecutable,
    pub(crate) phantom: PhantomData<(T, R)>,
}

pub trait ToHost {
    type HostTy;

    fn to_host(&self) -> Self::HostTy;
}

// This macro allows us to implement the run function for a series of tuples easily.
// This essentially a workaround for Rust lacking variadic types / generics.
macro_rules! impl_exec {
      ($($ty:tt),+) => {
        #[allow(non_snake_case, clippy::too_many_arguments)]
        impl<$($ty,)* R> Exec<($($ty,)*), R>
        where
            R: BufferForm,
            R::BufferTy: FromPjrtBuffer,
            $($ty: AsBuffer, )*
        {
            pub fn run(&self, client: &Client, $(mut $ty: impl BufferArg<$ty>,)*) -> Result<R::BufferTy, xla::Error> {
                paste! {
                $(
                    let [<buf_$ty>] = $ty.as_buffer(client);
                )*
                let mut res = self.exec.execute_b(&[$([<buf_$ty>],)*])?;
                let tuple = &mut res[0];
                $(
                    if $ty.is_mut_borrowed() {
                        let buf = tuple.pop().unwrap();
                        $ty.replace_buffer(buf);
                    }
                )*

                Ok(R::BufferTy::from_pjrt(res))
                }
            }
        }
      }
}

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
