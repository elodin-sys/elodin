use std::marker::PhantomData;
use crate::{FromHost, AsBuffer, FromPjrtBuffer, Client};

pub struct Exec<T, R> {
    pub(crate) exec: xla::PjRtLoadedExecutable,
    pub(crate) phantom: PhantomData<(T, R)>,
}

macro_rules! impl_exec {
      ($($ty:tt),+) => {
        #[allow(non_snake_case, clippy::too_many_arguments)]
        impl<$($ty,)* R> Exec<($($ty,)*), R>
        where
            R: ToHost,
            R::HostTy: FromPjrtBuffer,
            $($ty: FromHost + AsBuffer, )*
        {
            pub fn run(&self, client: &Client, $($ty: $ty::HostTy,)*) -> Result<R::HostTy, xla::Error> {
                $(
                let $ty = $ty::from_host(client, $ty);
                let $ty = $ty.as_buffer();
                )*
                let res = self.exec.execute_b(&[$($ty,)*])?;
                Ok(R::HostTy::from_pjrt(res))
            }
        }
      }
}

pub trait ToHost {
    type HostTy;
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