use std::marker::PhantomData;

use crate::drivers::DriverMode;
use crate::{Componentize, Decomponentize, System};

pub struct SystemFn<D: DriverMode, P, F> {
    f: F,
    _phantom: PhantomData<(D, P)>,
}

macro_rules! impl_system_fn {
    ($($ty:tt),+) => {
        impl<F, D, $($ty),*> System for SystemFn<D, ($($ty,)*), F>
        where
            F: for<'a> Fn($(&'a mut $ty,)*),
            D: DriverMode,
            $($ty: Componentize + Decomponentize + Default),+
        {

            type World = ($($ty,)*);

            type Driver = D;

            #[allow(non_snake_case)]
            fn update(&mut self, world: &mut Self::World) {
                let ($($ty,)*) = world;
                (self.f)($($ty),*)
            }
        }
        impl<F, $($ty),*> IntoSystem<($($ty,)*)> for F
            where F: for<'a> Fn($(&'a mut $ty,)*),
            $($ty: Componentize + Decomponentize + Default),+
        {

            type System<D: DriverMode> = SystemFn<D, ($($ty,)*), F>;
            fn into_system<D: DriverMode>(self) -> Self::System<D> {
                SystemFn {
                    f: self,
                    _phantom: PhantomData,
                }
            }
        }

    };
}

impl_system_fn!(T1);
impl_system_fn!(T1, T2);
impl_system_fn!(T1, T2, T3);
impl_system_fn!(T1, T2, T3, T4);
impl_system_fn!(T1, T2, T3, T4, T5);
impl_system_fn!(T1, T2, T3, T4, T5, T6);
impl_system_fn!(T1, T2, T3, T4, T5, T6, T7);
impl_system_fn!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_system_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_system_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_system_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_system_fn!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

pub trait IntoSystem<P> {
    type System<D: DriverMode>: System;
    fn into_system<D: DriverMode>(self) -> Self::System<D>;
}

impl<S: System> IntoSystem<()> for S {
    type System<D: DriverMode> = S;
    fn into_system<D: DriverMode>(self) -> Self::System<D> {
        self
    }
}
