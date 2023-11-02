use super::FromState;
use crate::Time;

pub trait Effector<T, S> {
    type Effect;

    fn effect(&self, time: Time, state: &S) -> Self::Effect;
}

pub trait StateEffect<S> {
    fn apply(&self, time: Time, init_state: &S, inc_state: &mut S);
}

macro_rules! impl_effector {
    ($($ty:tt),+) => {
        #[allow(non_snake_case)]
        impl<F, $($ty,)* E, S> Effector<($($ty, )*), S> for F
        where
            F: Fn($($ty, )*) -> E,
            $($ty: FromState<S>, )*
        {
            type Effect = E;

            fn effect(&self, time: Time, state: &S) -> Self::Effect {
                $(
                    let $ty = $ty::from_state(time, &state);
                )*
                (self)($($ty,)*)
            }
        }
    };
}

impl<F, E, S> Effector<(), S> for F
where
    F: Fn() -> E,
{
    type Effect = E;

    fn effect(&self, _time: Time, _state: &S) -> Self::Effect {
        (self)()
    }
}

impl_effector!(T1);
impl_effector!(T1, T2);
impl_effector!(T1, T2, T3);
impl_effector!(T1, T2, T3, T4);
impl_effector!(T1, T2, T3, T4, T5);
impl_effector!(T1, T2, T3, T4, T5, T6);
impl_effector!(T1, T2, T3, T4, T5, T6, T7);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12);
impl_effector!(T1, T2, T3, T4, T5, T6, T7, T9, T10, T11, T12, T13);

macro_rules! concrete_effector {
    ($concrete_name: ident, $trait_name: ident, $state: ty, $effect: ty) => {
        struct $concrete_name<ER, E> {
            effector: ER,
            _phantom: std::marker::PhantomData<(E,)>,
        }

        impl<ER, E> $concrete_name<ER, E> {
            fn new(effector: ER) -> Self {
                Self {
                    effector,
                    _phantom: std::marker::PhantomData,
                }
            }
        }
        impl<ER, T, Eff> $trait_name for $concrete_name<ER, T>
        where
            ER: for<'s> crate::effector::Effector<T, $state, Effect = Eff>,
            Eff: Into<$effect>,
        {
            fn effect<'s>(&self, time: crate::Time, state: $state) -> $effect {
                self.effector.effect(time, &state).into()
            }
        }

        pub trait $trait_name {
            fn effect<'s>(&self, time: crate::Time, state: $state) -> $effect;
        }
    };
}

pub(crate) use concrete_effector;
