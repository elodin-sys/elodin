use super::FromState;
use crate::Time;
use std::marker::PhantomData;

pub trait Effector<T, S> {
    type Effect;

    fn effect(&self, time: Time, state: &S) -> Self::Effect;
}

pub trait StateEffect<S> {
    fn apply(&self, time: Time, init_state: &S, inc_state: &mut S);
}

pub(crate) struct ErasedStateEffector<T, S, ER> {
    effector: ER,
    _phantom: PhantomData<(T, S)>,
}

impl<T, ER, E, S> ErasedStateEffector<T, S, ER>
where
    ER: Effector<T, S, Effect = E> + 'static,
    E: StateEffect<S> + 'static,
    S: 'static,
    T: 'static,
{
    pub(crate) fn boxed(effector: ER) -> Box<dyn StateEffect<S>> {
        Box::new(ErasedStateEffector {
            effector,
            _phantom: PhantomData,
        })
    }
}

impl<T, ER: Effector<T, S, Effect = E>, E: StateEffect<S>, S> StateEffect<S>
    for ErasedStateEffector<T, S, ER>
{
    fn apply(&self, time: Time, init_state: &S, inc_state: &mut S) {
        let effect = self.effector.effect(time, init_state);
        effect.apply(time, init_state, inc_state)
    }
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
