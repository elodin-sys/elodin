use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul},
};

pub mod effector;
pub mod forces;
pub mod sensor;
mod six_dof;
mod types;
pub mod xpbd;

use effector::{Effector, ErasedStateEffector, StateEffect};
pub use six_dof::*;
pub use types::*;

pub trait FromState<S> {
    fn from_state(time: Time, state: &S) -> Self;
}

pub struct Sim<S> {
    pub state: S,
    effectors: Vec<Box<dyn StateEffect<S>>>,
}

impl<S: Debug> Debug for Sim<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sim").field("state", &self.state).finish()
    }
}

impl<S> Sim<S>
where
    S: Default
        + Clone
        + Copy
        + Debug
        + Add<Output = S>
        + AddAssign<S>
        + State
        + TimeState
        + 'static,
    f64: Mul<S, Output = S>,
{
    pub fn new(state: S) -> Self {
        Self {
            state,
            effectors: vec![],
        }
    }

    pub fn effector<T, E, EF>(mut self, effector: E) -> Self
    where
        T: 'static,
        E: 'static,
        E: Effector<T, S, Effect = EF> + 'static,
        EF: StateEffect<S> + 'static,
    {
        self.effectors.push(ErasedStateEffector::boxed(effector));
        self
    }

    pub fn tick(&mut self, dt: f64) {
        let delta = rk4_step(
            |init_state| {
                let mut state = self.effectors.iter().fold(S::default(), |mut s, e| {
                    e.apply(init_state.time(), &init_state, &mut s);
                    s
                });
                init_state.step(&mut state);
                state
            },
            self.state,
            dt,
        );
        self.state += delta;
    }
}

pub fn rk4_step<F, S, A>(f: F, init_state: S, dt: f64) -> A
where
    F: Fn(S) -> A,
    S: Add<A, Output = S> + Copy + Debug,
    A: Add<A, Output = A> + Copy + Debug,
    f64: Mul<A, Output = A>,
{
    let k1 = f(init_state);
    let half_dt: f64 = dt / 2.0;
    let k2 = f(init_state + half_dt * k1);
    let k3 = f(init_state + half_dt * k2);
    let k4 = f(init_state + dt * k3);
    (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

pub trait State {
    fn step(&self, inc_state: &mut Self); // TODO: this feels kinda hacky or maybe just a bad name
}

#[cfg(test)]
mod tests;
