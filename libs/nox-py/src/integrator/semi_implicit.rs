use crate::globals::SimulationTimeStep;
use crate::{ComponentArray, ComponentGroup, ErasedSystem, IntoSystem, Query};
use crate::{System, SystemParam};
use core::ops::Add;
use core::ops::Mul;
use nox::Scalar;

/// Semi-implicit Euler integrator, typically used when you need a sympletic integrator
///
/// **WARNING**: This implementation makes a major assumption about the
/// differential equations you are integrating. It assumes that $dx/dt = v(t)$, and not some other function dependent on
/// $$v(t)$$. In other words we are integrating the system of equations:
/// $$\frac{dv}{dt} = a(t, x_t, v_t)$$
/// $$\frac{dx}{dt} = v_{t+1}$$
///
/// If for whatever reason that assumption doesn't hold for you, you might have to use another integrator.
pub fn semi_implicit_euler_with_dt<X, V, A>(dt: f64) -> impl System<Arg = (), Ret = ()>
where
    Query<X>: SystemParam<Item = Query<X>> + Clone,
    Query<V>: SystemParam<Item = Query<V>> + Clone,
    Query<A>: SystemParam<Item = Query<A>> + Clone,
    X: Add<V, Output = X> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = X>,
    V: Add<A, Output = V> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = V>,
    A: ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = A>,
    f64: Mul<V, Output = V>,
    f64: Mul<A, Output = A>,
{
    let step_v = move |query: Query<(V, A)>| -> Query<V> { query.map(|v, a| v + dt * a).unwrap() };
    let step_x = move |query: Query<(X, V)>| -> Query<X> { query.map(|x, v| x + dt * v).unwrap() };
    ErasedSystem::new(step_v.pipe(step_x))
}

/// Semi-implicit Euler integrator, typically used when you need a sympletic integrator
///
/// **WARNING**: This implementation makes a major assumption about the
/// differential equations you are integrating. It assumes that $dx/dt = v(t)$, and not some other function dependent on
/// $$v(t)$$. In other words we are integrating the system of equations:
/// $$\frac{dv}{dt} = a(t, x_t, v_t)$$
/// $$\frac{dx}{dt} = v_{t+1}$$
///
/// If for whatever reason that assumption doesn't hold for you, you might have to use another integrator.
pub fn semi_implicit_euler<X, V, A>() -> impl System<Arg = (), Ret = ()>
where
    Query<X>: SystemParam<Item = Query<X>> + Clone,
    Query<V>: SystemParam<Item = Query<V>> + Clone,
    Query<A>: SystemParam<Item = Query<A>> + Clone,
    X: Add<V, Output = X> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = X>,
    V: Add<A, Output = V> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = V>,
    A: ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = A>,
    Scalar<f64>: Mul<V, Output = V>,
    Scalar<f64>: Mul<A, Output = A>,
{
    let step_v = move |dt: ComponentArray<SimulationTimeStep>, query: Query<(V, A)>| -> Query<V> {
        let dt = dt.get(0).0;
        query.map(|v, a| v + dt.clone() * a).unwrap()
    };
    let step_x = move |dt: ComponentArray<SimulationTimeStep>, query: Query<(X, V)>| -> Query<X> {
        let dt = dt.get(0).0;
        query.map(|x, v| x + dt.clone() * v).unwrap()
    };
    ErasedSystem::new(step_v.pipe(step_x))
}
