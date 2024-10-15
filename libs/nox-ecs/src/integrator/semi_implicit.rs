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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, Component, World, WorldExt};
    use nox::{tensor, Op, OwnedRepr, Scalar, SpatialMotion, SpatialTransform};
    use nox_ecs_macros::ReprMonad;

    #[test]
    fn test_simple_semi_implicit_vertlet() {
        #[derive(Clone, Component, ReprMonad)]
        struct X<R: OwnedRepr = Op>(Scalar<f64, R>);

        impl Add<V> for X {
            type Output = X;

            fn add(self, v: V) -> Self::Output {
                X(self.0 + v.0)
            }
        }

        #[derive(Clone, Component, ReprMonad)]
        struct V<R: OwnedRepr = Op>(Scalar<f64, R>);

        impl Add<A> for V {
            type Output = V;

            fn add(self, v: A) -> Self::Output {
                V(self.0 + v.0)
            }
        }

        impl Mul<V> for Scalar<f64> {
            type Output = V;

            fn mul(self, rhs: V) -> Self::Output {
                V(self * rhs.0)
            }
        }

        #[derive(Clone, Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(Scalar<f64, R>);

        impl Mul<A> for Scalar<f64> {
            type Output = A;

            fn mul(self, rhs: A) -> Self::Output {
                A(self * rhs.0)
            }
        }

        #[derive(Archetype)]
        struct Body {
            x: X,
            v: V,
            a: A,
        }

        let mut world = World::default();
        world.spawn(Body {
            x: X(0.0.into()),
            v: V(0.0.into()),
            a: A(9.8.into()),
        });
        let builder = world
            .builder()
            .tick_pipeline(semi_implicit_euler::<X, V, A>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(col.typed_buf::<f64>().unwrap(), &[0.0006805555011111122])
    }

    #[test]
    fn test_six_dof() {
        #[derive(Clone, Component, ReprMonad)]
        struct X<R: OwnedRepr = Op>(SpatialTransform<f64, R>);
        #[derive(Clone, Component, ReprMonad)]
        struct V<R: OwnedRepr = Op>(SpatialMotion<f64, R>);
        #[derive(Clone, Component, ReprMonad)]
        struct A<R: OwnedRepr = Op>(SpatialMotion<f64, R>);

        impl Add<V> for X {
            type Output = X;

            fn add(self, v: V) -> Self::Output {
                X(self.0 + v.0)
            }
        }

        impl Add<A> for V {
            type Output = V;

            fn add(self, v: A) -> Self::Output {
                V(self.0 + v.0)
            }
        }

        impl Mul<V> for Scalar<f64> {
            type Output = V;

            fn mul(self, rhs: V) -> Self::Output {
                V(&self * rhs.0)
            }
        }

        impl Mul<A> for Scalar<f64> {
            type Output = A;

            fn mul(self, rhs: A) -> Self::Output {
                A(&self * rhs.0)
            }
        }

        #[derive(Archetype)]
        struct Body {
            x: X,
            v: V,
            a: A,
        }

        let mut world = World::default();
        world.spawn(Body {
            x: X(SpatialTransform {
                inner: tensor![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            v: V(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            a: A(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
        });
        let builder = world
            .builder()
            .tick_pipeline(semi_implicit_euler::<X, V, A>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(
            col.typed_buf::<f64>().unwrap(),
            &[1.0f64, 0.0, 0.0, 0.0, 0.008333333, 0.0, 0.0]
        )
    }
}
