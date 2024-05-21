use crate::{ComponentGroup, ErasedSystem, IntoSystem, Query};
use crate::{System, SystemParam};
use nox::IntoOp;
use std::ops::Add;
use std::ops::Mul;

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
    X: Add<V, Output = X> + ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = X>,
    V: Add<A, Output = V> + ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = V>,
    A: ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = A>,
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
    X: Add<V, Output = X> + ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = X>,
    V: Add<A, Output = V> + ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = V>,
    A: ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = A>,
    f64: Mul<V, Output = V>,
    f64: Mul<A, Output = A>,
{
    semi_implicit_euler_with_dt::<X, V, A>(1.0 / 60.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, Component, World, WorldExt};
    use nox::{
        nalgebra::{self, vector},
        Scalar, ScalarExt, SpatialMotion, SpatialTransform,
    };

    #[test]
    fn test_simple_semi_implicit_vertlet() {
        #[derive(Clone, Component)]
        struct X(Scalar<f64>);

        impl Add<V> for X {
            type Output = X;

            fn add(self, v: V) -> Self::Output {
                X(self.0 + v.0)
            }
        }

        #[derive(Clone, Component)]
        struct V(Scalar<f64>);

        impl Add<A> for V {
            type Output = V;

            fn add(self, v: A) -> Self::Output {
                V(self.0 + v.0)
            }
        }

        impl Mul<V> for f64 {
            type Output = V;

            fn mul(self, rhs: V) -> Self::Output {
                V(self * rhs.0)
            }
        }

        #[derive(Clone, Component)]
        struct A(Scalar<f64>);

        impl Mul<A> for f64 {
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
            x: X(0.0.constant()),
            v: V(0.0.constant()),
            a: A(9.8.constant()),
        });
        let builder = world
            .builder()
            .tick_pipeline(semi_implicit_euler::<X, V, A>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(col.typed_buf::<f64>().unwrap(), &[0.0027222222222222222])
    }

    #[test]
    fn test_six_dof() {
        #[derive(Clone, Component)]
        struct X(SpatialTransform<f64>);
        #[derive(Clone, Component)]
        struct V(SpatialMotion<f64>);
        #[derive(Clone, Component)]
        struct A(SpatialMotion<f64>);

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

        impl Mul<V> for f64 {
            type Output = V;

            fn mul(self, rhs: V) -> Self::Output {
                V(self * rhs.0)
            }
        }

        impl Mul<A> for f64 {
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
            x: X(SpatialTransform {
                inner: vector![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            v: V(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            a: A(SpatialMotion {
                inner: vector![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
        });
        let builder = world
            .builder()
            .tick_pipeline(semi_implicit_euler::<X, V, A>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(
            col.typed_buf::<f64>().unwrap(),
            &[1.0f64, 0.0, 0.0, 0.0, 0.016666666666666666, 0.0, 0.0]
        )
    }
}
