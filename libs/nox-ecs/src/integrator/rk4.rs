use crate::{ComponentGroup, Error, Query};
use crate::{IntoSystem, System, SystemParam};
use nox::IntoOp;
use std::ops::Add;
use std::sync::Arc;
use std::{marker::PhantomData, ops::Mul};

pub struct Rk4<U, DU, Pipe> {
    dt: f64,
    pipe: Arc<Pipe>,
    phantom_data: PhantomData<(U, DU)>,
}

impl<Pipe, U, DU> Rk4<U, DU, Pipe> {
    pub fn new(pipe: Pipe, dt: f64) -> Self {
        Self {
            dt,
            pipe: Arc::new(pipe),
            phantom_data: PhantomData,
        }
    }
}

pub trait Rk4Ext {
    fn rk4<U, DU>(self) -> Rk4<U, DU, Self>
    where
        Self: Sized;
    fn rk4_with_dt<U, DU>(self, dt: f64) -> Rk4<U, DU, Self>
    where
        Self: Sized;
}

impl<Sys> Rk4Ext for Sys
where
    Sys: System,
{
    fn rk4<U, DU>(self) -> Rk4<U, DU, Self>
    where
        Self: Sized,
    {
        Rk4::new(self, 1.0 / 60.0)
    }

    fn rk4_with_dt<U, DU>(self, dt: f64) -> Rk4<U, DU, Self>
    where
        Self: Sized,
    {
        Rk4::new(self, dt)
    }
}

impl<Pipe, U, DU> System for Rk4<U, DU, Pipe>
where
    Query<U>: SystemParam<Item = Query<U>> + Clone,
    Query<DU>: SystemParam<Item = Query<DU>> + Clone,
    U: Add<DU, Output = U> + ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = U>,
    DU: Add<DU, Output = DU> + ComponentGroup + IntoOp + for<'a> nox::FromBuilder<Item<'a> = DU>,
    f64: Mul<DU, Output = DU>,
    Pipe: System,
{
    type Arg = Pipe::Arg;
    type Ret = Pipe::Ret;

    fn init_builder(&self, builder: &mut crate::PipelineBuilder) -> Result<(), Error> {
        self.pipe.init_builder(builder)?;
        Query::<U>::init(builder)?;
        Query::<DU>::init(builder)
    }

    fn add_to_builder(&self, builder: &mut crate::PipelineBuilder) -> Result<(), Error> {
        let dt = self.dt;
        let init_u = Query::<U>::from_builder(builder);
        let f = |dt: f64| {
            let init_u = init_u.clone();
            move |du: Query<DU>| -> Query<U> {
                init_u
                    .clone()
                    .join_query(du.clone())
                    .map(|u, du| u + dt * du)
                    .unwrap()
            }
        };
        let mut step = |dt: f64| -> Result<Query<DU>, Error> {
            f(dt).pipe(self.pipe.clone()).add_to_builder(builder)?;
            Ok(Query::<DU>::from_builder(builder))
        };
        let k1 = step(0.0)?;
        let k2 = step(dt / 2.0)?;
        let k3 = step(dt / 2.0)?;
        let k4 = step(dt)?;
        let u = init_u
            .join_query(k1)
            .join_query(k2)
            .join_query(k3)
            .join_query(k4)
            .map(|(((du, k1), k2), k3), k4| du + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))?;
        u.insert_into_builder(builder);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, Component, World};
    use nox::nalgebra::vector;
    use nox::{nalgebra, Scalar, ScalarExt, SpatialMotion, SpatialTransform};
    use nox_ecs_macros::{ComponentGroup, FromBuilder, IntoOp};

    #[test]
    fn test_simple_rk4() {
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

        impl Add for V {
            type Output = V;

            fn add(self, v: V) -> Self::Output {
                V(self.0 + v.0)
            }
        }

        impl Mul<V> for f64 {
            type Output = V;

            fn mul(self, rhs: V) -> Self::Output {
                V(self * rhs.0)
            }
        }

        #[derive(Archetype)]
        struct Body {
            x: X,
            v: V,
        }

        let mut world = World::default();
        world.spawn(Body {
            x: X(0.0.constant()),
            v: V(10.0.constant()),
        });
        let builder = world.builder().tick_pipeline(().rk4::<X, V>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(col.typed_buf::<f64>().unwrap(), &[0.16666666666666669])
    }

    #[test]
    fn test_six_dof() {
        #[derive(Clone, Component)]
        struct X(SpatialTransform<f64>);
        #[derive(Clone, Component)]
        struct V(SpatialMotion<f64>);
        #[derive(Clone, Component)]
        struct A(SpatialMotion<f64>);

        #[derive(FromBuilder, ComponentGroup, IntoOp)]
        struct U {
            x: X,
            v: V,
        }

        #[derive(FromBuilder, ComponentGroup, IntoOp)]
        struct DU {
            v: V,
            a: A,
        }

        impl Add<DU> for U {
            type Output = U;

            fn add(self, v: DU) -> Self::Output {
                U {
                    x: X(self.x.0 + v.v.0),
                    v: V(self.v.0 + v.a.0),
                }
            }
        }

        impl Add for DU {
            type Output = DU;

            fn add(self, v: DU) -> Self::Output {
                DU {
                    v: V(self.v.0 + v.v.0),
                    a: A(self.a.0 + v.a.0),
                }
            }
        }

        impl Mul<DU> for f64 {
            type Output = DU;

            fn mul(self, rhs: DU) -> Self::Output {
                DU {
                    v: V(self * rhs.v.0),
                    a: A(self * rhs.a.0),
                }
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
        let builder = world.builder().tick_pipeline(().rk4::<U, DU>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(
            col.typed_buf::<f64>().unwrap(),
            &[1.0f64, 0.0, 0.0, 0.0, 0.016666666666666666, 0.0, 0.0]
        )
    }
}
