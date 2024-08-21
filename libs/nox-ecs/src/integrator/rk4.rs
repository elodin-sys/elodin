use crate::globals::SimulationTimeStep;
use crate::system::{CompiledSystem, IntoSystem, System, SystemBuilder, SystemParam};
use crate::{ComponentArray, ComponentGroup, Error, Query};
use impeller::World;
use nox::{IntoOp, Scalar};
use std::ops::Add;
use std::{marker::PhantomData, ops::Mul};

pub struct Rk4<U, DU, Pipe> {
    dt: Option<f64>,
    pipe: Pipe,
    phantom_data: PhantomData<(U, DU)>,
}

impl<Pipe, U, DU> Rk4<U, DU, Pipe> {
    pub fn new(pipe: Pipe, dt: Option<f64>) -> Self {
        Self {
            dt,
            pipe,
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
        Rk4::new(self, None)
    }

    fn rk4_with_dt<U, DU>(self, dt: f64) -> Rk4<U, DU, Self>
    where
        Self: Sized,
    {
        Rk4::new(self, Some(dt))
    }
}

impl<Pipe, U, DU> System for Rk4<U, DU, Pipe>
where
    Query<U>: SystemParam<Item = Query<U>> + Clone,
    Query<DU>: SystemParam<Item = Query<DU>> + Clone,
    U: Add<DU, Output = U>
        + ComponentGroup
        + IntoOp
        + for<'a> nox::FromBuilder<Item<'a> = U>
        + Send
        + Sync,
    DU: Add<DU, Output = DU>
        + ComponentGroup
        + IntoOp
        + for<'a> nox::FromBuilder<Item<'a> = DU>
        + Send
        + Sync,
    f64: Mul<DU, Output = DU>,
    Scalar<f64>: Mul<DU, Output = DU>,
    Pipe: System + Send + Sync,
{
    type Arg = ();
    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.pipe.init(builder)?;
        ComponentArray::<SimulationTimeStep>::init(builder)?;
        Query::<U>::init(builder)?;
        Query::<DU>::init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        let mut builder = SystemBuilder::new(world);
        let compiled_pipe = self.pipe.compile(world)?;
        self.init(&mut builder)?;
        let init_u = Query::<U>::param(&builder)?;
        let sim_dt = ComponentArray::<SimulationTimeStep>::param(&builder)?;
        let dt = self.dt.map(Scalar::from).unwrap_or_else(|| sim_dt.get(0).0);

        let step = |dt_factor: f64, builder: &mut SystemBuilder| -> Result<Query<DU>, Error> {
            let f = |dt: ComponentArray<SimulationTimeStep>,
                     init_u: Query<U>,
                     du: Query<DU>|
             -> Query<U> {
                let dt = &dt.get(0).0 * dt_factor;
                init_u
                    .clone()
                    .join_query(du.clone())
                    .map(|u, du| u + dt.clone() * du)
                    .unwrap()
            };
            f.into_system()
                .compile(world)?
                .insert_into_builder(builder)?;
            compiled_pipe.clone().insert_into_builder(builder)?;
            Query::<DU>::param(builder)
        };

        let k1 = step(0.0, &mut builder)?;
        init_u.insert_into_builder(&mut builder);
        let k2 = step(0.5, &mut builder)?;
        init_u.insert_into_builder(&mut builder);
        let k3 = step(0.5, &mut builder)?;
        init_u.insert_into_builder(&mut builder);
        let k4 = step(1.0, &mut builder)?;
        init_u.insert_into_builder(&mut builder);

        let u = init_u
            .join_query(k1)
            .join_query(k2)
            .join_query(k3)
            .join_query(k4.clone())
            .map(|(((u, k1), k2), k3), k4| {
                let du = (&dt * (1.0 / 6.0)) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                u + du
            })
            .unwrap();
        u.insert_into_builder(&mut builder);
        builder.to_compiled_system()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Archetype, Component, World, WorldExt};
    use nox::tensor;
    use nox::{Scalar, SpatialMotion, SpatialTransform};
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

        impl Mul<V> for Scalar<f64> {
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
            x: X(0.0.into()),
            v: V(10.0.into()),
        });
        let builder = world.builder().tick_pipeline(().rk4::<X, V>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(col.typed_buf::<f64>().unwrap(), &[0.08333333])
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

        impl Mul<DU> for Scalar<f64> {
            type Output = DU;

            fn mul(self, rhs: DU) -> Self::Output {
                DU {
                    v: V(&self * rhs.v.0),
                    a: A(&self * rhs.a.0),
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
                inner: tensor![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
            v: V(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 1.0, 0.0, 0.0].into(),
            }),
            a: A(SpatialMotion {
                inner: tensor![0.0, 0.0, 0.0, 0.0, 0.0, 0.0].into(),
            }),
        });
        let builder = world.builder().tick_pipeline(().rk4::<U, DU>());
        let world = builder.run();
        let col = world.column::<X>().unwrap();
        assert_eq!(
            col.typed_buf::<f64>().unwrap(),
            &[1.0f64, 0.0, 0.0, 0.0, 0.008333333, 0.0, 0.0]
        )
    }
}
