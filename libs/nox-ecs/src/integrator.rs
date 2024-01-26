use crate::{Component, Error};
use crate::{ComponentArray, System, SystemParam};
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

pub trait Rk4Ext<P> {
    fn rk4<U, DU>(self) -> Rk4<U, DU, Self>
    where
        Self: Sized;
    fn rk4_with_dt<U, DU>(self, dt: f64) -> Rk4<U, DU, Self>
    where
        Self: Sized;
}

impl<A, R, Sys> Rk4Ext<(A, R)> for Sys
where
    Sys: System<A, R>,
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

impl<Pipe, Arg, Ret, U, DU> System<Arg, Ret> for Rk4<U, DU, Pipe>
where
    ComponentArray<U>: SystemParam<Item = ComponentArray<U>> + Clone,
    ComponentArray<DU>: SystemParam<Item = ComponentArray<DU>> + Clone,
    U: Add<DU, Output = U> + Component,
    DU: Add<DU, Output = DU> + Component,
    f64: Mul<DU, Output = DU>,

    Pipe: System<Arg, Ret> + Clone,
{
    fn add_to_builder(&self, builder: &mut crate::PipelineBuilder) -> Result<(), Error> {
        ComponentArray::<U>::init(builder)?;
        ComponentArray::<DU>::init(builder)?;
        let dt = self.dt;
        let init_u = ComponentArray::<U>::from_builder(builder);
        let f = |dt: f64| {
            let init_u = init_u.clone();
            move |du: ComponentArray<DU>| -> ComponentArray<U> {
                init_u.join(&du).map(|u, du| u + dt * du).unwrap()
            }
        };
        let mut step = |dt: f64| -> Result<ComponentArray<DU>, Error> {
            f(dt).pipe(self.pipe.clone()).add_to_builder(builder)?;
            Ok(ComponentArray::<DU>::from_builder(builder))
        };
        let k1 = step(0.0)?;
        let k2 = step(dt / 2.0)?;
        let k3 = step(dt / 2.0)?;
        let k4 = step(dt)?;
        let u = init_u
            .join(&k1)
            .join(&k2)
            .join(&k3)
            .join(&k4)
            .map(|du, k1, k2, k3, k4| du + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))?;
        u.insert_into_builder(builder);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Component;
    use crate::{Archetype, World, WorldBuilder};
    use nox::Scalar;

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
            x: X::host(0.0),
            v: V::host(10.0),
        });
        let builder = WorldBuilder::new(world, ().rk4::<X, V>());
        let client = nox::Client::cpu().unwrap();
        let mut exec = builder.build(&client).unwrap();
        exec.run(&client).unwrap();
        let col = exec.column(X::component_id()).unwrap();
        assert_eq!(col.typed_buf::<f64>().unwrap(), &[0.16666666666666669])
    }
}
