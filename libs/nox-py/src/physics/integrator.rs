use crate::ecs::system::{CompiledSystem, IntoSystem, System, SystemBuilder, SystemParam};
use crate::ecs::{Query, World, component_array::ComponentArray, query::ComponentGroup};
use crate::physics::globals::SimulationTimeStep;
use nox::Scalar;
use std::ops::Add;
use std::{marker::PhantomData, ops::Mul};

// Integrator enum
pub enum Integrator {
    Rk4,
    SemiImplicit,
}

// RK4 Integrator
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
    U: Add<DU, Output = U> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = U> + Send + Sync,
    DU: Add<DU, Output = DU>
        + ComponentGroup
        + for<'a> nox::FromBuilder<Item<'a> = DU>
        + Send
        + Sync,
    f64: Mul<DU, Output = DU>,
    Scalar<f64>: Mul<DU, Output = DU>,
    Pipe: System + Send + Sync,
{
    type Arg = ();
    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), crate::Error> {
        self.pipe.init(builder)?;
        ComponentArray::<SimulationTimeStep>::init(builder)?;
        Query::<U>::init(builder)?;
        Query::<DU>::init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, crate::Error> {
        let mut builder = SystemBuilder::new(world);
        let compiled_pipe = self.pipe.compile(world)?;
        self.init(&mut builder)?;
        let init_u = Query::<U>::param(&builder)?;
        let sim_dt = ComponentArray::<SimulationTimeStep>::param(&builder)?;
        let dt = self.dt.map(Scalar::from).unwrap_or_else(|| sim_dt.get(0).0);

        let step =
            |dt_factor: f64, builder: &mut SystemBuilder| -> Result<Query<DU>, crate::Error> {
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

// Semi-implicit Euler Integrator
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
    crate::ErasedSystem::new(step_v.pipe(step_x))
}

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
    crate::ErasedSystem::new(step_v.pipe(step_x))
}
