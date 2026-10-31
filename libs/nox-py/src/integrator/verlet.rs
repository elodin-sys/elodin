use crate::World;
use crate::globals::SimulationTimeStep;
use crate::system::{CompiledSystem, System, SystemBuilder, SystemParam};
use crate::{ComponentArray, ComponentGroup, Error, Query};
use nox::Scalar;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

/// Velocity Verlet: second order, and symplectic when the acceleration `A` that `pipe`
/// writes depends only on position `X`.
pub struct VelocityVerlet<X, V, A, Pipe> {
    dt: Option<f64>,
    pipe: Pipe,
    phantom_data: PhantomData<(X, V, A)>,
}

impl<X, V, A, Pipe> VelocityVerlet<X, V, A, Pipe> {
    pub fn new(pipe: Pipe, dt: Option<f64>) -> Self {
        Self {
            dt,
            pipe,
            phantom_data: PhantomData,
        }
    }
}

impl<X, V, A, Pipe> System for VelocityVerlet<X, V, A, Pipe>
where
    Query<X>: SystemParam<Item = Query<X>>,
    Query<V>: SystemParam<Item = Query<V>> + Clone,
    Query<A>: SystemParam<Item = Query<A>> + Clone,
    X: Add<V, Output = X> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = X> + Send + Sync,
    V: Add<A, Output = V> + ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = V> + Send + Sync,
    A: ComponentGroup + for<'a> nox::FromBuilder<Item<'a> = A> + Send + Sync,
    Scalar<f64>: Mul<V, Output = V> + Mul<A, Output = A>,
    Pipe: System + Send + Sync,
{
    type Arg = ();
    type Ret = ();

    fn init(&self, builder: &mut SystemBuilder) -> Result<(), Error> {
        self.pipe.init(builder)?;
        ComponentArray::<SimulationTimeStep>::init(builder)?;
        Query::<X>::init(builder)?;
        Query::<V>::init(builder)?;
        Query::<A>::init(builder)
    }

    fn compile(&self, world: &World) -> Result<CompiledSystem, Error> {
        let mut builder = SystemBuilder::new(world);
        let compiled_pipe = self.pipe.compile(world)?;
        self.init(&mut builder)?;
        let x0 = Query::<X>::param(&builder)?;
        let v0 = Query::<V>::param(&builder)?;
        let sim_dt = ComponentArray::<SimulationTimeStep>::param(&builder)?;
        let dt = self.dt.map(Scalar::from).unwrap_or_else(|| sim_dt.get(0).0);
        let half_dt = &dt * 0.5;

        compiled_pipe.clone().insert_into_builder(&mut builder)?;
        let a0 = Query::<A>::param(&builder)?;
        let v_half = v0
            .clone()
            .join_query(a0.clone())
            .map(|v, a| v + half_dt.clone() * a)?;
        x0.join_query(v_half.clone())
            .map(|x, v| x + dt.clone() * v)?
            .insert_into_builder(&mut builder);
        // Evaluating at the Euler-predicted velocity rather than v_half keeps
        // velocity-dependent accelerations second order.
        v0.join_query(a0)
            .map(|v, a| v + dt.clone() * a)?
            .insert_into_builder(&mut builder);
        compiled_pipe.insert_into_builder(&mut builder)?;
        let a1 = Query::<A>::param(&builder)?;
        v_half
            .join_query(a1)
            .map(|v, a| v + half_dt.clone() * a)?
            .insert_into_builder(&mut builder);
        builder.to_compiled_system()
    }
}
