use nox::Op;
use nox::OwnedRepr;
use nox::Scalar;

use crate::ecs::{Archetype, Component, component_array::ComponentArray};

#[derive(Clone)]
pub struct SimulationTimeStep<R: OwnedRepr = Op>(pub Scalar<f64, R>);

#[derive(Clone)]
pub struct Tick<R: OwnedRepr = Op>(pub Scalar<u64, R>);

// Manual Component implementation for SimulationTimeStep
impl<R: OwnedRepr> impeller2::component::Component for SimulationTimeStep<R> {
    const NAME: &'static str = "simulation_time_step";
    
    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        Scalar::<f64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for SimulationTimeStep<R> 
where
    Self: nox::ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{}

impl<R: OwnedRepr> nox::ReprMonad<R> for SimulationTimeStep<R> {
    type Elem = <Scalar<f64, R> as nox::ReprMonad<R>>::Elem;
    type Dim = <Scalar<f64, R> as nox::ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = SimulationTimeStep<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        SimulationTimeStep(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        SimulationTimeStep(Scalar::from_inner(inner))
    }
}

// Manual Component implementation for Tick
impl<R: OwnedRepr> impeller2::component::Component for Tick<R> {
    const NAME: &'static str = "tick";
    
    fn schema() -> impeller2::schema::Schema<Vec<u64>> {
        Scalar::<u64, R>::schema()
    }
}

impl<R: OwnedRepr> Component for Tick<R> 
where
    Self: nox::ReprMonad<Op> + for<'a> nox::FromBuilder<Item<'a> = Self>
{}

impl<R: OwnedRepr> nox::ReprMonad<R> for Tick<R> {
    type Elem = <Scalar<u64, R> as nox::ReprMonad<R>>::Elem;
    type Dim = <Scalar<u64, R> as nox::ReprMonad<R>>::Dim;
    type Map<NewRepr: OwnedRepr> = Tick<NewRepr>;

    fn map<N: OwnedRepr>(
        self,
        func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
    ) -> Self::Map<N> {
        Tick(self.0.map(func))
    }

    fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
        self.0.into_inner()
    }

    fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
        self.0.inner()
    }

    fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
        Tick(Scalar::from_inner(inner))
    }
}

impl Default for SimulationTimeStep {
    fn default() -> Self {
        SimulationTimeStep(0.01.into())
    }
}

impl Tick {
    pub fn zero() -> Self {
        Tick(0.into())
    }
}

pub struct SystemGlobals {
    pub sim_tick: Tick,
    pub sim_time_step: SimulationTimeStep,
}

// Manual Archetype implementation for SystemGlobals
impl Archetype for SystemGlobals {
    fn components() -> Vec<(impeller2::schema::Schema<Vec<u64>>, impeller2_wkt::ComponentMetadata)> {
        use impeller2::component::Component;
        use crate::ecs::archetype::ComponentExt;
        vec![
            (<Tick>::schema(), <Tick>::metadata()),
            (<SimulationTimeStep>::schema(), <SimulationTimeStep>::metadata()),
        ]
    }

    fn insert_into_world(self, world: &mut crate::ecs::World) {
        self.sim_tick.insert_into_world(world);
        self.sim_time_step.insert_into_world(world);
    }
}

impl SystemGlobals {
    pub fn new(sim_time_step: f64) -> Self {
        SystemGlobals {
            sim_tick: Tick::zero(),
            sim_time_step: SimulationTimeStep(sim_time_step.into()),
        }
    }
}

pub fn increment_sim_tick(query: ComponentArray<Tick>) -> ComponentArray<Tick> {
    query.map(|tick: Tick| Tick(tick.0 + 1)).unwrap()
}

