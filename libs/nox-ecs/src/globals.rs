use nox::Op;
use nox::OwnedRepr;
use nox::Scalar;
use nox_ecs_macros::ReprMonad;

use crate::{Archetype, Component, ComponentArray};

#[derive(Component, Clone, ReprMonad)]
pub struct SimulationTimeStep<R: OwnedRepr = Op>(pub Scalar<f64, R>);

#[derive(Component, Clone, ReprMonad)]
pub struct Tick<R: OwnedRepr = Op>(pub Scalar<u64, R>);

impl Default for SimulationTimeStep {
    fn default() -> Self {
        SimulationTimeStep(0.01.into()) // TODO
                                        //SimulationTimeStep(DEFAULT_TIME_STEP.as_secs_f64().into())
    }
}

impl Tick {
    pub fn zero() -> Self {
        Tick(0.into())
    }
}

#[derive(Archetype)]
pub struct SystemGlobals {
    sim_tick: Tick,
    sim_time_step: SimulationTimeStep,
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
