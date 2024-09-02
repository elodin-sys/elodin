use impeller::DEFAULT_TIME_STEP;
use nox::Op;
use nox::Repr;
use nox::Scalar;
use nox_ecs_macros::ReprMonad;

use crate::{Archetype, Component, ComponentArray};

#[derive(Component, Clone, ReprMonad)]
pub struct SimulationTimeStep<R: Repr = Op>(pub Scalar<f64, R>);

#[derive(Component, Clone, ReprMonad)]
pub struct SimulationTick<R: Repr = Op>(pub Scalar<u64, R>);

impl Default for SimulationTimeStep {
    fn default() -> Self {
        SimulationTimeStep(DEFAULT_TIME_STEP.as_secs_f64().into())
    }
}

impl SimulationTick {
    pub fn zero() -> Self {
        SimulationTick(0.into())
    }
}

#[derive(Archetype)]
pub struct SystemGlobals {
    sim_tick: SimulationTick,
    sim_time_step: SimulationTimeStep,
}

impl SystemGlobals {
    pub fn new(sim_time_step: f64) -> Self {
        SystemGlobals {
            sim_tick: SimulationTick::zero(),
            sim_time_step: SimulationTimeStep(sim_time_step.into()),
        }
    }
}

pub fn increment_sim_tick(query: ComponentArray<SimulationTick>) -> ComponentArray<SimulationTick> {
    query
        .map(|tick: SimulationTick| SimulationTick(tick.0 + 1))
        .unwrap()
}
