mod runge_kutta;
mod semi_implicit;
mod verlet;

pub use runge_kutta::*;
pub use semi_implicit::*;
pub use verlet::*;

use crate::globals::StageTime;
use crate::system::SystemBuilder;
use crate::{ComponentArray, Error, Query};
use nox::Scalar;

pub enum Integrator {
    ExplicitRk(&'static ButcherTableau),
    SemiImplicit,
    Verlet,
}

fn set_stage_time(
    stage_time: &ComponentArray<StageTime>,
    t: Scalar<f64>,
    builder: &mut SystemBuilder,
) -> Result<(), Error> {
    Query::from(stage_time.map(move |_: StageTime| StageTime(t.clone()))?)
        .insert_into_builder(builder);
    Ok(())
}
