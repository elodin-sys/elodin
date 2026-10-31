mod runge_kutta;
mod semi_implicit;

pub use runge_kutta::*;
pub use semi_implicit::*;

pub enum Integrator {
    ExplicitRk(&'static ButcherTableau),
    SemiImplicit,
}
