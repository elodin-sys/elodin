mod runge_kutta;
mod semi_implicit;
mod verlet;

pub use runge_kutta::*;
pub use semi_implicit::*;
pub use verlet::*;

pub enum Integrator {
    ExplicitRk(&'static ButcherTableau),
    SemiImplicit,
    Verlet,
}
