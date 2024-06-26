mod rk4;
mod semi_implicit;

pub use rk4::*;
pub use semi_implicit::*;

pub enum Integrator {
    Rk4,
    SemiImplicit,
}
