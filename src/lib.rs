pub mod effector;
pub mod forces;
pub mod sensor;
pub mod six_dof;
mod types;
pub mod xpbd;

pub use types::*;

pub trait FromState<S> {
    fn from_state(time: Time, state: &S) -> Self;
}
