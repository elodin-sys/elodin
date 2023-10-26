mod bevy_transform;
pub mod body;
pub mod builder;
pub mod constraints;
pub mod effector;
pub mod forces;
pub mod hierarchy;
pub mod history;
pub mod monte_carlo;
pub mod plugin;
pub mod runner;
pub mod runtime;
pub mod sensor;
pub mod spatial;
pub mod systems;
pub mod tree;

mod types;

pub use types::*;

pub trait FromState<S> {
    fn from_state(time: Time, state: &S) -> Self;
}
