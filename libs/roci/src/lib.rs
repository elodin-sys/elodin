use drivers::DriverMode;

mod componentize;
mod decomponentize;

pub use componentize::*;
pub use conduit;
pub use decomponentize::*;
pub use roci_macros::{Componentize, Decomponentize, Metadatatize};

pub mod combinators;
#[cfg(feature = "csv")]
pub mod csv;
pub mod drivers;
#[cfg(feature = "tokio")]
pub mod tokio;
pub mod types;

#[cfg(feature = "std")]
pub mod metadata;
#[cfg(feature = "std")]
pub use metadata::Metadatatize;

pub trait System {
    type World: Default + Decomponentize + Componentize;
    const MAX_SIZE: usize = Self::World::MAX_SIZE;

    type Driver: DriverMode;

    fn init_world(&mut self) -> Self::World {
        Default::default()
    }

    fn update(&mut self, world: &mut Self::World);
}

pub const fn system_max_size<H: System>() -> usize {
    H::World::MAX_SIZE
}
