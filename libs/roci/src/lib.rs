use combinators::Pipe;
use drivers::DriverMode;

mod system_fn;

pub use impeller2;
pub use impeller2::com_de::{Componentize, Decomponentize};
pub use roci_macros::{Componentize, Decomponentize, Metadatatize};
pub use system_fn::*;

pub mod combinators;
#[cfg(feature = "csv")]
pub mod csv;
pub mod drivers;
pub mod types;

#[cfg(feature = "stellar")]
pub mod tcp;

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

    fn pipe<P, R: IntoSystem<P>>(self, right: R) -> Pipe<Self, R::System<Self::Driver>>
    where
        Self: Sized,
    {
        Pipe {
            left: self,
            right: right.into_system(),
        }
    }
}

pub const fn system_max_size<H: System>() -> usize {
    H::World::MAX_SIZE
}
