use core::convert::Infallible;

use combinators::Pipe;
use drivers::DriverMode;

mod system_fn;

pub use impeller2;
pub use impeller2::com_de::{Componentize, Decomponentize};
pub use impeller2::vtable::AsVTable;
pub use impeller2_wkt;
pub use impeller2_wkt::Metadatatize;
pub use db_macros::{AsVTable, Componentize, Decomponentize, Metadatatize};
pub use impeller2_stellar::{SinkExt, StreamExt, Subscription};
pub use system_fn::*;
pub use zerocopy;

pub mod combinators;
#[cfg(feature = "csv")]
pub mod csv;
pub mod drivers;

pub trait System {
    type World: Default + Decomponentize<Error = Infallible> + Componentize;
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
