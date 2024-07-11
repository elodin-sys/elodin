#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "bevy")]
pub mod bevy;
#[cfg(feature = "bevy")]
pub mod bevy_sync;
#[cfg(feature = "nox")]
pub mod nox;

pub mod assets;
pub mod client;
pub mod error;
#[cfg(feature = "std")]
pub mod query;
pub mod ser_de;
#[cfg(feature = "tokio")]
pub mod server;
pub mod types;
mod util;
#[cfg(feature = "well-known")]
pub mod well_known;

pub use assets::*;
pub use error::*;
pub use types::*;
pub use util::*;

#[doc(hidden)]
pub use const_fnv1a_hash;

#[cfg(feature = "std")]
pub use bytes;
pub use ndarray;

#[cfg(feature = "std")]
mod world;
#[cfg(feature = "std")]
pub use world::*;

#[cfg(all(feature = "std", feature = "polars"))]
mod polars;
#[cfg(all(feature = "std", feature = "polars"))]
pub use polars::PolarsWorld;

#[cfg(feature = "std")]
mod replay;

#[cfg(feature = "std")]
pub use replay::*;

#[cfg(feature = "std")]
mod system;
#[cfg(feature = "std")]
pub use system::*;
