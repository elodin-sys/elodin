use thiserror::Error;

#[cfg(feature = "bevy")]
pub mod bevy;
#[cfg(feature = "bevy")]
pub mod bevy_sync;
#[cfg(feature = "nox")]
pub mod nox;

pub mod client;
pub mod error;
pub mod query;
pub mod ser_de;
#[cfg(feature = "tokio")]
pub mod server;
pub mod types;
pub mod well_known;

pub use error::*;
pub use types::*;

#[doc(hidden)]
pub use const_fnv1a_hash;

pub use bytes;
pub use ndarray;
