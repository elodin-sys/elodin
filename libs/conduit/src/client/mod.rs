#[cfg(feature = "tokio")]
mod tokio_client;

#[cfg(feature = "tokio")]
pub use tokio_client::*;

#[cfg(feature = "embedded-io-async")]
pub mod embedded_async;

#[cfg(feature = "std")]
mod demux;

#[cfg(feature = "std")]
pub use demux::*;

#[cfg(feature = "std")]
use crate::{Packet, Payload};

#[cfg(feature = "std")]
pub struct MsgPair {
    pub msg: Msg<bytes::Bytes>,
    pub tx: flume::WeakSender<Packet<Payload<bytes::Bytes>>>,
}
