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

#[cfg(all(feature = "std", feature = "flume"))]
pub struct MsgPair {
    pub msg: Msg<bytes::Bytes>,
    pub tx: Option<flume::WeakSender<crate::Packet<crate::Payload<bytes::Bytes>>>>,
}
