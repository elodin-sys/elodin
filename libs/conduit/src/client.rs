use std::sync::Arc;

use bevy::utils::HashMap;
use bytes::{Buf, Bytes};

use crate::{
    ser_de::{ColumnValue, Slice},
    ColumnPayload, ControlMsg, Error, Metadata, Packet, Payload, StreamId,
};

#[derive(Clone, Default)]
pub struct Demux {
    streams: HashMap<StreamId, Arc<Metadata>>,
}

impl Demux {
    pub fn handle<B: Buf + Slice>(&mut self, packet: Packet<Payload<B>>) -> Result<Msg<B>, Error> {
        match packet.payload {
            Payload::ControlMsg(ControlMsg::Metadata {
                stream_id,
                metadata,
            }) => {
                self.streams.insert(stream_id, Arc::new(metadata.clone()));
                Ok(Msg::Control(ControlMsg::Metadata {
                    stream_id,
                    metadata,
                }))
            }
            Payload::ControlMsg(m) => Ok(Msg::Control(m)),
            Payload::Column(payload) => {
                let metadata = self
                    .streams
                    .get(&packet.stream_id)
                    .ok_or(Error::StreamNotFound(packet.stream_id))?;
                Ok(Msg::Column(ColumnMsg {
                    metadata: metadata.clone(),
                    payload,
                }))
            }
        }
    }
}

#[derive(Debug)]
pub enum Msg<B = Bytes> {
    Control(ControlMsg),
    Column(ColumnMsg<B>),
}

#[derive(Debug)]
pub struct ColumnMsg<B> {
    pub metadata: Arc<Metadata>,
    pub payload: ColumnPayload<B>,
}

impl ColumnMsg<Bytes> {
    pub fn iter(&self) -> impl Iterator<Item = Result<ColumnValue<'_>, Error>> + '_ {
        self.payload
            .as_ref()
            .into_iter(self.metadata.component_type.clone())
    }
}

pub struct MsgPair {
    pub msg: Msg<Bytes>,
    pub tx: flume::WeakSender<Packet<Payload<Bytes>>>,
}

#[cfg(feature = "tokio")]
pub use tokio_impl::*;

#[cfg(feature = "tokio")]
mod tokio_impl {
    use bytes::{Bytes, BytesMut};
    use futures::Sink;
    use std::io;
    use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
    use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

    use super::*;

    pub struct AsyncClient<T> {
        demux: Demux,
        inner: T,
    }

    impl<T> AsyncClient<T> {
        pub fn new(inner: T) -> Self {
            Self {
                inner,
                demux: Demux::default(),
            }
        }
    }

    impl<T> AsyncClient<T>
    where
        T: futures::stream::Stream<Item = Result<BytesMut, io::Error>> + Unpin,
    {
        pub async fn recv(&mut self) -> Result<Msg<Bytes>, Error> {
            use futures::stream::StreamExt;
            let buf = match self.inner.next().await {
                Some(Ok(m)) => m,
                Some(Err(err)) => return Err(err.into()),
                None => return Err(Error::ConnectionClosed),
            };
            let buf = buf.freeze();
            let packet = Packet::parse(buf)?;
            self.demux.handle(packet)
        }
    }

    impl<T> AsyncClient<T>
    where
        T: Sink<Bytes, Error = io::Error> + Unpin,
    {
        pub async fn send(
            &mut self,
            packet: Packet<Payload<impl Buf + Slice>>,
        ) -> Result<(), Error> {
            use futures::SinkExt;
            let mut buf = BytesMut::new();
            packet.write(&mut buf)?;
            self.inner.send(buf.freeze()).await?;
            Ok(())
        }
    }

    pub type ReaderClient<T> = AsyncClient<FramedRead<T, LengthDelimitedCodec>>;
    impl<T> ReaderClient<T>
    where
        T: tokio::io::AsyncRead + Unpin,
    {
        pub fn from_read_half(read_half: T) -> Self {
            AsyncClient::new(FramedRead::new(read_half, LengthDelimitedCodec::default()))
        }
    }

    pub type TcpReader = AsyncClient<FramedRead<OwnedReadHalf, LengthDelimitedCodec>>;

    pub type WriterClient<T> = AsyncClient<FramedWrite<T, LengthDelimitedCodec>>;

    impl<T> WriterClient<T>
    where
        T: tokio::io::AsyncWrite + Unpin,
    {
        pub fn from_write_half(write_half: T) -> Self {
            AsyncClient::new(FramedWrite::new(
                write_half,
                LengthDelimitedCodec::default(),
            ))
        }
    }

    pub type TcpWriter = AsyncClient<FramedWrite<OwnedWriteHalf, LengthDelimitedCodec>>;
}
