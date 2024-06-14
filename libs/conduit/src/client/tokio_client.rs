use bytes::{Buf, Bytes, BytesMut};
use std::io;
use tokio::net::{
    tcp::{OwnedReadHalf, OwnedWriteHalf},
    TcpStream,
};
use tokio_util::codec::{Framed, FramedRead, FramedWrite, LengthDelimitedCodec};

use crate::{
    client::{Demux, Msg},
    ser_de::Slice,
    Error, Packet, Payload,
};

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
            Some(Err(err)) => {
                return Err(err.into());
            }
            None => {
                return Err(Error::ConnectionClosed);
            }
        };
        let buf = buf.freeze();
        let packet = Packet::parse(buf)?;
        self.demux.handle(packet)
    }
}

impl<T> AsyncClient<T>
where
    T: futures::Sink<Bytes, Error = io::Error> + Unpin,
{
    pub async fn send(&mut self, packet: Packet<Payload<impl Buf + Slice>>) -> Result<(), Error> {
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

pub type TcpClient = AsyncClient<Framed<TcpStream, LengthDelimitedCodec>>;

impl<T> AsyncClient<Framed<T, LengthDelimitedCodec>>
where
    T: tokio::io::AsyncWrite + tokio::io::AsyncRead + Unpin,
{
    pub fn from_stream(stream: T) -> Self {
        AsyncClient::new(Framed::new(stream, LengthDelimitedCodec::default()))
    }
}
