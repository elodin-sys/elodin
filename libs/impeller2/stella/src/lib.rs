use impeller2::types::{LenPacket, OwnedPacket};
use stellarator::{
    buf::{IoBufMut, Slice},
    io::{AsyncRead, AsyncWrite, LengthDelReader},
    BufResult,
};

#[cfg(feature = "queue")]
pub mod queue;

pub struct PacketStream<R: AsyncRead> {
    reader: LengthDelReader<R>,
}

impl<R: AsyncRead> PacketStream<R> {
    pub fn new(reader: R) -> Self {
        let reader = LengthDelReader::new(reader);
        Self::from_reader(reader)
    }
    pub fn from_reader(reader: LengthDelReader<R>) -> Self {
        Self { reader }
    }

    pub async fn next<B: IoBufMut>(&mut self, buf: B) -> Result<OwnedPacket<Slice<B>>, Error> {
        let packet_buf = self.reader.recv(buf).await?;
        OwnedPacket::parse(packet_buf).map_err(Error::from)
    }
}

pub struct PacketSink<W: AsyncWrite> {
    writer: W,
}

impl<W: AsyncWrite> PacketSink<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub async fn send(&self, packet: LenPacket) -> BufResult<(), LenPacket> {
        let (res, inner) = self.writer.write_all(packet.inner).await;
        (res, LenPacket { inner })
    }
}

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
pub enum Error {
    #[error("{0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("{0}")]
    Stellerator(#[from] stellarator::Error),
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
}

#[cfg(test)]
mod tests;
