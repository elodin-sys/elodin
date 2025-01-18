use impeller2::types::{LenPacket, OwnedPacket};
use stellarator::{
    buf::IoBufMut,
    io::{AsyncRead, AsyncWrite, LengthDelReader},
    BufResult,
};

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
pub enum Error {
    #[error("{0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("{0}")]
    Stellerator(#[from] stellarator::Error),
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
}

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

    pub async fn next<B: IoBufMut>(&mut self, buf: B) -> Result<OwnedPacket<B>, Error> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2::types::MsgExt;
    use impeller2::types::{Msg, PacketId};
    use postcard::experimental::max_size::MaxSize;
    use serde::{Deserialize, Serialize};
    use std::net::SocketAddr;
    use stellarator::net::{TcpListener, TcpStream};

    #[derive(Serialize, Deserialize, MaxSize, PartialEq, Debug)]
    struct Foo {
        bar: u32,
    }

    impl Msg for Foo {
        const ID: PacketId = [0x1, 0x2, 0x3];
    }

    #[test]
    fn test_packet_echo() {
        stellarator::test!(async {
            let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let addr = listener.local_addr().unwrap();
            stellarator::spawn(async move {
                let sink = PacketSink::new(listener.accept().await.unwrap());
                let msg = Foo { bar: 0xBB }.to_len_packet();
                sink.send(msg).await.0.unwrap();
            });
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut stream = PacketStream::new(stream);
            let buf = vec![0; 128];
            let OwnedPacket::Msg(m) = stream.next(buf).await.unwrap() else {
                panic!("non msg pkt");
            };
            assert_eq!(m.id, Foo::ID);
            let foo: Foo = m.parse().unwrap();
            assert_eq!(foo, Foo { bar: 0xBB });
        })
    }
}
