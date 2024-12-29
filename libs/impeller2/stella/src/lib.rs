use impeller2::{
    com_de::Decomponentize,
    registry::VTableRegistry,
    types::{self, PacketId, PacketTy, PACKET_HEADER_LEN},
};
use serde::{Deserialize, Serialize};
use stellarator::{
    buf::{IoBuf, IoBufMut, Slice},
    io::{AsyncRead, AsyncWrite, LengthDelReader},
    BufResult,
};
use zerocopy::TryFromBytes;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("{0}")]
    Stellerator(#[from] stellarator::Error),
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("invalid packet")]
    InvalidPacket,
    #[error("vtable not found")]
    VTableNotFound,
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

    pub async fn next<B: IoBufMut>(&mut self, buf: B) -> Result<Packet<B>, Error> {
        let packet_buf = self.reader.recv(buf).await?;
        let types::Packet { packet_ty, id, .. } =
            types::Packet::try_ref_from_bytes(&packet_buf).unwrap();
        let packet_ty = *packet_ty;
        let id = *id;
        let buf = packet_buf
            .try_sub_slice(PACKET_HEADER_LEN..)
            .ok_or(Error::InvalidPacket)?;
        Ok(match packet_ty {
            PacketTy::Msg => Packet::Msg(MsgBuf { id, buf }),
            PacketTy::Table => Packet::Table(Table { id, buf }),
            PacketTy::TimeSeries => Packet::TimeSeries(TimeSeries { id, buf }),
        })
    }
}

pub struct Table<B: IoBuf> {
    pub id: PacketId,
    pub buf: Slice<B>,
}

impl<B: IoBuf> Table<B> {
    pub fn sink(
        &self,
        registry: &impl VTableRegistry,
        sink: &mut impl Decomponentize,
    ) -> Result<(), Error> {
        let vtable = registry.get(&self.id).ok_or(Error::VTableNotFound)?;
        vtable
            .parse_table(stellarator::buf::deref(&self.buf), sink)
            .map_err(Error::Impeller)
    }
}

pub struct MsgBuf<B: IoBuf> {
    pub id: PacketId,
    pub buf: Slice<B>,
}

impl<B: IoBuf> MsgBuf<B> {
    pub fn parse<'a, T: Deserialize<'a> + 'a>(&'a self) -> Result<T, Error> {
        let msg = postcard::from_bytes(stellarator::buf::deref(&self.buf))?;
        Ok(msg)
    }

    pub fn try_parse<'a, T: Msg + Deserialize<'a> + 'a>(&'a self) -> Option<Result<T, Error>> {
        if T::ID == self.id {
            return None;
        }
        Some(self.parse())
    }
}

pub struct TimeSeries<B: IoBuf> {
    pub id: PacketId,
    pub buf: Slice<B>,
}

pub enum Packet<B: IoBuf> {
    Msg(MsgBuf<B>),
    Table(Table<B>),
    TimeSeries(TimeSeries<B>),
}

impl<B: IoBuf> Packet<B> {
    pub fn into_buf(self) -> B {
        match self {
            Packet::Msg(msg_buf) => msg_buf.buf.into_inner(),
            Packet::Table(table) => table.buf.into_inner(),
            Packet::TimeSeries(time_series) => time_series.buf.into_inner(),
        }
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
        let (res, packet) = self.writer.write_all(packet).await;
        (res, packet)
    }
}

pub struct LenPacket {
    pub inner: Vec<u8>,
}

impl LenPacket {
    pub fn new(ty: PacketTy, id: PacketId, cap: usize) -> Self {
        let mut inner = Vec::with_capacity(cap + 16);
        inner.extend_from_slice(&(PACKET_HEADER_LEN as u64).to_le_bytes());
        inner.push(ty as u8);
        inner.extend_from_slice(&id);
        inner.extend_from_slice(&[0; 4]);
        Self { inner }
    }

    pub fn msg(id: PacketId, cap: usize) -> Self {
        Self::new(PacketTy::Msg, id, cap)
    }

    pub fn table(id: PacketId, cap: usize) -> Self {
        Self::new(PacketTy::Table, id, cap)
    }

    pub fn time_series(id: PacketId, cap: usize) -> Self {
        Self::new(PacketTy::TimeSeries, id, cap)
    }

    fn pkt_len(&self) -> u64 {
        let len = &self.inner[..8];
        u64::from_le_bytes(len.try_into().expect("len wrong size"))
    }

    pub fn push(&mut self, elem: u8) {
        let len = self.pkt_len() + 1;
        self.inner.push(elem);
        self.inner[..8].copy_from_slice(&len.to_le_bytes());
    }

    pub fn extend_from_slice(&mut self, buf: &[u8]) {
        let len = self.pkt_len() + buf.len() as u64;
        self.inner.extend_from_slice(buf);
        self.inner[..8].copy_from_slice(&len.to_le_bytes());
    }

    pub fn as_packet(&self) -> &impeller2::types::Packet {
        let len = self.pkt_len() as usize;
        impeller2::types::Packet::try_ref_from_bytes_with_elems(&self.inner[8..], len)
            .expect("len packet was not a valid `Packet`")
    }

    pub fn as_mut_packet(&mut self) -> &mut impeller2::types::Packet {
        let len = self.pkt_len() as usize;
        impeller2::types::Packet::try_mut_from_bytes_with_elems(&mut self.inner[8..], len)
            .expect("len packet was not a valid `Packet`")
    }

    pub fn clear(&mut self) {
        self.inner[..8].copy_from_slice(&(PACKET_HEADER_LEN as u64).to_le_bytes());
        self.inner.truncate(PACKET_HEADER_LEN + 8);
    }
}

unsafe impl IoBuf for LenPacket {
    fn stable_init_ptr(&self) -> *const u8 {
        self.inner.stable_init_ptr()
    }

    fn init_len(&self) -> usize {
        self.inner.init_len()
    }

    fn total_len(&self) -> usize {
        self.inner.total_len()
    }
}

impl postcard::ser_flavors::Flavor for LenPacket {
    type Output = LenPacket;

    fn try_push(&mut self, data: u8) -> postcard::Result<()> {
        self.push(data);
        Ok(())
    }

    fn finalize(self) -> postcard::Result<Self::Output> {
        Ok(self)
    }
}

pub trait Msg: Serialize {
    const ID: PacketId;

    fn to_len_packet(&self) -> LenPacket {
        let msg = LenPacket::msg(Self::ID, 0);
        postcard::serialize_with_flavor(&self, msg).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use super::*;
    use postcard::experimental::max_size::MaxSize;
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
            let Packet::Msg(m) = stream.next(buf).await.unwrap() else {
                panic!("non msg pkt");
            };
            assert_eq!(m.id, Foo::ID);
            let foo: Foo = m.parse().unwrap();
            assert_eq!(foo, Foo { bar: 0xBB });
        })
    }
}
