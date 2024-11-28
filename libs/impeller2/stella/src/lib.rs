use std::sync::Arc;

use impeller2::{
    com_de::Decomponentize,
    registry::VTableRegistry,
    types::{self, PacketId, PacketTy},
};
use serde::{Deserialize, Serialize};
use stellarator::{
    buf::{IoBuf, IoBufMut, Slice},
    io::{AsyncRead, AsyncWrite, LengthDelReader},
    BufResult,
};
use zerocopy::{Immutable, KnownLayout, TryFromBytes, Unaligned};

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
        let buf = packet_buf.try_sub_slice(4..).ok_or(Error::InvalidPacket)?;
        Ok(match packet_ty {
            PacketTy::Msg => Packet::Msg(MsgBuf { id, buf }),
            PacketTy::Table => Packet::Table(Table { id, buf }),
        })
    }
}

pub struct Table<B: IoBuf> {
    pub id: [u8; 3],
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
    pub id: [u8; 3],
    pub buf: Slice<B>,
}

impl<B: IoBuf> MsgBuf<B> {
    pub fn parse<'a, T: Deserialize<'a> + 'a>(&'a self) -> Result<T, Error> {
        let msg = postcard::from_bytes(stellarator::buf::deref(&self.buf))?;
        Ok(msg)
    }
}

pub enum Packet<B: IoBuf> {
    Msg(MsgBuf<B>),
    Table(Table<B>),
}

pub struct PacketSink<W: AsyncWrite> {
    writer: W,
}

impl<W: AsyncWrite> PacketSink<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub async fn send<P: AsLenPacket + FromLenPacket + ?Sized>(
        &self,
        packet: Arc<P>,
    ) -> BufResult<(), Arc<P>> {
        let packet = ArcLenPacket(packet.as_arc_packet());
        let (res, packet) = self.writer.write_all(packet).await;
        (res, P::from_arc_packet(packet.0))
    }
}

pub trait AsLenPacket {
    fn as_packet(&self) -> &'_ LenPacket;
    fn as_arc_packet(self: Arc<Self>) -> Arc<LenPacket>;
    fn as_box_packet(self: Box<Self>) -> Box<LenPacket>;
}

pub trait FromLenPacket {
    fn from_arc_packet(pkt: Arc<LenPacket>) -> Arc<Self>;
    fn from_box_packet(pkt: Box<LenPacket>) -> Box<Self>;
}

impl AsLenPacket for LenPacket {
    fn as_packet(&self) -> &'_ LenPacket {
        self
    }

    fn as_arc_packet(self: Arc<Self>) -> Arc<LenPacket> {
        self
    }

    fn as_box_packet(self: Box<Self>) -> Box<LenPacket> {
        self
    }
}

impl FromLenPacket for LenPacket {
    fn from_arc_packet(pkt: Arc<LenPacket>) -> Arc<Self> {
        pkt
    }

    fn from_box_packet(pkt: Box<LenPacket>) -> Box<Self> {
        pkt
    }
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout)]
#[repr(C)]
pub struct LenPacket {
    pub length: [u8; 8],
    pub packet: types::Packet,
}

#[repr(transparent)]
struct ArcLenPacket(Arc<LenPacket>);

unsafe impl IoBuf for ArcLenPacket {
    fn stable_init_ptr(&self) -> *const u8 {
        Arc::as_ptr(&self.0) as *const _
    }

    fn init_len(&self) -> usize {
        let len = u64::from_le_bytes(self.0.length) as usize;
        len + 4 + 8
    }

    fn total_len(&self) -> usize {
        let len = u64::from_le_bytes(self.0.length) as usize;
        len + 4 + 8
    }
}

pub trait Msg: Serialize + postcard::experimental::max_size::MaxSize {
    fn id(&self) -> [u8; 3];

    fn to_arc_len_packet(&self) -> Arc<LenPacket> {
        let mut buf = Arc::new_uninit_slice(Self::POSTCARD_MAX_SIZE + 4 + 8);
        let data = Arc::get_mut(&mut buf).expect("arc was cloned");
        for x in data {
            x.write(0);
        }
        let mut buf = unsafe { buf.assume_init() };
        let data = Arc::get_mut(&mut buf).expect("arc was cloned");
        const LEN_SIZE: usize = size_of::<u64>();
        data[LEN_SIZE] = PacketTy::Msg as u8;
        const PKT_HEADER_SIZE: usize = size_of::<PacketTy>() + size_of::<PacketId>();
        data[LEN_SIZE + size_of::<PacketTy>()..LEN_SIZE + PKT_HEADER_SIZE]
            .copy_from_slice(&self.id());
        let len = postcard::to_slice(self, &mut data[(PKT_HEADER_SIZE + LEN_SIZE)..])
            .expect("postcarf failed")
            .len() as u64
            + 4;
        data[..8].copy_from_slice(&len.to_le_bytes());
        unsafe { Arc::from_raw(Arc::into_raw(buf) as *const _) }
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
        fn id(&self) -> [u8; 3] {
            [0x1, 0x2, 0x3]
        }
    }

    #[test]
    fn test_packet_echo() {
        stellarator::test!(async {
            let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).unwrap();
            let addr = listener.local_addr().unwrap();
            stellarator::spawn(async move {
                let sink = PacketSink::new(listener.accept().await.unwrap());
                let msg = Foo { bar: 0xBB }.to_arc_len_packet();
                sink.send(msg).await.0.unwrap();
            });
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut stream = PacketStream::new(stream);
            let buf = vec![0; 128];
            let Packet::Msg(m) = stream.next(buf).await.unwrap() else {
                panic!("non msg pkt");
            };
            assert_eq!(m.id, [1, 2, 3]);
            let foo: Foo = m.parse().unwrap();
            assert_eq!(foo, Foo { bar: 0xBB });
        })
    }
}
