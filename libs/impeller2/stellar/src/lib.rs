use std::{
    marker::PhantomData,
    net::SocketAddr,
    ops::{Deref, DerefMut},
};

use impeller2::types::{
    IntoLenPacket, LenPacket, Msg, OwnedPacket, Request, RequestId, TryFromPacket,
};
use impeller2_wkt::ErrorResponse;
use stellarator::{
    BufResult,
    buf::{IoBufMut, Slice},
    io::{AsyncRead, AsyncWrite, GrowableBuf, LengthDelReader, OwnedReader, OwnedWriter, SplitExt},
    net::TcpStream,
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

    pub async fn next_grow<B: IoBufMut + GrowableBuf>(
        &mut self,
        buf: B,
    ) -> Result<OwnedPacket<Slice<B>>, Error> {
        let packet_buf = self.reader.recv_growable(buf).await?;
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

    pub async fn send(&self, packet: impl IntoLenPacket) -> BufResult<(), LenPacket> {
        let packet = packet.into_len_packet();
        let (res, inner) = self.writer.write_all(packet.inner).await;
        (res, LenPacket { inner })
    }
}

pub struct Client {
    resp_buf: Option<Vec<u8>>,
    pub tx: PacketSink<OwnedWriter<TcpStream>>,
    pub rx: PacketStream<OwnedReader<TcpStream>>,
    next_req_id: u8,
}

impl Client {
    pub async fn connect(addr: SocketAddr) -> Result<Self, Error> {
        let stream = TcpStream::connect(addr).await?;
        let (rx, tx) = stream.split();
        let tx = PacketSink::new(tx);
        let rx = PacketStream::new(rx);
        Ok(Client {
            tx,
            rx,
            next_req_id: 0,
            resp_buf: Some(vec![0u8; 256]),
        })
    }

    pub async fn send(&mut self, packet: impl IntoLenPacket) -> BufResult<(), LenPacket> {
        let len_pkt = packet.into_len_packet();
        self.tx.send(len_pkt).await
    }

    pub async fn request<R: Request + IntoLenPacket>(
        &mut self,
        req: R,
    ) -> Result<R::Reply<Slice<Vec<u8>>>, Error> {
        let req_id = self.next_req_id.wrapping_add(1);
        self.send(req.with_request_id(req_id)).await.0?;
        self.recv(req_id).await
    }

    pub async fn recv<O: TryFromPacket<Slice<Vec<u8>>>>(
        &mut self,
        req_id: RequestId,
    ) -> Result<O, Error> {
        loop {
            let buf = self.resp_buf.take().unwrap_or(vec![0u8; 256]);
            let pkt = self.rx.next_grow(buf).await?;
            if pkt.req_id() != req_id {
                println!("skipping msg because of mismatched req_id");
                self.resp_buf = Some(pkt.into_buf().into_inner());
                continue;
            }
            let res = match &pkt {
                OwnedPacket::Msg(m) if m.id == ErrorResponse::ID => {
                    match postcard::from_bytes::<ErrorResponse>(&m.buf) {
                        Ok(e) => Err(Error::Response(e)),
                        Err(e) => Err(Error::Postcard(e)),
                    }
                }
                pkt => O::try_from_packet(pkt).map_err(Error::from),
            };

            self.resp_buf = Some(pkt.into_buf().into_inner());
            return res;
        }
    }

    pub async fn stream<R: impeller2::types::Request + IntoLenPacket>(
        &mut self,
        req: R,
    ) -> Result<SubStream<'_, R::Reply<Slice<Vec<u8>>>>, Error> {
        let req_id = self.next_req_id.wrapping_add(1);
        self.send(req.with_request_id(req_id)).await.0?;
        Ok(SubStream {
            req_id,
            client: self,
            _phantom_data: PhantomData,
        })
    }
}

pub struct SubStream<'a, R> {
    req_id: RequestId,
    client: &'a mut Client,
    _phantom_data: PhantomData<R>,
}

impl<R: TryFromPacket<Slice<Vec<u8>>>> SubStream<'_, R> {
    pub async fn next(&mut self) -> Result<R, Error> {
        self.client.recv(self.req_id).await
    }
}

impl<R> Deref for SubStream<'_, R> {
    type Target = Client;

    fn deref(&self) -> &Self::Target {
        self.client
    }
}

impl<R> DerefMut for SubStream<'_, R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.client
    }
}

pub struct ReplyStream {}

#[derive(thiserror::Error, Debug, miette::Diagnostic)]
pub enum Error {
    #[error("{0}")]
    Impeller(#[from] impeller2::error::Error),
    #[error("{0}")]
    Stellar(#[from] stellarator::Error),
    #[error("postcard: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("invalid packet type")]
    InvalidPacketType,
    #[error("wait error {0}")]
    Wait(stellarator::sync::wait_map::WaitError),
    #[error("rx handle closed")]
    RxHandleClosed,
    #[error("{0}")]
    Response(ErrorResponse),
}

#[cfg(test)]
mod tests;
