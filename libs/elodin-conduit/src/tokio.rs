use futures::{stream, Sink, SinkExt, Stream, StreamExt, TryStreamExt};
use std::{io, mem::size_of, net::SocketAddr};
use tokio::net::{
    tcp::{OwnedReadHalf, OwnedWriteHalf},
    TcpStream,
};
use tokio_util::{
    bytes::{Bytes, BytesMut},
    codec::{Framed, FramedRead, FramedWrite, LengthDelimitedCodec},
};

use crate::{
    builder::Builder,
    parser::{ComponentPair, Parser},
    Component, ComponentBatch, ComponentData, EntityId, Error,
};

pub struct Client<T> {
    current_time: u64,
    transport: T,
}

impl<T> Client<T> {
    pub fn new(current_time: u64, transport: T) -> Self {
        Self {
            current_time,
            transport,
        }
    }
}

impl<T> Client<T>
where
    T: Sink<Bytes, Error = io::Error> + Unpin,
{
    pub async fn send(
        &mut self,
        entity_id: impl Into<EntityId>,
        component: impl Component,
    ) -> Result<(), Error> {
        self.send_with_time(self.current_time, entity_id, component)
            .await
    }

    pub async fn send_with_time<C: Component>(
        &mut self,
        time: u64,
        entity_id: impl Into<EntityId>,
        component: C,
    ) -> Result<(), Error> {
        let mut builder = Builder::new(BytesMut::with_capacity(26 + size_of::<C>()), time)?;
        builder.append_component(entity_id, component)?;
        self.send_builder(builder).await?;
        Ok(())
    }

    pub async fn send_data(&mut self, time: u64, data: ComponentData<'_>) -> Result<(), Error> {
        let mut builder = Builder::new(BytesMut::default(), time)?;
        builder.append_data(data)?;
        self.send_builder(builder).await?;
        Ok(())
    }

    pub async fn send_builder(&mut self, builder: Builder<BytesMut>) -> Result<(), Error> {
        self.transport.send(builder.into_buf().freeze()).await?;
        Ok(())
    }
}

impl<T: Stream<Item = Result<BytesMut, std::io::Error>> + Unpin> Client<T> {
    pub async fn recv(&mut self) -> Result<Option<ComponentBatch<'static>>, Error> {
        let Some(res) = self.transport.next().await else {
            return Ok(None);
        };
        let mut parser = Parser::new(res?).ok_or(Error::ParsingError)?;
        parser.parse_data_msg().ok_or(Error::ParsingError).map(Some)
    }

    pub async fn recv_parser(&mut self) -> Result<Option<Parser<Bytes>>, Error> {
        let Some(res) = self.transport.next().await else {
            return Ok(None);
        };
        Parser::new(res?.freeze())
            .ok_or(Error::ParsingError)
            .map(Some)
    }

    pub fn pair_stream(self) -> impl Stream<Item = Result<ComponentPair<'static>, Error>> {
        self.transport
            .map_err(Error::from)
            .and_then(|res| async {
                Parser::new(res)
                    .ok_or(Error::ParsingError)
                    .map(|p| stream::iter(p).map(Ok))
            })
            .try_flatten()
    }
}

pub type TcpClient = Client<Framed<TcpStream, LengthDelimitedCodec>>;

impl TcpClient {
    pub async fn connect(addr: SocketAddr, current_time: u64) -> Result<Self, Error> {
        let socket = TcpStream::connect(addr).await?;
        Ok(Client::new(
            current_time,
            Framed::new(socket, LengthDelimitedCodec::default()),
        ))
    }
}

pub type TcpReader = Client<FramedRead<OwnedReadHalf, LengthDelimitedCodec>>;
impl TcpReader {
    pub fn from_read_half(read_half: OwnedReadHalf, current_time: u64) -> Self {
        Client::new(
            current_time,
            FramedRead::new(read_half, LengthDelimitedCodec::default()),
        )
    }
}
pub type TcpWriter = Client<FramedWrite<OwnedWriteHalf, LengthDelimitedCodec>>;
impl TcpWriter {
    pub fn from_write_half(write_half: OwnedWriteHalf, current_time: u64) -> Self {
        Client::new(
            current_time,
            FramedWrite::new(write_half, LengthDelimitedCodec::default()),
        )
    }
}
