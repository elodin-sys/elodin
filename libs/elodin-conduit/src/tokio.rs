use futures::{stream, Sink, SinkExt, Stream, StreamExt, TryStreamExt};
use std::{io, mem::size_of, net::SocketAddr};
use tokio::net::TcpStream;
use tokio_util::{
    bytes::{Bytes, BytesMut},
    codec::{Framed, LengthDelimitedCodec},
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
        self.transport.send(builder.into_buf().freeze()).await?;
        Ok(())
    }

    pub async fn send_data(&mut self, time: u64, data: ComponentData<'_>) -> Result<(), Error> {
        let mut builder = Builder::new(BytesMut::default(), time)?;
        builder.append_data(data)?;
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

impl crate::builder::Extend for BytesMut {
    fn extend_from_slice(&mut self, slice: &[u8]) -> Result<(), Error> {
        self.extend_from_slice(slice);
        Ok(())
    }
}
