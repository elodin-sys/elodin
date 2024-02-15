use crate::{assets::Handle, Error, WorldExec};
use bytes::{Bytes, BytesMut};
use elodin_conduit::{
    bevy_sync::DEFAULT_SUB_FILTERS,
    builder::{encode_varint_usize, Builder},
    parser::{varint_max, ComponentPair, Parser},
    ComponentId, ComponentType, ComponentValue,
};
use nox::{FromBuilder, IntoOp};
use std::collections::BTreeMap;

struct ConnectionId(usize);

struct Subscription {
    connection_id: ConnectionId,
    sent_generation: usize,
}

pub struct ConduitExec {
    subscriptions: BTreeMap<ComponentId, Vec<Subscription>>,
    connections: Vec<Sender>,
    rx: flume::Receiver<RecvMsg>,
    exec: WorldExec,
}

struct Sender(flume::Sender<Builder<BytesMut>>);

impl Sender {
    fn send(
        &mut self,
        component_id: ComponentId,
        component_ty: ComponentType,
        len: usize,
        entities: &[u8],
        values: &[u8],
    ) -> Result<(), Error> {
        use elodin_conduit::builder::ComponentBuilder;
        let capacity = 26 + entities.len() + values.len();
        let mut builder = Builder::new(bytes::BytesMut::with_capacity(capacity), 0).unwrap();
        builder.append_builder(ComponentBuilder::new(
            (component_id, component_ty),
            (len, entities),
            values,
        ))?;
        self.0.send(builder).map_err(|_| Error::ChannelClosed)
    }
}

pub struct RecvMsg {
    parser: Parser<Bytes>,
    tx: flume::WeakSender<Builder<BytesMut>>,
}

impl ConduitExec {
    pub fn new(exec: WorldExec) -> (Self, flume::Sender<RecvMsg>) {
        let (tx, rx) = flume::unbounded();
        (
            Self {
                subscriptions: BTreeMap::new(),
                connections: Vec::new(),
                rx,
                exec,
            },
            tx,
        )
    }

    pub fn run(&mut self, client: &nox::Client) -> Result<(), Error> {
        self.exec.run(client)?;
        self.send()?;
        self.recv()?;
        Ok(())
    }

    pub fn send(&mut self) -> Result<(), Error> {
        for (comp_id, subs) in &mut self.subscriptions {
            for sub in subs.iter_mut() {
                if let Err(err) = send_sub(&mut self.connections, &mut self.exec, comp_id, sub) {
                    tracing::debug!(?err, ?comp_id, "send sub error")
                }
            }
        }
        Ok(())
    }

    pub fn recv(&mut self) -> Result<(), Error> {
        let Ok(RecvMsg { parser, tx }) = self.rx.try_recv() else {
            return Ok(());
        };
        for pair in parser {
            if let Err(err) = self.process_component_pair(&tx, pair) {
                tracing::warn!(?err, "error processing component pair");
            }
        }

        Ok(())
    }

    fn process_component_pair(
        &mut self,
        tx: &flume::WeakSender<Builder<BytesMut>>,
        ComponentPair {
            component_id,
            entity_id,
            value,
        }: ComponentPair,
    ) -> Result<(), Error> {
        if component_id == elodin_conduit::SUB_COMPONENT_ID {
            let ComponentValue::Filter(filter) = value else {
                return Err(Error::InvalidFilter);
            };
            let connection_id = ConnectionId(self.connections.len());
            self.connections
                .push(Sender(tx.upgrade().ok_or(Error::ChannelClosed)?));
            let subs = self
                .subscriptions
                .entry(ComponentId(filter.id))
                .or_default();
            subs.push(Subscription {
                connection_id,
                sent_generation: 0,
            });
        } else {
            let mut col = self.exec.column_mut(component_id)?;
            let Some(out) = col.entity_buf(entity_id) else {
                return Err(Error::EntityNotFound);
            };
            value.with_bytes(|bytes| {
                if bytes.len() != out.len() {
                    return Err(Error::ValueSizeMismatch);
                }
                out.copy_from_slice(bytes);
                Ok(())
            })?;
        }
        Ok(())
    }
}

fn send_sub(
    connections: &mut [Sender],
    exec: &mut WorldExec,
    comp_id: &ComponentId,
    sub: &mut Subscription,
) -> Result<(), Error> {
    let tx = connections
        .get_mut(sub.connection_id.0)
        .ok_or(Error::AssetNotFound)?;
    let _ = exec.column(*comp_id)?;
    let col = exec.cached_column(*comp_id)?;
    if col.column.buffer.asset {
        let Some(buf) = col.column.buffer.typed_buf::<u64>() else {
            // TODO: warn
            todo!()
        };
        let mut out = vec![];
        let mut changed = false;
        for id in buf.iter() {
            let gen = exec
                .world
                .host
                .assets
                .gen(Handle::<()>::new(*id))
                .ok_or(Error::AssetNotFound)?;
            if gen > sub.sent_generation {
                changed = true;
                sub.sent_generation = gen;
            }
        }
        if !changed {
            return Ok(());
        }
        for id in buf.iter() {
            let Some(value) = exec.world.host.assets.value(Handle::<()>::new(*id)) else {
                todo!("gracefully handle")
            };
            value.with_bytes(|bytes| {
                let mut arr = [0; varint_max::<usize>()];
                out.extend_from_slice(encode_varint_usize(bytes.len(), &mut arr));
                out.extend_from_slice(bytes)
            });
        }

        tx.send(
            *comp_id,
            ComponentType::Bytes,
            col.column.buffer.len,
            &col.entities.buf,
            &out,
        )?;
    } else {
        tx.send(
            *comp_id,
            col.column.buffer.component_type,
            col.column.buffer.len,
            &col.entities.buf,
            &col.column.buffer.buf,
        )?;
    }
    Ok(())
}

#[cfg(feature = "tokio")]
pub struct TokioServer {
    tx: flume::Sender<RecvMsg>,
    listener: tokio::net::TcpListener,
}

#[cfg(feature = "tokio")]
impl TokioServer {
    pub async fn bind(
        tx: flume::Sender<RecvMsg>,
        addr: std::net::SocketAddr,
    ) -> Result<Self, Error> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        Ok(Self { tx, listener })
    }

    pub async fn run(self) -> Result<(), Error> {
        loop {
            let (socket, _) = self.listener.accept().await?;
            let (rx_socket, tx_socket) = socket.into_split();
            let (tx, rx) = flume::unbounded();
            let tx_recv_msg = self.tx.clone();
            tokio::spawn(async move {
                for filter in DEFAULT_SUB_FILTERS {
                    let default_subs = Builder::filters(&[*filter]);
                    let default_subs = default_subs.into_buf();
                    let default_subs = Parser::new(default_subs.freeze()).unwrap();
                    tx_recv_msg
                        .send_async(RecvMsg {
                            parser: default_subs,
                            tx: tx.downgrade(),
                        })
                        .await
                        .map_err(|_| Error::ChannelClosed)?;
                }
                let mut rx_client = elodin_conduit::tokio::TcpReader::from_read_half(rx_socket, 0);
                loop {
                    let parser = match rx_client.recv_parser().await {
                        Ok(Some(parser)) => parser,
                        Ok(None) => {
                            continue;
                        }
                        Err(elodin_conduit::Error::ParsingError) => {
                            continue;
                        }
                        Err(err) => return Err::<(), Error>(Error::from(err)),
                    };
                    tx_recv_msg
                        .send_async(RecvMsg {
                            parser,
                            tx: tx.downgrade(),
                        })
                        .await
                        .map_err(|_| Error::ChannelClosed)?;
                }
            });
            tokio::spawn(async move {
                let mut tx_client = elodin_conduit::tokio::TcpWriter::from_write_half(tx_socket, 0);
                while let Ok(builder) = rx.recv_async().await {
                    tx_client.send_builder(builder).await?;
                }
                Ok::<(), Error>(())
            });
        }
    }
}

#[cfg(feature = "tokio")]
pub fn spawn_tcp_server(
    socket_addr: std::net::SocketAddr,
    exec: WorldExec,
    client: &nox::Client,
    tick_period: std::time::Duration,
) -> Result<(), Error> {
    use std::time::Instant;

    let (mut conduit_exec, rx) = ConduitExec::new(exec);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let server = TokioServer::bind(rx, socket_addr).await.unwrap();
            server.run().await
        })
        .unwrap();
    });
    loop {
        let start = Instant::now();
        conduit_exec.run(client)?;
        let sleep_time = tick_period.saturating_sub(start.elapsed());
        if !sleep_time.is_zero() {
            std::thread::sleep(sleep_time)
        }
    }
}

pub struct WorldPos(pub nox::SpatialTransform<f64>);
impl FromBuilder for WorldPos {
    type Item<'a> = Self;

    fn from_builder(builder: &nox::Builder) -> Self::Item<'_> {
        WorldPos(nox::SpatialTransform::from_builder(builder))
    }
}

impl IntoOp for WorldPos {
    fn into_op(self) -> nox::Noxpr {
        self.0.into_op()
    }
}

impl crate::Component for WorldPos {
    type Inner = nox::SpatialTransform<f64>;

    type HostTy = Self;

    fn host(val: Self::HostTy) -> Self {
        val
    }

    fn component_id() -> ComponentId {
        ComponentId::new("world_pos")
    }

    fn component_type() -> ComponentType {
        ComponentType::SpatialPosF64
    }
}
