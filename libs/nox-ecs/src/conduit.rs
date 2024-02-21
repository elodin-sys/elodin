use crate::{assets::Handle, Error, WorldExec};
use bytes::Bytes;
use elodin_conduit::{
    client::{Msg, MsgPair},
    query::{MetadataStore, QueryId},
    ser_de::ColumnValue,
    AssetId, ColumnPayload, ComponentId, ControlMsg, EntityId, Metadata, Packet, Payload, StreamId,
};
use tracing::warn;

use std::collections::{BTreeMap, HashMap};

struct ConnectionId(usize);

struct Subscription {
    stream_id: StreamId,
    connection_id: ConnectionId,
    sent_generation: usize,
}

pub struct ConduitExec {
    subscriptions: BTreeMap<ComponentId, Vec<Subscription>>,
    connections: Vec<flume::Sender<Packet<Payload<Bytes>>>>,
    metadata_store: MetadataStore,
    rx: flume::Receiver<MsgPair>,
    exec: WorldExec,
}

impl ConduitExec {
    pub fn new(exec: WorldExec) -> (Self, flume::Sender<MsgPair>) {
        let mut metadata_store = MetadataStore::default();
        for arch in exec.world.host.archetypes.values() {
            for (id, col) in &arch.columns {
                let metadata = Metadata {
                    component_id: *id,
                    component_type: col.buffer.component_type.clone(),
                    tags: HashMap::new(),
                };
                metadata_store.push(metadata);
            }
        }
        let (tx, rx) = flume::unbounded();
        (
            Self {
                subscriptions: BTreeMap::new(),
                connections: Vec::new(),
                rx,
                exec,
                metadata_store,
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
        let Ok(pair) = self.rx.try_recv() else {
            return Ok(());
        };
        if let Err(err) = self.process_msg_pair(pair) {
            tracing::warn!(?err, "error processing msg pair");
        }

        Ok(())
    }

    fn process_msg_pair(&mut self, MsgPair { msg, tx }: MsgPair) -> Result<(), Error> {
        match msg {
            Msg::Control(ControlMsg::Subscribe { query }) => {
                let ids = query.execute(&self.metadata_store);
                if ids.len() != 1 {
                    return Err(Error::InvalidQuery); // For now we only support ids with len 1
                }
                let QueryId::Component(id) = ids[0] else {
                    return Err(Error::InvalidQuery); // For now we only support ids with len 1
                };
                let connection_id = ConnectionId(self.connections.len());
                let tx = tx.upgrade().ok_or(Error::ChannelClosed)?;
                self.connections.push(tx.clone());
                let subs = self.subscriptions.entry(id).or_default();
                let stream_id = StreamId::rand();
                let Some(metadata) = self.metadata_store.get_metadata(&id) else {
                    warn!(?id, "component not found");
                    return Err(Error::ComponentNotFound);
                };
                tx.send(Packet {
                    stream_id: StreamId::CONTROL,
                    payload: Payload::ControlMsg(ControlMsg::Metadata {
                        stream_id,
                        metadata: metadata.clone(),
                    }),
                })
                .map_err(|_| Error::ChannelClosed)?;
                subs.push(Subscription {
                    connection_id,
                    sent_generation: 0,
                    stream_id,
                });
            }
            Msg::Control(_) => {}
            Msg::Column(col) => {
                for res in col.iter() {
                    let Ok(value) = res else {
                        tracing::warn!("error processing column value");
                        continue;
                    };
                    if let Err(err) = self.process_column_value(&col.metadata, value) {
                        tracing::warn!(?err, "error processing column value");
                    }
                }
            }
        }
        Ok(())
    }

    fn process_column_value(
        &mut self,
        metadata: &Metadata,
        column_value: ColumnValue<'_>,
    ) -> Result<(), Error> {
        let mut col = self.exec.column_mut(metadata.component_id)?;
        let Some(out) = col.entity_buf(column_value.entity_id) else {
            return Err(Error::EntityNotFound);
        };
        if let Some(bytes) = column_value.value.bytes() {
            if bytes.len() != out.len() {
                return Err(Error::ValueSizeMismatch);
            }
            out.copy_from_slice(bytes);
        }
        Ok(())
    }
}

fn send_sub(
    connections: &mut [flume::Sender<Packet<Payload<Bytes>>>],
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
        let mut changed = false;
        for id in buf.iter() {
            let gen = exec
                .world
                .host
                .assets
                .gen(Handle::<()>::new(AssetId(*id)))
                .ok_or(Error::AssetNotFound)?;
            if gen > sub.sent_generation {
                changed = true;
                sub.sent_generation = gen;
            }
        }
        if !changed {
            return Ok(());
        }
        let entities_buf = col.entities.typed_buf::<u64>().unwrap();
        for (id, entity_id) in buf.iter().zip(entities_buf.iter().copied()) {
            let Some(value) = exec
                .world
                .host
                .assets
                .value(Handle::<()>::new(AssetId(*id)))
            else {
                todo!("gracefully handle")
            };
            let packet = Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::Asset {
                    entity_id: EntityId(entity_id),
                    bytes: value.inner.clone(),
                    id: value.asset_id,
                }),
            };
            tx.send(packet).map_err(|_| Error::ChannelClosed)?;
        }
    } else {
        let packet = Packet {
            stream_id: sub.stream_id,
            payload: Payload::Column(ColumnPayload {
                time: 0,
                len: col.column.buffer.len as u32,
                entity_buf: Bytes::copy_from_slice(&col.entities.buf),
                value_buf: Bytes::copy_from_slice(&col.column.buffer.buf), // TODO: make the Vec<u8> here bytes so this is a ref-count
            }),
        };
        tx.send(packet).map_err(|_| Error::ChannelClosed)?;
    }
    Ok(())
}

#[cfg(feature = "tokio")]
pub fn spawn_tcp_server(
    socket_addr: std::net::SocketAddr,
    exec: WorldExec,
    client: &nox::Client,
    tick_period: std::time::Duration,
    check_canceled: impl Fn() -> bool,
) -> Result<(), Error> {
    use std::time::Instant;

    use elodin_conduit::server::TcpServer;

    let (mut conduit_exec, rx) = ConduitExec::new(exec);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let server = TcpServer::bind(rx, socket_addr).await.unwrap();
            server.run().await
        })
        .unwrap();
    });
    loop {
        let start = Instant::now();
        conduit_exec.run(client)?;
        if check_canceled() {
            break Ok(());
        }
        let sleep_time = tick_period.saturating_sub(start.elapsed());
        if !sleep_time.is_zero() {
            std::thread::sleep(sleep_time)
        }
    }
}
