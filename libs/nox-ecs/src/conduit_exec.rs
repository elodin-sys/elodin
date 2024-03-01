use crate::{assets::Handle, ColumnRef, ColumnStore, Error, WorldExec};
use bytes::Bytes;
use conduit::{
    client::{Msg, MsgPair},
    query::{MetadataStore, QueryId},
    ser_de::ColumnValue,
    ColumnPayload, ComponentId, ControlMsg, EntityId, Metadata, Packet, Payload, StreamId,
};
use tracing::warn;

type Connection = flume::Sender<Packet<Payload<Bytes>>>;

struct Subscription {
    component_id: ComponentId,
    stream_id: StreamId,
    connection: Connection,
    sent_generation: usize,
}

pub struct ConduitExec {
    subscriptions: Vec<Subscription>,
    connections: Vec<Connection>,
    metadata_store: MetadataStore,
    rx: flume::Receiver<MsgPair>,
    exec: WorldExec,
    playing: bool,
    state: State,
}

impl ConduitExec {
    pub fn new(exec: WorldExec, rx: flume::Receiver<MsgPair>) -> Self {
        let mut metadata_store = MetadataStore::default();
        for arch in exec.world.host.archetypes.values() {
            for col in arch.columns.values() {
                metadata_store.push(col.metadata.clone());
            }
        }
        Self {
            subscriptions: Vec::new(),
            connections: Vec::new(),
            rx,
            exec,
            metadata_store,
            playing: true,
            state: State::default(),
        }
    }

    pub fn time_step(&self) -> std::time::Duration {
        self.exec.time_step()
    }

    pub fn run(&mut self, client: &nox::Client) -> Result<(), Error> {
        if self.playing {
            match &mut self.state {
                State::Running => {
                    self.exec.run(client)?;
                }
                State::Replaying { index } => {
                    *index += 1;
                    if *index >= self.exec.history.worlds.len() {
                        self.state = State::Running;
                    }
                }
            }
        }
        self.send();
        self.recv();
        Ok(())
    }

    pub fn send(&mut self) {
        match self.state {
            State::Running => {
                let max_tick = self.exec.history.worlds.len();
                send(
                    &mut self.subscriptions,
                    &mut self.connections,
                    &mut self.exec,
                    max_tick,
                );
            }
            State::Replaying { index } => {
                let Some(mut polars) = &self.exec.history.worlds.get(index) else {
                    return;
                };
                send(
                    &mut self.subscriptions,
                    &mut self.connections,
                    &mut polars,
                    self.exec.history.worlds.len(),
                );
            }
        }
    }

    pub fn recv(&mut self) {
        while let Ok(pair) = self.rx.try_recv() {
            if let Err(err) = self.process_msg_pair(pair) {
                tracing::warn!(?err, "error processing msg pair");
            }
        }
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn add_connection(&mut self, conn: Connection) -> Result<(), Error> {
        let already_exits = self.connections.iter().any(|c| c.same_channel(&conn));
        if already_exits {
            tracing::debug!("connection already exists");
            return Ok(());
        }
        tracing::debug!("received connect, sending metadata");
        conn.send(Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::StartSim {
                metadata_store: self.metadata_store.clone(),
                time_step: self.exec.time_step(),
            }),
        })?;
        self.connections.push(conn);
        Ok(())
    }

    fn process_msg_pair(&mut self, MsgPair { msg, tx }: MsgPair) -> Result<(), Error> {
        let Some(tx) = tx.upgrade() else {
            tracing::debug!("channel closed");
            return Ok(());
        };
        match msg {
            Msg::Control(ControlMsg::Connect) => self.add_connection(tx)?,
            Msg::Control(ControlMsg::Subscribe { query }) => {
                let ids = query.execute(&self.metadata_store);
                if ids.len() != 1 {
                    return Err(Error::InvalidQuery); // For now we only support ids with len 1
                }
                let QueryId::Component(id) = ids[0] else {
                    return Err(Error::InvalidQuery); // For now we only support ids with len 1
                };
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
                self.subscriptions.push(Subscription {
                    component_id: id,
                    connection: tx,
                    sent_generation: 0,
                    stream_id,
                });
            }
            Msg::Control(ControlMsg::SetPlaying(playing)) => self.playing = playing,
            Msg::Control(ControlMsg::Rewind(index)) => {
                self.state = State::Replaying {
                    index: index as usize,
                }
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

fn send(
    subscriptions: &mut Vec<Subscription>,
    connections: &mut Vec<Connection>,
    exec: &mut impl ColumnStore,
    max_tick: usize,
) {
    // drop connections and subscriptions if the connection is closed
    connections.retain_mut(|con| {
        con.send(Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::Tick {
                tick: exec.tick(),
                max_tick: max_tick as u64,
            }),
        })
        .inspect_err(|err| {
            tracing::debug!(?err, "send tick error, dropping connection");
        })
        .is_ok()
    });
    subscriptions.retain_mut(|sub| {
        send_sub(exec, sub)
            .inspect_err(|err| {
                tracing::debug!(?err, "send sub error, dropping connection");
            })
            .is_ok()
    });
}

fn send_sub(exec: &mut impl ColumnStore, sub: &mut Subscription) -> Result<(), Error> {
    let comp_id = sub.component_id;
    exec.transfer_column(comp_id)?;
    let col = exec.column(comp_id)?;
    if col.is_asset() {
        let Some(assets) = exec.assets() else {
            return Ok(());
        };
        let buf = col.value_buf();
        let Ok(buf) = bytemuck::try_cast_slice(&buf) else {
            // TODO: warn
            todo!()
        };

        let mut changed = false;
        for id in buf.iter() {
            let gen = assets
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
        let entities_buf = col.entity_buf();
        let entities_buf = bytemuck::cast_slice(&entities_buf);
        for (id, entity_id) in buf.iter().zip(entities_buf.iter().copied()) {
            let Some(value) = assets.value(Handle::<()>::new(*id)) else {
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
            sub.connection
                .send(packet)
                .map_err(|_| Error::ChannelClosed)?;
        }
    } else {
        let packet = Packet {
            stream_id: sub.stream_id,
            payload: Payload::Column(ColumnPayload {
                time: exec.tick(),
                len: col.len() as u32,
                entity_buf: Bytes::copy_from_slice(&col.entity_buf()),
                value_buf: Bytes::copy_from_slice(&col.value_buf()), // TODO: make the Vec<u8> here bytes so this is a ref-count
            }),
        };
        sub.connection
            .send(packet)
            .map_err(|_| Error::ChannelClosed)?;
    }
    Ok(())
}

#[derive(Default)]
enum State {
    #[default]
    Running,
    Replaying {
        index: usize,
    },
}

#[cfg(feature = "tokio")]
pub fn spawn_tcp_server(
    socket_addr: std::net::SocketAddr,
    exec: WorldExec,
    client: &nox::Client,
    check_canceled: impl Fn() -> bool,
) -> Result<(), Error> {
    use std::time::Instant;

    use conduit::server::TcpServer;

    let time_step = exec.time_step();
    let (tx, rx) = flume::unbounded();
    let mut conduit_exec = ConduitExec::new(exec, rx);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let server = TcpServer::bind(tx, socket_addr).await.unwrap();
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
        let sleep_time = time_step.saturating_sub(start.elapsed());
        if !sleep_time.is_zero() {
            std::thread::sleep(sleep_time)
        }
    }
}
