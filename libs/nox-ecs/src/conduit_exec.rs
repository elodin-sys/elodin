use crate::{Compiled, Error, WorldExec};
use bytes::{BufMut, Bytes, BytesMut};
use conduit::{
    client::{Msg, MsgPair},
    query::MetadataStore,
    ColumnPayload, ComponentId, ControlMsg, EntityId, Handle, Packet, Payload, StreamId,
};
use std::mem;
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
    exec: WorldExec<Compiled>,
    playing: bool,
    state: State,
}

impl ConduitExec {
    pub fn new(exec: WorldExec<Compiled>, rx: flume::Receiver<MsgPair>) -> Self {
        let mut metadata_store = MetadataStore::default();
        for (_, metadata) in exec.world.component_map.values() {
            metadata_store.push(metadata.clone());
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

    pub fn run(&mut self) -> Result<(), Error> {
        if self.playing {
            match &mut self.state {
                State::Running => {
                    self.exec.run()?;
                }
                State::Replaying { index } => {
                    *index += 1;
                    if *index >= self.exec.tick() {
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
        let tick = match self.state {
            State::Running => self.exec.tick(),
            State::Replaying { index } => index,
        };
        // drop connections and subscriptions if the connection is closed
        self.connections.retain_mut(|con| {
            con.send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::Tick {
                    tick,
                    max_tick: self.exec.tick(),
                }),
            })
            .inspect_err(|err| {
                tracing::debug!(?err, "send tick error, dropping connection");
            })
            .is_ok()
        });
        self.subscriptions.retain_mut(|sub| {
            send_sub(&self.exec, sub, self.state, &[])
                .inspect_err(|err| {
                    tracing::debug!(?err, "send sub error, dropping connection");
                })
                .is_ok()
        });
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
                entity_ids: self.exec.world.entity_ids(),
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
                let id = query.component_id;
                if !query.with_component_ids.is_empty() {
                    return Err(Error::InvalidQuery); // For now we only support ids with len 1
                }
                if !query.entity_ids.is_empty() {
                    return Err(Error::InvalidQuery); // for nowe we don't support
                };
                let stream_id = StreamId::rand();
                let Some(metadata) = self.metadata_store.get_metadata(&id) else {
                    warn!(?id, "component not found");
                    return Err(Error::ComponentNotFound);
                };
                tx.send(Packet {
                    stream_id: StreamId::CONTROL,
                    payload: Payload::ControlMsg(ControlMsg::OpenStream {
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
            Msg::Control(ControlMsg::Rewind(index)) => self.state = State::Replaying { index },
            Msg::Control(ControlMsg::Query { time_range, query }) => {
                let time_range = time_range.start as usize
                    ..(time_range.end as usize).min(self.exec.world.history.len());
                if !query.with_component_ids.is_empty() {
                    return Err(Error::InvalidQuery); // For now we only support ids with len 1
                }
                let stream_id = StreamId::rand();
                let Some(metadata) = self.metadata_store.get_metadata(&query.component_id) else {
                    warn!(?query.component_id, "component not found");
                    return Err(Error::ComponentNotFound);
                };
                tx.send(Packet {
                    stream_id: StreamId::CONTROL,
                    payload: Payload::ControlMsg(ControlMsg::OpenStream {
                        stream_id,
                        metadata: metadata.clone(),
                    }),
                })
                .map_err(|_| Error::ChannelClosed)?;
                let mut sub = Subscription {
                    component_id: query.component_id,
                    stream_id,
                    connection: tx.clone(),
                    sent_generation: usize::MAX,
                };
                for index in time_range {
                    send_sub(
                        &self.exec,
                        &mut sub,
                        State::Replaying {
                            index: index as u64,
                        },
                        &query.entity_ids,
                    )?;
                }
            }
            Msg::Control(_) => {}
            Msg::Column(new_col) => {
                // NOTE: the entity ids in `new_col` can be a subset of the ones in `col`,
                // but the order must be the same
                let mut col_ref = self
                    .exec
                    .world
                    .column_by_id_mut(new_col.metadata.component_id())
                    .ok_or(Error::ComponentNotFound)?;
                let mut col = col_ref.iter();
                let updates = new_col
                    .iter()
                    .filter_map(|res| {
                        let value = res
                            .inspect_err(|err| {
                                tracing::warn!(?err, "error processing column value")
                            })
                            .ok()?;
                        // `col` is only ever scanned once because the iterator state is preserved across calls to `position`
                        let offset = col.position(|(entity_id, _)| entity_id == value.entity_id)?;
                        Some((offset, value.value))
                    })
                    .collect::<Vec<_>>();
                drop(col);
                for (offset, value) in updates {
                    if let Err(err) = col_ref.update(offset, value) {
                        tracing::warn!(?err, "error processing column value");
                    }
                }
            }
        }
        Ok(())
    }
}

fn send_sub(
    exec: &WorldExec<Compiled>,
    sub: &mut Subscription,
    state: State,
    entity_ids: &[EntityId],
) -> Result<(), Error> {
    let comp_id = sub.component_id;
    let (time, col) = match state {
        State::Running => (
            exec.tick(),
            exec.world
                .column_by_id(comp_id)
                .ok_or(Error::ComponentNotFound)?,
        ),
        State::Replaying { index } => (
            index,
            exec.world
                .column_at_tick(comp_id, index)
                .ok_or(Error::ComponentNotFound)?,
        ),
    };

    if col.metadata.asset {
        let mut changed = false;
        for (_, id) in col.typed_iter::<u64>() {
            let gen = exec
                .world
                .assets
                .gen(Handle::<()>::new(id))
                .ok_or(Error::AssetNotFound)?;
            if gen > sub.sent_generation {
                changed = true;
                sub.sent_generation = gen;
            }
        }
        if !changed {
            return Ok(());
        }
        for (entity_id, id) in col.typed_iter::<u64>() {
            let Some(value) = exec.world.assets.value(Handle::<()>::new(id)) else {
                todo!("gracefully handle")
            };
            let packet = Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::Asset {
                    entity_id,
                    bytes: value.inner.clone(),
                    id: value.asset_id,
                    asset_index: id,
                }),
            };
            sub.connection
                .send(packet)
                .map_err(|_| Error::ChannelClosed)?;
        }
    } else {
        let packet = if entity_ids.is_empty() {
            Packet {
                stream_id: sub.stream_id,
                payload: Payload::Column(ColumnPayload {
                    time,
                    len: col.len() as u32,
                    entity_buf: col.entities.clone().into(),
                    value_buf: col.column.clone().into(),
                }),
            }
        } else {
            let col_entity_ids: &[EntityId] = bytemuck::cast_slice(col.entities);
            let mut entity_iter = col_entity_ids.iter();
            let mut entity_buf = BytesMut::with_capacity(mem::size_of::<u64>() * entity_ids.len());
            let comp_size = col.metadata.component_type.size();
            let mut value_buf = BytesMut::with_capacity(comp_size * entity_ids.len());
            let mut len: usize = 0;
            for id in entity_ids {
                let Some(index) = entity_iter.position(|entity_id| *entity_id == *id) else {
                    continue;
                };
                len += 1;
                entity_buf.put_u64_le(id.0);
                value_buf
                    .extend_from_slice(&col.column[index * comp_size..(index + 1) * comp_size]);
            }
            Packet {
                stream_id: sub.stream_id,
                payload: Payload::Column(ColumnPayload {
                    time,
                    len: len as u32,
                    entity_buf: entity_buf.freeze(),
                    value_buf: value_buf.freeze(),
                }),
            }
        };
        sub.connection
            .send(packet)
            .map_err(|_| Error::ChannelClosed)?;
    }
    Ok(())
}

#[derive(Default, Copy, Clone)]
enum State {
    #[default]
    Running,
    Replaying {
        index: u64,
    },
}

#[cfg(feature = "tokio")]
pub fn spawn_tcp_server(
    socket_addr: std::net::SocketAddr,
    exec: WorldExec,
    client: nox::Client,
    check_canceled: impl Fn() -> bool,
) -> Result<(), Error> {
    use std::time::Instant;

    use conduit::server::TcpServer;

    let time_step = exec.time_step();
    let (tx, rx) = flume::unbounded();
    let exec = exec.compile(client)?;
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
        conduit_exec.run()?;
        if check_canceled() {
            break Ok(());
        }
        let sleep_time = time_step.saturating_sub(start.elapsed());
        if !sleep_time.is_zero() {
            std::thread::sleep(sleep_time)
        }
    }
}
