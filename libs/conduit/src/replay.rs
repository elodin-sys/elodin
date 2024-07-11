use bytes::{BufMut, Bytes, BytesMut};
use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use tracing::warn;

use crate::client::{ColumnMsg, MsgPair};
use crate::{
    client::Msg, query::MetadataStore, world::World, ColumnPayload, ComponentId, ControlMsg,
    EntityId, Error, Handle, Metadata, Packet, Payload, Query, StreamId,
};

#[derive(Debug, Clone)]
pub struct Connection {
    pub tx: flume::Sender<Packet<Payload<Bytes>>>,
    pub state: ConnectionState,
    pub playing: bool,
}

impl Connection {
    pub fn new(tx: flume::Sender<Packet<Payload<Bytes>>>) -> Self {
        Self {
            tx,
            state: Default::default(),
            playing: true,
        }
    }

    pub fn tick(&mut self, world: &World) -> Option<u64> {
        if self.playing {
            Some(self.state.tick(world.tick))
        } else {
            self.state.load_tick()
        }
    }

    pub fn load_tick(&self, world: &World) -> Option<u64> {
        Some(self.state.load_tick().unwrap_or(world.tick))
    }

    pub fn send(
        &self,
        msg: Packet<Payload<Bytes>>,
    ) -> Result<(), flume::SendError<Packet<Payload<Bytes>>>> {
        self.tx.send(msg)
    }
}

// #[derive(Debug, Clone)]
// pub enum ConnectionState {
//     Replaying { index: u64 },
//     Live,
// }

#[derive(Debug, Clone)]
pub struct ConnectionState(pub Arc<AtomicU64>);
impl Default for ConnectionState {
    fn default() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }
}

impl ConnectionState {
    const LIVE: u64 = u64::MAX;

    fn load_tick(&self) -> Option<u64> {
        let tick = self.0.load(std::sync::atomic::Ordering::SeqCst);
        if tick == Self::LIVE {
            None
        } else {
            Some(tick)
        }
    }

    fn tick(&self, world_tick: u64) -> u64 {
        let mut tick = self.0.load(std::sync::atomic::Ordering::SeqCst);
        if tick == Self::LIVE {
            world_tick
        } else {
            loop {
                let new_tick = (tick + 1).min(world_tick);
                match self.0.compare_exchange_weak(
                    tick,
                    new_tick,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                ) {
                    Err(val) => {
                        if val == Self::LIVE {
                            return world_tick;
                        }
                        tick = val;
                    }
                    Ok(_) => return new_tick,
                }
            }
        }
    }
}

#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
#[derive(Debug, Clone)]
pub struct Replay {
    sub_manager: SubscriptionManager,
    proxy: Proxy,
    in_rx: flume::Receiver<Packet<Payload<Bytes>>>,
    connection: Connection,
    world: World,
}

#[derive(Debug, Clone)]
pub struct Subscription {
    component_id: ComponentId,
    stream_id: StreamId,
    pub connection: Connection,
    sent_generation: usize,
}

#[derive(Debug, Clone)]
pub struct SubscriptionManager {
    pub subscriptions: Vec<Subscription>,
    pub metadata_store: MetadataStore,
}

#[derive(Debug, Clone)]
struct Proxy {
    streams: HashMap<StreamId, Arc<Metadata>>,
    out_rx: flume::Receiver<Packet<Payload<Bytes>>>,
    mp_tx: flume::Sender<MsgPair>,
    in_tx: flume::Sender<Packet<Payload<Bytes>>>,
}

#[cfg(feature = "bevy")]
pub fn serve_replay(mut replay: bevy::prelude::ResMut<Replay>) {
    if let Err(err) = replay.run() {
        warn!(?err, "error serving replay data");
    }
}

impl Proxy {
    fn flush(&mut self) -> Result<(), Error> {
        while let Ok(packet) = self.out_rx.try_recv() {
            let msg = match packet.payload {
                Payload::ControlMsg(ControlMsg::OpenStream {
                    stream_id,
                    metadata,
                }) => {
                    self.streams.insert(stream_id, Arc::new(metadata.clone()));
                    Msg::Control(ControlMsg::OpenStream {
                        stream_id,
                        metadata,
                    })
                }
                Payload::ControlMsg(msg) => Msg::Control(msg),
                Payload::Column(msg) => {
                    let metadata = self
                        .streams
                        .get(&packet.stream_id)
                        .ok_or(Error::StreamNotFound(packet.stream_id))?;
                    Msg::Column(ColumnMsg {
                        metadata: metadata.clone(),
                        payload: msg,
                    })
                }
            };
            self.mp_tx.send(MsgPair {
                msg,
                tx: Some(self.in_tx.downgrade()),
            })?;
        }
        Ok(())
    }
}

impl Replay {
    pub fn new(world: World, mp_tx: flume::Sender<MsgPair>) -> Self {
        let mut metadata_store = MetadataStore::default();
        for (_, metadata) in world.component_map.values() {
            metadata_store.push(metadata.clone());
        }

        let (in_tx, in_rx) = flume::unbounded();
        let (out_tx, out_rx) = flume::unbounded();
        out_tx
            .send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::StartSim {
                    metadata_store: metadata_store.clone(),
                    time_step: world.sim_time_step.0,
                    entity_ids: world.entity_ids(),
                }),
            })
            .unwrap();

        let proxy = Proxy {
            streams: HashMap::new(),
            out_rx,
            mp_tx,
            in_tx,
        };

        Self {
            sub_manager: SubscriptionManager::new(metadata_store),
            proxy,
            in_rx,
            connection: Connection::new(out_tx.clone()),
            world,
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        while let Ok(packet) = self.in_rx.try_recv() {
            match packet.payload {
                Payload::ControlMsg(msg) => self.process_msg(msg)?,
                Payload::Column(_) => {}
            }
        }

        let Some(tick) = self.connection.tick(&self.world) else {
            return Ok(());
        };
        if let Err(err) = self.connection.send(Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::Tick {
                tick,
                max_tick: self.world.tick,
                simulating: false,
            }),
        }) {
            tracing::debug!(?err, "send tick error, dropping connection");
        }
        self.sub_manager.send(&self.world);
        self.proxy.flush()?;
        Ok(())
    }

    fn process_msg(&mut self, msg: ControlMsg) -> Result<(), Error> {
        match msg {
            ControlMsg::Subscribe { query } => {
                self.sub_manager.subscribe(query, self.connection.clone())?;
            }
            ControlMsg::SetPlaying(playing) => {
                self.connection.playing = playing;
            }
            ControlMsg::Rewind(index) => {
                self.connection.state.0.store(index, Ordering::SeqCst);
            }
            ControlMsg::Query { time_range, query } => {
                self.sub_manager
                    .query(time_range, query, &self.world, self.connection.clone())?;
            }
            _ => {}
        }
        Ok(())
    }
}

impl SubscriptionManager {
    pub fn new(metadata_store: MetadataStore) -> Self {
        Self {
            subscriptions: Vec::new(),
            metadata_store,
        }
    }

    pub fn send(&mut self, world: &World) {
        self.subscriptions.retain_mut(|sub| {
            let Some(tick) = sub.connection.load_tick(world) else {
                return true;
            };
            send_sub(world, sub, tick, &[])
                .inspect_err(|err| {
                    tracing::debug!(?err, "send sub error, dropping connection");
                })
                .is_ok()
        });
    }

    pub fn subscribe(&mut self, query: Query, connection: Connection) -> Result<(), Error> {
        let id = query.component_id;
        if !query.with_component_ids.is_empty() {
            return Err(Error::InvalidQuery); // For now we only support ids with len 1
        }
        if !query.entity_ids.is_empty() {
            return Err(Error::InvalidQuery); // for now, we we don't support
        };
        let stream_id = StreamId::rand();
        let Some(metadata) = self.metadata_store.get_metadata(&id) else {
            warn!(?id, "component not found");
            return Err(Error::ComponentNotFound);
        };
        connection
            .send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::OpenStream {
                    stream_id,
                    metadata: metadata.clone(),
                }),
            })
            .map_err(|_| Error::ConnectionClosed)?;
        self.subscriptions.push(Subscription {
            component_id: id,
            connection,
            sent_generation: 0,
            stream_id,
        });
        Ok(())
    }

    pub fn query(
        &mut self,
        time_range: Range<u64>,
        query: Query,
        world: &World,
        connection: Connection,
    ) -> Result<(), Error> {
        let time_range = time_range.start..(time_range.end).min(world.tick);
        if !query.with_component_ids.is_empty() {
            return Err(Error::InvalidQuery); // For now we only support ids with len 1
        }
        let stream_id = StreamId::rand();
        let Some(metadata) = self.metadata_store.get_metadata(&query.component_id) else {
            warn!(?query.component_id, "component not found");
            return Err(Error::ComponentNotFound);
        };
        connection
            .send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::OpenStream {
                    stream_id,
                    metadata: metadata.clone(),
                }),
            })
            .map_err(|_| Error::ConnectionClosed)?;
        let mut sub = Subscription {
            component_id: query.component_id,
            stream_id,
            connection,
            sent_generation: usize::MAX,
        };
        for index in time_range {
            send_sub(world, &mut sub, index, &query.entity_ids)?;
        }
        Ok(())
    }
}

fn send_sub(
    world: &World,
    sub: &mut Subscription,
    tick: u64,
    entity_ids: &[EntityId],
) -> Result<(), Error> {
    let col = world
        .column_at_tick(sub.component_id, tick)
        .ok_or(Error::ComponentNotFound)?;
    if col.metadata.asset {
        let mut changed = false;
        for (_, id) in col.typed_iter::<u64>() {
            let gen = world
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
            let Some(value) = world.assets.value(Handle::<()>::new(id)) else {
                todo!("gracefully handle")
            };
            let packet = Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(ControlMsg::Asset {
                    entity_id,
                    bytes: value.inner.clone(),
                    component_id: sub.component_id,
                    asset_index: id,
                }),
            };
            sub.connection
                .send(packet)
                .map_err(|_| Error::ConnectionClosed)?;
        }
    } else {
        let packet = if entity_ids.is_empty() {
            Packet {
                stream_id: sub.stream_id,
                payload: Payload::Column(ColumnPayload {
                    time: tick,
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
                    time: tick,
                    len: len as u32,
                    entity_buf: entity_buf.freeze(),
                    value_buf: value_buf.freeze(),
                }),
            }
        };
        sub.connection
            .send(packet)
            .map_err(|_| Error::ConnectionClosed)?;
    }
    Ok(())
}
