use bytes::{BufMut, Bytes, BytesMut};
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

pub type Connection = flume::Sender<Packet<Payload<Bytes>>>;

#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
#[derive(Debug, Clone)]
pub struct Replay {
    sub_manager: SubscriptionManager,
    proxy: Proxy,
    in_rx: flume::Receiver<Packet<Payload<Bytes>>>,
    out_tx: flume::Sender<Packet<Payload<Bytes>>>,
    world: World,
    tick: u64,
    playing: bool,
}

#[derive(Debug, Clone)]
pub struct Subscription {
    component_id: ComponentId,
    stream_id: StreamId,
    connection: Connection,
    sent_generation: usize,
}

#[derive(Debug, Clone)]
pub struct SubscriptionManager {
    subscriptions: Vec<Subscription>,
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
                    time_step: world.time_step.0,
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
            out_tx,
            world,
            tick: 0,
            playing: true,
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        while let Ok(packet) = self.in_rx.try_recv() {
            match packet.payload {
                Payload::ControlMsg(msg) => self.process_msg(msg)?,
                Payload::Column(_) => {}
            }
        }
        if self.playing {
            self.out_tx
                .send(Packet {
                    stream_id: StreamId::CONTROL,
                    payload: Payload::ControlMsg(ControlMsg::Tick {
                        tick: self.tick,
                        max_tick: self.world.tick,
                    }),
                })
                .map_err(|_| Error::ConnectionClosed)?;
            self.sub_manager.send(&self.world, self.tick);
            self.tick += 1;
        }
        self.proxy.flush()?;
        if self.tick >= self.world.tick {
            self.playing = false;
        }
        Ok(())
    }

    fn process_msg(&mut self, msg: ControlMsg) -> Result<(), Error> {
        let tx = self.out_tx.clone();
        match msg {
            ControlMsg::Subscribe { query } => {
                self.sub_manager.subscribe(query, tx)?;
            }
            ControlMsg::SetPlaying(playing) => self.playing = playing,
            ControlMsg::Rewind(index) => self.tick = index,
            ControlMsg::Query { time_range, query } => {
                self.sub_manager.query(time_range, query, &self.world, tx)?;
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

    pub fn send(&mut self, world: &World, tick: u64) {
        self.subscriptions.retain_mut(|sub| {
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
