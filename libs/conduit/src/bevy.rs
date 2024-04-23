use crate::client::ColumnMsg;
use crate::client::Msg;
use crate::client::MsgPair;
use crate::query::{MetadataStore, QueryId};
use crate::ser_de::ColumnValue;
use crate::well_known::EntityMetadata;
use crate::Asset;
use crate::AssetId;
use crate::ColumnPayload;
use crate::ComponentType;
use crate::Error;
use crate::{
    Component, ComponentId, ComponentValue, ControlMsg, EntityId, Packet, Payload, StreamId,
};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bytes::Bytes;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::time::Duration;
use tracing::warn;

use bevy::{
    app::AppExit,
    ecs::query::Changed,
    prelude::{
        Commands, Deref, DerefMut, Entity, EventWriter, Plugin, Query, Res, ResMut, Resource,
        Update,
    },
};
use big_space::GridCell;

#[derive(bevy::prelude::Component, Debug, Default)]
pub struct ComponentValueMap(pub BTreeMap<ComponentId, ComponentValue<'static>>);

#[derive(bevy::prelude::Component)]
pub struct Received;

#[derive(bevy::prelude::Resource)]
pub struct MaxTick(pub u64);

#[derive(bevy::prelude::Resource)]
pub struct Tick(pub u64);

#[derive(bevy::prelude::Resource)]
pub struct TimeStep(pub Duration);

impl ColumnMsg<Bytes> {
    pub fn load_into_bevy(
        &self,
        entity_map: &mut EntityMap,
        component_map: &ComponentMap,
        commands: &mut Commands,
        value_map: &mut Query<&mut ComponentValueMap>,
    ) {
        let component_id = self.metadata.component_id();

        for res in self.iter() {
            let Ok(ColumnValue { entity_id, value }) = res else {
                warn!("error parsing column value");
                continue;
            };

            let e = if let Some(entity) = entity_map.0.get(&entity_id) {
                let Some(e) = commands.get_entity(*entity) else {
                    return;
                };
                e.id()
            } else {
                let e = commands.spawn((
                    EntityMetadata {
                        name: format!("{:?}", entity_id),
                        color: Color::WHITE.into(),
                    },
                    entity_id,
                    Transform::default(),
                    GridCell::<i128>::default(),
                    ComponentValueMap::default(),
                ));
                entity_map.0.insert(entity_id, e.id());
                e.id()
            };

            if let Ok(mut value_map) = value_map.get_mut(e) {
                value_map.0.insert(component_id, value.clone().into_owned());
            } else {
                warn!("no component value map");
            }

            let Some(adapter) = component_map.0.get(&component_id) else {
                continue;
            };

            adapter.insert(commands, entity_map, entity_id, value);
        }
    }
}

#[derive(Resource, Default, Deref)]
pub struct EntityMap(pub HashMap<EntityId, Entity>);

#[derive(Resource, Default)]
pub struct ComponentMap(pub HashMap<ComponentId, Box<dyn ComponentAdapter + Send + Sync>>);

pub trait ComponentAdapter {
    fn get(&self, world: &World, entity: Entity) -> Option<ComponentValue>;
    fn insert(
        &self,
        commands: &mut Commands,
        map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentValue,
    );
}

struct StaticComponentAdapter<C> {
    _phantom: PhantomData<C>,
}

impl<C> Default for StaticComponentAdapter<C> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<C> ComponentAdapter for StaticComponentAdapter<C>
where
    C: Component + bevy::prelude::Component + std::fmt::Debug,
{
    fn get(&self, world: &World, entity: Entity) -> Option<ComponentValue> {
        Some(world.get_entity(entity)?.get::<C>()?.component_value())
    }

    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentValue,
    ) {
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Some(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            return;
        };
        if let Some(c) = C::from_component_value(value) {
            e.insert((c, entity_id, Received));
        }
    }
}

struct StaticResourceAdapter<C> {
    _phantom: PhantomData<C>,
}

impl<C> Default for StaticResourceAdapter<C> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<C: Resource + Component + std::fmt::Debug> ComponentAdapter for StaticResourceAdapter<C> {
    fn get(&self, world: &World, _entity_id: Entity) -> Option<ComponentValue> {
        Some(world.get_resource::<C>()?.component_value())
    }

    fn insert(
        &self,
        commands: &mut Commands,
        _map: &mut EntityMap,
        _entity_id: EntityId,
        value: ComponentValue,
    ) {
        if let Some(r) = C::from_component_value(value) {
            commands.insert_resource(r)
        }
    }
}

#[derive(Resource, Default)]
pub struct AssetMap(pub HashMap<AssetId, Box<dyn AssetAdapter + Send + Sync>>);

pub trait AssetAdapter {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: &[u8],
    );
}

pub trait AppExt {
    fn add_conduit_component<C>(&mut self) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug;

    fn add_conduit_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter + Send + Sync>,
    ) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug;

    fn add_conduit_asset<R: Asset + bevy::prelude::Component + Debug>(
        &mut self,
        adapter: Box<dyn AssetAdapter + Send + Sync>,
    ) -> &mut Self;
}

impl AppExt for bevy::app::App {
    fn add_conduit_component<C>(&mut self) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug,
    {
        self.add_conduit_component_with_adapter::<C>(Box::<StaticComponentAdapter<C>>::default())
    }

    fn add_conduit_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter + Send + Sync>,
    ) -> &mut Self
    where
        C: Component + bevy::prelude::Component + std::fmt::Debug,
    {
        let mut metadata = self
            .world
            .get_resource_or_insert_with(MetadataStore::default);
        metadata.push(C::metadata());

        let mut map = self
            .world
            .get_resource_or_insert_with(|| ComponentMap(HashMap::default()));

        map.0.insert(C::component_id(), adapter);
        self.add_systems(Update, sync_component::<C>);
        self.add_systems(Update, query_component::<C>);
        self
    }

    fn add_conduit_asset<R: Asset + bevy::prelude::Component + Debug>(
        &mut self,
        adapter: Box<dyn AssetAdapter + Send + Sync>,
    ) -> &mut Self {
        let mut map = self.world.get_resource_or_insert_with(AssetMap::default);
        map.0.insert(R::ASSET_ID, adapter);
        self.add_systems(Update, sync_asset::<R>);
        self.add_systems(Update, query_asset::<R>);
        self
    }
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct ConduitRx(pub flume::Receiver<MsgPair>);

#[derive(Component)]
pub struct Persistent;

#[derive(SystemParam)]
pub struct RecvSystemArgs<'w, 's> {
    rx: Res<'w, ConduitRx>,
    entity_map: ResMut<'w, EntityMap>,
    event: EventWriter<'w, Msg>,
    commands: Commands<'w, 's>,
    value_map: Query<'w, 's, &'static mut ComponentValueMap>,
    metadata_store: ResMut<'w, MetadataStore>,
    time_step_res: ResMut<'w, TimeStep>,
    sim_peer: ResMut<'w, SimPeer>,
    subscriptions: ResMut<'w, Subscriptions>,
    subscribe_event: EventWriter<'w, SubscribeEvent>,
    persistent_query: Query<'w, 's, &'static mut Parent, With<Persistent>>,
    component_map: Res<'w, ComponentMap>,
    children: Query<'w, 's, &'static Children>,
    asset_map: Res<'w, AssetMap>,
    exit: EventWriter<'w, AppExit>,
    max_tick_res: ResMut<'w, MaxTick>,
    tick_res: ResMut<'w, Tick>,
}

fn recv_system(args: RecvSystemArgs) {
    let RecvSystemArgs {
        rx,
        mut entity_map,
        mut event,
        mut commands,
        mut value_map,
        mut metadata_store,
        mut time_step_res,
        mut sim_peer,
        mut subscriptions,
        mut subscribe_event,
        persistent_query,
        component_map,
        children,
        asset_map,
        mut exit,
        mut max_tick_res,
        mut tick_res,
    } = args;

    while let Ok(MsgPair { msg, tx }) = rx.try_recv() {
        let Some(tx) = tx.upgrade() else { continue };
        match &msg {
            Msg::Control(ControlMsg::StartSim {
                metadata_store: new_metadata_store,
                time_step,
                entity_ids,
            }) => {
                tracing::debug!("received startsim, sending subscribe messages");
                *metadata_store = new_metadata_store.clone();
                *time_step_res = TimeStep(*time_step);
                for id in metadata_store.component_index.keys() {
                    let packet = Packet {
                        stream_id: StreamId::CONTROL,
                        payload: Payload::ControlMsg::<Bytes>(ControlMsg::sub_component_id(*id)),
                    };
                    tx.send(packet).unwrap();
                }
                let _ = sim_peer.tx.insert(tx);

                for (_, &entity) in entity_map.0.iter() {
                    if let Ok(children) = children.get(entity) {
                        for child in children.iter() {
                            if persistent_query.get(*child).is_err() {
                                if let Some(entity) = commands.get_entity(*child) {
                                    entity.despawn_recursive();
                                }
                            }
                        }
                    }
                }
                entity_map.0.retain(|id, entity| {
                    if entity_ids.contains(id) {
                        true
                    } else {
                        if let Some(entity) = commands.get_entity(*entity) {
                            entity.despawn_recursive();
                        }
                        false
                    }
                });
                value_map.iter_mut().for_each(|mut map| {
                    map.0.clear();
                });
            }
            Msg::Control(ControlMsg::Subscribe { query }) => {
                let subscription = Subscription {
                    stream_id: StreamId::rand(),
                    tx,
                };
                let ids = query.execute(&metadata_store);
                if ids.len() != 1 {
                    warn!("only single id queries are supported for now");
                    continue;
                }
                let QueryId::Component(id) = ids[0] else {
                    warn!("only single id queries are supported for now");
                    continue;
                };
                subscriptions.subscribe(id, subscription.clone());
                subscribe_event.send(SubscribeEvent {
                    query: query.clone(),
                    subscription,
                });
            }
            Msg::Control(ControlMsg::Asset {
                id,
                bytes,
                entity_id,
            }) => {
                let Some(adapter) = asset_map.0.get(id) else {
                    warn!(?id, "unknown asset type");
                    continue;
                };
                adapter.insert(&mut commands, entity_map.as_mut(), *entity_id, bytes);
            }
            Msg::Control(ControlMsg::Exit) => {
                exit.send(AppExit);
            }
            Msg::Control(ControlMsg::Tick { tick, max_tick }) => {
                max_tick_res.0 = *max_tick;
                tick_res.0 = *tick;
            }
            Msg::Control(_) => {}
            Msg::Column(col) => {
                col.load_into_bevy(
                    entity_map.as_mut(),
                    component_map.as_ref(),
                    &mut commands,
                    &mut value_map,
                );
            }
        }
        event.send(msg);
    }
}

#[derive(Clone)]
pub struct ConduitSubscribePlugin {
    rx: flume::Receiver<MsgPair>,
}

impl ConduitSubscribePlugin {
    pub fn new(rx: flume::Receiver<MsgPair>) -> Self {
        Self { rx }
    }

    pub fn pair() -> (ConduitSubscribePlugin, flume::Sender<MsgPair>) {
        let (tx, rx) = flume::unbounded();
        (ConduitSubscribePlugin { rx }, tx)
    }
}

impl Plugin for ConduitSubscribePlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(EntityMap::default());
        app.insert_resource(ComponentMap::default());
        app.insert_resource(AssetMap::default());
        app.insert_resource(MaxTick(0));
        app.insert_resource(Tick(0));
        app.insert_resource(TimeStep(Duration::default()));
        app.insert_resource(ConduitRx(self.rx.clone()));
        app.add_event::<SubscribeEvent>();
        app.add_event::<ControlMsg>();
        app.add_event::<Msg>();
        app.add_event::<ColumnPayloadMsg>();
        app.add_systems(Update, control_msg);
        app.add_systems(Update, recv_system);
        app.add_systems(PostUpdate, column_payload_msg);
    }
}

#[allow(clippy::type_complexity)]
pub fn sync_component<T>(
    query: Query<(&EntityId, &T), (Changed<T>, Without<Received>)>,
    subs: ResMut<Subscriptions>,
) where
    T: bevy::prelude::Component + Component + Debug,
{
    let component_id = T::component_id();
    if query.is_empty() {
        return;
    }
    let Some(subs) = subs.get(&component_id) else {
        return;
    };
    let iter = query.iter().map(|(entity_id, val)| ColumnValue {
        entity_id: *entity_id,
        value: val.component_value(),
    });
    let Ok(col) = crate::ColumnPayload::try_from_value_iter(0, iter) else {
        warn!("error creating column payload packet");
        return;
    };
    for sub in subs.iter() {
        let _ = sub.tx.send(Packet {
            stream_id: sub.stream_id,
            payload: Payload::Column(col.clone()),
        });
    }
}

pub fn query_component<T: bevy::prelude::Component + Component + Debug>(
    mut events: EventReader<SubscribeEvent>,
    store: Res<MetadataStore>,
    query: Query<(&EntityId, &T)>,
) {
    let component_id = T::component_id();
    for event in events.read() {
        let ids = event.query.execute(&store);
        if ids.len() != 1 {
            warn!("only single id queries are supported right now");
            continue;
        }
        if ids[0] != QueryId::Component(component_id) {
            continue;
        }
        let Some(metadata) = store.get_metadata(&component_id) else {
            warn!(?component_id, "component lacks metadata");
            continue;
        };

        let _ = event.subscription.tx.send(Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::Metadata {
                stream_id: event.subscription.stream_id,
                metadata: metadata.clone(),
            }),
        });

        let iter = query.iter().map(|(entity_id, val)| ColumnValue {
            entity_id: *entity_id,
            value: val.component_value(),
        });
        let Ok(col) = crate::ColumnPayload::try_from_value_iter(0, iter) else {
            warn!("error creating column payload packet");
            continue;
        };

        let _ = event.subscription.tx.send(Packet {
            stream_id: event.subscription.stream_id,
            payload: Payload::Column(col),
        });
    }
}

pub fn query_asset<T>(
    mut events: EventReader<SubscribeEvent>,
    store: Res<MetadataStore>,
    query: Query<(&EntityId, &T)>,
) where
    T: Asset + bevy::prelude::Component + Debug,
{
    let asset_id = T::ASSET_ID;
    for event in events.read() {
        let ids = event.query.execute(&store);
        if ids.len() != 1 {
            warn!("only single id queries are supported right now");
            continue;
        }
        if ids[0] != QueryId::Component(ComponentId(asset_id.0)) {
            continue;
        }

        let msgs = query
            .iter()
            .map(|(id, val)| {
                let bytes = postcard::to_allocvec(&val)?;
                Ok(ControlMsg::Asset {
                    id: asset_id,
                    entity_id: *id,
                    bytes: bytes.into(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()
            .unwrap();

        for msg in msgs.into_iter() {
            let _ = event.subscription.tx.send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(msg),
            });
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn sync_asset<T>(
    query: Query<(&EntityId, &T), (Changed<T>, Without<Received>)>,
    subs: ResMut<Subscriptions>,
) where
    T: Asset + bevy::prelude::Component + Debug,
{
    let asset_id = T::ASSET_ID;
    if query.is_empty() {
        return;
    }
    let Some(subs) = subs.get(&ComponentId(asset_id.0)) else {
        return;
    };
    let msgs = query
        .iter()
        .map(|(id, val)| {
            let bytes = postcard::to_allocvec(&val)?;
            Ok(ControlMsg::Asset {
                id: asset_id,
                entity_id: *id,
                bytes: bytes.into(),
            })
        })
        .collect::<Result<Vec<_>, Error>>()
        .unwrap();
    for sub in subs.iter() {
        for msg in msgs.iter().cloned() {
            let _ = sub.tx.send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(msg),
            });
        }
    }
}

pub fn control_msg(mut reader: EventReader<ControlMsg>, sim_peer: Res<SimPeer>) {
    for msg in reader.read() {
        if let Some(tx) = &sim_peer.tx {
            let _ = tx.send(Packet {
                stream_id: StreamId::CONTROL,
                payload: Payload::ControlMsg(msg.clone()),
            });
        }
    }
}

pub fn column_payload_msg(mut reader: EventReader<ColumnPayloadMsg>, sim_peer: Res<SimPeer>) {
    for msg in reader.read() {
        if let Some(tx) = &sim_peer.tx {
            let stream_id = StreamId::rand();
            let _ = tx.send(Packet {
                stream_id,
                payload: Payload::Column(msg.payload.clone()),
            });
        }
    }
}

#[derive(Event)]
pub struct ColumnPayloadMsg {
    pub component_id: ComponentId,
    pub component_type: ComponentType,
    pub payload: ColumnPayload<Bytes>,
}

#[derive(Event)]
pub struct SubscribeEvent {
    query: crate::Query,
    subscription: Subscription,
}

#[derive(Debug)]
pub struct SendError;

#[derive(Resource, Default, Clone)]
pub struct Subscriptions {
    map: HashMap<ComponentId, Vec<Subscription>>,
}

impl Subscriptions {
    pub fn get(&self, key: &ComponentId) -> Option<&Vec<Subscription>> {
        self.map.get(key)
    }

    pub fn subscribe(&mut self, key: ComponentId, subscription: Subscription) {
        let subs = self.map.entry(key).or_default();
        subs.push(subscription);
    }
}

#[derive(Clone)]
pub struct Subscription {
    stream_id: StreamId,
    tx: flume::Sender<Packet<Payload<Bytes>>>,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct SimPeer {
    tx: Option<flume::Sender<Packet<Payload<Bytes>>>>,
}
