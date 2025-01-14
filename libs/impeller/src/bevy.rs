use crate::client::ColumnMsg;
use crate::client::Msg;
use crate::client::MsgPair;
use crate::query::MetadataStore;
use crate::ser_de::ColumnValue;
use crate::well_known::EntityMetadata;
use crate::well_known::WorldPos;
use crate::Asset;
use crate::ColumnPayload;
use crate::ComponentExt;
use crate::ComponentType;
use crate::Error;
use crate::Metadata;
use crate::ValueRepr;
use crate::{
    Component, ComponentId, ComponentValue, ControlMsg, EntityId, Packet, Payload, StreamId,
};
use bevy::ecs::system::SystemParam;
use bevy::math::DQuat;
use bevy::math::DVec3;
use bevy::prelude::*;
use bytes::Bytes;
use nox::Tensor;
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

pub trait BevyComponent:
    Component + ValueRepr + bevy::prelude::Component + std::fmt::Debug
{
}
impl<T> BevyComponent for T where
    T: Component + ValueRepr + bevy::prelude::Component + std::fmt::Debug
{
}

#[derive(bevy::prelude::Component, Debug, Default)]
pub struct ComponentValueMap(pub BTreeMap<ComponentId, ComponentValue<'static>>);

#[derive(bevy::prelude::Component)]
pub struct Received;

#[derive(bevy::prelude::Resource)]
pub struct MaxTick(pub u64);

#[derive(bevy::prelude::Resource)]
pub struct Tick(pub u64);

#[derive(bevy::prelude::Resource)]
pub struct Simulating(pub bool);

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
    fn get<'a>(&'a self, world: &'a World, entity: Entity) -> Option<ComponentValue<'a>>;
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

impl<C: BevyComponent> ComponentAdapter for StaticComponentAdapter<C> {
    fn get<'a>(&'a self, world: &'a World, entity: Entity) -> Option<ComponentValue<'a>> {
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

impl<C: Resource + BevyComponent> ComponentAdapter for StaticResourceAdapter<C> {
    fn get<'a>(&'a self, world: &'a World, _entity_id: Entity) -> Option<ComponentValue<'a>> {
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
pub struct AssetMap(pub HashMap<ComponentId, Box<dyn AssetAdapter + Send + Sync>>);

pub trait AssetAdapter {
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        asset_index: u64,
        value: &[u8],
    );
}

pub trait AppExt {
    fn add_impeller_component<C: BevyComponent>(&mut self) -> &mut Self;

    fn add_impeller_component_with_adapter<C: BevyComponent>(
        &mut self,
        adapter: Box<dyn ComponentAdapter + Send + Sync>,
    ) -> &mut Self;

    fn add_impeller_asset<R: Asset + bevy::prelude::Component + Debug>(
        &mut self,
        adapter: Box<dyn AssetAdapter + Send + Sync>,
    ) -> &mut Self;
}

impl AppExt for bevy::app::App {
    fn add_impeller_component<C: BevyComponent>(&mut self) -> &mut Self {
        self.add_impeller_component_with_adapter::<C>(Box::<StaticComponentAdapter<C>>::default())
    }

    fn add_impeller_component_with_adapter<C: BevyComponent>(
        &mut self,
        adapter: Box<dyn ComponentAdapter + Send + Sync>,
    ) -> &mut Self {
        let mut metadata = self
            .world_mut()
            .get_resource_or_insert_with(MetadataStore::default);
        metadata.push(C::metadata());

        let mut map = self
            .world_mut()
            .get_resource_or_insert_with(|| ComponentMap(HashMap::default()));

        map.0.insert(C::COMPONENT_ID, adapter);
        self.add_systems(Update, sync_component::<C>);
        self.add_systems(Update, query_component::<C>);
        self
    }

    fn add_impeller_asset<R: Asset + bevy::prelude::Component + Debug>(
        &mut self,
        adapter: Box<dyn AssetAdapter + Send + Sync>,
    ) -> &mut Self {
        let mut map = self
            .world_mut()
            .get_resource_or_insert_with(AssetMap::default);
        map.0.insert(R::COMPONENT_ID, adapter);
        self.add_systems(Update, sync_asset::<R>);
        self.add_systems(Update, query_asset::<R>);
        self
    }
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct ImpellerRx(pub flume::Receiver<MsgPair>);

#[derive(Component)]
pub struct Persistent;

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct ImpellerMsgSender(pub flume::Sender<Msg>);
#[derive(Debug, Resource, Deref, DerefMut)]
pub struct ImpellerMsgReceiver(pub flume::Receiver<Msg>);

#[derive(SystemParam)]
pub struct RecvSystemArgs<'w, 's> {
    rx: Res<'w, ImpellerRx>,
    entity_map: ResMut<'w, EntityMap>,
    event: Res<'w, ImpellerMsgSender>,
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
    simulating_res: ResMut<'w, Simulating>,
}

fn recv_system(args: RecvSystemArgs) {
    let RecvSystemArgs {
        rx,
        mut entity_map,
        event,
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
        mut simulating_res,
    } = args;

    while let Ok(MsgPair { msg, tx }) = rx.try_recv() {
        let Some(tx) = tx.and_then(|tx| tx.upgrade()) else {
            continue;
        };
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
                    if tx.send(packet).is_err() {
                        continue;
                    }
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
                subscriptions.subscribe(query.component_id, subscription.clone());
                subscribe_event.send(SubscribeEvent {
                    query: query.clone(),
                    subscription,
                });
            }
            Msg::Control(ControlMsg::Asset {
                component_id,
                bytes,
                entity_id,
                asset_index,
            }) => {
                let Some(adapter) = asset_map.0.get(component_id) else {
                    warn!(?component_id, "unknown asset type");
                    continue;
                };
                adapter.insert(
                    &mut commands,
                    entity_map.as_mut(),
                    *entity_id,
                    *asset_index,
                    bytes,
                );
            }
            Msg::Control(ControlMsg::Exit) => {
                exit.send(AppExit::Success);
            }
            Msg::Control(ControlMsg::Tick {
                tick,
                max_tick,
                simulating,
            }) => {
                max_tick_res.0 = *max_tick;
                tick_res.0 = *tick;
                simulating_res.0 = *simulating;
            }
            Msg::Control(_) => {}
            Msg::Column(col) => {
                if tick_res.0 == col.payload.time {
                    col.load_into_bevy(
                        entity_map.as_mut(),
                        component_map.as_ref(),
                        &mut commands,
                        &mut value_map,
                    );
                }
            }
        }
        let _ = event.send(msg);
    }
}

#[derive(Clone)]
pub struct ImpellerSubscribePlugin {
    rx: flume::Receiver<MsgPair>,
}

impl ImpellerSubscribePlugin {
    pub fn new(rx: flume::Receiver<MsgPair>) -> Self {
        Self { rx }
    }

    pub fn pair() -> (ImpellerSubscribePlugin, flume::Sender<MsgPair>) {
        let (tx, rx) = flume::unbounded();
        (ImpellerSubscribePlugin { rx }, tx)
    }
}

impl Plugin for ImpellerSubscribePlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        let (tx, rx) = flume::unbounded();
        app.insert_resource(EntityMap::default());
        app.insert_resource(ComponentMap::default());
        app.insert_resource(AssetMap::default());
        app.insert_resource(MaxTick(0));
        app.insert_resource(Tick(0));
        app.insert_resource(Simulating(false));
        app.insert_resource(TimeStep(Duration::default()));
        app.insert_resource(ImpellerRx(self.rx.clone()));
        app.insert_resource(ImpellerMsgSender(tx));
        app.insert_resource(ImpellerMsgReceiver(rx));
        app.add_event::<SubscribeEvent>();
        app.add_event::<ControlMsg>();
        app.add_event::<ColumnPayloadMsg>();
        app.add_systems(Update, control_msg);
        app.add_systems(Update, recv_system);
        app.add_systems(PostUpdate, column_payload_msg);
    }
}

#[allow(clippy::type_complexity)]
pub fn sync_component<T: BevyComponent>(
    query: Query<(&EntityId, &T), (Changed<T>, Without<Received>)>,
    subs: ResMut<Subscriptions>,
) {
    let component_id = T::COMPONENT_ID;
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

pub fn query_component<T: BevyComponent>(
    mut events: EventReader<SubscribeEvent>,
    store: Res<MetadataStore>,
    query: Query<(&EntityId, &T)>,
) {
    let component_id = T::COMPONENT_ID;
    for event in events.read() {
        if event.query.component_id != component_id {
            continue;
        }
        let Some(metadata) = store.get_metadata(&component_id) else {
            warn!(?component_id, "component lacks metadata");
            continue;
        };

        let _ = event.subscription.tx.send(Packet {
            stream_id: StreamId::CONTROL,
            payload: Payload::ControlMsg(ControlMsg::OpenStream {
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

pub fn query_asset<T>(mut events: EventReader<SubscribeEvent>, query: Query<(&EntityId, &T)>)
where
    T: Asset + bevy::prelude::Component + Debug,
{
    let component_id = T::COMPONENT_ID;
    for event in events.read() {
        if event.query.component_id != component_id {
            continue;
        }

        let msgs = query
            .iter()
            .map(|(id, val)| {
                let bytes = postcard::to_allocvec(&val)?;
                Ok(ControlMsg::Asset {
                    component_id,
                    entity_id: *id,
                    bytes: bytes.into(),
                    asset_index: 0, // TODO: fix me
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
    let component_id = T::COMPONENT_ID;
    if query.is_empty() {
        return;
    }
    let Some(subs) = subs.get(&component_id) else {
        return;
    };
    let msgs = query
        .iter()
        .map(|(id, val)| {
            let bytes = postcard::to_allocvec(&val)?;
            Ok(ControlMsg::Asset {
                component_id,
                entity_id: *id,
                bytes: bytes.into(),
                asset_index: 0, // TODO: fix me
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
            let _ = tx.send(Packet::start_stream(
                stream_id,
                Metadata {
                    name: msg.component_name.clone().into(),
                    component_type: msg.component_type.clone(),
                    tags: Some(HashMap::default()),
                    asset: false,
                },
            ));
            let _ = tx.send(Packet {
                stream_id,
                payload: Payload::Column(msg.payload.clone()),
            });
        }
    }
}

#[derive(Event)]
pub struct ColumnPayloadMsg {
    pub component_name: String,
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

impl WorldPos {
    pub fn bevy_pos(&self) -> DVec3 {
        let [x, y, z] = self.pos.parts().map(Tensor::into_buf);
        DVec3::new(x, z, -y)
    }

    pub fn bevy_att(&self) -> DQuat {
        let [i, j, k, w] = self.att.parts().map(Tensor::into_buf);
        let x = i;
        let y = k;
        let z = -j;
        DQuat::from_xyzw(x, y, z, w)
    }
}
