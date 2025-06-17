use bevy::{
    app::Plugin,
    ecs::system::SystemId,
    log::warn,
    prelude::{Command, In, InRef, IntoSystem, Mut, System},
};
use bevy::{ecs::system::SystemParam, prelude::World};
use bevy::{
    ecs::system::SystemState,
    prelude::{Commands, Component, Deref, DerefMut, Entity, Query, ResMut, Resource},
};
use impeller2::types::IntoLenPacket;
use impeller2::types::RequestId;
use impeller2::{
    com_de::Decomponentize,
    component::Asset,
    registry::HashMapRegistry,
    types::{OwnedPacket, PacketId, Request},
};
use impeller2::{
    schema::Schema,
    types::{ComponentId, ComponentView, EntityId, LenPacket, Msg, Timestamp},
};
use impeller2_bbq::{AsyncArcQueueRx, RxExt};
use impeller2_wkt::{
    AssetId, BodyAxes, ComponentMetadata, CurrentTimestamp, DbSettings, DumpAssets, DumpMetadata,
    DumpMetadataResp, DumpSchema, DumpSchemaResp, EarliestTimestamp, EntityMetadata, ErrorResponse,
    FixedRateBehavior, GetDbSettings, GetEarliestTimestamp, Glb, IsRecording, LastUpdated, Line3d,
    Material, Mesh, Panel, Stream, StreamBehavior, StreamId, StreamTimestamp, SubscribeLastUpdated,
    VTableMsg, VectorArrow, WorldPos,
};
use nox::array::ArrayViewExt;
use serde::de::DeserializeOwned;
use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    marker::PhantomData,
    time::Duration,
};
use stellarator_buf::Slice;

pub use impeller2_bbq::PacketGrantR;
pub use impeller2_wkt::ComponentValue;
pub use impeller2_wkt::ElementValueMut;

#[cfg(feature = "tcp")]
mod tcp;
#[cfg(feature = "tcp")]
pub use tcp::*;

pub const QUEUE_LEN: usize = 64 * 1024 * 1024;

#[derive(Resource)]
pub struct PacketRx(AsyncArcQueueRx);

impl From<AsyncArcQueueRx> for PacketRx {
    fn from(rx: AsyncArcQueueRx) -> Self {
        Self(rx)
    }
}

impl PacketRx {
    #[inline]
    pub fn try_recv_pkt(&mut self) -> Option<OwnedPacket<PacketGrantR>> {
        self.0.try_recv_pkt()
    }
}

#[derive(Resource)]
pub struct PacketTx(pub thingbuf::mpsc::Sender<Option<LenPacket>>);

impl PacketTx {
    pub fn send_msg(&self, msg: impl Msg) {
        let pkt = msg.into_len_packet();
        let _ = self.0.try_send(Some(pkt));
    }
}

fn sink_inner(
    world: &mut World,
    packet_rx: &mut PacketRx,
    vtable_registry: &mut HashMapRegistry,
    packet_handlers: &mut PacketHandlers,
    world_sink_state: &mut SystemState<WorldSink>,
) -> Result<(), impeller2::error::Error> {
    let mut count = 0;
    while let Some(pkt) = packet_rx.try_recv_pkt() {
        if count > 2048 {
            return Ok(());
        }
        count += 1;
        {
            let pkt_id = match &pkt {
                OwnedPacket::Msg(m) => m.id,
                OwnedPacket::Table(table) => table.id,
                OwnedPacket::TimeSeries(time_series) => time_series.id,
            };
            let handler = world
                .get_resource_mut::<PacketIdHandlers>()
                .and_then(|mut handlers| handlers.remove(&pkt_id));
            if let Some(handler) = handler {
                if let Err(err) = world.run_system_with(handler, &pkt) {
                    bevy::log::error!(?err, "packet id handler error");
                }
            }
        }

        {
            let req_id = match &pkt {
                OwnedPacket::Msg(m) => m.req_id,
                OwnedPacket::Table(table) => table.req_id,
                OwnedPacket::TimeSeries(time_series) => time_series.req_id,
            };

            let handler = world
                .get_resource_mut::<RequestIdHandlers>()
                .and_then(|mut handlers| handlers.remove(&req_id));
            if let Some(handler) = handler {
                match world.run_system_with(handler, &pkt) {
                    Ok(completed) => {
                        if !completed {
                            world
                                .get_resource_mut::<RequestIdHandlers>()
                                .and_then(|mut handlers| handlers.insert(req_id, handler));
                        }
                    }
                    Err(err) => {
                        bevy::log::error!(?err, "packet id handler error");
                    }
                }
            }
        }

        for handler in packet_handlers.0.iter() {
            if let Err(err) = world.run_system_with(*handler, (&pkt, vtable_registry)) {
                bevy::log::error!(?err, "packet handler error");
            }
        }
        let mut world_sink = world_sink_state.get_mut(world);
        match &pkt {
            OwnedPacket::Msg(m) if m.id == VTableMsg::ID => {
                let vtable = m.parse::<VTableMsg>()?;
                vtable_registry.map.insert(vtable.id, vtable.vtable);
            }
            OwnedPacket::Msg(m) if m.id == ComponentMetadata::ID => {
                let metadata = m.parse::<ComponentMetadata>()?;
                world_sink
                    .metadata_reg
                    .insert(metadata.component_id, metadata);
            }
            OwnedPacket::Msg(m) if m.id == DumpMetadataResp::ID => {
                let metadata = m.parse::<DumpMetadataResp>()?;
                for metadata in metadata.component_metadata.into_iter() {
                    world_sink
                        .metadata_reg
                        .insert(metadata.component_id, metadata);
                }
                for metadata in metadata.entity_metadata.into_iter() {
                    let mut e = if let Some(entity) = world_sink.entity_map.get(&metadata.entity_id)
                    {
                        let Ok(e) = world_sink.commands.get_entity(*entity) else {
                            continue;
                        };
                        e
                    } else {
                        let e = world_sink
                            .commands
                            .spawn((metadata.entity_id, ComponentValueMap::default()));
                        world_sink.entity_map.insert(metadata.entity_id, e.id());
                        e
                    };
                    e.insert(metadata.clone());
                }
            }
            OwnedPacket::Msg(m) if m.id == LastUpdated::ID => {
                let m = m.parse::<LastUpdated>()?;
                *world_sink.max_tick = m;
            }
            OwnedPacket::Msg(m) if m.id == DbSettings::ID => {
                let settings = m.parse::<DbSettings>()?;
                world_sink.recording.0 = settings.recording;
            }
            OwnedPacket::Msg(m) if m.id == DumpSchemaResp::ID => {
                let dump_schema = m.parse::<DumpSchemaResp>()?;
                world_sink.schema_reg.0.extend(dump_schema.schemas);
            }
            OwnedPacket::Table(table) if table.id == world_sink.current_stream_id.packet_id() => {
                let _ = table.sink(&*vtable_registry, &mut world_sink)?;
            }
            OwnedPacket::Table(_) => {}
            OwnedPacket::Msg(m) if m.id == impeller2_wkt::Asset::ID => {
                let asset = m.parse::<impeller2_wkt::Asset>()?;
                world_sink.asset_store.insert(asset.id, asset.buf.to_vec());
            }
            OwnedPacket::Msg(m) if m.id == EarliestTimestamp::ID => {
                let earliest_timestamp = m.parse::<EarliestTimestamp>()?;
                world_sink.commands.insert_resource(earliest_timestamp);
            }
            OwnedPacket::Msg(m) if m.id == StreamTimestamp::ID => {
                let stream_timestamp = m.parse::<StreamTimestamp>()?;
                if stream_timestamp.stream_id == world_sink.current_stream_id.0 {
                    world_sink.current_timestamp.0 = stream_timestamp.timestamp;
                }
            }
            OwnedPacket::Msg(_) => {}
            OwnedPacket::TimeSeries(_) => {}
        }
        world_sink_state.apply(world);
    }
    Ok(())
}

pub fn sink(world: &mut World, world_sink_state: &mut SystemState<WorldSink>) {
    world.resource_scope(|world, mut packet_rx: Mut<PacketRx>| {
        world.resource_scope(|world, mut vtable_reg: Mut<HashMapRegistry>| {
            world.resource_scope(|world, mut packet_handlers: Mut<PacketHandlers>| {
                if let Err(err) = sink_inner(
                    world,
                    &mut packet_rx,
                    &mut vtable_reg,
                    &mut packet_handlers,
                    world_sink_state,
                ) {
                    bevy::log::error!(?err, "sink failed")
                }
            })
        })
    })
}

#[allow(clippy::type_complexity)]
#[derive(Resource, Default, Deref, DerefMut)]
pub struct PacketIdHandlers(
    pub HashMap<PacketId, SystemId<InRef<'static, OwnedPacket<PacketGrantR>>, ()>>,
);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct PacketHandlers(pub Vec<SystemId<PacketHandlerInput<'static>, ()>>);

pub struct PacketHandlerInput<'a> {
    pub packet: &'a OwnedPacket<PacketGrantR>,
    pub registry: &'a HashMapRegistry,
}

impl bevy::prelude::SystemInput for PacketHandlerInput<'_> {
    type Param<'i> = PacketHandlerInput<'i>;

    type Inner<'i> = (&'i OwnedPacket<PacketGrantR>, &'i HashMapRegistry);

    fn wrap((packet, registry): Self::Inner<'_>) -> Self::Param<'_> {
        PacketHandlerInput { packet, registry }
    }
}

#[allow(clippy::type_complexity)]
#[derive(Resource, Default, Deref, DerefMut)]
pub struct RequestIdHandlers(
    pub HashMap<RequestId, SystemId<InRef<'static, OwnedPacket<PacketGrantR>>, bool>>,
);

#[derive(Component, Default, DerefMut, Deref)]
pub struct ComponentValueMap(pub BTreeMap<ComponentId, ComponentValue>);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct EntityMap(pub HashMap<EntityId, Entity>);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ComponentMetadataRegistry(pub HashMap<ComponentId, ComponentMetadata>);

impl ComponentMetadataRegistry {
    #[inline(always)]
    pub fn get_metadata(&self, component_id: &ComponentId) -> Option<&ComponentMetadata> {
        self.get(component_id)
    }
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ComponentSchemaRegistry(pub HashMap<ComponentId, Schema<Vec<u64>>>);

#[derive(SystemParam)]
pub struct WorldSink<'w, 's> {
    query: Query<'w, 's, &'static mut ComponentValueMap>,
    commands: Commands<'w, 's>,
    entity_map: ResMut<'w, EntityMap>,
    asset_store: ResMut<'w, AssetStore>,
    metadata_reg: ResMut<'w, ComponentMetadataRegistry>,
    asset_adapters: ResMut<'w, AssetAdapters>,
    component_adapters: ResMut<'w, ComponentAdapters>,
    max_tick: ResMut<'w, LastUpdated>,
    current_stream_id: ResMut<'w, CurrentStreamId>,
    recording: ResMut<'w, IsRecording>,
    current_timestamp: ResMut<'w, CurrentTimestamp>,
    schema_reg: ResMut<'w, ComponentSchemaRegistry>,
}

impl Decomponentize for WorldSink<'_, '_> {
    type Error = core::convert::Infallible;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: impeller2::types::EntityId,
        view: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Infallible> {
        let e = if let Some(entity) = self.entity_map.get(&entity_id) {
            let Ok(e) = self.commands.get_entity(*entity) else {
                return Ok(());
            };
            e.id()
        } else {
            let e = self
                .commands
                .spawn((
                    entity_id,
                    ComponentValueMap::default(),
                    EntityMetadata {
                        entity_id,
                        name: entity_id.to_string(),
                        metadata: Default::default(),
                    },
                ))
                .id();
            self.entity_map.insert(entity_id, e);
            e
        };
        let Ok(mut value_map) = self.query.get_mut(e) else {
            return Ok(());
        };
        if let Some(value) = value_map.0.get_mut(&component_id) {
            value.copy_from_view(view);
        } else {
            value_map
                .0
                .insert(component_id, ComponentValue::from_view(view));
        }
        if self
            .metadata_reg
            .get_metadata(&component_id)
            .map(|m| m.asset)
            .unwrap_or_default()
        {
            let Some(adapter) = self.asset_adapters.get(&component_id) else {
                return Ok(());
            };
            let Some(asset_id) = view.as_asset_id() else {
                return Ok(());
            };
            let Some(asset) = self.asset_store.get(&asset_id) else {
                return Ok(());
            };
            adapter.insert(
                &mut self.commands,
                &mut self.entity_map,
                entity_id,
                asset,
                asset_id,
            );
        } else {
            let Some(adapter) = self.component_adapters.get(&component_id) else {
                return Ok(());
            };
            adapter.insert(&mut self.commands, &mut self.entity_map, entity_id, view);
        }
        Ok(())
    }
}

pub trait ComponentViewExt {
    fn as_asset_id(&self) -> Option<u64>;
}

impl ComponentViewExt for ComponentView<'_> {
    fn as_asset_id(&self) -> Option<u64> {
        let ComponentView::U64(arr) = self else {
            return None;
        };
        Some(arr.get(0))
    }
}

#[derive(Resource, Debug, Deref, DerefMut)]
pub struct CurrentStreamId(pub StreamId);

impl CurrentStreamId {
    pub fn rand() -> CurrentStreamId {
        CurrentStreamId(fastrand::u64(..))
    }

    pub fn packet_id(&self) -> PacketId {
        self.0.to_le_bytes()[..2].try_into().unwrap()
    }
}

#[derive(Resource, Debug, Deref, DerefMut, Default)]
pub struct AssetStore(HashMap<AssetId, Vec<u8>>);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct AssetAdapters(HashMap<ComponentId, Box<dyn AssetAdapter>>);

pub trait AssetAdapter: Send + Sync {
    fn insert(
        &self,
        commands: &mut Commands,
        map: &mut EntityMap,
        entity_id: EntityId,
        asset: &[u8],
        asset_id: AssetId,
    );
}

pub trait ComponentAdapter: Send + Sync {
    fn insert(
        &self,
        commands: &mut Commands,
        map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentView<'_>,
    );
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct ComponentAdapters(HashMap<ComponentId, Box<dyn ComponentAdapter>>);

pub struct StaticComponentAdapter<C>(PhantomData<C>);
impl<C> Default for StaticComponentAdapter<C> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<C> ComponentAdapter for StaticComponentAdapter<C>
where
    C: impeller2::component::Component + Decomponentize + Default + Component,
{
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        value: ComponentView<'_>,
    ) {
        let mut val = C::default();
        let _ = val.apply_value(C::COMPONENT_ID, entity_id, value, None);
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Ok(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            return;
        };
        e.insert(val);
    }
}

#[derive(bevy::prelude::Component, Debug, Clone)]
pub struct AssetHandle<T> {
    id: u64,
    phantom_data: PhantomData<T>,
}

impl<T> AssetHandle<T> {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            phantom_data: PhantomData,
        }
    }
}

impl<T> PartialEq for AssetHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for AssetHandle<T> {}

impl<T> core::hash::Hash for AssetHandle<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub struct PostcardAssetAdapter<T: DeserializeOwned + Asset>(PhantomData<T>);

impl<T: DeserializeOwned + Asset> Default for PostcardAssetAdapter<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: DeserializeOwned + Asset + Send + Sync + Component> AssetAdapter
    for PostcardAssetAdapter<T>
{
    fn insert(
        &self,
        commands: &mut Commands,
        entity_map: &mut EntityMap,
        entity_id: EntityId,
        asset: &[u8],
        asset_id: AssetId,
    ) {
        let Ok(asset) = postcard::from_bytes::<T>(asset) else {
            let name = std::any::type_name::<T>();
            warn!(asset.id = ?asset_id, adapter = name, buf = ?asset, "failed to deserialize asset");
            return;
        };
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Ok(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            let e = commands.spawn((entity_id, ComponentValueMap::default()));
            entity_map.0.insert(entity_id, e.id());
            e
        };
        e.insert_if_new(asset).insert_if_new(AssetHandle::<T> {
            id: asset_id,
            phantom_data: PhantomData,
        });
    }
}

pub trait AppExt {
    fn add_impeller_component<C>(&mut self) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component;

    fn add_impeller_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter>,
    ) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component;

    fn add_impeller_asset_with_adapter<R>(
        &mut self,
        adapter: Box<dyn AssetAdapter + Send + Sync>,
    ) -> &mut Self
    where
        R: Asset;

    fn add_impeller_asset<R>(&mut self) -> &mut Self
    where
        R: Asset + Send + Sync + Component + DeserializeOwned;
}

impl AppExt for bevy::app::App {
    fn add_impeller_component<C>(&mut self) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component,
    {
        self.add_impeller_component_with_adapter::<C>(Box::<StaticComponentAdapter<C>>::default())
    }

    fn add_impeller_component_with_adapter<C>(
        &mut self,
        adapter: Box<dyn ComponentAdapter>,
    ) -> &mut Self
    where
        C: impeller2::component::Component + Decomponentize + Default + Component,
    {
        let mut map = self
            .world_mut()
            .get_resource_or_insert_with(ComponentAdapters::default);

        map.0.insert(C::COMPONENT_ID, adapter);
        self
    }

    fn add_impeller_asset_with_adapter<R: Asset>(
        &mut self,
        adapter: Box<dyn AssetAdapter + Send + Sync>,
    ) -> &mut Self {
        let mut map = self
            .world_mut()
            .get_resource_or_insert_with(AssetAdapters::default);

        map.0.insert(
            ComponentId::new(&format!("asset_handle_{}", R::NAME)),
            adapter,
        );
        self
    }

    fn add_impeller_asset<R>(&mut self) -> &mut Self
    where
        R: Asset + Send + Sync + Component + DeserializeOwned,
    {
        self.add_impeller_asset_with_adapter::<R>(Box::new(PostcardAssetAdapter::<R>::default()))
    }
}

#[derive(Resource, Default)]
pub struct RequestIdAlloc(RequestId);

impl RequestIdAlloc {
    pub fn alloc_next_id(&mut self) -> RequestId {
        self.0 = self.0.wrapping_add(1);
        self.0
    }
}

pub struct DefaultAdaptersPlugin;

impl Plugin for DefaultAdaptersPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_impeller_asset::<Glb>();
        app.add_impeller_asset::<Mesh>();
        app.add_impeller_asset::<VectorArrow>();
        app.add_impeller_asset::<BodyAxes>();
        app.add_impeller_asset::<Panel>();
        app.add_impeller_asset::<Material>();
        app.add_impeller_asset::<Line3d>();

        app.add_impeller_component::<WorldPos>();
        app.add_impeller_component::<CurrentTimestamp>();
        app.add_impeller_component::<impeller2_wkt::SimulationTimeStep>();
    }
}

pub struct Impeller2Plugin;

impl Plugin for Impeller2Plugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(DefaultAdaptersPlugin)
            .insert_resource(impeller2_wkt::SimulationTimeStep(0.001))
            .insert_resource(impeller2_wkt::CurrentTimestamp(Timestamp::EPOCH))
            .insert_resource(impeller2_wkt::LastUpdated(Timestamp::now()))
            .insert_resource(impeller2_wkt::EarliestTimestamp(Timestamp::now()))
            .init_resource::<IsRecording>()
            .init_resource::<EntityMap>()
            .init_resource::<ComponentMetadataRegistry>()
            .init_resource::<ComponentSchemaRegistry>()
            .init_resource::<AssetStore>()
            .init_resource::<HashMapRegistry>()
            .init_resource::<PacketIdHandlers>()
            .init_resource::<PacketHandlers>()
            .init_resource::<RequestIdHandlers>()
            .init_resource::<RequestIdAlloc>();
    }
}

pub struct ReqHandlerCommand<S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = ()>> {
    request: LenPacket,
    packet_id: PacketId,
    system: S,
}

impl<S> Command for ReqHandlerCommand<S>
where
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = ()>,
{
    fn apply(self, world: &mut World) {
        let system_id = world.register_system(self.system);
        let mut handlers = world
            .get_resource_mut::<PacketIdHandlers>()
            .expect("missing packet handlers");
        handlers.insert(self.packet_id, system_id);
        let tx = world
            .get_resource_mut::<PacketTx>()
            .expect("missing packet handlers");
        if let Err(err) = tx.0.try_send(Some(self.request)) {
            let mut handlers = world
                .get_resource_mut::<PacketIdHandlers>()
                .expect("missing packet handlers");
            handlers.remove(&self.packet_id);
            bevy::log::warn!(?err, "failed to send msg");
        }
    }
}

pub struct ReplyHandlerCommand<
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = bool>,
> {
    request: LenPacket,
    system: S,
}

impl<S> Command for ReplyHandlerCommand<S>
where
    S: System<In = InRef<'static, OwnedPacket<PacketGrantR>>, Out = bool>,
{
    fn apply(self, world: &mut World) {
        let system_id = world.register_system(self.system);
        let mut alloc = world
            .get_resource_mut::<RequestIdAlloc>()
            .expect("missing packet handlers");
        let req_id = alloc.alloc_next_id();

        let mut handlers = world
            .get_resource_mut::<RequestIdHandlers>()
            .expect("missing packet handlers");
        handlers.insert(req_id, system_id);
        let tx = world
            .get_resource_mut::<PacketTx>()
            .expect("missing packet handlers");
        if let Err(err) = tx.0.try_send(Some(self.request.with_request_id(req_id))) {
            let mut handlers = world
                .get_resource_mut::<RequestIdHandlers>()
                .expect("missing packet handlers");
            handlers.remove(&req_id);
            bevy::log::warn!(?err, "failed to send msg");
        }
    }
}

pub trait CommandsExt {
    fn send_req_with_handler<S, M>(&mut self, msg: impl Msg, packet_id: PacketId, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, (), M>;

    fn send_req_reply<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        M::Reply<Slice<Vec<u8>>>: Msg + DeserializeOwned + 'static,
        S: IntoSystem<In<Result<M::Reply<Slice<Vec<u8>>>, ErrorResponse>>, bool, Marker>;

    fn send_req_reply_raw<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, bool, Marker>;
}

impl CommandsExt for Commands<'_, '_> {
    fn send_req_with_handler<S, M>(&mut self, msg: impl Msg, packet_id: PacketId, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, (), M>,
    {
        let system = S::into_system(handler);
        let cmd = ReqHandlerCommand {
            request: msg.into_len_packet(),
            packet_id,
            system,
        };
        self.queue(cmd);
    }

    fn send_req_reply<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        M::Reply<Slice<Vec<u8>>>: DeserializeOwned + Msg + 'static,
        S: IntoSystem<In<Result<M::Reply<Slice<Vec<u8>>>, ErrorResponse>>, bool, Marker>,
    {
        fn adapter<R: Msg + DeserializeOwned>(
            msg: InRef<OwnedPacket<PacketGrantR>>,
        ) -> Result<R, ErrorResponse> {
            match &*msg {
                OwnedPacket::Msg(m) if m.id == ErrorResponse::ID => {
                    let Ok(m) = m.parse::<ErrorResponse>() else {
                        return Err(ErrorResponse {
                            description: "parse failed".to_string(),
                        });
                    };
                    Err(m)
                }
                OwnedPacket::Msg(m) if m.id == R::ID => {
                    let Ok(m) = m.parse::<R>() else {
                        return Err(ErrorResponse {
                            description: "parse failed".to_string(),
                        });
                    };
                    Ok(m)
                }
                _ => Err(ErrorResponse {
                    description: "wrong msg type".to_string(),
                }),
            }
        }
        let system = adapter::<M::Reply<Slice<Vec<u8>>>>.pipe(handler);
        let system = IntoSystem::into_system(system);

        let cmd = ReplyHandlerCommand {
            request: msg.into_len_packet(),
            system,
        };
        self.queue(cmd);
    }

    fn send_req_reply_raw<S, M: Msg + Request, Marker>(&mut self, msg: M, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<PacketGrantR>>, bool, Marker>,
    {
        let system = IntoSystem::into_system(handler);

        let cmd = ReplyHandlerCommand {
            request: msg.into_len_packet(),
            system,
        };
        self.queue(cmd);
    }
}

pub fn new_connection_packets(stream_id: StreamId) -> impl Iterator<Item = LenPacket> {
    [
        Stream {
            behavior: StreamBehavior::FixedRate(FixedRateBehavior {
                initial_timestamp: impeller2_wkt::InitialTimestamp::Earliest,
                timestep: Duration::from_secs_f64(1.0 / 60.0).as_nanos() as u64,
                frequency: 60,
            }),
            id: stream_id,
        }
        .into_len_packet(),
        Stream {
            behavior: StreamBehavior::RealTime,
            id: fastrand::u64(..),
        }
        .into_len_packet(),
        GetEarliestTimestamp.into_len_packet(),
        DumpAssets.into_len_packet(),
        DumpMetadata.into_len_packet(),
        GetDbSettings.into_len_packet(),
        SubscribeLastUpdated.into_len_packet(),
        DumpSchema.into_len_packet(),
    ]
    .into_iter()
}

pub trait ComponentValueExt {
    fn indexed_iter_mut<'i>(
        &'i mut self,
    ) -> Box<dyn Iterator<Item = (&'i smallvec::SmallVec<[usize; 4]>, ElementValueMut<'i>)> + 'i>;
}
impl ComponentValueExt for ComponentValue {
    fn indexed_iter_mut<'i>(
        &'i mut self,
    ) -> Box<dyn Iterator<Item = (&'i smallvec::SmallVec<[usize; 4]>, ElementValueMut<'i>)> + 'i>
    {
        match self {
            ComponentValue::U8(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U8(x))),
            ),
            ComponentValue::U16(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U16(x))),
            ),
            ComponentValue::U32(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U32(x))),
            ),
            ComponentValue::U64(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U64(x))),
            ),
            ComponentValue::I8(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I8(x))),
            ),
            ComponentValue::I16(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I16(x))),
            ),
            ComponentValue::I32(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I32(x))),
            ),
            ComponentValue::I64(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I64(x))),
            ),
            ComponentValue::Bool(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::Bool(x))),
            ),
            ComponentValue::F32(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::F32(x))),
            ),
            ComponentValue::F64(array) => Box::new(
                array
                    .indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::F64(x))),
            ),
        }
    }
}
