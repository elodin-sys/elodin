use bevy::{
    app::Plugin,
    ecs::{
        hierarchy::ChildOf,
        system::{EntityCommands, SystemId},
    },
    prelude::{Command, In, InRef, IntoSystem, Message, Mut, System},
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
    registry::HashMapRegistry,
    types::{OwnedPacket, PacketId, Request},
};
use impeller2::{
    schema::Schema,
    types::{ComponentId, ComponentView, LenPacket, Msg, Timestamp},
};
use impeller2_bbq::{AsyncArcQueueRx, RxExt};
use impeller2_wkt::{
    ComponentMetadata, CurrentTimestamp, DbConfig, DumpMetadata, DumpMetadataResp, DumpSchema,
    DumpSchemaResp, EarliestTimestamp, ErrorResponse, GetDbSettings, GetEarliestTimestamp,
    GetTimeSeries, IsRecording, LastUpdated, Stream, StreamBehavior, StreamId, StreamTimestamp,
    SubscribeLastUpdated, VTableMsg, WorldPos,
};
use serde::de::DeserializeOwned;
use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    marker::PhantomData,
};
use stellarator_buf::Slice;

pub use impeller2_bbq::PacketGrantR;
pub use impeller2_wkt::ComponentValue;
pub use impeller2_wkt::ElementValueMut;
pub use impeller2_wkt::{ComponentPart, ComponentPath};

#[cfg(feature = "tcp")]
mod tcp;
#[cfg(feature = "tcp")]
pub use tcp::*;

/// Size of the BBQ queue for incoming packets.
/// Increased from 64MB to 256MB to handle large Arrow IPC responses from SQL queries.
pub const QUEUE_LEN: usize = 256 * 1024 * 1024;

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

#[derive(Debug, Message)]
pub enum DbMessage {
    UpdateConfig,
}

/// Per-component time-series cache. Stores raw component values keyed by
/// timestamp so the Editor can display data at any `CurrentTimestamp` without
/// a DB round-trip.
#[derive(Resource, Default)]
pub struct TelemetryCache {
    components: HashMap<ComponentId, BTreeMap<Timestamp, ComponentValue>>,
    generation: u64,
}

impl TelemetryCache {
    pub fn insert(&mut self, component_id: ComponentId, ts: Timestamp, value: ComponentValue) {
        self.components
            .entry(component_id)
            .or_default()
            .insert(ts, value);
        self.generation = self.generation.wrapping_add(1);
    }

    pub fn get_at_or_before(
        &self,
        component_id: &ComponentId,
        ts: Timestamp,
    ) -> Option<&ComponentValue> {
        let series = self.components.get(component_id)?;
        series.range(..=ts).next_back().map(|(_, v)| v)
    }

    pub fn component_ids(&self) -> impl Iterator<Item = &ComponentId> {
        self.components.keys()
    }
}

/// Decomponentize implementation that collects component values into a
/// Vec for later insertion into the TelemetryCache.
struct CacheCollector {
    collected: Vec<(ComponentId, Timestamp, ComponentValue)>,
}

impl Decomponentize for CacheCollector {
    type Error = core::convert::Infallible;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        view: ComponentView<'_>,
        timestamp: Option<Timestamp>,
    ) -> Result<(), Infallible> {
        if let Some(ts) = timestamp {
            let value = ComponentValue::from_view(view);
            self.collected.push((component_id, ts, value));
        }
        Ok(())
    }
}

/// Bevy system that reads the TelemetryCache at `CurrentTimestamp` and
/// overwrites entity `ComponentValue` components and adapted components
/// (like `WorldPos`) with the cached data. This allows the viewport to
/// display data at any timestamp the user scrubs to, without waiting for
/// the DB stream to deliver it.
pub fn apply_cached_data(
    current_ts: bevy::prelude::Res<CurrentTimestamp>,
    cache: bevy::prelude::Res<TelemetryCache>,
    mut entity_map: ResMut<EntityMap>,
    mut query: Query<&mut ComponentValue>,
    adapters: bevy::prelude::Res<ComponentAdapters>,
    mut commands: Commands,
    mut last_applied: bevy::prelude::Local<(Timestamp, u64)>,
) {
    let ts = current_ts.0;
    let cache_gen = cache.generation;
    if ts == last_applied.0 && cache_gen == last_applied.1 {
        return;
    }
    *last_applied = (ts, cache_gen);
    for component_id in cache.component_ids().copied().collect::<Vec<_>>() {
        let Some(value) = cache.get_at_or_before(&component_id, ts) else {
            continue;
        };
        let Some(&entity) = entity_map.get(&component_id) else {
            continue;
        };
        if let Ok(mut cv) = query.get_mut(entity) {
            *cv = value.clone();
        } else {
            commands.entity(entity).insert(value.clone());
        }
        if let Some(adapter) = adapters.get(&component_id) {
            let view = value.as_view();
            adapter.insert(&mut commands, &mut entity_map, component_id, view);
        }
    }
}

/// Tracks which components have had their backfill requests sent.
#[derive(Resource, Default)]
pub struct BackfillState {
    requested: std::collections::HashSet<ComponentId>,
}

const BACKFILL_CHUNK_SIZE: usize = 4096;

/// Bevy system that sends paginated `GetTimeSeries` requests for each
/// known component to populate the `TelemetryCache` with historical data.
/// Runs every frame but only sends requests for components not yet requested.
pub fn backfill_cache(
    metadata_reg: bevy::prelude::Res<ComponentMetadataRegistry>,
    schema_reg: bevy::prelude::Res<ComponentSchemaRegistry>,
    mut backfill: ResMut<BackfillState>,
    mut commands: Commands,
) {
    for (&component_id, _metadata) in metadata_reg.iter() {
        if backfill.requested.contains(&component_id) {
            continue;
        }
        if !schema_reg.0.contains_key(&component_id) {
            continue;
        }
        backfill.requested.insert(component_id);
        send_backfill_page(&mut commands, component_id, Timestamp(i64::MIN));
    }
}

fn send_backfill_page(commands: &mut Commands, component_id: ComponentId, start: Timestamp) {
    let msg = GetTimeSeries {
        id: PacketId::default(),
        range: start..Timestamp(i64::MAX),
        component_id,
        limit: Some(BACKFILL_CHUNK_SIZE),
    };

    commands.send_req_reply_raw::<_, GetTimeSeries, _>(
        msg,
        move |pkt: bevy::prelude::InRef<OwnedPacket<PacketGrantR>>,
              mut cache: ResMut<TelemetryCache>,
              schema_reg: bevy::prelude::Res<ComponentSchemaRegistry>,
              mut cmds: Commands| {
            let OwnedPacket::TimeSeries(ts) = &*pkt else {
                return true;
            };
            let Ok(timestamps) = ts.timestamps() else {
                return true;
            };
            let Ok(data) = ts.data() else {
                return true;
            };

            let Some(schema) = schema_reg.0.get(&component_id) else {
                return true;
            };

            let elem_size = schema.size();
            let count = timestamps.len();
            let mut last_ts = start;

            for (i, &timestamp) in timestamps.iter().enumerate() {
                let offset = i * elem_size;
                if offset + elem_size > data.len() {
                    break;
                }
                let bytes = &data[offset..offset + elem_size];
                if let Ok(view) = impeller2::types::ComponentView::try_from_bytes_shape(
                    bytes,
                    schema.shape(),
                    schema.prim_type(),
                ) {
                    let value = ComponentValue::from_view(view);
                    cache.insert(component_id, timestamp, value);
                    last_ts = timestamp;
                }
            }

            if count >= BACKFILL_CHUNK_SIZE {
                send_backfill_page(&mut cmds, component_id, Timestamp(last_ts.0 + 1));
            }
            true
        },
    );
}

fn sink_inner(
    world: &mut World,
    packet_rx: &mut PacketRx,
    vtable_registry: &mut HashMapRegistry,
    packet_handlers: &mut PacketHandlers,
    world_sink_state: &mut SystemState<WorldSink>,
) -> Result<(), impeller2::error::Error> {
    let mut count = 0;
    let mut pending_cache_entries: Vec<(ComponentId, Timestamp, ComponentValue)> = Vec::new();
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
                if let Err(err) = world.unregister_system(handler) {
                    bevy::log::error!(?err, "unregister packet handler error");
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
                // Create entity and register path so the component appears in UI
                let path = ComponentPath::from_name(&metadata.name);
                try_insert_entity(
                    &mut world_sink.entity_map,
                    &mut world_sink.metadata_reg,
                    &mut world_sink.commands,
                    path.path.last().unwrap(),
                );
                world_sink.path_reg.0.insert(metadata.component_id, path);
                world_sink
                    .metadata_reg
                    .insert(metadata.component_id, metadata);
            }
            OwnedPacket::Msg(m) if m.id == DumpMetadataResp::ID => {
                let metadata = m.parse::<DumpMetadataResp>()?;
                for metadata in metadata.component_metadata.into_iter() {
                    let path = ComponentPath::from_name(&metadata.name);
                    try_insert_entity(
                        &mut world_sink.entity_map,
                        &mut world_sink.metadata_reg,
                        &mut world_sink.commands,
                        path.path.last().unwrap(),
                    );
                    world_sink.path_reg.0.insert(metadata.component_id, path);
                    world_sink
                        .metadata_reg
                        .insert(metadata.component_id, metadata);
                }
                *world_sink.db_config = metadata.db_config.clone();
                world_sink.commands.write_message(DbMessage::UpdateConfig);
            }
            OwnedPacket::Msg(m) if m.id == LastUpdated::ID => {
                let m = m.parse::<LastUpdated>()?;
                *world_sink.max_tick = m;
            }
            OwnedPacket::Msg(m) if m.id == DbConfig::ID => {
                let config = m.parse::<DbConfig>()?;
                world_sink.recording.0 = config.recording;
            }
            OwnedPacket::Msg(m) if m.id == DumpSchemaResp::ID => {
                let dump_schema = m.parse::<DumpSchemaResp>()?;
                world_sink.schema_reg.0.extend(dump_schema.schemas);
            }
            OwnedPacket::Table(table) => {
                let mut collector = CacheCollector {
                    collected: Vec::new(),
                };
                let _ = table.sink(vtable_registry, &mut collector);
                pending_cache_entries.extend(collector.collected);
                let _ = table.sink(vtable_registry, &mut world_sink)?;
            }
            OwnedPacket::Msg(m) if m.id == EarliestTimestamp::ID => {
                let new_earliest = m.parse::<EarliestTimestamp>()?;
                let is_first = world_sink.earliest_timestamp.0 == Timestamp(i64::MAX);
                *world_sink.earliest_timestamp = new_earliest;
                if is_first {
                    world_sink.current_timestamp.0 = new_earliest.0;
                }
            }
            OwnedPacket::Msg(m) if m.id == StreamTimestamp::ID => {
                let _ = m;
            }
            OwnedPacket::Msg(_) => {}
            OwnedPacket::TimeSeries(_) => {}
        }
        world_sink_state.apply(world);

        if !pending_cache_entries.is_empty()
            && let Some(mut cache) = world.get_resource_mut::<TelemetryCache>()
        {
            for (cid, ts, val) in pending_cache_entries.drain(..) {
                cache.insert(cid, ts, val);
            }
        }
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
pub struct EntityMap(pub HashMap<ComponentId, Entity>);

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

#[derive(Resource, Default, Deref, DerefMut)]
pub struct ComponentPathRegistry(pub HashMap<ComponentId, ComponentPath>);

#[derive(SystemParam)]
pub struct WorldSink<'w, 's> {
    commands: Commands<'w, 's>,
    entity_map: ResMut<'w, EntityMap>,
    metadata_reg: ResMut<'w, ComponentMetadataRegistry>,
    max_tick: ResMut<'w, LastUpdated>,
    earliest_timestamp: ResMut<'w, EarliestTimestamp>,
    recording: ResMut<'w, IsRecording>,
    current_timestamp: ResMut<'w, CurrentTimestamp>,
    schema_reg: ResMut<'w, ComponentSchemaRegistry>,
    path_reg: ResMut<'w, ComponentPathRegistry>,
    db_config: ResMut<'w, DbConfig>,
}

#[allow(clippy::needless_lifetimes)] // removing these lifetimes causes an internal compiler error, so here we are
fn try_insert_entity<'a, 'w, 's>(
    entity_map: &mut EntityMap,
    metadata_reg: &mut ComponentMetadataRegistry,
    commands: &'a mut Commands<'w, 's>,
    component_path: &ComponentPart,
) -> Option<EntityCommands<'a>> {
    let component_id = component_path.id;
    if let Some(entity) = entity_map.get(&component_id) {
        let Ok(e) = commands.get_entity(*entity) else {
            return None;
        };
        Some(e)
    } else {
        let mut e = commands.spawn((component_id, ComponentValueMap::default()));
        let metadata = metadata_reg
            .entry(component_id)
            .or_insert_with(|| ComponentMetadata {
                component_id,
                name: component_path.name.to_string(),
                metadata: Default::default(),
            })
            .clone();
        e.insert(metadata.clone());

        entity_map.insert(component_id, e.id());
        Some(e)
    }
}

impl Decomponentize for WorldSink<'_, '_> {
    type Error = core::convert::Infallible;
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        _view: ComponentView<'_>,
        _timestamp: Option<Timestamp>,
    ) -> Result<(), Infallible> {
        let Some(path) = self.path_reg.get(&component_id) else {
            return Ok(());
        };

        let Some(part) = path.path.last() else {
            return Ok(());
        };

        // Ensure the entity exists (creates if needed).
        try_insert_entity(
            &mut self.entity_map,
            &mut self.metadata_reg,
            &mut self.commands,
            part,
        );

        // Build parent-child hierarchy.
        let mut last_entity: Option<Entity> = None;
        for parent in path.path.iter() {
            let Some(mut e) = try_insert_entity(
                &mut self.entity_map,
                &mut self.metadata_reg,
                &mut self.commands,
                parent,
            ) else {
                continue;
            };
            if let Some(last_entity) = last_entity {
                e.insert(ChildOf(last_entity));
            }
            last_entity = Some(e.id());
        }

        // ComponentValue and adapter writes (WorldPos, etc.) are handled
        // exclusively by apply_cached_data from the TelemetryCache.
        Ok(())
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

pub trait ComponentAdapter: Send + Sync {
    fn insert(
        &self,
        commands: &mut Commands,
        map: &mut EntityMap,
        component_id: ComponentId,
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
        component_id: ComponentId,
        value: ComponentView<'_>,
    ) {
        let mut val = C::default();
        let _ = val.apply_value(C::COMPONENT_ID, value, None);
        let mut e = if let Some(entity) = entity_map.0.get(&component_id) {
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
}

#[derive(Resource)]
pub struct RequestIdAlloc(RequestId);

impl Default for RequestIdAlloc {
    fn default() -> Self {
        // Start at 1 to avoid request ID 0, which is reserved for streaming
        RequestIdAlloc(1)
    }
}

impl RequestIdAlloc {
    pub fn alloc_next_id(&mut self) -> RequestId {
        self.0 = self.0.wrapping_add(1);
        // Skip request ID 0, which is reserved for streaming messages
        if self.0 == 0 {
            self.0 = 1;
        }
        self.0
    }
}

pub struct DefaultAdaptersPlugin;

impl Plugin for DefaultAdaptersPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_impeller_component::<WorldPos>();
        app.add_impeller_component::<CurrentTimestamp>();
        app.add_impeller_component::<impeller2_wkt::SimulationTimeStep>();
    }
}

pub struct Impeller2Plugin;

impl Plugin for Impeller2Plugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_message::<DbMessage>()
            .add_plugins(DefaultAdaptersPlugin)
            .insert_resource(impeller2_wkt::SimulationTimeStep(0.001))
            .insert_resource(impeller2_wkt::CurrentTimestamp(Timestamp::EPOCH))
            .insert_resource(impeller2_wkt::LastUpdated(Timestamp(i64::MIN)))
            .insert_resource(impeller2_wkt::EarliestTimestamp(Timestamp(i64::MAX)))
            .init_resource::<IsRecording>()
            .init_resource::<EntityMap>()
            .init_resource::<ComponentMetadataRegistry>()
            .init_resource::<ComponentSchemaRegistry>()
            .init_resource::<ComponentPathRegistry>()
            .init_resource::<HashMapRegistry>()
            .init_resource::<PacketIdHandlers>()
            .init_resource::<PacketHandlers>()
            .init_resource::<RequestIdHandlers>()
            .init_resource::<RequestIdAlloc>()
            .init_resource::<DbConfig>()
            .init_resource::<TelemetryCache>()
            .init_resource::<BackfillState>();
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
        // Warn if we're about to overwrite an existing handler - this indicates
        // request ID collision (IDs being reused before previous queries complete)
        if let Some(_old) = handlers.insert(req_id, system_id) {
            bevy::log::warn!(
                req_id,
                "RequestId collision: overwriting existing handler! This may cause query failures."
            );
        }
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
                other => {
                    let desc = match other {
                        OwnedPacket::Msg(m) => {
                            format!("wrong msg type: got id={:?}, expected id={:?}", m.id, R::ID)
                        }
                        OwnedPacket::Table(_) => "wrong msg type: got Table".to_string(),
                        OwnedPacket::TimeSeries(_) => "wrong msg type: got TimeSeries".to_string(),
                    };
                    Err(ErrorResponse { description: desc })
                }
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
        // RealTimeBatched delivers all component data whenever new data
        // arrives (batched per last_updated change).  For recorded DBs this
        // sends one table then blocks — historical data is loaded via
        // GetTimeSeries backfill (triggered after DumpMetadata).
        Stream {
            behavior: StreamBehavior::RealTimeBatched,
            id: stream_id,
        }
        .into_len_packet(),
        GetEarliestTimestamp.into_len_packet(),
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
