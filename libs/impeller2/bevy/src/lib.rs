use bevy::{
    app::Plugin,
    ecs::system::SystemId,
    log::warn,
    prelude::{Command, InRef, IntoSystem, Mut, System},
};
use bevy::{ecs::system::SystemParam, prelude::World};
use bevy::{
    ecs::system::SystemState,
    prelude::{Commands, Component, Deref, DerefMut, Entity, Query, ResMut, Resource},
};
use impeller2::types::{
    ComponentId, ComponentView, ElementValue, EntityId, LenPacket, Msg, MsgExt,
};
use impeller2::types::{FilledRecycle, MaybeFilledPacket};
use impeller2::util::concat_str;
use impeller2::{
    com_de::Decomponentize,
    component::Asset,
    registry::HashMapRegistry,
    types::{OwnedPacket, PacketId, PrimType},
};
use impeller2_wkt::{
    AssetId, BodyAxes, ComponentMetadata, DbSettings, DumpAssets, DumpMetadata, DumpMetadataResp,
    EntityMetadata, GetDbSettings, Glb, IsRecording, Line3d, Material, MaxTick, Mesh, Panel,
    Stream, StreamFilter, StreamId, SubscribeMaxTick, Tick, VTableMsg, VectorArrow, WorldPos,
};
use nox::{Array, ArrayBuf, Dyn};
use serde::de::DeserializeOwned;
use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    ops::Deref,
    time::Duration,
};
use thingbuf::mpsc;

#[cfg(feature = "tcp")]
mod tcp;
#[cfg(feature = "tcp")]
pub use tcp::*;

#[derive(Resource)]
pub struct PacketRx(pub mpsc::Receiver<MaybeFilledPacket, FilledRecycle>);

#[derive(Resource, DerefMut, Deref)]
pub struct PacketTx(pub mpsc::Sender<Option<LenPacket>>);

impl PacketTx {
    pub fn send_msg(&self, msg: impl Msg) {
        let pkt = msg.to_len_packet();
        if self.0.try_send(Some(pkt)).is_err() {
            bevy::log::warn!("packet tx full");
        }
    }
}

fn sink_inner(
    world: &mut World,
    packet_rx: &mut PacketRx,
    vtable_registry: &mut HashMapRegistry,
    packet_id_handlers: &mut PacketIdHandlers,
    packet_handlers: &mut PacketHandlers,
    world_sink_state: &mut SystemState<WorldSink>,
) -> Result<(), impeller2::error::Error> {
    while let Ok(pkt) = packet_rx.0.try_recv_ref() {
        let MaybeFilledPacket::Packet(ref pkt) = pkt.deref() else {
            continue;
        };
        let pkt_id = match &pkt {
            OwnedPacket::Msg(m) => m.id,
            OwnedPacket::Table(table) => table.id,
            OwnedPacket::TimeSeries(time_series) => time_series.id,
        };
        if let Some(handler) = packet_id_handlers.remove(&pkt_id) {
            if let Err(err) = world.run_system_with_input(handler, pkt) {
                bevy::log::error!(?err, "packet id handler error");
            }
        }
        for handler in packet_handlers.0.iter() {
            if let Err(err) = world.run_system_with_input(*handler, (pkt, vtable_registry)) {
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
                        let Some(e) = world_sink.commands.get_entity(*entity) else {
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
            OwnedPacket::Msg(m) if m.id == MaxTick::ID => {
                let m = m.parse::<MaxTick>()?;
                *world_sink.max_tick = m;
            }
            OwnedPacket::Msg(m) if m.id == DbSettings::ID => {
                let settings = m.parse::<DbSettings>()?;
                world_sink.recording.0 = settings.recording;
            }
            OwnedPacket::Table(table) if table.id == world_sink.current_stream_id.packet_id() => {
                table.sink(&*vtable_registry, &mut world_sink)?;
            }
            OwnedPacket::Table(_) => {}
            OwnedPacket::Msg(m) if m.id == impeller2_wkt::Asset::ID => {
                let asset = m.parse::<impeller2_wkt::Asset>()?;
                world_sink.asset_store.insert(asset.id, asset.buf.to_vec());
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
            world.resource_scope(|world, mut packet_id_handlers: Mut<PacketIdHandlers>| {
                world.resource_scope(|world, mut packet_handlers: Mut<PacketHandlers>| {
                    if let Err(err) = sink_inner(
                        world,
                        &mut packet_rx,
                        &mut vtable_reg,
                        &mut packet_id_handlers,
                        &mut packet_handlers,
                        world_sink_state,
                    ) {
                        bevy::log::error!(?err, "sink failed")
                    }
                })
            })
        })
    })
}

#[allow(clippy::type_complexity)]
#[derive(Resource, Default, Deref, DerefMut)]
pub struct PacketIdHandlers(
    pub HashMap<PacketId, SystemId<InRef<'static, OwnedPacket<Vec<u8>>>, ()>>,
);

#[derive(Resource, Default, Deref, DerefMut)]
pub struct PacketHandlers(pub Vec<SystemId<PacketHandlerInput<'static>, ()>>);

pub struct PacketHandlerInput<'a> {
    pub packet: &'a OwnedPacket<Vec<u8>>,
    pub registry: &'a HashMapRegistry,
}

impl bevy::prelude::SystemInput for PacketHandlerInput<'_> {
    type Param<'i> = PacketHandlerInput<'i>;

    type Inner<'i> = (&'i OwnedPacket<Vec<u8>>, &'i HashMapRegistry);

    fn wrap((packet, registry): Self::Inner<'_>) -> Self::Param<'_> {
        PacketHandlerInput { packet, registry }
    }
}

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

#[derive(SystemParam)]
pub struct WorldSink<'w, 's> {
    query: Query<'w, 's, &'static mut ComponentValueMap>,
    commands: Commands<'w, 's>,
    entity_map: ResMut<'w, EntityMap>,
    asset_store: ResMut<'w, AssetStore>,
    metadata_reg: ResMut<'w, ComponentMetadataRegistry>,
    asset_adapters: ResMut<'w, AssetAdapters>,
    component_adapters: ResMut<'w, ComponentAdapters>,
    max_tick: ResMut<'w, MaxTick>,
    current_stream_id: ResMut<'w, CurrentStreamId>,
    recording: ResMut<'w, IsRecording>,
}

impl Decomponentize for WorldSink<'_, '_> {
    fn apply_value(
        &mut self,
        component_id: ComponentId,
        entity_id: impeller2::types::EntityId,
        view: ComponentView<'_>,
    ) {
        let e = if let Some(entity) = self.entity_map.get(&entity_id) {
            let Some(e) = self.commands.get_entity(*entity) else {
                return;
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
            return;
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
                return;
            };
            let Some(asset_id) = view.as_asset_id() else {
                return;
            };
            let Some(asset) = self.asset_store.get(&asset_id) else {
                return;
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
                return;
            };
            adapter.insert(&mut self.commands, &mut self.entity_map, entity_id, view);
        }
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
        // if arr.shape().is_empty() {
        //     return None;
        // }
        Some(arr.get(0))
    }
}

#[derive(Clone, Debug)]
pub enum ComponentValue {
    U8(Array<u8, Dyn>),
    U16(Array<u16, Dyn>),
    U32(Array<u32, Dyn>),
    U64(Array<u64, Dyn>),
    I8(Array<i8, Dyn>),
    I16(Array<i16, Dyn>),
    I32(Array<i32, Dyn>),
    I64(Array<i64, Dyn>),
    Bool(Array<bool, Dyn>),
    F32(Array<f32, Dyn>),
    F64(Array<f64, Dyn>),
}

impl ComponentValue {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::U8(arr) => arr.shape(),
            Self::U16(arr) => arr.shape(),
            Self::U32(arr) => arr.shape(),
            Self::U64(arr) => arr.shape(),
            Self::I8(arr) => arr.shape(),
            Self::I16(arr) => arr.shape(),
            Self::I32(arr) => arr.shape(),
            Self::I64(arr) => arr.shape(),
            Self::Bool(arr) => arr.shape(),
            Self::F32(arr) => arr.shape(),
            Self::F64(arr) => arr.shape(),
        }
    }
    pub fn copy_from_view(&mut self, view: ComponentView<'_>) {
        match (self, view) {
            (Self::U8(arr), ComponentView::U8(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::U16(arr), ComponentView::U16(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::U32(arr), ComponentView::U32(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::U64(arr), ComponentView::U64(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I8(arr), ComponentView::I8(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I16(arr), ComponentView::I16(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I32(arr), ComponentView::I32(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::I64(arr), ComponentView::I64(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::Bool(arr), ComponentView::Bool(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::F32(arr), ComponentView::F32(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            (Self::F64(arr), ComponentView::F64(view)) => {
                arr.buf.as_mut_buf().copy_from_slice(view.buf());
            }
            _ => panic!("Incompatible component value and view types"),
        }
    }

    pub fn from_view(view: ComponentView<'_>) -> Self {
        match view {
            ComponentView::U8(view) => Self::U8(view.to_dyn_owned()),
            ComponentView::U16(view) => Self::U16(view.to_dyn_owned()),
            ComponentView::U32(view) => Self::U32(view.to_dyn_owned()),
            ComponentView::U64(view) => Self::U64(view.to_dyn_owned()),
            ComponentView::I8(view) => Self::I8(view.to_dyn_owned()),
            ComponentView::I16(view) => Self::I16(view.to_dyn_owned()),
            ComponentView::I32(view) => Self::I32(view.to_dyn_owned()),
            ComponentView::I64(view) => Self::I64(view.to_dyn_owned()),
            ComponentView::Bool(view) => Self::Bool(view.to_dyn_owned()),
            ComponentView::F32(view) => Self::F32(view.to_dyn_owned()),
            ComponentView::F64(view) => Self::F64(view.to_dyn_owned()),
        }
    }

    pub fn indexed_iter_mut<'i>(
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

    pub fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = ElementValue> + 'i> {
        match self {
            ComponentValue::U8(u8) => {
                Box::new(u8.buf.as_buf().iter().map(|&x| ElementValue::U8(x)))
            }
            ComponentValue::U16(u16) => {
                Box::new(u16.buf.as_buf().iter().map(|&x| ElementValue::U16(x)))
            }
            ComponentValue::U32(u32) => {
                Box::new(u32.buf.as_buf().iter().map(|&x| ElementValue::U32(x)))
            }
            ComponentValue::U64(u64) => {
                Box::new(u64.buf.as_buf().iter().map(|&x| ElementValue::U64(x)))
            }
            ComponentValue::I8(i8) => {
                Box::new(i8.buf.as_buf().iter().map(|&x| ElementValue::I8(x)))
            }
            ComponentValue::I16(i16) => {
                Box::new(i16.buf.as_buf().iter().map(|&x| ElementValue::I16(x)))
            }
            ComponentValue::I32(i32) => {
                Box::new(i32.buf.as_buf().iter().map(|&x| ElementValue::I32(x)))
            }
            ComponentValue::I64(i64) => {
                Box::new(i64.buf.as_buf().iter().map(|&x| ElementValue::I64(x)))
            }
            ComponentValue::Bool(bool) => {
                Box::new(bool.buf.as_buf().iter().map(|&x| ElementValue::Bool(x)))
            }
            ComponentValue::F32(f32) => {
                Box::new(f32.buf.as_buf().iter().map(|&x| ElementValue::F32(x)))
            }
            ComponentValue::F64(f64) => {
                Box::new(f64.buf.as_buf().iter().map(|&x| ElementValue::F64(x)))
            }
        }
    }

    pub fn get(&self, i: usize) -> Option<ElementValue> {
        match self {
            ComponentValue::U8(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U8(x)),
            ComponentValue::U16(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U16(x)),
            ComponentValue::U32(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U32(x)),
            ComponentValue::U64(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::U64(x)),
            ComponentValue::I8(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I8(x)),
            ComponentValue::I16(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I16(x)),
            ComponentValue::I32(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I32(x)),
            ComponentValue::I64(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::I64(x)),
            ComponentValue::Bool(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::Bool(x)),
            ComponentValue::F32(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::F32(x)),
            ComponentValue::F64(x) => x.buf.as_buf().get(i).map(|&x| ElementValue::F64(x)),
        }
    }

    pub fn prim_type(&self) -> PrimType {
        match self {
            ComponentValue::U8(_) => PrimType::U8,
            ComponentValue::U16(_) => PrimType::U16,
            ComponentValue::U32(_) => PrimType::U32,
            ComponentValue::U64(_) => PrimType::U64,
            ComponentValue::I8(_) => PrimType::I8,
            ComponentValue::I16(_) => PrimType::I16,
            ComponentValue::I32(_) => PrimType::I32,
            ComponentValue::I64(_) => PrimType::I64,
            ComponentValue::Bool(_) => PrimType::Bool,
            ComponentValue::F32(_) => PrimType::F32,
            ComponentValue::F64(_) => PrimType::F64,
        }
    }
}

pub enum ElementValueMut<'a> {
    U8(&'a mut u8),
    U16(&'a mut u16),
    U32(&'a mut u32),
    U64(&'a mut u64),
    I8(&'a mut i8),
    I16(&'a mut i16),
    I32(&'a mut i32),
    I64(&'a mut i64),
    F64(&'a mut f64),
    F32(&'a mut f32),
    Bool(&'a mut bool),
}

#[derive(Resource, Debug, Deref, DerefMut)]
pub struct CurrentStreamId(pub StreamId);

impl CurrentStreamId {
    pub fn rand() -> CurrentStreamId {
        CurrentStreamId(fastrand::u64(..))
    }

    pub fn packet_id(&self) -> PacketId {
        self.0.to_le_bytes()[..3].try_into().unwrap()
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
        val.apply_value(C::COMPONENT_ID, entity_id, value);
        let mut e = if let Some(entity) = entity_map.0.get(&entity_id) {
            let Some(e) = commands.get_entity(*entity) else {
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
            let Some(e) = commands.get_entity(*entity) else {
                return;
            };
            e
        } else {
            let e = commands.spawn((entity_id, ComponentValueMap::default()));
            entity_map.0.insert(entity_id, e.id());
            e
        };
        e.insert(asset).insert(AssetHandle::<T> {
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
            ComponentId::new(concat_str!("asset_handle_", R::NAME)),
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
        app.add_impeller_component::<Tick>();
        app.add_impeller_component::<impeller2_wkt::SimulationTimeStep>();
    }
}

pub struct Impeller2Plugin;

impl Plugin for Impeller2Plugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(DefaultAdaptersPlugin)
            .insert_resource(impeller2_wkt::SimulationTimeStep(0.001))
            .insert_resource(impeller2_wkt::Tick(0))
            .insert_resource(impeller2_wkt::MaxTick(0))
            .init_resource::<IsRecording>()
            .init_resource::<EntityMap>()
            .init_resource::<ComponentMetadataRegistry>()
            .init_resource::<AssetStore>()
            .init_resource::<HashMapRegistry>()
            .init_resource::<PacketIdHandlers>()
            .init_resource::<PacketHandlers>();
    }
}

pub struct ReqHandlerCommand<S: System<In = InRef<'static, OwnedPacket<Vec<u8>>>, Out = ()>> {
    request: LenPacket,
    packet_id: PacketId,
    system: S,
}

impl<S> Command for ReqHandlerCommand<S>
where
    S: System<In = InRef<'static, OwnedPacket<Vec<u8>>>, Out = ()>,
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
        if let Err(err) = tx.try_send(Some(self.request)) {
            let mut handlers = world
                .get_resource_mut::<PacketIdHandlers>()
                .expect("missing packet handlers");
            handlers.remove(&self.packet_id);
            bevy::log::warn!(?err, "failed to send msg");
        }
    }
}

pub trait CommandsExt {
    fn send_req_with_handler<S, M>(&mut self, msg: impl Msg, packet_id: PacketId, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<Vec<u8>>>, (), M>;
}

impl CommandsExt for Commands<'_, '_> {
    fn send_req_with_handler<S, M>(&mut self, msg: impl Msg, packet_id: PacketId, handler: S)
    where
        S: IntoSystem<InRef<'static, OwnedPacket<Vec<u8>>>, (), M>,
    {
        let system = S::into_system(handler);
        let cmd = ReqHandlerCommand {
            request: msg.to_len_packet(),
            packet_id,
            system,
        };
        self.queue(cmd);
    }
}

pub fn new_connection_packets(stream_id: StreamId) -> impl Iterator<Item = LenPacket> {
    [
        Stream {
            filter: StreamFilter {
                component_id: None,
                entity_id: None,
            },
            time_step: None,
            start_tick: Some(0),
            id: stream_id,
        }
        .to_len_packet(),
        Stream {
            filter: StreamFilter {
                component_id: None,
                entity_id: None,
            },
            time_step: Some(Duration::ZERO),
            start_tick: None,
            id: fastrand::u64(..),
        }
        .to_len_packet(),
        DumpAssets.to_len_packet(),
        DumpMetadata.to_len_packet(),
        GetDbSettings.to_len_packet(),
        SubscribeMaxTick.to_len_packet(),
    ]
    .into_iter()
}
