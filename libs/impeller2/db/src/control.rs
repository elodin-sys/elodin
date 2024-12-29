use impeller2::{
    schema::Schema,
    table::{Entry, VTable},
    types::{ComponentId, EntityId, PacketId},
};
use impeller2_stella::Msg;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::{collections::HashMap, ops::Range};

#[derive(Serialize, Deserialize)]
pub struct VTableMsg {
    pub id: PacketId,
    pub vtable: VTable<Vec<Entry>, Vec<u8>>,
}

impl Msg for VTableMsg {
    const ID: PacketId = [224, 0, 0];
}

#[derive(Serialize, Deserialize)]
pub struct Stream {
    pub filter: StreamFilter,
    pub time_step: Duration,
    pub start_tick: Option<u64>,
    pub id: StreamId,
}

pub type StreamId = u64;

#[derive(Serialize, Deserialize)]
pub struct StreamFilter {
    pub component_id: Option<ComponentId>,
    pub entity_id: Option<EntityId>,
}

impl Msg for Stream {
    const ID: PacketId = [224, 0, 1];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetStreamState {
    pub id: StreamId,
    pub playing: Option<bool>,
    pub tick: Option<u64>,
}

impl Msg for SetStreamState {
    const ID: PacketId = [224, 0, 2];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetTimeSeries {
    pub id: PacketId,
    pub range: Range<u64>,
    pub entity_id: EntityId,
    pub component_id: ComponentId,
}

impl Msg for GetTimeSeries {
    const ID: PacketId = [224, 0, 3];
}

#[derive(Serialize, Deserialize)]
pub struct SchemaMsg(pub Schema<Vec<u8>>);
impl Msg for SchemaMsg {
    const ID: PacketId = [224, 0, 4];
}

#[derive(Serialize, Deserialize)]
pub struct GetSchema {
    pub component_id: ComponentId,
}

impl Msg for GetSchema {
    const ID: PacketId = [224, 0, 5];
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GetComponentMetadata {
    pub component_id: ComponentId,
}

impl Msg for GetComponentMetadata {
    const ID: PacketId = [224, 0, 6];
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GetEntityMetadata {
    pub entity_id: EntityId,
}

impl Msg for GetEntityMetadata {
    const ID: PacketId = [224, 0, 7];
}

#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct Metadata {
    pub metadata: HashMap<String, MetadataValue>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum MetadataValue {
    Unit,
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SetComponentMetadata {
    pub component_id: ComponentId,
    pub metadata: Metadata,
}

impl Msg for SetComponentMetadata {
    const ID: PacketId = [224, 0, 8];
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SetEntityMetadata {
    pub entity_id: EntityId,
    pub metadata: Metadata,
}

impl Msg for SetEntityMetadata {
    const ID: PacketId = [224, 0, 9];
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ComponentMetadata {
    pub component_id: ComponentId,
    pub metadata: Metadata,
}

impl Msg for ComponentMetadata {
    const ID: PacketId = [224, 0, 11];
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EntityMetadata {
    pub entity_id: EntityId,
    pub metadata: Metadata,
}

impl Msg for EntityMetadata {
    const ID: PacketId = [224, 0, 11];
}
