use impeller2::{
    schema::Schema,
    table::{Entry, VTable},
    types::{ComponentId, EntityId, Msg, MsgExt, PacketId},
};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::{borrow::Cow, time::Duration};

use crate::{
    metadata::{ComponentMetadata, EntityMetadata},
    MaxTick,
};

use crate::AssetId;

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
    #[serde(default)]
    pub filter: StreamFilter,
    pub time_step: Option<Duration>,
    #[serde(default)]
    pub start_tick: Option<u64>,
    #[serde(default)]
    pub id: StreamId,
}

pub type StreamId = u64;

#[derive(Serialize, Deserialize, Default)]
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
    pub time_step: Option<Duration>,
}

impl SetStreamState {
    pub fn rewind(id: StreamId, tick: u64) -> Self {
        Self {
            id,
            playing: None,
            tick: Some(tick),
            time_step: None,
        }
    }
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
pub struct SchemaMsg(pub Schema<Vec<u64>>);
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

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub struct SetComponentMetadata(pub ComponentMetadata);

impl Msg for SetComponentMetadata {
    const ID: PacketId = [224, 0, 8];
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub struct SetEntityMetadata(pub EntityMetadata);

impl Msg for SetEntityMetadata {
    const ID: PacketId = [224, 0, 9];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetAsset<'a> {
    pub id: AssetId,
    pub buf: Cow<'a, [u8]>,
}

impl Msg for SetAsset<'_> {
    const ID: PacketId = [224, 0, 12];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetAsset {
    pub id: AssetId,
}

impl Msg for GetAsset {
    const ID: PacketId = [224, 0, 13];
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DumpMetadata;

impl Msg for DumpMetadata {
    const ID: PacketId = [224, 0, 14];
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DumpMetadataResp {
    pub component_metadata: Vec<ComponentMetadata>,
    pub entity_metadata: Vec<EntityMetadata>,
}

impl Msg for DumpMetadataResp {
    const ID: PacketId = [224, 0, 15];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DumpAssets;

impl Msg for DumpAssets {
    const ID: PacketId = [224, 0, 16];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubscribeMaxTick;

impl Msg for SubscribeMaxTick {
    const ID: PacketId = [224, 0, 17];
}

impl Msg for MaxTick {
    const ID: PacketId = [224, 0, 18];
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SetDbSettings {
    pub recording: Option<bool>,
    pub time_step: Option<Duration>,
}

impl Msg for SetDbSettings {
    const ID: PacketId = [224, 0, 19];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbSettings {
    pub recording: bool,
    pub time_step: Duration,
    pub default_stream_time_step: Duration,
}

impl Msg for DbSettings {
    const ID: PacketId = [224, 0, 20];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetDbSettings;

impl Msg for GetDbSettings {
    const ID: PacketId = [224, 0, 21];
}

#[derive(Serialize, Deserialize)]
pub struct NewConnection;

impl Msg for NewConnection {
    const ID: PacketId = [224, 255, 1];
}

#[cfg(feature = "mlua")]
impl mlua::UserData for SetStreamState {
    fn add_methods<T: mlua::UserDataMethods<Self>>(methods: &mut T) {
        methods.add_method("msg", |_, this, ()| {
            let msg = this.to_len_packet().inner;
            Ok(msg)
        });
    }
}

#[cfg(feature = "mlua")]
impl mlua::UserData for SetAsset<'_> {
    fn add_methods<T: mlua::UserDataMethods<Self>>(methods: &mut T) {
        methods.add_method("msg", |_, this, ()| {
            let msg = this.to_len_packet().inner;
            Ok(msg)
        });
    }
}

#[cfg(feature = "mlua")]
impl mlua::UserData for SetComponentMetadata {
    fn add_methods<T: mlua::UserDataMethods<Self>>(methods: &mut T) {
        methods.add_method("msg", |_, this, ()| {
            let msg = this.to_len_packet().inner;
            Ok(msg)
        });
    }
}

#[cfg(feature = "mlua")]
impl mlua::FromLua for SetComponentMetadata {
    fn from_lua(value: mlua::Value, lua: &mlua::Lua) -> mlua::Result<Self> {
        mlua::LuaSerdeExt::from_value(lua, value)
    }
}

#[cfg(feature = "mlua")]
impl mlua::UserData for SetEntityMetadata {
    fn add_methods<T: mlua::UserDataMethods<Self>>(methods: &mut T) {
        methods.add_method("msg", |_, this, ()| {
            let msg = this.to_len_packet().inner;
            Ok(msg)
        });
    }
}

#[cfg(feature = "mlua")]
impl mlua::FromLua for SetEntityMetadata {
    fn from_lua(value: mlua::Value, lua: &mlua::Lua) -> mlua::Result<Self> {
        mlua::LuaSerdeExt::from_value(lua, value)
    }
}
