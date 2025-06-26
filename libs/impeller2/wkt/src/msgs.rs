use impeller2::{
    buf::IoBuf,
    schema::Schema,
    types::{
        ComponentId, Msg, MsgBuf, OwnedTable, OwnedTimeSeries, PacketId, Request, Timestamp,
        TryFromPacket,
    },
    vtable::{Field, Op, VTable},
};
use postcard_schema::schema::owned::OwnedNamedType;
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf, time::Duration};
use std::{collections::HashMap, ops::Range};

use crate::{LastUpdated, metadata::ComponentMetadata};

use crate::AssetId;

#[derive(Serialize, Deserialize, Clone, postcard_schema::Schema)]
pub struct VTableMsg {
    pub id: PacketId,
    pub vtable: VTable<Vec<Op>, Vec<u8>, Vec<Field>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct Stream {
    #[serde(default)]
    pub behavior: StreamBehavior,
    #[serde(default)]
    pub id: StreamId,
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct VTableStream {
    pub id: PacketId,
}

impl Request for VTableStream {
    type Reply<B: IoBuf + Clone> = StreamReply<B>;
}

#[derive(Clone)]
pub enum StreamReply<B: IoBuf> {
    Table(OwnedTable<B>),
    VTable(VTableMsg),
}

impl<B: IoBuf + Clone> TryFromPacket<B> for StreamReply<B> {
    fn try_from_packet(
        packet: &impeller2::types::OwnedPacket<B>,
    ) -> Result<Self, impeller2::error::Error> {
        match packet {
            impeller2::types::OwnedPacket::Msg(m) if m.id == VTableMsg::ID => {
                let msg = m.parse::<VTableMsg>()?;
                Ok(Self::VTable(msg))
            }
            impeller2::types::OwnedPacket::Table(table) => Ok(Self::Table(table.clone())),
            _ => Err(impeller2::error::Error::InvalidPacket),
        }
    }
}

impl Request for Stream {
    type Reply<B: IoBuf + Clone> = StreamReply<B>;
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, postcard_schema::Schema)]
pub struct FixedRateOp {
    pub stream_id: StreamId,
    pub behavior: FixedRateBehavior,
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct FixedRateBehavior {
    pub initial_timestamp: InitialTimestamp,
    /// The time interval between each tick in nanoseconds
    pub timestep: u64,
    /// The number of ticks per second
    pub frequency: u64,
}

impl Default for FixedRateBehavior {
    fn default() -> Self {
        Self {
            initial_timestamp: Default::default(),
            timestep: (1e9 / 60.0) as u64,
            frequency: 60,
        }
    }
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, postcard_schema::Schema)]
pub enum InitialTimestamp {
    #[default]
    Earliest,
    Latest,
    Manual(Timestamp),
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, postcard_schema::Schema)]
pub enum StreamBehavior {
    #[default]
    RealTime,
    FixedRate(FixedRateBehavior),
}

pub type StreamId = u64;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetStreamState {
    pub id: StreamId,
    pub playing: Option<bool>,
    pub timestamp: Option<Timestamp>,
    pub time_step: Option<Duration>,
    pub frequency: Option<u64>,
}

impl SetStreamState {
    pub fn rewind(id: StreamId, tick: Timestamp) -> Self {
        Self {
            id,
            playing: None,
            timestamp: Some(tick),
            time_step: None,
            frequency: None,
        }
    }
}

impl Msg for SetStreamState {
    const ID: PacketId = [224, 2];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetTimeSeries {
    pub id: PacketId,
    pub range: Range<Timestamp>,
    pub component_id: ComponentId,
    pub limit: Option<usize>,
}

impl Msg for GetTimeSeries {
    const ID: PacketId = [224, 3];
}

impl Request for GetTimeSeries {
    type Reply<B: IoBuf + Clone> = OwnedTimeSeries<B>;
}

#[derive(Serialize, Deserialize)]
pub struct SchemaMsg(pub Schema<Vec<u64>>);
impl Msg for SchemaMsg {
    const ID: PacketId = [224, 4];
}

#[derive(Serialize, Deserialize)]
pub struct GetSchema {
    pub component_id: ComponentId,
}

impl Msg for GetSchema {
    const ID: PacketId = [224, 5];
}

impl Request for GetSchema {
    type Reply<B: IoBuf + Clone> = SchemaMsg;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct GetComponentMetadata {
    pub component_id: ComponentId,
}

impl Msg for GetComponentMetadata {
    const ID: PacketId = [224, 6];
}

impl Request for GetComponentMetadata {
    type Reply<B: IoBuf + Clone> = crate::ComponentMetadata;
}

#[derive(Clone, Serialize, Deserialize, Debug, postcard_schema::Schema)]
#[serde(transparent)]
pub struct SetComponentMetadata(pub ComponentMetadata);

impl SetComponentMetadata {
    pub fn new(component_id: impl Into<ComponentId>, name: impl ToString) -> Self {
        let component_id = component_id.into();
        let name = name.to_string();
        Self(ComponentMetadata {
            component_id,
            metadata: Default::default(),
            asset: false,
            name,
        })
    }

    pub fn metadata(mut self, metadata: std::collections::HashMap<String, String>) -> Self {
        self.0.metadata = metadata;
        self
    }

    pub fn asset(mut self, asset: bool) -> Self {
        self.0.asset = asset;
        self
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetAsset<'a> {
    pub id: AssetId,
    pub buf: Cow<'a, [u8]>,
}

impl SetAsset<'static> {
    pub fn new(id: AssetId, asset: impl Serialize) -> Result<Self, postcard::Error> {
        let buf = postcard::to_allocvec(&asset)?;
        Ok(Self {
            id,
            buf: buf.into(),
        })
    }
}

impl Msg for SetAsset<'_> {
    const ID: PacketId = [224, 12];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetAsset {
    pub id: AssetId,
}

impl Msg for GetAsset {
    const ID: PacketId = [224, 13];
}

impl Request for GetAsset {
    type Reply<B: IoBuf + Clone> = crate::Asset<'static>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DumpMetadata;

impl Msg for DumpMetadata {
    const ID: PacketId = [224, 14];
}

impl Request for DumpMetadata {
    type Reply<B: IoBuf + Clone> = DumpMetadataResp;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DumpMetadataResp {
    pub component_metadata: Vec<ComponentMetadata>,
    pub msg_metadata: Vec<MsgMetadata>,
}

impl Msg for DumpMetadataResp {
    const ID: PacketId = [224, 15];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DumpAssets;

impl Msg for DumpAssets {
    const ID: PacketId = [224, 16];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubscribeLastUpdated;

impl Msg for SubscribeLastUpdated {
    const ID: PacketId = [224, 17];
}

impl Msg for LastUpdated {
    const ID: PacketId = [224, 18];
}

impl Request for SubscribeLastUpdated {
    type Reply<B: IoBuf + Clone> = LastUpdated;
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SetDbSettings {
    pub recording: Option<bool>,
    pub time_step: Option<Duration>,
}

impl Msg for SetDbSettings {
    const ID: PacketId = [224, 19];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbSettings {
    pub recording: bool,
    pub time_step: Duration,
    pub default_stream_time_step: Duration,
}

impl Msg for DbSettings {
    const ID: PacketId = [224, 20];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetDbSettings;

impl Msg for GetDbSettings {
    const ID: PacketId = [224, 21];
}

#[derive(Serialize, Deserialize)]
pub struct NewConnection;

impl Msg for NewConnection {
    const ID: PacketId = [225, 1];
}

macro_rules! impl_user_data_msg {
    ($t: ty) => {
        #[cfg(feature = "mlua")]
        impl mlua::UserData for $t {
            fn add_methods<T: mlua::UserDataMethods<Self>>(methods: &mut T) {
                methods.add_method("msg", |_, this, ()| {
                    use impeller2::types::IntoLenPacket;
                    let msg = this.into_len_packet().inner;
                    Ok(msg)
                });
            }
        }
        #[cfg(feature = "mlua")]
        impl mlua::FromLua for $t {
            fn from_lua(value: mlua::Value, lua: &mlua::Lua) -> mlua::Result<Self> {
                mlua::LuaSerdeExt::from_value(lua, value)
            }
        }
    };
}

impl_user_data_msg!(VTableStream);
impl_user_data_msg!(VTableMsg);
impl_user_data_msg!(Stream);
impl_user_data_msg!(MsgStream);
impl_user_data_msg!(SetAsset<'_>);
impl_user_data_msg!(SetStreamState);
impl_user_data_msg!(SetComponentMetadata);
impl_user_data_msg!(UdpUnicast);
impl_user_data_msg!(UdpVTableStream);

#[derive(Serialize, Deserialize)]
pub struct GetEarliestTimestamp;

impl Msg for GetEarliestTimestamp {
    const ID: PacketId = [224, 22];
}

#[derive(Serialize, Deserialize, Clone, Copy)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
pub struct EarliestTimestamp(pub Timestamp);

impl Msg for EarliestTimestamp {
    const ID: PacketId = [224, 23];
}

#[derive(Serialize, Deserialize, Clone, Copy)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
pub struct DumpSchema;

impl Msg for DumpSchema {
    const ID: PacketId = [224, 24];
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DumpSchemaResp {
    pub schemas: HashMap<ComponentId, Schema<Vec<u64>>>,
}

impl Msg for DumpSchemaResp {
    const ID: PacketId = [224, 25];
}

impl Request for DumpSchema {
    type Reply<B: IoBuf + Clone> = DumpSchemaResp;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StreamTimestamp {
    pub timestamp: Timestamp,
    pub stream_id: StreamId,
}

impl Msg for StreamTimestamp {
    const ID: PacketId = [224, 26];
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[repr(transparent)]
pub struct SQLQuery(pub String);

impl Msg for SQLQuery {
    const ID: PacketId = [224, 27];
}

impl_user_data_msg!(SQLQuery);

#[derive(Clone, Serialize, Deserialize, Debug)]
#[repr(transparent)]
pub struct ArrowIPC<'a> {
    pub batch: Option<Cow<'a, [u8]>>,
}

impl Msg for ArrowIPC<'_> {
    const ID: PacketId = [224, 28];
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ErrorResponse {
    pub description: String,
}

impl std::fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description)
    }
}

impl Msg for ErrorResponse {
    const ID: PacketId = [224, 29];
}

impl Request for SQLQuery {
    type Reply<B: IoBuf + Clone> = ArrowIPC<'static>;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MsgMetadata {
    pub name: String,
    pub schema: OwnedNamedType,
    pub metadata: HashMap<String, String>,
}

impl Msg for MsgMetadata {
    const ID: PacketId = [224, 30];
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SetMsgMetadata {
    pub id: PacketId,
    pub metadata: MsgMetadata,
}

impl Msg for SetMsgMetadata {
    const ID: PacketId = [224, 31];
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct MsgStream {
    pub msg_id: PacketId,
}

impl Request for MsgStream {
    type Reply<B: IoBuf + Clone> = MsgBuf<B>;
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct FixedRateMsgStream {
    pub msg_id: PacketId,
    pub fixed_rate: FixedRateOp,
}

impl Request for FixedRateMsgStream {
    type Reply<B: IoBuf + Clone> = MsgBuf<B>;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetMsgMetadata {
    pub msg_id: PacketId,
}

impl Msg for GetMsgMetadata {
    const ID: PacketId = [224, 33];
}

impl Request for GetMsgMetadata {
    type Reply<B: IoBuf + Clone> = MsgMetadata;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GetMsgs {
    pub msg_id: PacketId,
    pub range: Range<Timestamp>,
    pub limit: Option<usize>,
}

impl Msg for GetMsgs {
    const ID: PacketId = [224, 34];
}

impl Request for GetMsgs {
    type Reply<B: IoBuf + Clone> = MsgBatch;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MsgBatch {
    pub data: Vec<(Timestamp, Vec<u8>)>,
}

impl Msg for MsgBatch {
    const ID: PacketId = [224, 35];
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct UdpUnicast {
    pub stream: Stream,
    pub addr: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct UdpVTableStream {
    pub id: PacketId,
    pub addr: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct SaveArchive {
    pub path: PathBuf,
    pub format: ArchiveFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
pub struct ArchiveSaved {
    pub path: PathBuf,
}

impl Request for SaveArchive {
    type Reply<B: IoBuf + Clone> = ArchiveSaved;
}

#[derive(Serialize, Deserialize, Debug, Clone, postcard_schema::Schema)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveFormat {
    ArrowIpc,
    Parquet,
    Csv,
}

#[derive(Serialize, Deserialize, Default, Debug, Clone, postcard_schema::Schema)]
pub struct MeanOp {
    pub window: u16,
}
