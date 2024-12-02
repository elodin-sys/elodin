use impeller2::{
    table::{Entry, VTable},
    types::{ComponentId, EntityId, PacketId},
};
use impeller2_stella::Msg;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize, Deserialize)]
pub struct VTableMsg {
    pub id: PacketId,
    pub vtable: VTable<Vec<Entry>, Vec<u8>>,
}

impl Msg for VTableMsg {
    const ID: PacketId = [224, 0, 0, 0, 0, 0, 0];
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
    const ID: PacketId = [224, 0, 0, 0, 0, 0, 1];
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetStreamState {
    pub id: StreamId,
    pub playing: Option<bool>,
    pub tick: Option<u64>,
}

impl Msg for SetStreamState {
    const ID: PacketId = [224, 0, 0, 0, 0, 0, 2];
}
