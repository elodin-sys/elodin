use impeller2::types::{Msg, PacketId};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

pub type AssetId = u64;

#[derive(Serialize, Deserialize, Debug)]
pub struct Asset<'a> {
    pub id: AssetId,
    pub buf: Cow<'a, [u8]>,
}

impl Msg for Asset<'_> {
    const ID: PacketId = [224, 0, 14];
}
