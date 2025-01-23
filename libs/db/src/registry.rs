use impeller2::{
    table::{Entry, VTable},
    types::PacketId,
};
use std::collections::HashMap;

#[derive(Default)]
pub struct VTableRegistry {
    pub map: HashMap<PacketId, VTable<Vec<Entry>, Vec<u8>>>,
}

impl impeller2::registry::VTableRegistry for VTableRegistry {
    type EntryBuf = Vec<Entry>;

    type DataBuf = Vec<u8>;

    fn get(
        &self,
        id: &PacketId,
    ) -> Option<&impeller2::table::VTable<Self::EntryBuf, Self::DataBuf>> {
        self.map.get(id)
    }
}
