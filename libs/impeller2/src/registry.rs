use crate::{
    buf::Buf,
    table::{Entry, VTable},
    types::PacketId,
};

pub trait VTableRegistry {
    type EntryBuf: Buf<Entry>;
    type DataBuf: Buf<u8>;
    fn get(&self, id: &PacketId) -> Option<&VTable<Self::EntryBuf, Self::DataBuf>>;
}

#[cfg(feature = "std")]
mod std {
    use super::*;
    use ::std::collections::HashMap;
    use alloc::vec::Vec;

    #[derive(Default)]
    #[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
    pub struct HashMapRegistry {
        pub map: HashMap<PacketId, VTable<Vec<Entry>, Vec<u8>>>,
    }

    impl VTableRegistry for HashMapRegistry {
        type EntryBuf = Vec<Entry>;

        type DataBuf = Vec<u8>;

        fn get(
            &self,
            id: &PacketId,
        ) -> Option<&crate::table::VTable<Self::EntryBuf, Self::DataBuf>> {
            self.map.get(id)
        }
    }
}

#[cfg(feature = "std")]
pub use std::*;
