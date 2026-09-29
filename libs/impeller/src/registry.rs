use crate::{
    buf::Buf,
    types::PacketId,
    vtable::{Field, Op, VTable},
};

pub trait VTableRegistry {
    type Ops: Buf<Op>;
    type Fields: Buf<Field>;
    type Data: Buf<u8>;
    fn get(&self, id: &PacketId) -> Option<&VTable<Self::Ops, Self::Data, Self::Fields>>;
}

#[cfg(feature = "std")]
mod std {
    use super::*;
    use ::std::collections::HashMap;
    use alloc::vec::Vec;

    #[derive(Default)]
    #[cfg_attr(feature = "bevy", derive(bevy::prelude::Resource))]
    pub struct HashMapRegistry {
        pub map: HashMap<PacketId, VTable>,
    }

    impl VTableRegistry for HashMapRegistry {
        type Ops = Vec<Op>;

        type Fields = Vec<Field>;

        type Data = Vec<u8>;

        fn get(&self, id: &PacketId) -> Option<&VTable<Self::Ops, Self::Data, Self::Fields>> {
            self.map.get(id)
        }
    }
}

#[cfg(feature = "std")]
pub use std::*;
