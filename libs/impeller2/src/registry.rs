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
