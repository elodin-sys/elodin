use crate::{
    buf::Buf,
    table::{Entry, VTable},
};

pub trait VTableRegistry {
    type EntryBuf: Buf<Entry>;
    type DataBuf: Buf<u8>;
    fn get(&self, id: &[u8; 3]) -> Option<&VTable<Self::EntryBuf, Self::DataBuf>>;
}
