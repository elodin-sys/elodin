use impeller2::{
    buf::Buf,
    error::Error,
    table::{Entry, VTable, VTableBuilder},
};

pub trait AsVTable {
    fn populate_vtable_builder<EntryBuf: Buf<Entry>, DataBuf: Buf<u8>>(
        builder: &mut VTableBuilder<EntryBuf, DataBuf>,
    ) -> Result<(), Error>;
    fn as_vtable() -> VTable<Vec<Entry>, Vec<u8>> {
        let mut builder = VTableBuilder::<Vec<Entry>, Vec<u8>>::default();
        Self::populate_vtable_builder(&mut builder).expect("failed to populate builder");
        builder.build()
    }
}
