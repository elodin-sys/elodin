use impeller2::{
    error::Error,
    vtable::{
        VTable,
        builder::{FieldBuilder, vtable},
    },
};

pub trait AsVTable {
    fn populate_vtable_fields(builder: &mut Vec<FieldBuilder>) -> Result<(), Error>;
    fn as_vtable() -> VTable {
        let mut fields = vec![];
        Self::populate_vtable_fields(&mut fields).expect("vtable failed to form");
        vtable(fields)
    }
}
