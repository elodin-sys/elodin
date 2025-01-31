use std::{fs::File, io::Write, path::PathBuf, sync::Arc};

use arrow::{
    array::{ArrayRef, ArrowPrimitiveType, FixedSizeListArray, RecordBatch},
    buffer::{Buffer, ScalarBuffer},
    datatypes::*,
    ipc::writer::FileWriter,
};
use elodin_db::ComponentSchema;
use impeller2::types::PrimType;
use impeller2_wkt::ComponentMetadata;

use crate::Error;

pub fn write_ipc(
    buf: &[u8],
    schema: &ComponentSchema,
    metadata: &ComponentMetadata,
    path: PathBuf,
) -> Result<(), Error> {
    let (field, array) = to_arrow(buf, schema, metadata);
    let fields = vec![field];
    let columns = vec![array];

    let record_batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
    let mut file = File::create(path)?;
    let mut ipc_writer = FileWriter::try_new_buffered(&mut file, &record_batch.schema())?;
    ipc_writer.write(&record_batch)?;
    ipc_writer.finish()?;
    ipc_writer
        .into_inner()?
        .into_inner()
        .map_err(|err| err.into_error())?
        .flush()?;
    Ok(())
}

fn to_arrow(
    buf: &[u8],
    schema: &ComponentSchema,
    metadata: &ComponentMetadata,
) -> (FieldRef, ArrayRef) {
    let size = schema.shape.iter().product::<u64>() as i32;
    let array = match schema.prim_type {
        PrimType::F64 => array_ref::<Float64Type>(buf),
        PrimType::F32 => array_ref::<Float32Type>(buf),
        PrimType::U64 => array_ref::<UInt64Type>(buf),
        PrimType::U32 => array_ref::<UInt32Type>(buf),
        PrimType::U16 => array_ref::<UInt16Type>(buf),
        PrimType::U8 => array_ref::<UInt8Type>(buf),
        PrimType::I64 => array_ref::<Int64Type>(buf),
        PrimType::I32 => array_ref::<Int32Type>(buf),
        PrimType::I16 => array_ref::<Int16Type>(buf),
        PrimType::I8 => array_ref::<Int8Type>(buf),
        PrimType::Bool => todo!(),
    };

    let inner_field = Arc::new(Field::new(
        metadata.name.to_string(),
        array.data_type().clone(),
        false,
    ));

    if schema.shape.is_empty() {
        return (inner_field, array);
    }

    let field = Arc::new(Field::new_fixed_size_list(
        metadata.name.to_string(),
        inner_field.clone(),
        size,
        false,
    ));
    let array = FixedSizeListArray::new(inner_field, size, array, None);
    let array = Arc::new(array);
    (field, array)
}

fn array_ref<P: ArrowPrimitiveType>(buf: &[u8]) -> ArrayRef
where
    P::Native: bytemuck::Pod,
{
    let buf = bytemuck::cast_slice::<_, P::Native>(buf);
    let len = buf.len();
    let buf = Buffer::from_slice_ref(buf);
    let buf = ScalarBuffer::<P::Native>::new(buf, 0, len);
    let array = arrow::array::PrimitiveArray::<P>::new(buf, None);
    Arc::new(array)
}
