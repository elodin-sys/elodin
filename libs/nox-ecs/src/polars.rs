use elodin_db::ComponentSchema;
use impeller2::types::PrimType;
use impeller2_wkt::ComponentMetadata;
use polars::{prelude::ArrowDataType, series::Series};
use polars_arrow::array::{Array, PrimitiveArray};

use crate::{ColumnRef, Error};

pub fn to_series(
    buf: &[u8],
    schema: &ComponentSchema,
    metadata: &ComponentMetadata,
) -> Result<Series, Error> {
    let array = match schema.prim_type {
        PrimType::F64 => tensor_array(&schema.dim, schema.prim_type, prim_array::<f64>(buf)),
        PrimType::F32 => tensor_array(&schema.dim, schema.prim_type, prim_array::<f32>(buf)),
        PrimType::U64 => tensor_array(&schema.dim, schema.prim_type, prim_array::<u64>(buf)),
        PrimType::U32 => tensor_array(&schema.dim, schema.prim_type, prim_array::<u32>(buf)),
        PrimType::U16 => tensor_array(&schema.dim, schema.prim_type, prim_array::<u16>(buf)),
        PrimType::U8 => tensor_array(&schema.dim, schema.prim_type, prim_array::<u8>(buf)),
        PrimType::I64 => tensor_array(&schema.dim, schema.prim_type, prim_array::<i64>(buf)),
        PrimType::I32 => tensor_array(&schema.dim, schema.prim_type, prim_array::<i32>(buf)),
        PrimType::I16 => tensor_array(&schema.dim, schema.prim_type, prim_array::<i16>(buf)),
        PrimType::I8 => tensor_array(&schema.dim, schema.prim_type, prim_array::<i8>(buf)),
        PrimType::Bool => todo!(),
    };
    Series::from_arrow(&metadata.name, array).map_err(Error::from)
}

pub fn prim_array<T: polars_arrow::types::NativeType>(buf: &[u8]) -> Box<dyn Array> {
    let buf = bytemuck::cast_slice::<_, T>(buf);
    Box::new(PrimitiveArray::from_slice(buf))
}

pub fn arrow_data_type(ty: PrimType) -> ArrowDataType {
    match ty {
        PrimType::U8 => ArrowDataType::UInt8,
        PrimType::U16 => ArrowDataType::UInt16,
        PrimType::U32 => ArrowDataType::UInt32,
        PrimType::U64 => ArrowDataType::UInt64,
        PrimType::I8 => ArrowDataType::Int8,
        PrimType::I16 => ArrowDataType::Int16,
        PrimType::I32 => ArrowDataType::Int32,
        PrimType::I64 => ArrowDataType::Int64,
        PrimType::F32 => ArrowDataType::Float32,
        PrimType::F64 => ArrowDataType::Float64,
        PrimType::Bool => ArrowDataType::Boolean,
    }
}

fn tensor_array(shape: &[usize], ty: PrimType, inner: Box<dyn Array>) -> Box<dyn Array> {
    let data_type = arrow_data_type(ty);
    if shape.is_empty() {
        return inner;
    }
    let data_type = ArrowDataType::FixedSizeList(
        Box::new(polars_arrow::datatypes::Field::new(
            "inner", data_type, false,
        )),
        shape.iter().product(),
    );
    Box::new(polars_arrow::array::FixedSizeListArray::new(
        data_type, inner, None,
    ))
    // let metadata = HashMap::from_iter([(
    //     "ARROW:extension:metadata".to_string(),
    //     format!("{{ \"shape\": {:?} }}", shape),
    // )]);
    // (data_type, Some(metadata))
}

impl<'a, B: 'a + AsRef<[u8]>> ColumnRef<'a, B> {
    pub fn series(&self) -> Result<Series, Error> {
        to_series(self.column.as_ref(), self.schema, self.metadata)
    }
}
