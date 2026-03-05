use ndarray::ArrayViewD;

pub enum DynArrayView<'a> {
    F64(ArrayViewD<'a, f64>),
    F32(ArrayViewD<'a, f32>),
    U64(ArrayViewD<'a, u64>),
    U32(ArrayViewD<'a, u32>),
    U16(ArrayViewD<'a, u16>),
    U8(ArrayViewD<'a, u8>),
    I64(ArrayViewD<'a, i64>),
    I32(ArrayViewD<'a, i32>),
    I16(ArrayViewD<'a, i16>),
    I8(ArrayViewD<'a, i8>),
    Bool(ArrayViewD<'a, bool>),
}
