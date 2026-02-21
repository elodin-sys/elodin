use impeller2::{schema::Schema, types::PrimType};
use nox::{ArrayTy, xla::ElementType};

pub trait SchemaExt {
    fn element_type(&self) -> ElementType;
    fn to_array_ty(&self) -> ArrayTy;
}

pub trait PrimTypeExt {
    fn to_element_type(&self) -> ElementType;
}

impl PrimTypeExt for PrimType {
    fn to_element_type(&self) -> ElementType {
        match self {
            impeller2::types::PrimType::U8 => ElementType::U8,
            impeller2::types::PrimType::U16 => ElementType::U16,
            impeller2::types::PrimType::U32 => ElementType::U32,
            impeller2::types::PrimType::U64 => ElementType::U64,
            impeller2::types::PrimType::I8 => ElementType::S8,
            impeller2::types::PrimType::I16 => ElementType::S16,
            impeller2::types::PrimType::I32 => ElementType::S32,
            impeller2::types::PrimType::I64 => ElementType::S64,
            impeller2::types::PrimType::Bool => ElementType::Pred,
            impeller2::types::PrimType::F32 => ElementType::F32,
            impeller2::types::PrimType::F64 => ElementType::F64,
        }
    }
}

impl SchemaExt for elodin_db::ComponentSchema {
    fn element_type(&self) -> ElementType {
        self.prim_type.to_element_type()
    }

    fn to_array_ty(&self) -> ArrayTy {
        ArrayTy {
            element_type: self.element_type(),
            shape: self.dim.iter().map(|x| *x as i64).collect(),
        }
    }
}

impl SchemaExt for Schema<Vec<u64>> {
    fn element_type(&self) -> ElementType {
        match self.prim_type() {
            impeller2::types::PrimType::U8 => ElementType::U8,
            impeller2::types::PrimType::U16 => ElementType::U16,
            impeller2::types::PrimType::U32 => ElementType::U32,
            impeller2::types::PrimType::U64 => ElementType::U64,
            impeller2::types::PrimType::I8 => ElementType::S8,
            impeller2::types::PrimType::I16 => ElementType::S16,
            impeller2::types::PrimType::I32 => ElementType::S32,
            impeller2::types::PrimType::I64 => ElementType::S64,
            impeller2::types::PrimType::Bool => ElementType::Pred,
            impeller2::types::PrimType::F32 => ElementType::F32,
            impeller2::types::PrimType::F64 => ElementType::F64,
        }
    }

    fn to_array_ty(&self) -> ArrayTy {
        ArrayTy {
            element_type: self.element_type(),
            shape: self.shape().iter().map(|x| *x as i64).collect(),
        }
    }
}
