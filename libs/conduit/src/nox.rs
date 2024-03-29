use nox::{
    xla::{ElementType, PjRtBuffer},
    ArrayTy, Client,
};

use crate::{ComponentType, ComponentValue, PrimitiveTy};

impl ComponentValue<'_> {
    pub fn to_pjrt_buf(&self, client: &Client) -> Result<PjRtBuffer, nox::Error> {
        let ComponentType {
            primitive_ty,
            shape,
        } = self.ty();
        let element_ty = primitive_ty.element_type();
        let bytes = self.bytes().ok_or(nox::Error::OutOfBoundsAccess)?;
        client
            .copy_raw_host_buffer(element_ty, bytes, &shape)
            .map_err(nox::Error::from)
    }
}

impl PrimitiveTy {
    #[inline]
    pub fn element_type(self) -> ElementType {
        match self {
            PrimitiveTy::U8 => ElementType::U8,
            PrimitiveTy::U16 => ElementType::U16,
            PrimitiveTy::U32 => ElementType::U32,
            PrimitiveTy::U64 => ElementType::U64,
            PrimitiveTy::I8 => ElementType::S8,
            PrimitiveTy::I16 => ElementType::S16,
            PrimitiveTy::I32 => ElementType::S32,
            PrimitiveTy::I64 => ElementType::S64,
            PrimitiveTy::Bool => ElementType::Pred,
            PrimitiveTy::F32 => ElementType::F32,
            PrimitiveTy::F64 => ElementType::F64,
        }
    }
}

impl From<ComponentType> for ArrayTy {
    fn from(val: ComponentType) -> Self {
        ArrayTy {
            element_type: val.primitive_ty.element_type(),
            shape: val.shape,
        }
    }
}
