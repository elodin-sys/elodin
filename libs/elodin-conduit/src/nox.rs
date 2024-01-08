use nox::{
    xla::{ElementType, PjRtBuffer},
    Client,
};

use crate::{ComponentType, ComponentValue};

impl ComponentValue<'_> {
    pub fn to_pjrt_buf(&self, client: &Client) -> Result<PjRtBuffer, nox::Error> {
        let dims: heapless::Vec<usize, 2> = match self {
            ComponentValue::U8(_)
            | ComponentValue::U16(_)
            | ComponentValue::U32(_)
            | ComponentValue::U64(_)
            | ComponentValue::I8(_)
            | ComponentValue::I16(_)
            | ComponentValue::I32(_)
            | ComponentValue::I64(_)
            | ComponentValue::Bool(_)
            | ComponentValue::F32(_)
            | ComponentValue::F64(_) => heapless::Vec::new(),
            ComponentValue::String(b) => heapless::Vec::from_slice(&[b.len()]).unwrap(),
            ComponentValue::Bytes(b) => heapless::Vec::from_slice(&[b.len()]).unwrap(),
            ComponentValue::Vector3F32(_) | ComponentValue::Vector3F64(_) => {
                heapless::Vec::from_slice(&[3]).unwrap()
            }
            ComponentValue::Matrix3x3F32(_) | ComponentValue::Matrix3x3F64(_) => {
                heapless::Vec::from_slice(&[3, 3]).unwrap()
            }
            ComponentValue::QuaternionF32(_) | ComponentValue::QuaternionF64(_) => {
                heapless::Vec::from_slice(&[4]).unwrap()
            }
            ComponentValue::SpatialPosF32(_) | ComponentValue::SpatialPosF64(_) => {
                heapless::Vec::from_slice(&[7]).unwrap()
            }
            ComponentValue::SpatialMotionF32(_) | ComponentValue::SpatialMotionF64(_) => {
                heapless::Vec::from_slice(&[6]).unwrap()
            }
            ComponentValue::Filter(_) => heapless::Vec::from_slice(&[9]).unwrap(),
        };
        let element_ty = self.ty().element_type();
        self.with_bytes(|bytes| {
            client
                .0
                .copy_raw_host_buffer(element_ty, bytes, &dims)
                .map_err(nox::Error::from)
        })
    }
}

impl ComponentType {
    #[inline]
    pub fn element_type(self) -> ElementType {
        match self {
            ComponentType::U8 => ElementType::U8,
            ComponentType::U16 => ElementType::U16,
            ComponentType::U32 => ElementType::U32,
            ComponentType::U64 => ElementType::U64,
            ComponentType::I8 => ElementType::S8,
            ComponentType::I16 => ElementType::S16,
            ComponentType::I32 => ElementType::S32,
            ComponentType::I64 => ElementType::S64,
            ComponentType::Bool => ElementType::U8,
            ComponentType::F32 => ElementType::F32,
            ComponentType::F64 => ElementType::F64,
            ComponentType::String => ElementType::U8,
            ComponentType::Bytes => ElementType::U8,
            ComponentType::Vector3F32
            | ComponentType::Matrix3x3F32
            | ComponentType::SpatialPosF32
            | ComponentType::SpatialMotionF32
            | ComponentType::QuaternionF32 => ElementType::F32,
            ComponentType::Vector3F64
            | ComponentType::Matrix3x3F64
            | ComponentType::QuaternionF64
            | ComponentType::SpatialPosF64
            | ComponentType::SpatialMotionF64 => ElementType::F64,
            ComponentType::Filter => ElementType::U8,
        }
    }

    pub fn dims(self) -> heapless::Vec<i64, 2> {
        match self {
            ComponentType::U8
            | ComponentType::U16
            | ComponentType::U32
            | ComponentType::U64
            | ComponentType::I8
            | ComponentType::I16
            | ComponentType::I32
            | ComponentType::I64
            | ComponentType::Bool
            | ComponentType::F32
            | ComponentType::F64 => heapless::Vec::from_slice(&[]).unwrap(),
            ComponentType::String | ComponentType::Bytes => {
                heapless::Vec::from_slice(&[-1]).unwrap()
            }
            ComponentType::Vector3F32 | ComponentType::Vector3F64 => {
                heapless::Vec::from_slice(&[3]).unwrap()
            }
            ComponentType::Matrix3x3F32 | ComponentType::Matrix3x3F64 => {
                heapless::Vec::from_slice(&[3, 3]).unwrap()
            }
            ComponentType::QuaternionF32 | ComponentType::QuaternionF64 => {
                heapless::Vec::from_slice(&[4]).unwrap()
            }
            ComponentType::SpatialPosF32 | ComponentType::SpatialPosF64 => {
                heapless::Vec::from_slice(&[7]).unwrap()
            }
            ComponentType::SpatialMotionF32 | ComponentType::SpatialMotionF64 => {
                heapless::Vec::from_slice(&[6]).unwrap()
            }
            ComponentType::Filter => heapless::Vec::from_slice(&[9]).unwrap(),
        }
    }
}
