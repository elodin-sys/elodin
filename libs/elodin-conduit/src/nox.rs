use nox::{
    xla::{ElementType, PjRtBuffer},
    Client,
};

use crate::ComponentValue;

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
        let element_ty = match self {
            ComponentValue::U8(_) => ElementType::U8,
            ComponentValue::U16(_) => ElementType::U16,
            ComponentValue::U32(_) => ElementType::U32,
            ComponentValue::U64(_) => ElementType::U64,
            ComponentValue::I8(_) => ElementType::S8,
            ComponentValue::I16(_) => ElementType::S16,
            ComponentValue::I32(_) => ElementType::S32,
            ComponentValue::I64(_) => ElementType::S64,
            ComponentValue::Bool(_) => ElementType::U8,
            ComponentValue::F32(_) => ElementType::F32,
            ComponentValue::F64(_) => ElementType::F64,
            ComponentValue::String(_) => ElementType::U8,
            ComponentValue::Bytes(_) => ElementType::U8,
            ComponentValue::Vector3F32(_)
            | ComponentValue::Matrix3x3F32(_)
            | ComponentValue::SpatialPosF32(_)
            | ComponentValue::SpatialMotionF32(_)
            | ComponentValue::QuaternionF32(_) => ElementType::F32,
            ComponentValue::Vector3F64(_)
            | ComponentValue::Matrix3x3F64(_)
            | ComponentValue::QuaternionF64(_)
            | ComponentValue::SpatialPosF64(_)
            | ComponentValue::SpatialMotionF64(_) => ElementType::F64,
            ComponentValue::Filter(_) => ElementType::U8,
        };
        self.with_bytes(|bytes| {
            client
                .0
                .buffer_from_host_raw_bytes(element_ty, bytes, &dims, None)
                .map_err(nox::Error::from)
        })
    }
}
