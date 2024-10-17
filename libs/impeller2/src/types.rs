use nox::ArrayView;
use serde::{Deserialize, Serialize};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes, Unaligned};

use crate::error::Error;

#[derive(
    Serialize,
    Deserialize,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    IntoBytes,
    Immutable,
    Hash,
    FromBytes,
)]
#[repr(transparent)]
pub struct ComponentId(u64);

impl ComponentId {
    pub const fn new(str: &str) -> Self {
        ComponentId(const_fnv1a_hash::fnv1a_hash_str_64(str))
    }
}

#[derive(
    Serialize,
    Deserialize,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    IntoBytes,
    Immutable,
    Hash,
    FromBytes,
)]
#[repr(transparent)]
pub struct EntityId(pub u64);

impl TryFrom<&[u8]> for EntityId {
    type Error = Error;

    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        let bytes: [u8; 8] = bytes.try_into().map_err(|_| Error::BufferUnderflow)?;
        Ok(EntityId(u64::from_le_bytes(bytes)))
    }
}

#[derive(
    Serialize,
    Deserialize,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    TryFromBytes,
    Immutable,
    IntoBytes,
)]
#[repr(u64)]
pub enum PrimType {
    U8 = 0,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    F32,
    F64,
}

impl PrimType {
    pub const fn size(&self) -> usize {
        match self {
            PrimType::U8 => core::mem::size_of::<u8>(),
            PrimType::U16 => core::mem::size_of::<u16>(),
            PrimType::U32 => core::mem::size_of::<u32>(),
            PrimType::U64 => core::mem::size_of::<u64>(),
            PrimType::I8 => core::mem::size_of::<i8>(),
            PrimType::I16 => core::mem::size_of::<i16>(),
            PrimType::I32 => core::mem::size_of::<i32>(),
            PrimType::I64 => core::mem::size_of::<i64>(),
            PrimType::Bool => core::mem::size_of::<bool>(),
            PrimType::F32 => core::mem::size_of::<f32>(),
            PrimType::F64 => core::mem::size_of::<f64>(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum ComponentView<'a> {
    U8(ArrayView<'a, u8>),
    U16(ArrayView<'a, u16>),
    U32(ArrayView<'a, u32>),
    U64(ArrayView<'a, u64>),
    I8(ArrayView<'a, i8>),
    I16(ArrayView<'a, i16>),
    I32(ArrayView<'a, i32>),
    I64(ArrayView<'a, i64>),
    Bool(ArrayView<'a, bool>),
    F32(ArrayView<'a, f32>),
    F64(ArrayView<'a, f64>),
}

impl<'a> ComponentView<'a> {
    pub fn try_from_bytes_shape(
        buf: &'a [u8],
        shape: &'a [usize],
        ty: PrimType,
    ) -> Result<Self, Error> {
        let len = shape.iter().try_fold(1usize, |x, &xs| {
            x.checked_mul(xs).ok_or(Error::OffsetOverflow)
        })?;

        match ty {
            PrimType::U8 => {
                let (buf, _) = <[u8]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::U8(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::U16 => {
                let (buf, _) = <[u16]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::U16(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::U32 => {
                let (buf, _) = <[u32]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::U32(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::U64 => {
                let (buf, _) = <[u64]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::U64(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::I8 => {
                let (buf, _) = <[i8]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::I8(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::I16 => {
                let (buf, _) = <[i16]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::I16(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::I32 => {
                let (buf, _) = <[i32]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::I32(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::I64 => {
                let (buf, _) = <[i64]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::I64(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::Bool => {
                let (buf, _) = <[bool]>::try_ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::Bool(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::F32 => {
                let (buf, _) = <[f32]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::F32(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
            PrimType::F64 => {
                let (buf, _) = <[f64]>::ref_from_prefix_with_elems(buf, len)?;
                Ok(Self::F64(ArrayView::from_buf_shape_unchecked(buf, shape)))
            }
        }
    }
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout, PartialEq, Debug)]
#[repr(u8)]
pub enum PacketTy {
    Msg = 0,
    Table = 1,
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout)]
#[repr(C)]
pub struct Packet {
    pub packet_ty: PacketTy,
    pub id: [u8; 3],
    pub body: [u8],
}
