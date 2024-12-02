use core::fmt::Display;

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
pub struct ComponentId(pub u64);

impl ComponentId {
    pub const fn new(str: &str) -> Self {
        ComponentId(const_fnv1a_hash::fnv1a_hash_str_64(str))
    }
}

impl Display for ComponentId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
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

impl Display for EntityId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

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

#[derive(Clone, Copy, Debug)]
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
    pub fn shape(&self) -> &[usize] {
        match *self {
            Self::U8(ref view) => view.shape(),
            Self::U16(ref view) => view.shape(),
            Self::U32(ref view) => view.shape(),
            Self::U64(ref view) => view.shape(),
            Self::I8(ref view) => view.shape(),
            Self::I16(ref view) => view.shape(),
            Self::I32(ref view) => view.shape(),
            Self::I64(ref view) => view.shape(),
            Self::Bool(ref view) => view.shape(),
            Self::F32(ref view) => view.shape(),
            Self::F64(ref view) => view.shape(),
        }
    }

    pub fn prim_type(&self) -> PrimType {
        match *self {
            Self::U8(_) => PrimType::U8,
            Self::U16(_) => PrimType::U16,
            Self::U32(_) => PrimType::U32,
            Self::U64(_) => PrimType::U64,
            Self::I8(_) => PrimType::I8,
            Self::I16(_) => PrimType::I16,
            Self::I32(_) => PrimType::I32,
            Self::I64(_) => PrimType::I64,
            Self::Bool(_) => PrimType::Bool,
            Self::F32(_) => PrimType::F32,
            Self::F64(_) => PrimType::F64,
        }
    }

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
    pub fn as_bytes(&self) -> &[u8] {
        match *self {
            Self::U8(ref view) => view.as_bytes(),
            Self::U16(ref view) => view.as_bytes(),
            Self::U32(ref view) => view.as_bytes(),
            Self::U64(ref view) => view.as_bytes(),
            Self::I8(ref view) => view.as_bytes(),
            Self::I16(ref view) => view.as_bytes(),
            Self::I32(ref view) => view.as_bytes(),
            Self::I64(ref view) => view.as_bytes(),
            Self::Bool(ref view) => view.as_bytes(),
            Self::F32(ref view) => view.as_bytes(),
            Self::F64(ref view) => view.as_bytes(),
        }
    }
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout, PartialEq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum PacketTy {
    Msg = 0,
    Table = 1,
}

pub type PacketId = [u8; 7];

pub const PACKET_HEADER_LEN: usize = 8;

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout)]
#[repr(C)]
pub struct Packet {
    pub packet_ty: PacketTy,
    pub id: PacketId,
    pub body: [u8],
}
