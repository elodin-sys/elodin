use core::fmt::Display;

use nox::ArrayView;
use serde::{Deserialize, Serialize};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes, Unaligned};

use crate::error::Error;
use stellarator_buf::{IoBuf, Slice};

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
        ComponentId(const_fnv1a_hash::fnv1a_hash_str_64(str) & !(1u64 << 63)) // NOTE: we mask the last bit so the number can fit in an i64, and thus be lua compatible
    }
}

impl From<&'_ str> for ComponentId {
    fn from(value: &'_ str) -> Self {
        ComponentId::new(value)
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
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
#[repr(transparent)]
pub struct EntityId(pub u64);

impl From<u64> for EntityId {
    fn from(value: u64) -> Self {
        EntityId(value)
    }
}

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

#[cfg(feature = "mlua")]
impl mlua::UserData for EntityId {}
#[cfg(feature = "mlua")]
impl mlua::UserData for ComponentId {}

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
    pub const fn padding(&self, offset: usize) -> usize {
        let align = self.alignment();
        let offset_rounded = (offset + align - 1) & !(align - 1);
        offset_rounded - offset
    }

    pub const fn alignment(&self) -> usize {
        match self {
            PrimType::U8 => core::mem::align_of::<u8>(),
            PrimType::U16 => core::mem::align_of::<u16>(),
            PrimType::U32 => core::mem::align_of::<u32>(),
            PrimType::U64 => core::mem::align_of::<u64>(),
            PrimType::I8 => core::mem::align_of::<i8>(),
            PrimType::I16 => core::mem::align_of::<i16>(),
            PrimType::I32 => core::mem::align_of::<i32>(),
            PrimType::I64 => core::mem::align_of::<i64>(),
            PrimType::Bool => core::mem::align_of::<bool>(),
            PrimType::F32 => core::mem::align_of::<f32>(),
            PrimType::F64 => core::mem::align_of::<f64>(),
        }
    }

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

    pub const fn as_str(self) -> &'static str {
        match self {
            PrimType::U8 => "u8",
            PrimType::U16 => "u16",
            PrimType::U32 => "u32",
            PrimType::U64 => "u64",
            PrimType::I8 => "i8",
            PrimType::I16 => "i16",
            PrimType::I32 => "i32",
            PrimType::I64 => "i64",
            PrimType::Bool => "bool",
            PrimType::F32 => "f32",
            PrimType::F64 => "f64",
        }
    }
}

impl core::fmt::Display for PrimType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            PrimType::U8 => "u8",
            PrimType::U16 => "u16",
            PrimType::U32 => "u32",
            PrimType::U64 => "u64",
            PrimType::I8 => "i8",
            PrimType::I16 => "i16",
            PrimType::I32 => "i32",
            PrimType::I64 => "i64",
            PrimType::Bool => "bool",
            PrimType::F32 => "f32",
            PrimType::F64 => "f64",
        };
        core::fmt::Display::fmt(s, f)
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

    #[cfg(feature = "std")]
    pub fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = ElementValue> + 'i> {
        match self {
            ComponentView::U8(u8) => Box::new(u8.buf().iter().map(|&x| ElementValue::U8(x))),
            ComponentView::U16(u16) => Box::new(u16.buf().iter().map(|&x| ElementValue::U16(x))),
            ComponentView::U32(u32) => Box::new(u32.buf().iter().map(|&x| ElementValue::U32(x))),
            ComponentView::U64(u64) => Box::new(u64.buf().iter().map(|&x| ElementValue::U64(x))),
            ComponentView::I8(i8) => Box::new(i8.buf().iter().map(|&x| ElementValue::I8(x))),
            ComponentView::I16(i16) => Box::new(i16.buf().iter().map(|&x| ElementValue::I16(x))),
            ComponentView::I32(i32) => Box::new(i32.buf().iter().map(|&x| ElementValue::I32(x))),
            ComponentView::I64(i64) => Box::new(i64.buf().iter().map(|&x| ElementValue::I64(x))),
            ComponentView::Bool(bool) => {
                Box::new(bool.buf().iter().map(|&x| ElementValue::Bool(x)))
            }
            ComponentView::F32(f32) => Box::new(f32.buf().iter().map(|&x| ElementValue::F32(x))),
            ComponentView::F64(f64) => Box::new(f64.buf().iter().map(|&x| ElementValue::F64(x))),
        }
    }

    pub fn get(&self, i: usize) -> Option<ElementValue> {
        match self {
            Self::U8(x) => x.buf().get(i).map(|&x| ElementValue::U8(x)),
            Self::U16(x) => x.buf().get(i).map(|&x| ElementValue::U16(x)),
            Self::U32(x) => x.buf().get(i).map(|&x| ElementValue::U32(x)),
            Self::U64(x) => x.buf().get(i).map(|&x| ElementValue::U64(x)),
            Self::I8(x) => x.buf().get(i).map(|&x| ElementValue::I8(x)),
            Self::I16(x) => x.buf().get(i).map(|&x| ElementValue::I16(x)),
            Self::I32(x) => x.buf().get(i).map(|&x| ElementValue::I32(x)),
            Self::I64(x) => x.buf().get(i).map(|&x| ElementValue::I64(x)),
            Self::Bool(x) => x.buf().get(i).map(|&x| ElementValue::Bool(x)),
            Self::F32(x) => x.buf().get(i).map(|&x| ElementValue::F32(x)),
            Self::F64(x) => x.buf().get(i).map(|&x| ElementValue::F64(x)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum ElementValue {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F64(f64),
    F32(f32),
    Bool(bool),
}

impl ElementValue {
    pub fn as_f64(&self) -> f64 {
        match *self {
            ElementValue::U8(x) => x as f64,
            ElementValue::U16(x) => x as f64,
            ElementValue::U32(x) => x as f64,
            ElementValue::U64(x) => x as f64,
            ElementValue::I8(x) => x as f64,
            ElementValue::I16(x) => x as f64,
            ElementValue::I32(x) => x as f64,
            ElementValue::I64(x) => x as f64,
            ElementValue::F64(x) => x,
            ElementValue::F32(x) => x as f64,
            ElementValue::Bool(x) => {
                if x {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn as_f32(&self) -> f32 {
        match *self {
            ElementValue::U8(x) => x as f32,
            ElementValue::U16(x) => x as f32,
            ElementValue::U32(x) => x as f32,
            ElementValue::U64(x) => x as f32,
            ElementValue::I8(x) => x as f32,
            ElementValue::I16(x) => x as f32,
            ElementValue::I32(x) => x as f32,
            ElementValue::I64(x) => x as f32,
            ElementValue::F32(x) => x,
            ElementValue::F64(x) => x as f32,
            ElementValue::Bool(x) => {
                if x {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout, PartialEq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum PacketTy {
    Msg = 0,
    Table = 1,
    TimeSeries = 2,
}

pub type PacketId = [u8; 3];
pub type RequestId = [u8; 4];

pub const PACKET_HEADER_LEN: usize = 8;

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout)]
#[repr(C)]
pub struct Packet {
    pub packet_ty: PacketTy,
    pub id: PacketId,
    pub request_id: RequestId,
    pub body: [u8],
}

pub trait Msg: Serialize {
    const ID: PacketId;
}

#[derive(Clone)]
pub struct LenPacket {
    pub inner: Vec<u8>,
}

impl LenPacket {
    pub fn new(ty: PacketTy, id: PacketId, cap: usize) -> Self {
        let mut inner = Vec::with_capacity(cap + 16);
        inner.extend_from_slice(&(PACKET_HEADER_LEN as u64).to_le_bytes());
        inner.push(ty as u8);
        inner.extend_from_slice(&id);
        inner.extend_from_slice(&[0; 4]);
        Self { inner }
    }

    pub fn msg(id: PacketId, cap: usize) -> Self {
        Self::new(PacketTy::Msg, id, cap)
    }

    pub fn table(id: PacketId, cap: usize) -> Self {
        Self::new(PacketTy::Table, id, cap)
    }

    pub fn time_series(id: PacketId, cap: usize) -> Self {
        Self::new(PacketTy::TimeSeries, id, cap)
    }

    fn pkt_len(&self) -> u64 {
        let len = &self.inner[..8];
        u64::from_le_bytes(len.try_into().expect("len wrong size"))
    }

    pub fn pad_for_type(&mut self, ty: PrimType) {
        for _ in 0..ty.padding(self.inner.len()) {
            self.push(0);
        }
    }

    pub fn push(&mut self, elem: u8) {
        let len = self.pkt_len() + 1;
        self.inner.push(elem);
        self.inner[..8].copy_from_slice(&len.to_le_bytes());
    }

    pub fn extend_from_slice(&mut self, buf: &[u8]) {
        let len = self.pkt_len() + buf.len() as u64;
        self.inner.extend_from_slice(buf);
        self.inner[..8].copy_from_slice(&len.to_le_bytes());
    }

    pub fn as_packet(&self) -> &Packet {
        let len = self.pkt_len() as usize;
        Packet::try_ref_from_bytes_with_elems(&self.inner[8..], len)
            .expect("len packet was not a valid `Packet`")
    }

    pub fn as_mut_packet(&mut self) -> &mut Packet {
        let len = self.pkt_len() as usize;
        Packet::try_mut_from_bytes_with_elems(&mut self.inner[8..], len)
            .expect("len packet was not a valid `Packet`")
    }

    pub fn clear(&mut self) {
        self.inner[..8].copy_from_slice(&(PACKET_HEADER_LEN as u64).to_le_bytes());
        self.inner.truncate(PACKET_HEADER_LEN + 8);
    }
}

impl postcard::ser_flavors::Flavor for LenPacket {
    type Output = LenPacket;

    fn try_push(&mut self, data: u8) -> postcard::Result<()> {
        self.push(data);
        Ok(())
    }

    fn finalize(self) -> postcard::Result<Self::Output> {
        Ok(self)
    }
}

pub trait MsgExt: Msg {
    fn to_len_packet(&self) -> LenPacket {
        let msg = LenPacket::msg(Self::ID, 0);
        postcard::serialize_with_flavor(&self, msg).unwrap()
    }
}

impl<M: Msg> MsgExt for M {}

#[derive(Debug)]
pub enum OwnedPacket<B: IoBuf> {
    Msg(MsgBuf<B>),
    Table(OwnedTable<B>),
    TimeSeries(TimeSeries<B>),
}

impl<B: IoBuf> OwnedPacket<B> {
    pub fn parse(packet_buf: Slice<B>) -> Result<Self, Error> {
        let Packet { packet_ty, id, .. } = Packet::try_ref_from_bytes(&packet_buf)?;
        let packet_ty = *packet_ty;
        let id = *id;
        let buf = packet_buf
            .try_sub_slice(PACKET_HEADER_LEN..)
            .ok_or(Error::InvalidPacket)?;
        Ok(match packet_ty {
            PacketTy::Msg => OwnedPacket::Msg(MsgBuf { id, buf }),
            PacketTy::Table => OwnedPacket::Table(OwnedTable { id, buf }),
            PacketTy::TimeSeries => OwnedPacket::TimeSeries(TimeSeries { id, buf }),
        })
    }
}

#[derive(Debug)]
pub struct OwnedTable<B: IoBuf> {
    pub id: PacketId,
    pub buf: Slice<B>,
}

impl<B: IoBuf> OwnedTable<B> {
    pub fn sink(
        &self,
        registry: &impl crate::registry::VTableRegistry,
        sink: &mut impl crate::com_de::Decomponentize,
    ) -> Result<(), Error> {
        let vtable = registry.get(&self.id).ok_or(Error::VTableNotFound)?;
        vtable.parse_table(stellarator_buf::deref(&self.buf), sink)
    }
}

#[derive(Debug)]
pub struct MsgBuf<B: IoBuf> {
    pub id: PacketId,
    pub buf: Slice<B>,
}

impl<B: IoBuf> MsgBuf<B> {
    pub fn parse<'a, T: Deserialize<'a> + 'a>(&'a self) -> Result<T, Error> {
        let msg = postcard::from_bytes(stellarator_buf::deref(&self.buf))?;
        Ok(msg)
    }

    pub fn try_parse<'a, T: Msg + Deserialize<'a> + 'a>(&'a self) -> Option<Result<T, Error>> {
        if T::ID == self.id {
            return None;
        }
        Some(self.parse())
    }
}

#[derive(Debug)]
pub struct TimeSeries<B: IoBuf> {
    pub id: PacketId,
    pub buf: Slice<B>,
}

impl<B: IoBuf> OwnedPacket<B> {
    pub fn into_buf(self) -> B {
        match self {
            Self::Msg(msg_buf) => msg_buf.buf.into_inner(),
            Self::Table(table) => table.buf.into_inner(),
            Self::TimeSeries(time_series) => time_series.buf.into_inner(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padding() {
        assert_eq!(PrimType::F64.padding(1), 7);
        assert_eq!(PrimType::F64.padding(11), 5);
        assert_eq!(PrimType::F32.padding(2), 2);
        assert_eq!(PrimType::F32.padding(22), 2);
        assert_eq!(PrimType::F32.padding(0), 0);
        assert_eq!(PrimType::U8.padding(5), 0);
        assert_eq!(PrimType::U16.padding(12), 0);
        assert_eq!(PrimType::U16.padding(11), 1);
    }
}
