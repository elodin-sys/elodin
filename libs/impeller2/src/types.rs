use core::{
    fmt::Display,
    mem,
    ops::{Add, AddAssign, Sub},
    sync::atomic::AtomicI64,
    time::Duration,
};

use nox::ArrayView;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, TryFromBytes, Unaligned};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{buf::ByteBufExt, error::Error};
use stellarator_buf::{AtomicValue, IoBuf, Slice};

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
    postcard_schema::Schema,
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
    postcard_schema::Schema,
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
#[serde(rename_all = "kebab-case")]
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
            PrimType::U8 => mem::align_of::<u8>(),
            PrimType::U16 => mem::align_of::<u16>(),
            PrimType::U32 => mem::align_of::<u32>(),
            PrimType::U64 => mem::align_of::<u64>(),
            PrimType::I8 => mem::align_of::<i8>(),
            PrimType::I16 => mem::align_of::<i16>(),
            PrimType::I32 => mem::align_of::<i32>(),
            PrimType::I64 => mem::align_of::<i64>(),
            PrimType::Bool => mem::align_of::<bool>(),
            PrimType::F32 => mem::align_of::<f32>(),
            PrimType::F64 => mem::align_of::<f64>(),
        }
    }

    pub const fn size(&self) -> usize {
        match self {
            PrimType::U8 => mem::size_of::<u8>(),
            PrimType::U16 => mem::size_of::<u16>(),
            PrimType::U32 => mem::size_of::<u32>(),
            PrimType::U64 => mem::size_of::<u64>(),
            PrimType::I8 => mem::size_of::<i8>(),
            PrimType::I16 => mem::size_of::<i16>(),
            PrimType::I32 => mem::size_of::<i32>(),
            PrimType::I64 => mem::size_of::<i64>(),
            PrimType::Bool => mem::size_of::<bool>(),
            PrimType::F32 => mem::size_of::<f32>(),
            PrimType::F64 => mem::size_of::<f64>(),
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

impl<'a> From<ComponentView<'a>> for i64 {
    fn from(value: ComponentView<'a>) -> Self {
        match value {
            ComponentView::I64(view) => view.buf()[0],
            _ => panic!("invalid component type"),
        }
    }
}

impl<'a> From<ComponentView<'a>> for f32 {
    fn from(value: ComponentView<'a>) -> Self {
        match value {
            ComponentView::F32(view) => view.buf()[0],
            _ => panic!("invalid component type"),
        }
    }
}

impl<'a> From<ComponentView<'a>> for [f32; 3] {
    fn from(value: ComponentView<'a>) -> Self {
        match value {
            ComponentView::F32(view) => view.buf().try_into().unwrap(),
            _ => panic!("invalid component type"),
        }
    }
}

impl core::fmt::Display for ComponentView<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ComponentView::U8(array) => {
                f.write_str("u8")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::U16(array) => {
                f.write_str("u16")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::U32(array) => {
                f.write_str("u32")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::U64(array) => {
                f.write_str("u64")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::I8(array) => {
                f.write_str("i8")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::I16(array) => {
                f.write_str("i16")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::I32(array) => {
                f.write_str("i32")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::I64(array) => {
                f.write_str("i64")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::Bool(array) => {
                f.write_str("bool")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::F32(array) => {
                f.write_str("f32")?;
                core::fmt::Display::fmt(array, f)
            }
            ComponentView::F64(array) => {
                f.write_str("f64")?;
                core::fmt::Display::fmt(array, f)
            }
        }
    }
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

#[derive(
    TryFromBytes, Unaligned, Immutable, KnownLayout, PartialEq, Debug, Clone, Copy, IntoBytes,
)]
#[repr(u8)]
pub enum PacketTy {
    Msg = 0,
    Table = 1,
    TimeSeries = 2,
}

pub type PacketId = [u8; 2];

pub type RequestId = u8;

pub const PACKET_HEADER_LEN: usize = 4;

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout, Debug)]
#[repr(C)]
pub struct PacketHeader {
    pub packet_ty: PacketTy,
    pub id: PacketId,
    pub req_id: RequestId,
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout, Debug)]
#[repr(C)]
pub struct Packet {
    pub header: PacketHeader,
    pub body: [u8],
}

#[derive(TryFromBytes, Unaligned, Immutable, KnownLayout)]
#[repr(C)]
pub struct TimeSeries {
    pub len: [u8; 8],
    pub body: [u8],
}

impl TimeSeries {
    fn len(&self) -> Result<usize, Error> {
        u64::from_le_bytes(self.len)
            .try_into()
            .map_err(|_| Error::OffsetOverflow)
    }
}

pub trait Msg: Serialize {
    const ID: PacketId;
}

impl<T: Serialize + postcard_schema::Schema> Msg for T {
    const ID: PacketId = const_fnv1a_hash::fnv1a_hash_str_16_xor(T::SCHEMA.name).to_le_bytes();
}

pub const fn msg_id(name: &str) -> PacketId {
    let bytes = const_fnv1a_hash::fnv1a_hash_str_16_xor(name).to_le_bytes();
    if bytes[0] == 224 {
        [223, bytes[1]]
    } else {
        bytes
    }
}

#[cfg(feature = "alloc")]
pub trait IntoLenPacket {
    fn into_len_packet(self) -> LenPacket;

    fn with_request_id(self, request_id: RequestId) -> LenPacket
    where
        Self: Sized,
    {
        self.into_len_packet().with_request_id(request_id)
    }
}

#[cfg(feature = "alloc")]
impl<M: Msg> IntoLenPacket for &'_ M {
    fn into_len_packet(self) -> LenPacket {
        let mut msg = LenPacket::msg(M::ID, 0);
        postcard::serialize_with_flavor(&self, &mut msg).expect("postcard failed");
        msg
    }
}

#[cfg(feature = "alloc")]
impl IntoLenPacket for LenPacket {
    fn into_len_packet(self) -> LenPacket {
        self
    }
}

#[cfg(feature = "alloc")]
#[derive(Clone)]
pub struct LenPacket {
    pub inner: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl LenPacket {
    pub fn new(ty: PacketTy, id: PacketId, cap: usize) -> Self {
        let mut inner = Vec::with_capacity(cap + 8);
        inner.extend_from_slice(&(PACKET_HEADER_LEN as u32).to_le_bytes());
        inner.push(ty as u8);
        inner.extend_from_slice(&id);
        inner.push(0);
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

    pub fn set_request_id(&mut self, request_id: u8) {
        self.inner[core::mem::offset_of!(Packet, header.req_id) + 4] = request_id;
    }

    pub fn with_request_id(mut self, request_id: u8) -> Self {
        self.inner[core::mem::offset_of!(Packet, header.req_id) + 4] = request_id;
        self
    }

    fn pkt_len(&self) -> u32 {
        let len = &self.inner[..4];
        u32::from_le_bytes(len.try_into().expect("len wrong size"))
    }

    pub fn pad_for_type(&mut self, ty: PrimType) {
        for _ in 0..ty.padding(self.inner.len()) {
            self.push(0);
        }
    }

    pub fn push(&mut self, elem: u8) {
        let len = self.pkt_len() + 1;
        self.inner.push(elem);
        self.inner[..4].copy_from_slice(&len.to_le_bytes());
    }

    pub fn push_aligned<V: zerocopy::IntoBytes + zerocopy::Immutable>(&mut self, val: V) {
        let old_len = self.inner.len();
        let _ = self.inner.push_aligned(val);
        let new_len = self.inner.len();
        let len = self.pkt_len() + (new_len - old_len) as u32;
        self.inner[..4].copy_from_slice(&len.to_le_bytes());
    }

    pub fn extend_aligned<V: zerocopy::IntoBytes + zerocopy::Immutable>(&mut self, val: &[V]) {
        let old_len = self.inner.len();
        let _ = self.inner.extend_aligned(val);
        let new_len = self.inner.len();
        let len = self.pkt_len() + (new_len - old_len) as u32;
        self.inner[..4].copy_from_slice(&len.to_le_bytes());
    }

    pub fn extend_from_slice(&mut self, buf: &[u8]) {
        let len = self.pkt_len() + buf.len() as u32;
        self.inner.extend_from_slice(buf);
        self.inner[..4].copy_from_slice(&len.to_le_bytes());
    }

    #[inline]
    pub fn as_packet(&self) -> &Packet {
        let len = self.pkt_len() as usize;
        Packet::try_ref_from_bytes_with_elems(
            &self.inner[4..],
            len.saturating_sub(PACKET_HEADER_LEN),
        )
        .expect("len packet was not a valid `Packet`")
    }

    #[inline]
    pub fn as_mut_packet(&mut self) -> &mut Packet {
        let len = self.pkt_len() as usize;
        Packet::try_mut_from_bytes_with_elems(
            &mut self.inner[4..],
            len.saturating_sub(PACKET_HEADER_LEN),
        )
        .expect("len packet was not a valid `Packet`")
    }

    pub fn clear(&mut self) {
        self.inner[..4].copy_from_slice(&(PACKET_HEADER_LEN as u32).to_le_bytes());
        self.inner.truncate(PACKET_HEADER_LEN + 4);
    }
}

#[cfg(feature = "alloc")]
impl postcard::ser_flavors::Flavor for &'_ mut LenPacket {
    type Output = ();

    fn try_push(&mut self, data: u8) -> postcard::Result<()> {
        self.push(data);
        Ok(())
    }

    fn finalize(self) -> postcard::Result<Self::Output> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum OwnedPacket<B: IoBuf> {
    Msg(MsgBuf<B>),
    Table(OwnedTable<B>),
    TimeSeries(OwnedTimeSeries<B>),
}

impl<B: IoBuf> OwnedPacket<B> {
    pub fn parse_with_offset(packet_buf: B, offset: usize) -> Result<Self, Error> {
        let buf = &stellarator_buf::deref(&packet_buf)[offset..];
        let Packet {
            header:
                PacketHeader {
                    packet_ty,
                    req_id,
                    id,
                },
            ..
        } = Packet::try_ref_from_bytes(buf)?;
        let packet_ty = *packet_ty;
        let id = *id;
        let req_id = *req_id;
        let buf = packet_buf
            .try_slice(PACKET_HEADER_LEN + offset..)
            .ok_or(Error::InvalidPacket)?;
        Ok(match packet_ty {
            PacketTy::Msg => OwnedPacket::Msg(MsgBuf { id, req_id, buf }),
            PacketTy::Table => OwnedPacket::Table(OwnedTable { id, req_id, buf }),
            PacketTy::TimeSeries => {
                let time_series = TimeSeries::try_ref_from_bytes(&buf)?;
                let len = time_series.len()?;
                OwnedPacket::TimeSeries(OwnedTimeSeries {
                    id,
                    req_id,
                    buf,
                    len,
                })
            }
        })
    }
    pub fn parse(packet_buf: B) -> Result<Self, Error> {
        Self::parse_with_offset(packet_buf, 0)
    }

    pub fn req_id(&self) -> RequestId {
        match self {
            OwnedPacket::Msg(msg_buf) => msg_buf.req_id,
            OwnedPacket::Table(table) => table.req_id,
            OwnedPacket::TimeSeries(time_series) => time_series.req_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OwnedTable<B: IoBuf> {
    pub id: PacketId,
    pub req_id: RequestId,
    pub buf: Slice<B>,
}

impl<B: IoBuf> OwnedTable<B> {
    pub fn sink<D: crate::com_de::Decomponentize>(
        &self,
        registry: &impl crate::registry::VTableRegistry,
        sink: &mut D,
    ) -> Result<Result<(), D::Error>, Error> {
        let vtable = registry.get(&self.id).ok_or(Error::VTableNotFound)?;
        vtable.apply(stellarator_buf::deref(&self.buf), sink)
    }
}

#[derive(Debug, Clone)]
pub struct MsgBuf<B: IoBuf> {
    pub id: PacketId,
    pub req_id: RequestId,
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

#[derive(Debug, Clone)]
pub struct OwnedTimeSeries<B: IoBuf> {
    pub id: PacketId,
    pub req_id: RequestId,
    pub len: usize,
    pub buf: Slice<B>,
}

impl<B: IoBuf> OwnedTimeSeries<B> {
    pub fn timestamps(&self) -> Result<&[Timestamp], Error> {
        let start = size_of::<u64>();
        let end = start + self.len * size_of::<i64>();
        let timestamps = stellarator_buf::deref(&self.buf)
            .get(start..end)
            .ok_or(Error::BufferUnderflow)?;
        Ok(<[Timestamp]>::ref_from_bytes(timestamps)?)
    }

    pub fn data(&self) -> Result<&[u8], Error> {
        let start = size_of::<u64>();
        let end = start + self.len * size_of::<i64>();
        stellarator_buf::deref(&self.buf)
            .get(end..)
            .ok_or(Error::BufferUnderflow)
    }
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

#[derive(
    Copy,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    IntoBytes,
    Immutable,
    FromBytes,
    Serialize,
    Deserialize,
    Default,
    postcard_schema::Schema,
)]
#[repr(transparent)]
pub struct Timestamp(pub i64);

#[cfg(feature = "hifitime")]
impl From<Timestamp> for hifitime::Epoch {
    fn from(val: Timestamp) -> Self {
        hifitime::Epoch::from_unix_milliseconds((val.0 as f64) / 1000.0)
    }
}

#[cfg(feature = "hifitime")]
impl From<hifitime::Epoch> for Timestamp {
    fn from(epoch: hifitime::Epoch) -> Self {
        Timestamp((epoch.to_unix_milliseconds() * 1000.0) as i64)
    }
}

impl Timestamp {
    pub const EPOCH: Timestamp = Timestamp(0);

    #[cfg(feature = "std")]
    pub fn now() -> Self {
        std::time::SystemTime::now().into()
    }

    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(i64::from_le_bytes(bytes))
    }

    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }
}

impl Add<Duration> for Timestamp {
    type Output = Timestamp;

    fn add(self, rhs: Duration) -> Self::Output {
        Timestamp(self.0 + rhs.as_micros() as i64)
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Timestamp;

    fn sub(self, rhs: Duration) -> Self::Output {
        Timestamp(self.0 - rhs.as_micros() as i64)
    }
}

impl AddAssign<Duration> for Timestamp {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.as_micros() as i64;
    }
}

impl AtomicValue for Timestamp {
    type Atomic = AtomicI64;
    type Value = i64;

    fn atomic(self) -> Self::Atomic {
        AtomicI64::new(self.0)
    }

    fn load(atomic: &Self::Atomic, order: core::sync::atomic::Ordering) -> Self {
        Timestamp(atomic.load(order))
    }

    fn store(atomic: &Self::Atomic, val: Self, order: core::sync::atomic::Ordering) {
        atomic.store(val.0, order);
    }

    fn swap(atomic: &Self::Atomic, val: Self, order: core::sync::atomic::Ordering) -> Self {
        Timestamp(atomic.swap(val.0, order))
    }

    fn from_value(val: Self::Value) -> Self {
        Timestamp(val)
    }
}

#[cfg(feature = "std")]
impl From<std::time::SystemTime> for Timestamp {
    fn from(value: std::time::SystemTime) -> Self {
        match value.duration_since(std::time::SystemTime::UNIX_EPOCH) {
            Ok(dur) => Self(dur.as_micros() as i64),
            Err(err) => Self(-(err.duration().as_micros() as i64)),
        }
    }
}

pub trait Request {
    type Reply<B: IoBuf + Clone>: TryFromPacket<B>;
}

pub trait TryFromPacket<B: IoBuf>: Sized {
    fn try_from_packet(packet: &OwnedPacket<B>) -> Result<Self, Error>;
}

impl<M: Msg + DeserializeOwned, B: IoBuf> TryFromPacket<B> for M {
    fn try_from_packet(packet: &OwnedPacket<B>) -> Result<Self, Error> {
        match packet {
            OwnedPacket::Msg(m) if m.id == M::ID => m.parse::<M>(),
            _ => Err(Error::InvalidPacket),
        }
    }
}

impl<B: IoBuf + Clone> TryFromPacket<B> for MsgBuf<B> {
    fn try_from_packet(packet: &OwnedPacket<B>) -> Result<Self, Error> {
        match packet {
            OwnedPacket::Msg(m) => Ok(m.clone()),
            _ => Err(Error::InvalidPacket),
        }
    }
}

impl<B: IoBuf + Clone> TryFromPacket<B> for OwnedTimeSeries<B> {
    fn try_from_packet(packet: &OwnedPacket<B>) -> Result<Self, Error> {
        match packet {
            OwnedPacket::TimeSeries(m) => Ok(m.clone()),
            _ => Err(Error::InvalidPacket),
        }
    }
}

impl<B: IoBuf + Clone> TryFromPacket<B> for OwnedPacket<B> {
    fn try_from_packet(packet: &OwnedPacket<B>) -> Result<Self, Error> {
        Ok(packet.clone())
    }
}

impl<B: IoBuf + Clone> TryFromPacket<B> for OwnedTable<B> {
    fn try_from_packet(packet: &OwnedPacket<B>) -> Result<Self, Error> {
        match packet {
            OwnedPacket::Table(m) => Ok(m.clone()),
            _ => Err(Error::InvalidPacket),
        }
    }
}

impl<R: Request> Request for &'_ R {
    type Reply<B: IoBuf + Clone> = R::Reply<B>;
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
