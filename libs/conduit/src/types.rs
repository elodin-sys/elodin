use bytes::Bytes;
use ndarray::{CowArray, IxDyn};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    hash::Hash,
    mem::size_of,
    time::Duration,
};

use crate::query::MetadataStore;

#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    bevy::prelude::Component,
)]
#[repr(transparent)]
pub struct EntityId(pub u64);

impl EntityId {
    pub fn component_id() -> ComponentId {
        ComponentId::new("entity_id")
    }
}

impl From<u64> for EntityId {
    fn from(val: u64) -> Self {
        EntityId(val)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ComponentId(pub u64);

impl ComponentId {
    pub const fn new(str: &str) -> Self {
        ComponentId(crate::const_fnv1a_hash::fnv1a_hash_str_64(str))
    }
}

impl From<u64> for ComponentId {
    fn from(val: u64) -> Self {
        ComponentId(val)
    }
}

#[derive(
    Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[repr(transparent)]
pub struct GraphId(pub u64);

impl From<u64> for GraphId {
    fn from(val: u64) -> Self {
        GraphId(val)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct AssetId(pub u64);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct StreamId(pub u32);

impl StreamId {
    pub const CONTROL: StreamId = StreamId(0);

    pub fn rand() -> Self {
        StreamId(fastrand::u32(1..))
    }
}

#[repr(u8)]
#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, IntoPrimitive, TryFromPrimitive, PartialEq, Eq,
)]
pub enum PrimitiveTy {
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

impl fmt::Display for PrimitiveTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimitiveTy::U8 => write!(f, "u8"),
            PrimitiveTy::U16 => write!(f, "u16"),
            PrimitiveTy::U32 => write!(f, "u32"),
            PrimitiveTy::U64 => write!(f, "u64"),
            PrimitiveTy::I8 => write!(f, "i8"),
            PrimitiveTy::I16 => write!(f, "i16"),
            PrimitiveTy::I32 => write!(f, "i32"),
            PrimitiveTy::I64 => write!(f, "i64"),
            PrimitiveTy::Bool => write!(f, "bool"),
            PrimitiveTy::F32 => write!(f, "f32"),
            PrimitiveTy::F64 => write!(f, "f64"),
        }
    }
}

impl PrimitiveTy {
    fn size(&self) -> usize {
        match self {
            PrimitiveTy::U8 => size_of::<u8>(),
            PrimitiveTy::U16 => size_of::<u16>(),
            PrimitiveTy::U32 => size_of::<u32>(),
            PrimitiveTy::U64 => size_of::<u64>(),
            PrimitiveTy::I8 => size_of::<i8>(),
            PrimitiveTy::I16 => size_of::<i16>(),
            PrimitiveTy::I32 => size_of::<i32>(),
            PrimitiveTy::I64 => size_of::<i64>(),
            PrimitiveTy::Bool => size_of::<bool>(),
            PrimitiveTy::F32 => size_of::<f32>(),
            PrimitiveTy::F64 => size_of::<f64>(),
        }
    }
}

type Shape = SmallVec<[i64; 4]>;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ComponentType {
    pub primitive_ty: PrimitiveTy,
    pub shape: Shape,
}

impl fmt::Display for ComponentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.is_empty() {
            return write!(f, "{}", self.primitive_ty);
        }
        write!(f, "{}:[", self.primitive_ty)?;
        for (i, dim) in self.shape.iter().enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl ComponentType {
    pub fn size(&self) -> usize {
        if self.shape.is_empty() {
            return self.primitive_ty.size();
        }
        self.shape.iter().map(|i| *i as usize).product::<usize>() * self.primitive_ty.size()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComponentValue<'a> {
    U8(ndarray::CowArray<'a, u8, IxDyn>),
    U16(ndarray::CowArray<'a, u16, IxDyn>),
    U32(ndarray::CowArray<'a, u32, IxDyn>),
    U64(ndarray::CowArray<'a, u64, IxDyn>),
    I8(ndarray::CowArray<'a, i8, IxDyn>),
    I16(ndarray::CowArray<'a, i16, IxDyn>),
    I32(ndarray::CowArray<'a, i32, IxDyn>),
    I64(ndarray::CowArray<'a, i64, IxDyn>),
    Bool(ndarray::CowArray<'a, bool, IxDyn>),
    F32(ndarray::CowArray<'a, f32, IxDyn>),
    F64(ndarray::CowArray<'a, f64, IxDyn>),
}

impl<'a> ComponentValue<'a> {
    pub fn primitive_ty(&self) -> PrimitiveTy {
        match self {
            ComponentValue::U8(_) => PrimitiveTy::U8,
            ComponentValue::U16(_) => PrimitiveTy::U16,
            ComponentValue::U32(_) => PrimitiveTy::U32,
            ComponentValue::U64(_) => PrimitiveTy::U64,
            ComponentValue::I8(_) => PrimitiveTy::I8,
            ComponentValue::I16(_) => PrimitiveTy::I16,
            ComponentValue::I32(_) => PrimitiveTy::I32,
            ComponentValue::I64(_) => PrimitiveTy::I64,
            ComponentValue::Bool(_) => PrimitiveTy::Bool,
            ComponentValue::F32(_) => PrimitiveTy::F32,
            ComponentValue::F64(_) => PrimitiveTy::F64,
        }
    }

    pub fn shape(&self) -> SmallVec<[i64; 4]> {
        match self {
            ComponentValue::U8(a) => a.shape(),
            ComponentValue::U16(a) => a.shape(),
            ComponentValue::U32(a) => a.shape(),
            ComponentValue::U64(a) => a.shape(),
            ComponentValue::I8(a) => a.shape(),
            ComponentValue::I16(a) => a.shape(),
            ComponentValue::I32(a) => a.shape(),
            ComponentValue::I64(a) => a.shape(),
            ComponentValue::Bool(a) => a.shape(),
            ComponentValue::F32(a) => a.shape(),
            ComponentValue::F64(a) => a.shape(),
        }
        .iter()
        .map(|n| *n as _)
        .collect()
    }

    pub fn ty(&self) -> ComponentType {
        ComponentType {
            primitive_ty: self.primitive_ty(),
            shape: self.shape(),
        }
    }

    pub fn into_owned(self) -> ComponentValue<'static> {
        match self {
            ComponentValue::U8(a) => ComponentValue::U8(CowArray::from(a.into_owned())),
            ComponentValue::U16(a) => ComponentValue::U16(CowArray::from(a.into_owned())),
            ComponentValue::U32(a) => ComponentValue::U32(CowArray::from(a.into_owned())),
            ComponentValue::U64(a) => ComponentValue::U64(CowArray::from(a.into_owned())),
            ComponentValue::I8(a) => ComponentValue::I8(CowArray::from(a.into_owned())),
            ComponentValue::I16(a) => ComponentValue::I16(CowArray::from(a.into_owned())),
            ComponentValue::I32(a) => ComponentValue::I32(CowArray::from(a.into_owned())),
            ComponentValue::I64(a) => ComponentValue::I64(CowArray::from(a.into_owned())),
            ComponentValue::Bool(a) => ComponentValue::Bool(CowArray::from(a.into_owned())),
            ComponentValue::F32(a) => ComponentValue::F32(CowArray::from(a.into_owned())),
            ComponentValue::F64(a) => ComponentValue::F64(CowArray::from(a.into_owned())),
        }
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            ComponentValue::U8(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::U16(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::U32(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::U64(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::I8(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::I16(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::I32(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::I64(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::Bool(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::F32(b) => b.as_slice().map(bytemuck::cast_slice),
            ComponentValue::F64(b) => b.as_slice().map(bytemuck::cast_slice),
        }
    }

    pub fn iter<'i>(&'i self) -> Box<dyn Iterator<Item = ElementValue> + 'i> {
        match self {
            ComponentValue::U8(u8) => Box::new(u8.iter().map(|&x| ElementValue::U8(x))),
            ComponentValue::U16(u16) => Box::new(u16.iter().map(|&x| ElementValue::U16(x))),
            ComponentValue::U32(u32) => Box::new(u32.iter().map(|&x| ElementValue::U32(x))),
            ComponentValue::U64(u64) => Box::new(u64.iter().map(|&x| ElementValue::U64(x))),
            ComponentValue::I8(i8) => Box::new(i8.iter().map(|&x| ElementValue::I8(x))),
            ComponentValue::I16(i16) => Box::new(i16.iter().map(|&x| ElementValue::I16(x))),
            ComponentValue::I32(i32) => Box::new(i32.iter().map(|&x| ElementValue::I32(x))),
            ComponentValue::I64(i64) => Box::new(i64.iter().map(|&x| ElementValue::I64(x))),
            ComponentValue::Bool(bool) => Box::new(bool.iter().map(|&x| ElementValue::Bool(x))),
            ComponentValue::F32(f32) => Box::new(f32.iter().map(|&x| ElementValue::F32(x))),
            ComponentValue::F64(f64) => Box::new(f64.iter().map(|&x| ElementValue::F64(x))),
        }
    }

    pub fn iter_mut<'i>(&'i mut self) -> Box<dyn Iterator<Item = ElementValueMut<'i>> + 'i> {
        match self {
            ComponentValue::U8(u8) => Box::new(u8.iter_mut().map(ElementValueMut::U8)),
            ComponentValue::U16(u16) => Box::new(u16.iter_mut().map(ElementValueMut::U16)),
            ComponentValue::U32(u32) => Box::new(u32.iter_mut().map(ElementValueMut::U32)),
            ComponentValue::U64(u64) => Box::new(u64.iter_mut().map(ElementValueMut::U64)),
            ComponentValue::I8(i8) => Box::new(i8.iter_mut().map(ElementValueMut::I8)),
            ComponentValue::I16(i16) => Box::new(i16.iter_mut().map(ElementValueMut::I16)),
            ComponentValue::I32(i32) => Box::new(i32.iter_mut().map(ElementValueMut::I32)),
            ComponentValue::I64(i64) => Box::new(i64.iter_mut().map(ElementValueMut::I64)),
            ComponentValue::Bool(bool) => Box::new(bool.iter_mut().map(ElementValueMut::Bool)),
            ComponentValue::F32(f32) => Box::new(f32.iter_mut().map(ElementValueMut::F32)),
            ComponentValue::F64(f64) => Box::new(f64.iter_mut().map(ElementValueMut::F64)),
        }
    }

    pub fn indexed_iter_mut<'i>(
        &'i mut self,
    ) -> Box<dyn Iterator<Item = (IxDyn, ElementValueMut<'i>)> + 'i> {
        match self {
            ComponentValue::U8(u8) => Box::new(
                u8.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U8(x))),
            ),
            ComponentValue::U16(u16) => Box::new(
                u16.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U16(x))),
            ),
            ComponentValue::U32(u32) => Box::new(
                u32.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U32(x))),
            ),
            ComponentValue::U64(u64) => Box::new(
                u64.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::U64(x))),
            ),
            ComponentValue::I8(i8) => Box::new(
                i8.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I8(x))),
            ),
            ComponentValue::I16(i16) => Box::new(
                i16.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I16(x))),
            ),
            ComponentValue::I32(i32) => Box::new(
                i32.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I32(x))),
            ),
            ComponentValue::I64(i64) => Box::new(
                i64.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::I64(x))),
            ),
            ComponentValue::Bool(bool) => Box::new(
                bool.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::Bool(x))),
            ),
            ComponentValue::F32(f32) => Box::new(
                f32.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::F32(x))),
            ),
            ComponentValue::F64(f64) => Box::new(
                f64.indexed_iter_mut()
                    .map(|(i, x)| (i, ElementValueMut::F64(x))),
            ),
        }
    }
}

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

impl fmt::Display for ElementValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElementValue::U8(x) => write!(f, "{}", x),
            ElementValue::U16(x) => write!(f, "{}", x),
            ElementValue::U32(x) => write!(f, "{}", x),
            ElementValue::U64(x) => write!(f, "{}", x),
            ElementValue::I8(x) => write!(f, "{}", x),
            ElementValue::I16(x) => write!(f, "{}", x),
            ElementValue::I32(x) => write!(f, "{}", x),
            ElementValue::I64(x) => write!(f, "{}", x),
            ElementValue::F64(x) => write!(f, "{}", x),
            ElementValue::F32(x) => write!(f, "{}", x),
            ElementValue::Bool(x) => write!(f, "{}", x),
        }
    }
}

pub enum ElementValueMut<'a> {
    U8(&'a mut u8),
    U16(&'a mut u16),
    U32(&'a mut u32),
    U64(&'a mut u64),
    I8(&'a mut i8),
    I16(&'a mut i16),
    I32(&'a mut i32),
    I64(&'a mut i64),
    F64(&'a mut f64),
    F32(&'a mut f32),
    Bool(&'a mut bool),
}

pub trait Component {
    const NAME: &'static str;
    const ASSET: bool;

    fn component_id() -> ComponentId {
        ComponentId::new(Self::NAME)
    }
    fn component_type() -> ComponentType;
    fn component_value<'a>(&self) -> ComponentValue<'a>;
    fn from_component_value(value: ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized;
    fn metadata() -> Metadata {
        Metadata {
            component_id: Self::component_id(),
            component_type: Self::component_type(),
            tags: HashMap::new(),
            asset: Self::ASSET,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Packet<T = Bytes> {
    pub stream_id: StreamId,
    pub payload: T,
}

#[derive(Debug, PartialEq)]
pub enum Payload<B> {
    ControlMsg(ControlMsg),
    Column(ColumnPayload<B>),
}

impl<T> Packet<Payload<T>> {
    pub fn metadata(stream_id: StreamId, metadata: Metadata) -> Self {
        let payload = Payload::ControlMsg(ControlMsg::Metadata {
            stream_id,
            metadata,
        });
        Packet {
            stream_id: StreamId::CONTROL,
            payload,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Event))]
pub enum ControlMsg {
    Connect,
    StartSim {
        metadata_store: MetadataStore,
        time_step: Duration,
        entity_ids: HashSet<EntityId>,
    },
    Subscribe {
        query: Query,
    },
    Metadata {
        stream_id: StreamId,
        metadata: Metadata,
    },
    Asset {
        id: AssetId,
        entity_id: EntityId,
        bytes: Bytes,
    },
    SetPlaying(bool),
    Rewind(u64),
    Tick {
        tick: u64,
        max_tick: u64,
    },
    Exit,
}

impl ControlMsg {
    pub fn sub_component_id(id: ComponentId) -> Self {
        ControlMsg::Subscribe {
            query: Query::ComponentId(id),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Query {
    All,
    Metadata(MetadataQuery), // TODO(sphw): figure out
    ComponentId(ComponentId),
    With(ComponentId),
    And(Vec<Query>),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum MetadataQuery {
    And(Vec<MetadataQuery>),
    Equals(MetadataPair),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub component_id: ComponentId,
    pub component_type: ComponentType,
    pub tags: HashMap<String, TagValue>,
    pub asset: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, Hash, PartialEq, Eq)]
pub enum TagValue {
    Unit,
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
}

impl TagValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            TagValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Hash, PartialEq, Eq)]
pub struct MetadataPair(pub String, pub TagValue);

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Event))]
pub struct ColumnPayload<B = Bytes> {
    pub time: u64,
    pub len: u32,
    pub entity_buf: B,
    pub value_buf: B,
}

pub trait Asset: Serialize {
    const ASSET_ID: AssetId;
    fn asset_id(&self) -> AssetId;
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn test_component_ty_display() {
        let ty = ComponentType {
            primitive_ty: PrimitiveTy::F64,
            shape: smallvec![3, 4],
        };
        let shapeless_ty = ComponentType {
            primitive_ty: PrimitiveTy::F64,
            shape: smallvec![],
        };
        assert_eq!(ty.to_string(), "f64:[3,4]");
        assert_eq!(shapeless_ty.to_string(), "f64");
    }
}
