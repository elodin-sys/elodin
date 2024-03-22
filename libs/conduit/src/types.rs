use bytes::Bytes;
use ndarray::{CowArray, IxDyn};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{collections::HashMap, fmt, hash::Hash, mem::size_of, time::Duration};

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
}

pub trait Component {
    const NAME: &'static str;
    fn component_id() -> ComponentId {
        let name = Self::NAME;
        let ty = Self::component_type();
        ComponentId::new(&format!("{name}:{ty}"))
    }
    fn component_type() -> ComponentType;
    fn component_value<'a>(&self) -> ComponentValue<'a>;
    fn from_component_value(value: ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized;
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
