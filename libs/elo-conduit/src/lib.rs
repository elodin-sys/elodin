use std::{
    borrow::Cow,
    hash::{Hash, Hasher},
};

use nalgebra::{Matrix3, Quaternion, Vector3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "bevy")]
pub mod bevy;
#[cfg(feature = "tokio")]
pub mod tokio;

pub mod builder;
pub mod parser;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataMsg<'a> {
    pub time: u64,
    pub components: Vec<ComponentData<'a>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentData<'a> {
    pub component_id: ComponentId,
    pub storage: Vec<(EntityId, ComponentValue<'a>)>,
}

impl<'a> ComponentData<'a> {
    pub fn into_owned(self) -> ComponentData<'static> {
        ComponentData {
            component_id: self.component_id,
            storage: self
                .storage
                .into_iter()
                .map(|(id, v)| (id, v.into_owned()))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EntityId(u64);

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ComponentId(u64);

impl From<u64> for ComponentId {
    fn from(val: u64) -> Self {
        ComponentId(val)
    }
}

impl From<String> for ComponentId {
    fn from(val: String) -> Self {
        let mut hasher = ahash::AHasher::default();
        val.hash(&mut hasher);
        ComponentId(hasher.finish())
    }
}

impl<'a> From<&'a [u8]> for ComponentId {
    fn from(val: &'a [u8]) -> Self {
        let mut hasher = ahash::AHasher::default();
        val.hash(&mut hasher);
        ComponentId(hasher.finish())
    }
}

impl From<Vec<u8>> for ComponentId {
    fn from(val: Vec<u8>) -> Self {
        let mut hasher = ahash::AHasher::default();
        val.hash(&mut hasher);
        ComponentId(hasher.finish())
    }
}

impl<const N: usize> From<[u8; N]> for ComponentId {
    fn from(val: [u8; N]) -> Self {
        let mut hasher = ahash::AHasher::default();
        val.hash(&mut hasher);
        ComponentId(hasher.finish())
    }
}

impl From<u64> for EntityId {
    fn from(val: u64) -> Self {
        EntityId(val)
    }
}

#[repr(u8)]
#[derive(Clone, Debug, Serialize, Deserialize, IntoPrimitive, TryFromPrimitive)]
pub enum ComponentType {
    // Primatives
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

    // Variable Size
    String,
    Bytes,

    // Tensors
    Vector3F32,
    Vector3F64,
    Matrix3x3F32,
    Matrix3x3F64,
    QuaternionF32,
    QuaternionF64,
    SpatialPosF32,
    SpatialPosF64,
    SpatialMotionF32,
    SpatialMotionF64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ComponentValue<'a> {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Bool(bool),
    F32(f32),
    F64(f64),

    // Variable Size
    String(Cow<'a, str>),
    Bytes(std::borrow::Cow<'a, [u8]>),

    // Tensors
    Vector3F32(Vector3<f32>),
    Vector3F64(Vector3<f64>),
    Matrix3x3F32(Matrix3<f32>),
    Matrix3x3F64(Matrix3<f64>),
    QuaternionF32(Quaternion<f32>),
    QuaternionF64(Quaternion<f64>),
    SpatialPosF32((Quaternion<f32>, Vector3<f32>)),
    SpatialPosF64((Quaternion<f64>, Vector3<f64>)),
    SpatialMotionF32((Vector3<f32>, Vector3<f32>)),
    SpatialMotionF64((Vector3<f64>, Vector3<f64>)),
}

impl<'a> ComponentValue<'a> {
    pub fn into_owned(self) -> ComponentValue<'static> {
        match self {
            ComponentValue::String(s) => ComponentValue::String(Cow::Owned(s.into_owned())),
            ComponentValue::Bytes(b) => ComponentValue::Bytes(Cow::Owned(b.into_owned())),
            ComponentValue::U8(a) => ComponentValue::U8(a),
            ComponentValue::U16(a) => ComponentValue::U16(a),
            ComponentValue::U32(a) => ComponentValue::U32(a),
            ComponentValue::U64(a) => ComponentValue::U64(a),
            ComponentValue::I8(a) => ComponentValue::I8(a),
            ComponentValue::I16(a) => ComponentValue::I16(a),
            ComponentValue::I32(a) => ComponentValue::I32(a),
            ComponentValue::I64(a) => ComponentValue::I64(a),
            ComponentValue::Bool(a) => ComponentValue::Bool(a),
            ComponentValue::F32(a) => ComponentValue::F32(a),
            ComponentValue::F64(a) => ComponentValue::F64(a),
            ComponentValue::Vector3F32(a) => ComponentValue::Vector3F32(a),
            ComponentValue::Vector3F64(a) => ComponentValue::Vector3F64(a),
            ComponentValue::Matrix3x3F32(a) => ComponentValue::Matrix3x3F32(a),
            ComponentValue::Matrix3x3F64(a) => ComponentValue::Matrix3x3F64(a),
            ComponentValue::QuaternionF32(a) => ComponentValue::QuaternionF32(a),
            ComponentValue::QuaternionF64(a) => ComponentValue::QuaternionF64(a),
            ComponentValue::SpatialPosF32(a) => ComponentValue::SpatialPosF32(a),
            ComponentValue::SpatialPosF64(a) => ComponentValue::SpatialPosF64(a),
            ComponentValue::SpatialMotionF32(a) => ComponentValue::SpatialMotionF32(a),
            ComponentValue::SpatialMotionF64(a) => ComponentValue::SpatialMotionF64(a),
        }
    }
}

impl<'a> DataMsg<'a> {
    pub fn into_owned(self) -> DataMsg<'static> {
        DataMsg {
            time: self.time,
            components: self
                .components
                .into_iter()
                .map(ComponentData::into_owned)
                .collect(),
        }
    }
}

pub trait Component {
    fn component_id() -> ComponentId;
    fn component_type() -> ComponentType;
    fn component_value<'a>(&self) -> ComponentValue<'a>;
    fn from_component_value(value: ComponentValue<'_>) -> Option<Self>
    where
        Self: Sized;
}

macro_rules! impl_component_copy {
    ($varient: tt, $ty: ty, $id: tt) => {
        impl Component for $ty {
            fn component_id() -> ComponentId {
                ComponentId($id)
            }

            fn component_type() -> ComponentType {
                ComponentType::$varient
            }

            fn component_value<'a>(&self) -> ComponentValue<'a> {
                ComponentValue::$varient(*self)
            }

            fn from_component_value(value: ComponentValue<'_>) -> Option<Self> {
                let ComponentValue::$varient(v) = value else {
                    return None;
                };
                Some(v)
            }
        }
    };
}

impl_component_copy!(U8, u8, 0x0003_1001);
impl_component_copy!(U16, u16, 0x0003_1002);
impl_component_copy!(U32, u32, 0x0003_1003);
impl_component_copy!(U64, u64, 0x0003_1004);
impl_component_copy!(I8, i8, 0x0003_1005);
impl_component_copy!(I16, i16, 0x0003_1006);
impl_component_copy!(I32, i32, 0x0003_1007);
impl_component_copy!(I64, i64, 0x0003_1008);
impl_component_copy!(F32, f32, 0x0003_1007);
impl_component_copy!(F64, f64, 0x0003_1008);

#[derive(Debug, Error)]
pub enum Error {
    #[error("postcard {0}")]
    Postcard(#[from] postcard::Error),
    #[error("io {0}")]
    Io(#[from] std::io::Error),
    #[error("entity and component iters length must match")]
    EntityComponentLengthMismatch,
    #[error("buffer overflow")]
    BufferOverflow,
    #[error("parsing error")]
    ParsingError,
}
