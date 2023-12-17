use std::{
    borrow::Cow,
    hash::{Hash, Hasher},
};

use nalgebra::{Matrix3, Quaternion, Vector3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentBatch<'a> {
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

    pub fn subscribe(filter: ComponentFilter) -> Self {
        ComponentData {
            component_id: SUB_COMPONENT_ID,
            storage: vec![(EntityId(0), ComponentValue::Filter(filter))],
        }
    }
}

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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ComponentId(pub u64);

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
    Filter,
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
    Filter(ComponentFilter),
}

impl<'a> ComponentValue<'a> {
    pub fn ty(&self) -> ComponentType {
        match self {
            ComponentValue::U8(_) => ComponentType::U8,
            ComponentValue::U16(_) => ComponentType::U16,
            ComponentValue::U32(_) => ComponentType::U32,
            ComponentValue::U64(_) => todo!(),
            ComponentValue::I8(_) => ComponentType::I8,
            ComponentValue::I16(_) => ComponentType::I16,
            ComponentValue::I32(_) => ComponentType::I32,
            ComponentValue::I64(_) => ComponentType::I64,
            ComponentValue::Bool(_) => ComponentType::Bool,
            ComponentValue::F32(_) => ComponentType::F32,
            ComponentValue::F64(_) => ComponentType::F64,
            ComponentValue::String(_) => ComponentType::String,
            ComponentValue::Bytes(_) => ComponentType::Bytes,
            ComponentValue::Vector3F32(_) => ComponentType::Vector3F32,
            ComponentValue::Vector3F64(_) => ComponentType::Vector3F64,
            ComponentValue::Matrix3x3F32(_) => ComponentType::Matrix3x3F32,
            ComponentValue::Matrix3x3F64(_) => ComponentType::Matrix3x3F64,
            ComponentValue::QuaternionF32(_) => ComponentType::QuaternionF32,
            ComponentValue::QuaternionF64(_) => ComponentType::QuaternionF64,
            ComponentValue::SpatialPosF32(_) => ComponentType::SpatialPosF32,
            ComponentValue::SpatialPosF64(_) => ComponentType::SpatialPosF64,
            ComponentValue::SpatialMotionF32(_) => ComponentType::SpatialMotionF32,
            ComponentValue::SpatialMotionF64(_) => ComponentType::SpatialMotionF64,
            ComponentValue::Filter(_) => ComponentType::Filter,
        }
    }
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
            ComponentValue::Filter(filter) => ComponentValue::Filter(filter),
        }
    }
}

impl<'a> ComponentBatch<'a> {
    pub fn into_owned(self) -> ComponentBatch<'static> {
        ComponentBatch {
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

#[macro_export]
macro_rules! raw_id {
    ($($oct: literal):*) => {
        {
            const fn cid() -> u64 {
                let mut out: u64 = 0;
                let mut i = 64;
                $(
                    assert!(i > 0, "too many octets");
                    i -= 8;
                    let o = $oct as u64;
                    out |= (o << i);
                )*
                out
            }
            const OUT: u64 = cid();
            OUT
        }
    };
    ($($oct: literal):*;$str:ident) => {
        {
            const fn cid() -> u64 {
                let mut out: u64 = 0;
                let mut i: u64 = 64;
                $(
                    assert!(i > 0, "too many octets");
                    i -= 8;
                    let o = $oct as u64;
                    out |= (o << i);
                )*
                let string: &'static str = std::stringify!($str);
                let hash = $crate::const_fnv1a_hash::fnv1a_hash_str_64(string);
                let mask = u64::MAX >> (64 - i);
                let hash = hash & mask;
                out | hash
            }
            const OUT: u64 = cid();
            OUT
        }
    };
}

#[macro_export]
macro_rules! cid {
    ($($oct: literal):*) => {
        {
            $crate::ComponentId($crate::raw_id!($($oct):*))
        }
    };
    ($($oct: literal):*;$str:ident) => {
        {
            $crate::ComponentId($crate::raw_id!($($oct):*;$str))
        }
    };
}

#[macro_export]
macro_rules! eid {
    ($($oct: literal):*) => {
        {
            $crate::EntityId($crate::raw_id!($($oct):*))
        }
    };
    ($($oct: literal):*;$str:ident) => {
        {
            $crate::EntityId($crate::raw_id!($($oct):*;$str))
        }
    };
}

#[macro_export]
macro_rules! cid_mask {
    ($($oct: literal):*) => {
        {
            let id = $crate::cid!($($oct):*);

                let mut mask_len = 0;
                $(
                    let _= $oct;
                    mask_len += 8;
                )*
                $crate::ComponentFilter { id: id.0, mask_len }
        }
    };
    ($($oct: literal):*;$str:ident) => {
        {
            let id = $crate::cid!($($oct):*;$str);
            $crate::ComponentFilter { id: id.0, mask_len: 64}
        }
    }

}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentFilter {
    pub id: u64,
    pub mask_len: u8,
}

impl ComponentFilter {
    const fn mask(&self) -> u64 {
        if self.mask_len == 0 {
            0
        } else {
            !0 << (64 - self.mask_len)
        }
    }
}

impl ComponentFilter {
    pub const fn apply(&self, ComponentId(id): ComponentId) -> bool {
        id & self.mask() == self.id
    }
}

macro_rules! impl_component_prim {
    ($varient: tt, $ty: ty, $id: expr) => {
        impl Component for $ty {
            fn component_id() -> ComponentId {
                $id
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

impl_component_prim!(U8, u8, cid!(31:0:1));
impl_component_prim!(U16, u16, cid!(31:0:2));
impl_component_prim!(U32, u32, cid!(31:0:3));
impl_component_prim!(U64, u64, cid!(31:0:4));
impl_component_prim!(I8, i8, cid!(31:0:5));
impl_component_prim!(I16, i16, cid!(31:0:6));
impl_component_prim!(I32, i32, cid!(31:0:7));
impl_component_prim!(I64, i64, cid!(31:0:8));
impl_component_prim!(F32, f32, cid!(31:0:9));
impl_component_prim!(F64, f64, cid!(31:0:10));
impl_component_prim!(Matrix3x3F32, nalgebra::Matrix3::<f32>, cid!(31:0:9));
impl_component_prim!(Matrix3x3F64, nalgebra::Matrix3::<f64>, cid!(31:0:9));
impl_component_prim!(Vector3F32, nalgebra::Vector3::<f32>, cid!(31:0:9));
impl_component_prim!(Vector3F64, nalgebra::Vector3::<f64>, cid!(31:0:9));
impl_component_prim!(QuaternionF32, nalgebra::Quaternion::<f32>, cid!(31:0:9));
impl_component_prim!(QuaternionF64, nalgebra::Quaternion::<f64>, cid!(31:0:9));

impl Component for Vec<u8> {
    fn component_id() -> ComponentId {
        cid!(31:0:10)
    }

    fn component_type() -> ComponentType {
        ComponentType::Bytes
    }

    fn component_value<'a>(&self) -> ComponentValue<'a> {
        ComponentValue::Bytes(Cow::Owned(self.clone()))
    }

    fn from_component_value(value: ComponentValue<'_>) -> Option<Self> {
        let ComponentValue::Bytes(v) = value else {
            return None;
        };
        Some(v.to_vec())
    }
}

pub const SUB_COMPONENT_ID: ComponentId = cid!(31;sub);

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_oct_macro() {
        assert_eq!(
            cid!(1:2:3:4:5:6:7:8),
            ComponentId(u64::from_be_bytes([1, 2, 3, 4, 5, 6, 7, 8]))
        );
        assert_eq!(
            cid!(1:2:3:4:5:6:7),
            ComponentId(u64::from_be_bytes([1, 2, 3, 4, 5, 6, 7, 0]))
        );

        assert_eq!(
            &cid!(31;test).0.to_be_bytes()[..],
            &[31, 230, 230, 239, 25, 124, 43, 37]
        );
        assert_eq!(
            &cid!(31:0;test).0.to_be_bytes()[..],
            [31, 0, 230, 239, 25, 124, 43, 37]
        );
    }

    #[test]
    fn test_mask_macro() {
        let filter = cid_mask!(31);
        assert_eq!(filter.id, u64::from_be_bytes([31, 0, 0, 0, 0, 0, 0, 0]));
        assert_eq!(
            filter.mask(),
            u64::from_be_bytes([255, 0, 0, 0, 0, 0, 0, 0])
        );
        assert!(filter.apply(cid!(31:2:3:4)));
        assert!(!filter.apply(cid!(32:2:3:4)));
        let filter = cid_mask!(31;test);
        assert!(filter.apply(cid!(31;test)));
        assert!(!filter.apply(cid!(31;foo)));
    }
}
