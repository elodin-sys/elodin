use std::str::FromStr;

use crate::error::{Error, Result};
use num_derive::FromPrimitive;

pub trait ArrayElement: Copy {
    const TY: ElementType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

/// The primitive types supported by XLA. `S8` is a signed 1 byte integer,
/// `U32` is an unsigned 4 bytes integer, etc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, FromPrimitive)]
#[repr(i32)]
pub enum PrimitiveType {
    Invalid = 0,
    Pred = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    Bf16 = 16,
    F64 = 12,
    C64 = 15,
    C128 = 18,
    Tuple = 13,
    OpaqueType = 14,
    Token = 17,
}

impl PrimitiveType {
    pub fn element_type(self) -> Result<ElementType> {
        match self {
            Self::Pred => Ok(ElementType::Pred),
            Self::S8 => Ok(ElementType::S8),
            Self::S16 => Ok(ElementType::S16),
            Self::S32 => Ok(ElementType::S32),
            Self::S64 => Ok(ElementType::S64),
            Self::U8 => Ok(ElementType::U8),
            Self::U16 => Ok(ElementType::U16),
            Self::U32 => Ok(ElementType::U32),
            Self::U64 => Ok(ElementType::U64),
            Self::F16 => Ok(ElementType::F16),
            Self::F32 => Ok(ElementType::F32),
            Self::Bf16 => Ok(ElementType::Bf16),
            Self::F64 => Ok(ElementType::F64),
            Self::C64 => Ok(ElementType::C64),
            Self::C128 => Ok(ElementType::C128),
            Self::Invalid | Self::Tuple | Self::OpaqueType | Self::Token => {
                Err(Error::NotAnElementType { got: self })
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    Bf16,
    F64,
    C64,
    C128,
}

impl FromStr for ElementType {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "bool" => Ok(Self::Pred),
            "int8" => Ok(Self::S8),
            "int16" => Ok(Self::S16),
            "int32" => Ok(Self::S32),
            "int64" => Ok(Self::S64),
            "uint8" => Ok(Self::U8),
            "uint16" => Ok(Self::U16),
            "uint32" => Ok(Self::U32),
            "uint64" => Ok(Self::U64),
            "float16" => Ok(Self::F16),
            "float32" => Ok(Self::F32),
            "bf16" => Ok(Self::Bf16),
            "float64" => Ok(Self::F64),
            "complex64" => Ok(Self::C64),
            "complex128" => Ok(Self::C128),
            _ => Err(()),
        }
    }
}

impl ElementType {
    /// The size for this element type in bytes.
    pub fn element_size_in_bytes(&self) -> usize {
        match self {
            Self::Pred => 1,
            Self::S8 => 1,
            Self::S16 => 2,
            Self::S32 => 4,
            Self::S64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::F64 => 8,
            Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            Self::Pred => PrimitiveType::Pred,
            Self::S8 => PrimitiveType::S8,
            Self::S16 => PrimitiveType::S16,
            Self::S32 => PrimitiveType::S32,
            Self::S64 => PrimitiveType::S64,
            Self::U8 => PrimitiveType::U8,
            Self::U16 => PrimitiveType::U16,
            Self::U32 => PrimitiveType::U32,
            Self::U64 => PrimitiveType::U64,
            Self::F16 => PrimitiveType::F16,
            Self::F32 => PrimitiveType::F32,
            Self::Bf16 => PrimitiveType::Bf16,
            Self::F64 => PrimitiveType::F64,
            Self::C64 => PrimitiveType::C64,
            Self::C128 => PrimitiveType::C128,
        }
    }
}

// Dummy F16 type.
#[derive(Copy, Clone, Debug)]
pub struct F16;

impl ArrayElement for F16 {
    const TY: ElementType = ElementType::F16;
    const ELEMENT_SIZE_IN_BYTES: usize = 2;
    const ZERO: Self = Self;
}

// Dummy BF16 type.
#[derive(Copy, Clone, Debug)]
pub struct Bf16;

impl ArrayElement for Bf16 {
    const TY: ElementType = ElementType::Bf16;
    const ELEMENT_SIZE_IN_BYTES: usize = 2;
    const ZERO: Self = Self;
}

macro_rules! element_type {
    ($ty:ty, $v:ident, $sz:tt) => {
        impl ArrayElement for $ty {
            const TY: ElementType = ElementType::$v;
            const ELEMENT_SIZE_IN_BYTES: usize = $sz;
            const ZERO: Self = 0 as Self;
        }
    };
}

element_type!(u8, U8, 1);
element_type!(u16, U16, 2);
element_type!(u32, U32, 4);
element_type!(u64, U64, 8);
element_type!(i8, S8, 1);
element_type!(i16, S16, 2);
element_type!(i32, S32, 4);
element_type!(i64, S64, 8);
element_type!(f32, F32, 4);
element_type!(f64, F64, 8);
