//! Provides a Rust-native literal type for storing constant data.
//!
//! This replaces the XLA-dependent Literal type for use with IREE-based execution.

use zerocopy::{FromBytes, Immutable, IntoBytes};
use alloc::vec::Vec;
use smallvec::SmallVec;

#[cfg(feature = "std")]
use std::str::FromStr;

/// Array element type enumeration.
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

#[cfg(feature = "std")]
impl FromStr for ElementType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
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
}

/// Trait for types that can be stored in arrays.
pub trait ArrayElement: Copy {
    const TY: ElementType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

macro_rules! impl_array_element {
    ($ty:ty, $v:ident, $sz:tt, $zero:expr) => {
        impl ArrayElement for $ty {
            const TY: ElementType = ElementType::$v;
            const ELEMENT_SIZE_IN_BYTES: usize = $sz;
            const ZERO: Self = $zero;
        }
    };
}

impl_array_element!(u8, U8, 1, 0);
impl_array_element!(u16, U16, 2, 0);
impl_array_element!(u32, U32, 4, 0);
impl_array_element!(u64, U64, 8, 0);
impl_array_element!(i8, S8, 1, 0);
impl_array_element!(i16, S16, 2, 0);
impl_array_element!(i32, S32, 4, 0);
impl_array_element!(i64, S64, 8, 0);
impl_array_element!(f32, F32, 4, 0.0);
impl_array_element!(f64, F64, 8, 0.0);
impl_array_element!(bool, Pred, 1, false);

/// Trait for types that can be directly converted to literals.
/// This replaces the XLA NativeType trait.
pub trait NativeType: ArrayElement + IntoBytes + Immutable {
    /// Create a scalar literal from this value.
    fn literal(self) -> Literal {
        Literal::scalar(self)
    }
    
    /// Create a 1D literal from a slice.
    fn create_r1(slice: &[Self]) -> Literal {
        Literal::vector(slice)
    }
}

impl NativeType for u8 {}
impl NativeType for u16 {}
impl NativeType for u32 {}
impl NativeType for u64 {}
impl NativeType for i8 {}
impl NativeType for i16 {}
impl NativeType for i32 {}
impl NativeType for i64 {}
impl NativeType for f32 {}
impl NativeType for f64 {}

/// A Rust-native literal type that stores constant data with type information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Literal {
    data: Vec<u8>,
    element_type: ElementType,
    shape: SmallVec<[i64; 4]>,
}

impl Default for Literal {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            element_type: ElementType::F32,
            shape: SmallVec::new(),
        }
    }
}

impl Literal {
    /// Create a new literal from raw bytes with type and shape information.
    pub fn new(data: Vec<u8>, element_type: ElementType, shape: SmallVec<[i64; 4]>) -> Self {
        Self {
            data,
            element_type,
            shape,
        }
    }

    /// Create a scalar literal from a single value.
    pub fn scalar<T: ArrayElement + IntoBytes + Immutable>(val: T) -> Self {
        Self {
            data: val.as_bytes().to_vec(),
            element_type: T::TY,
            shape: SmallVec::new(),
        }
    }

    /// Create a 1D vector literal from a slice.
    pub fn vector<T: ArrayElement + IntoBytes + Immutable>(vals: &[T]) -> Self {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.as_bytes().iter().copied()).collect();
        Self {
            data,
            element_type: T::TY,
            shape: smallvec::smallvec![vals.len() as i64],
        }
    }

    /// Create an N-dimensional array literal from a flat buffer and shape.
    pub fn array<T: ArrayElement + IntoBytes + Immutable>(vals: &[T], shape: SmallVec<[i64; 4]>) -> Self {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.as_bytes().iter().copied()).collect();
        Self {
            data,
            element_type: T::TY,
            shape,
        }
    }

    /// Get the raw byte buffer.
    pub fn raw_buf(&self) -> &[u8] {
        &self.data
    }

    /// Get the element type.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    /// Get the shape.
    pub fn shape(&self) -> &SmallVec<[i64; 4]> {
        &self.shape
    }

    /// Get the number of elements.
    pub fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product::<i64>() as usize
        }
    }

    /// Get a typed buffer view.
    pub fn typed_buf<T: ArrayElement + FromBytes + Immutable>(&self) -> Result<&[T], LiteralError> {
        if self.element_type != T::TY {
            return Err(LiteralError::ElementTypeMismatch {
                expected: T::TY,
                actual: self.element_type,
            });
        }
        <[T]>::ref_from_bytes(&self.data).map_err(|_| LiteralError::CastError)
    }

    /// Reshape the literal to a new shape.
    pub fn reshape(&self, new_shape: SmallVec<[i64; 4]>) -> Result<Self, LiteralError> {
        let old_size: i64 = if self.shape.is_empty() { 1 } else { self.shape.iter().product() };
        let new_size: i64 = if new_shape.is_empty() { 1 } else { new_shape.iter().product() };
        
        if old_size != new_size {
            return Err(LiteralError::ShapeMismatch { old_size, new_size });
        }
        
        Ok(Self {
            data: self.data.clone(),
            element_type: self.element_type,
            shape: new_shape,
        })
    }
}

/// Errors that can occur with Literal operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LiteralError {
    #[error("element type mismatch: expected {expected:?}, got {actual:?}")]
    ElementTypeMismatch {
        expected: ElementType,
        actual: ElementType,
    },
    
    #[error("failed to cast buffer")]
    CastError,
    
    #[error("shape mismatch: old size {old_size}, new size {new_size}")]
    ShapeMismatch {
        old_size: i64,
        new_size: i64,
    },
}
