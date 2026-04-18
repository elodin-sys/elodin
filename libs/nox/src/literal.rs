use alloc::vec::Vec;
use core::str::FromStr;
use smallvec::SmallVec;
use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::Error;

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

pub trait ArrayElement: Copy {
    const TY: ElementType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

pub trait NativeType: ArrayElement + IntoBytes + Immutable + FromBytes {
    fn literal(self) -> Literal;
    fn create_r1(slice: &[Self]) -> Literal;
}

#[derive(Clone, Debug)]
pub struct Literal {
    data: Vec<u8>,
    element_type: ElementType,
    shape: SmallVec<[i64; 4]>,
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        self.element_type == other.element_type
            && self.shape == other.shape
            && self.data == other.data
    }
}

impl Eq for Literal {}

impl Literal {
    pub fn new(data: Vec<u8>, element_type: ElementType, shape: SmallVec<[i64; 4]>) -> Self {
        Self {
            data,
            element_type,
            shape,
        }
    }

    pub fn raw_buf(&self) -> &[u8] {
        &self.data
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn shape(&self) -> &SmallVec<[i64; 4]> {
        &self.shape
    }

    pub fn element_count(&self) -> usize {
        self.shape
            .iter()
            .map(|&d| d as usize)
            .product::<usize>()
            .max(1)
    }

    pub fn typed_buf<T: ArrayElement + FromBytes + Immutable>(&self) -> Result<&[T], Error> {
        if self.element_type != T::TY {
            return Err(Error::IncompatibleDType);
        }
        <[T]>::ref_from_bytes(&self.data).map_err(|_| Error::IncompatibleDType)
    }

    pub fn reshape(&self, dims: &[i64]) -> Result<Literal, Error> {
        let new_count: i64 = dims.iter().product();
        let old_count: i64 = self.shape.iter().product::<i64>().max(1);
        if new_count != old_count {
            return Err(Error::IncompatibleDType);
        }
        Ok(Literal {
            data: self.data.clone(),
            element_type: self.element_type,
            shape: SmallVec::from_slice(dims),
        })
    }

    pub fn scalar<T: NativeType>(val: T) -> Literal {
        val.literal()
    }

    pub fn vector<T: NativeType>(vals: &[T]) -> Literal {
        T::create_r1(vals)
    }
}

macro_rules! impl_array_element {
    ($ty:ty, $variant:ident, $size:expr) => {
        impl ArrayElement for $ty {
            const TY: ElementType = ElementType::$variant;
            const ELEMENT_SIZE_IN_BYTES: usize = $size;
            const ZERO: Self = 0 as Self;
        }
    };
}

impl_array_element!(u8, U8, 1);
impl_array_element!(u16, U16, 2);
impl_array_element!(u32, U32, 4);
impl_array_element!(u64, U64, 8);
impl_array_element!(i8, S8, 1);
impl_array_element!(i16, S16, 2);
impl_array_element!(i32, S32, 4);
impl_array_element!(i64, S64, 8);
impl_array_element!(f32, F32, 4);
impl_array_element!(f64, F64, 8);

impl ArrayElement for bool {
    const TY: ElementType = ElementType::Pred;
    const ELEMENT_SIZE_IN_BYTES: usize = 1;
    const ZERO: Self = false;
}

macro_rules! impl_native_type {
    ($ty:ty) => {
        impl NativeType for $ty {
            fn literal(self) -> Literal {
                Literal {
                    data: self.as_bytes().to_vec(),
                    element_type: <$ty as ArrayElement>::TY,
                    shape: SmallVec::new(),
                }
            }

            fn create_r1(slice: &[Self]) -> Literal {
                let mut shape = SmallVec::new();
                shape.push(slice.len() as i64);
                Literal {
                    data: slice.as_bytes().to_vec(),
                    element_type: <$ty as ArrayElement>::TY,
                    shape,
                }
            }
        }
    };
}

impl_native_type!(u8);
impl_native_type!(u16);
impl_native_type!(u32);
impl_native_type!(u64);
impl_native_type!(i8);
impl_native_type!(i16);
impl_native_type!(i32);
impl_native_type!(i64);
impl_native_type!(f32);
impl_native_type!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_literal() {
        let lit = 42.0f64.literal();
        assert_eq!(lit.element_type(), ElementType::F64);
        assert!(lit.shape().is_empty());
        let buf = lit.typed_buf::<f64>().unwrap();
        assert_eq!(buf, &[42.0]);
    }

    #[test]
    fn test_vector_literal() {
        let lit = Literal::vector(&[1.0f32, 2.0, 3.0]);
        assert_eq!(lit.element_type(), ElementType::F32);
        assert_eq!(lit.shape().as_slice(), &[3]);
        let buf = lit.typed_buf::<f32>().unwrap();
        assert_eq!(buf, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reshape() {
        let lit = Literal::vector(&[1i32, 2, 3, 4, 5, 6]);
        let reshaped = lit.reshape(&[2, 3]).unwrap();
        assert_eq!(reshaped.shape().as_slice(), &[2, 3]);
        assert_eq!(reshaped.typed_buf::<i32>().unwrap(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_element_type_from_str() {
        assert_eq!("float64".parse::<ElementType>(), Ok(ElementType::F64));
        assert_eq!("int32".parse::<ElementType>(), Ok(ElementType::S32));
        assert_eq!("bool".parse::<ElementType>(), Ok(ElementType::Pred));
    }
}
