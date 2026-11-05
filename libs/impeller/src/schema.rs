use serde::{Deserialize, Serialize};

#[cfg(target_pointer_width = "64")]
use zerocopy::{FromBytes, IntoBytes};

use crate::{buf::Buf, error::Error, types::PrimType};

#[cfg(feature = "alloc")]
pub type DefaultSizeBuf = Vec<u64>;
#[cfg(not(feature = "alloc"))]
pub type DefaultSizeBuf = heapless::Vec<u64, 6>;

#[cfg(all(not(target_pointer_width = "64"), feature = "alloc"))]
type UsizeShape = alloc::vec::Vec<usize>;
#[cfg(all(not(target_pointer_width = "64"), not(feature = "alloc")))]
type UsizeShape = heapless::Vec<usize, 8>;

#[derive(Serialize, Debug)]
pub struct Schema<S: Buf<u64> = DefaultSizeBuf> {
    prim_type: PrimType,
    #[serde(bound(deserialize = ""))]
    shape: S,
    /// On 32-bit, `u64` dims cannot alias `&[usize]`; keep a converted copy.
    #[serde(skip)]
    #[cfg(not(target_pointer_width = "64"))]
    usize_shape: UsizeShape,
}

impl<D: Buf<u64> + Clone> Clone for Schema<D> {
    fn clone(&self) -> Self {
        Self {
            prim_type: self.prim_type,
            shape: self.shape.clone(),
            #[cfg(not(target_pointer_width = "64"))]
            usize_shape: self.usize_shape.clone(),
        }
    }
}

impl<D: Buf<u64> + PartialEq> PartialEq for Schema<D> {
    fn eq(&self, other: &Self) -> bool {
        self.prim_type == other.prim_type && self.shape == other.shape
    }
}

#[derive(Deserialize)]
struct SchemaDe<S: Buf<u64>> {
    prim_type: PrimType,
    #[serde(bound(deserialize = ""))]
    shape: S,
}

impl<'de, D: Buf<u64>> Deserialize<'de> for Schema<D> {
    fn deserialize<De: serde::Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let de = SchemaDe::deserialize(deserializer)?;
        Ok(Self::from_parts(de.prim_type, de.shape))
    }
}

impl<D: Buf<u64>> Schema<D> {
    pub fn new<T, I>(prim_type: PrimType, shape: I) -> Result<Self, Error>
    where
        T: DimElem,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut data = D::default();
        for dim in shape {
            data.push(dim.into_u64())?;
        }
        Ok(Self::from_parts(prim_type, data))
    }

    fn from_parts(prim_type: PrimType, shape: D) -> Self {
        #[cfg(not(target_pointer_width = "64"))]
        let usize_shape = usize_shape_from_dims(shape.as_slice());
        Self {
            prim_type,
            shape,
            #[cfg(not(target_pointer_width = "64"))]
            usize_shape,
        }
    }

    pub fn prim_type(&self) -> PrimType {
        self.prim_type
    }

    pub fn dim(&self) -> &[u64] {
        self.shape.as_slice()
    }

    pub fn shape(&self) -> &[usize] {
        #[cfg(target_pointer_width = "64")]
        {
            let bytes = self.shape.as_slice().as_bytes();
            <[usize]>::ref_from_bytes(bytes).expect("u64 dims alias usize on 64-bit")
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            self.usize_shape.as_slice()
        }
    }

    pub fn element_count(&self) -> usize {
        self.dim().iter().copied().product::<u64>() as usize
    }

    pub fn size(&self) -> usize {
        self.element_count() * self.prim_type.size()
    }
}

#[cfg(all(not(target_pointer_width = "64"), feature = "alloc"))]
fn usize_shape_from_dims(dims: &[u64]) -> UsizeShape {
    dims.iter().copied().map(|d| d as usize).collect()
}

#[cfg(all(not(target_pointer_width = "64"), not(feature = "alloc")))]
fn usize_shape_from_dims(dims: &[u64]) -> UsizeShape {
    let mut out = heapless::Vec::new();
    for &d in dims {
        let _ = out.push(d as usize);
    }
    out
}

pub trait DimElem {
    fn into_u64(self) -> u64;
}

impl DimElem for u64 {
    fn into_u64(self) -> u64 {
        self
    }
}

impl DimElem for usize {
    fn into_u64(self) -> u64 {
        self as u64
    }
}

impl<T: DimElem + Copy> DimElem for &'_ T {
    fn into_u64(self) -> u64 {
        (*self).into_u64()
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn shape_len_matches_dim_len() {
        let schema = Schema::<Vec<u64>>::new(PrimType::F64, [3u64]).unwrap();
        assert_eq!(schema.dim(), &[3]);
        assert_eq!(schema.shape().len(), schema.dim().len());
        assert_eq!(schema.shape(), &[3]);
        assert_eq!(schema.element_count(), 3);
        assert_eq!(schema.size(), 24);
    }

    #[test]
    fn shape_matrix_element_count() {
        let schema = Schema::<Vec<u64>>::new(PrimType::F32, [2u64, 3]).unwrap();
        assert_eq!(schema.shape(), &[2, 3]);
        assert_eq!(schema.element_count(), 6);
        assert_eq!(schema.size(), 24);
    }

    #[test]
    fn deserialize_preserves_shape() {
        let schema = Schema::<Vec<u64>>::new(PrimType::F64, [3u64]).unwrap();
        let bytes = postcard::to_allocvec(&schema).unwrap();
        let decoded: Schema<Vec<u64>> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.shape(), &[3]);
        assert_eq!(decoded.element_count(), 3);
    }
}
