use core::mem::align_of;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use zerocopy::{Immutable, IntoBytes};

use crate::error::Error;

pub trait Buf<T>: Serialize + DeserializeOwned + for<'de> Deserialize<'de> + Default {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];

    fn extend_from_slice(&mut self, other: &[T]) -> Result<(), Error>
    where
        T: Clone;
    fn push(&mut self, elem: T) -> Result<(), Error>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait ByteBufExt {
    fn pad_for_alignment<V: IntoBytes + Immutable>(&mut self) -> Result<(), Error>;
    fn push_aligned<V: IntoBytes + Immutable>(&mut self, val: V) -> Result<usize, Error>;
    fn extend_aligned<V: IntoBytes + Immutable>(&mut self, slice: &[V]) -> Result<usize, Error>;
    fn extend_from_iter_aligned<V: IntoBytes + Immutable>(
        &mut self,
        iter: impl Iterator<Item = V>,
    ) -> Result<usize, Error>;
}

impl<B: Buf<u8>> ByteBufExt for B {
    fn pad_for_alignment<V: IntoBytes + Immutable>(&mut self) -> Result<(), Error> {
        let alignment = align_of::<V>();
        let current_len = self.as_slice().len();
        let padding = (alignment - (current_len % alignment)) % alignment;

        for _ in 0..padding {
            self.push(0)?;
        }
        Ok(())
    }
    fn push_aligned<V: IntoBytes + Immutable>(&mut self, val: V) -> Result<usize, Error> {
        self.pad_for_alignment::<V>()?;
        let offset = self.as_slice().len();
        self.extend_from_slice(val.as_bytes())?;

        Ok(offset)
    }

    fn extend_aligned<V: IntoBytes + Immutable>(&mut self, slice: &[V]) -> Result<usize, Error> {
        self.pad_for_alignment::<V>()?;
        let offset = self.as_slice().len();
        for val in slice {
            self.extend_from_slice(val.as_bytes())?;
        }

        Ok(offset)
    }

    fn extend_from_iter_aligned<V: IntoBytes + Immutable>(
        &mut self,
        iter: impl Iterator<Item = V>,
    ) -> Result<usize, Error> {
        self.pad_for_alignment::<V>()?;

        let offset = self.as_slice().len();
        for val in iter {
            self.extend_from_slice(val.as_bytes())?;
        }

        Ok(offset)
    }
}

#[cfg(feature = "alloc")]
impl<T: Serialize + DeserializeOwned> Buf<T> for alloc::vec::Vec<T> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self[..].iter()
    }

    fn as_slice(&self) -> &[T] {
        &self[..]
    }

    fn extend_from_slice(&mut self, other: &[T]) -> Result<(), Error>
    where
        T: Clone,
    {
        self.extend_from_slice(other);
        Ok(())
    }

    fn push(&mut self, elem: T) -> Result<(), Error> {
        self.push(elem);
        Ok(())
    }

    fn len(&self) -> usize {
        alloc::vec::Vec::len(self)
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self[..]
    }
}
impl<T: Serialize + DeserializeOwned, const N: usize> Buf<T> for heapless::Vec<T, N> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self[..].iter()
    }

    fn as_slice(&self) -> &[T] {
        &self[..]
    }

    fn extend_from_slice(&mut self, other: &[T]) -> Result<(), Error>
    where
        T: Clone,
    {
        self.extend_from_slice(other)
            .map_err(|_| Error::BufferOverflow)
    }

    fn push(&mut self, elem: T) -> Result<(), Error> {
        self.push(elem).map_err(|_| Error::BufferOverflow)
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self[..]
    }
}
