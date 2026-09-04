//! Traits to abstract working with mutable buffers of data
pub use stellarator_buf::*;

use core::mem::align_of;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::error::Error;

/// A mutable buffer of data - essentially an abstraction of types that look like std::Vec
pub trait Buf<T>: Serialize + DeserializeOwned + for<'de> Deserialize<'de> + Default {
    /// Returns an iterator over the elements of the buffer.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;

    /// Returns a slice of the buffer's elements.
    fn as_slice(&self) -> &[T];

    /// Returns a mutable slice of the buffer's elements.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Extends the buffer with the elements of another slice.
    ///
    /// Returns an error if the buffer is full.
    fn extend_from_slice(&mut self, other: &[T]) -> Result<(), Error>
    where
        T: Clone;

    /// Pushes an element onto the buffer
    ///
    /// Returns an error if the buffer is full.
    fn push(&mut self, elem: T) -> Result<(), Error>;

    /// Returns the number of elements in the buffer.
    fn len(&self) -> usize;

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the buffer.
    ///
    /// This usually keeps the capacity of the buffer unchanged, and just sets the length to zero.
    fn clear(&mut self);
}

pub trait ByteBufExt {
    /// Pushes `0u8` to the buffer until it is aligned to the size of `V`.
    fn pad_for_alignment<V: IntoBytes + Immutable>(&mut self) -> Result<(), Error>;

    /// Pushes a value to the buffer, aligned to the size of `V`.
    fn push_aligned<V: IntoBytes + Immutable>(&mut self, val: V) -> Result<usize, Error>;

    /// Pushes a slice of values to the buffer, aligned to the size of `V`.
    fn extend_aligned<V: IntoBytes + Immutable>(&mut self, slice: &[V]) -> Result<usize, Error>;

    /// Pushes values from an iterator to the buffer, each aligned to the size of `V`.
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

    fn clear(&mut self) {
        self.clear();
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

    fn clear(&mut self) {
        self.clear()
    }
}

/// A buffer that can store up to 12 bytes inline, or up to u32::MAX bytes in an external buffer
///
/// This is a generic version of ["Umbra Strings"](https://cedardb.com/blog/german_strings/) aka German Strings.
/// At a high level this buffer stores either inline data or a pointer to external data. It uses a clever layout
/// to make this efficient.
#[repr(C)]
#[derive(FromBytes, KnownLayout, Immutable, IntoBytes)]
pub struct UmbraBuf {
    pub len: u32,
    pub data: UmbraBufData,
}

impl UmbraBuf {
    /// Creates a new `UmbraBuf` with inline data.
    ///
    /// Assumes that len <= 12, in debug mode panics if len > 12.
    pub fn with_inline(len: u32, data: [u8; 12]) -> Self {
        debug_assert!(len <= 12, "inline is only valid len <= 12");
        Self {
            len,
            data: UmbraBufData { inline: data },
        }
    }

    /// Creates a new UmbraBuf that points to an offset
    ///
    /// Assumes that len > 12, in debug mode panics if len <= 12.
    ///
    /// Prefix is the first 4 bytes of the payload
    pub fn with_offset(len: u32, prefix: [u8; 4], offset: u32) -> Self {
        Self::with_segment_offset(len, prefix, 0, offset)
    }

    pub fn with_segment_offset(len: u32, prefix: [u8; 4], segment_index: u32, offset: u32) -> Self {
        debug_assert!(len > 12, "offset is only valid for lens above 12");
        Self {
            len,
            data: UmbraBufData {
                offset: LongBufOffset {
                    prefix,
                    segment_index,
                    offset,
                },
            },
        }
    }

    /// Returns the offset of the UmbraBuf if it is an offset buffer
    pub fn offset(&self) -> Option<u32> {
        self.long_offset().map(|o| o.offset)
    }

    /// Returns the external-buffer segment if this is an offset buffer.
    pub fn segment_index(&self) -> Option<u32> {
        self.long_offset().map(|o| o.segment_index)
    }

    /// First four payload bytes copied into a long `UmbraBuf`.
    pub fn prefix(&self) -> Option<[u8; 4]> {
        self.long_offset().map(|o| o.prefix)
    }

    fn long_offset(&self) -> Option<LongBufOffset> {
        if self.len <= 12 {
            return None;
        }
        LongBufOffset::read_from_bytes(&self.as_bytes()[4..]).ok()
    }
}

/// The internal union that stores the data for [`UmbraBuf`]
#[derive(FromBytes, KnownLayout, Immutable)]
#[repr(C)]
pub union UmbraBufData {
    pub inline: [u8; 12],
    pub offset: LongBufOffset,
}

unsafe impl IntoBytes for UmbraBufData {
    fn only_derive_is_allowed_to_implement_this_trait()
    where
        Self: Sized,
    {
        // gottem
    }
}

/// The offset data for an [`UmbraBuf`]
#[derive(FromBytes, KnownLayout, Immutable, IntoBytes, Clone, Copy)]
#[repr(C)]
pub struct LongBufOffset {
    /// A copy of the first 4 bytes of the data stored in the long buf
    pub prefix: [u8; 4],
    /// Which external buffer stores the payload.
    ///
    /// On-disk this was always the middle word (`_index`, 0 for single-segment
    /// logs). Do not reorder these fields.
    pub segment_index: u32,
    /// The offset into the buffer that contains the data
    pub offset: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_buf_exposes_segment_and_offset() {
        let buf = UmbraBuf::with_segment_offset(20, [1, 2, 3, 4], 7, 99);
        assert_eq!(buf.segment_index(), Some(7));
        assert_eq!(buf.offset(), Some(99));
        let inline = UmbraBuf::with_inline(4, [0; 12]);
        assert_eq!(inline.segment_index(), None);
        assert_eq!(inline.offset(), None);
        assert_eq!(inline.prefix(), None);
    }

    #[test]
    fn historical_single_segment_bytes_keep_offset_in_last_word() {
        let mut raw = [0u8; 16];
        raw[0..4].copy_from_slice(&20u32.to_le_bytes());
        raw[4..8].copy_from_slice(&[1, 2, 3, 4]);
        raw[8..12].copy_from_slice(&0u32.to_le_bytes());
        raw[12..16].copy_from_slice(&99u32.to_le_bytes());
        let buf = UmbraBuf::read_from_bytes(&raw).unwrap();
        assert_eq!(buf.prefix(), Some([1, 2, 3, 4]));
        assert_eq!(buf.segment_index(), Some(0));
        assert_eq!(buf.offset(), Some(99));
    }
}
