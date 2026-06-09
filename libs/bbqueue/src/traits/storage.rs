//! "Storage" functionality
//!
//! This trait defines how the "data" part of the ring buffer is stored.
//!
//! This is typically "inline", e.g. stored as an owned `[u8; N]` array,
//! or heap allocated, e.g. as a `Box<[u8]>`.
//!
//! Inline storage is useful for static allocation, or cases where a fixed
//! buffer is useful. Heap storage is useful when you need dynamically sized
//! storage, e.g. of a size provided from CLI args or a configuration file
//! at runtime.

use const_init::ConstInit;
use core::{cell::UnsafeCell, mem::MaybeUninit, ptr::NonNull};

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};

/// Trait for providing access to the storage
///
/// Must always return the same ptr/len forever.
pub trait Storage {
    /// Return a pointer and length pair of the underlying storage.
    ///
    /// ## Safety
    ///
    /// Implementations of this function MUST always return the same pointer an length
    /// for all calls.
    ///
    /// The ownership of the data pointed to by this ptr+len are exclusive to the [`BBQueue`](crate::queue::BBQueue)
    /// that contains this [`Storage`].
    ///
    /// The pointee does not necessarily need to be initialized.
    unsafe fn ptr_len(&self) -> (NonNull<u8>, usize);
}

/// A marker trait that the item is BOTH storage and can be initialized as a constant
///
/// This allows for making `static` versions of the bbqueue.
pub trait ConstStorage: Storage + ConstInit {}
impl<T> ConstStorage for T where T: Storage + ConstInit {}

/// Inline/array-ful storage
#[repr(transparent)]
pub struct Inline<const N: usize> {
    buf: UnsafeCell<MaybeUninit<[u8; N]>>,
}

unsafe impl<const N: usize> Sync for Inline<N> {}

impl<const N: usize> Inline<N> {
    /// Create a new inline storage, e.g. `[u8; N]`.
    pub const fn new() -> Self {
        Self {
            buf: UnsafeCell::new(MaybeUninit::zeroed()),
        }
    }
}

impl<const N: usize> Default for Inline<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Storage for Inline<N> {
    unsafe fn ptr_len(&self) -> (NonNull<u8>, usize) {
        if N == 0 {
            return (NonNull::dangling(), N);
        }
        let ptr: *mut MaybeUninit<[u8; N]> = self.buf.get();
        let ptr: *mut u8 = ptr.cast();
        // SAFETY: UnsafeCell and MaybeUninit are both repr transparent, cast is
        // sound to get to first byte element
        let nn_ptr = unsafe { NonNull::new_unchecked(ptr) };
        (nn_ptr, N)
    }
}

#[allow(clippy::declare_interior_mutable_const)]
impl<const N: usize> ConstInit for Inline<N> {
    const INIT: Self = Self::new();
}

impl<const N: usize> Storage for &'_ Inline<N> {
    unsafe fn ptr_len(&self) -> (NonNull<u8>, usize) {
        let len = N;

        let ptr: *mut MaybeUninit<[u8; N]> = self.buf.get();
        let ptr: *mut u8 = ptr.cast();
        let nn_ptr = unsafe { NonNull::new_unchecked(ptr) };

        (nn_ptr, len)
    }
}

/// Boxed/heap-ful storage
#[cfg(feature = "alloc")]
pub struct BoxedSlice {
    buf: Box<[UnsafeCell<MaybeUninit<u8>>]>,
}

#[cfg(feature = "alloc")]
unsafe impl Sync for BoxedSlice {}

#[cfg(feature = "alloc")]
impl BoxedSlice {
    /// Create a new BoxedSlice with capacity `len`.
    pub fn new(len: usize) -> Self {
        let buf: Box<[UnsafeCell<MaybeUninit<u8>>]> = {
            let mut v: Vec<UnsafeCell<MaybeUninit<u8>>> = Vec::with_capacity(len);
            // Fields are already MaybeUninit, so valid capacity is valid len
            unsafe {
                v.set_len(len);
            }
            // We can zero each field now
            v.iter_mut().for_each(|val| {
                *val = UnsafeCell::new(MaybeUninit::zeroed());
            });
            v.into_boxed_slice()
        };
        Self { buf }
    }
}

#[cfg(feature = "alloc")]
impl Storage for BoxedSlice {
    unsafe fn ptr_len(&self) -> (NonNull<u8>, usize) {
        let len = self.buf.len();

        let ptr: *const UnsafeCell<MaybeUninit<u8>> = self.buf.as_ptr();
        let ptr: *mut MaybeUninit<u8> = UnsafeCell::raw_get(ptr);
        let ptr: *mut u8 = ptr.cast();
        let nn_ptr = unsafe { NonNull::new_unchecked(ptr) };

        (nn_ptr, len)
    }
}

#[cfg(test)]
mod test {
    use super::{Inline, Storage};

    #[test]
    fn provenance_slice() {
        let sli = Inline::<64>::new();
        let sli = &sli;
        let (ptr, len) = unsafe { <&Inline<64> as Storage>::ptr_len(&sli) };

        // This test ensures that obtaining the pointer for ptr_len through a single
        // element is sound
        for i in 0..len {
            unsafe {
                ptr.as_ptr().add(i).write(i as u8);
            }
        }
    }
}
