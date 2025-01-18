use std::{
    mem::MaybeUninit,
    ops::{Bound, Deref, DerefMut, Range, RangeBounds},
    ptr::NonNull,
};

/// A buffer that is safe to pass to io_uring
///
/// Buffers that implement this trait must have a stable pointer to their contents.
/// Practically this means they must either be heap allocated or allocated with a static lifetime.
///
/// # Safety
///
/// Implementors of this trait must ensure that `stable_init_ptr` is stable for the entire lifetime of the buffer.
/// io_uring will ruin your day if you free the memory in the buffer before the operation is complete
/// `init_len` also must only contain initialized memory; returning a too large `init_len` will lead to UB
pub unsafe trait IoBuf: Unpin + 'static + Send {
    /// A stable pointer to the initialized data in the buffer
    fn stable_init_ptr(&self) -> *const u8;

    /// The length of initialized data in the buffer
    fn init_len(&self) -> usize;

    /// The total length of the buffer, including potentially uninitialized data
    fn total_len(&self) -> usize;

    fn try_slice(self, range: impl RangeBounds<usize>) -> Option<Slice<Self>>
    where
        Self: Sized,
    {
        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        if begin >= self.init_len() {
            return None;
        }

        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.init_len(),
        };

        if !(end <= self.init_len() || begin <= self.init_len()) {
            return None;
        }

        Some(Slice {
            inner: self,
            range: begin..end,
        })
    }
}

/// A buffer this is safe to pass to io_uring, in a mutable fashion
///
/// Buffers that implement this trait must have a stable pointer to their contents.
/// Practically this means they must either be heap allocated or allocated with a static lifetime.
///
/// # Safety
///
/// Implementors of this trait must ensure that `stable_init_ptr` is stable for the entire lifetime of the buffer.
/// io_uring will ruin your day if you free the memory in the buffer before the operation is complete
pub unsafe trait IoBufMut: IoBuf {
    /// A stable pointer to the buffer's contents, contains potentially uninitialized memory
    fn stable_mut_ptr(&mut self) -> NonNull<MaybeUninit<u8>>;
    /// Set the initialized length of memory
    ///
    /// # Safety
    /// The implementor must ensure that `len` does not extend into uninitialized memory. If it does you will get UB.
    unsafe fn set_init(&mut self, len: usize);
}

pub fn deref(buf: &impl IoBuf) -> &[u8] {
    // source: https://github.com/tokio-rs/tokio-uring/blob/7761222aa7f4bd48c559ca82e9535d47aac96d53/src/buf/mod.rs#L21
    // Safety: the `IoBuf` trait is marked as unsafe and is expected to be
    // implemented correctly.
    unsafe { std::slice::from_raw_parts(buf.stable_init_ptr(), buf.init_len()) }
}

pub fn deref_mut(buf: &mut impl IoBufMut) -> &mut [u8] {
    // source: https://github.com/tokio-rs/tokio-uring/blob/7761222aa7f4bd48c559ca82e9535d47aac96d53/src/buf/mod.rs#L28
    // Safety: the `IoBuf` trait is marked as unsafe and is expected to be
    // implemented correctly.
    unsafe {
        std::slice::from_raw_parts_mut(buf.stable_mut_ptr().as_ptr() as *mut u8, buf.init_len())
    }
}

unsafe impl IoBuf for Vec<u8> {
    fn stable_init_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn init_len(&self) -> usize {
        self.len()
    }

    fn total_len(&self) -> usize {
        self.capacity()
    }
}

unsafe impl IoBufMut for Vec<u8> {
    fn stable_mut_ptr(&mut self) -> NonNull<MaybeUninit<u8>> {
        NonNull::new(self.as_ptr() as *mut MaybeUninit<u8>).unwrap()
    }

    unsafe fn set_init(&mut self, len: usize) {
        self.set_len(len);
    }
}

unsafe impl IoBuf for &'static [u8] {
    fn stable_init_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn init_len(&self) -> usize {
        self.len()
    }

    fn total_len(&self) -> usize {
        self.len()
    }
}

unsafe impl<const N: usize> IoBuf for &'static [u8; N] {
    fn stable_init_ptr(&self) -> *const u8 {
        self.as_ptr()
    }

    fn init_len(&self) -> usize {
        self.len()
    }

    fn total_len(&self) -> usize {
        self.len()
    }
}

/// A slice of an [`IoBuf`]
///
/// IoBuf requires ownership of buffers, so `Slice` allows a user
/// to slice an `IoBuf` without loosing
pub struct Slice<T: IoBuf> {
    inner: T,
    range: Range<usize>,
}

impl<T: IoBuf> Slice<T> {
    /// The range of the data in the slice
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Consumes `Slice` returning the inner buffer
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Sets the [`Slice`] to the entire length of the inner [`IoBuf`]
    pub fn reset(&mut self) {
        self.range = 0..self.inner.init_len()
    }

    /// Attempts to set the range of the [`Slice`]
    ///
    /// Returns `false` if the range exceeds the bounds of the inner buf
    pub fn try_set_range(&mut self, range: Range<usize>) -> bool {
        if range.start <= self.inner.init_len() || range.end <= self.inner.init_len() {
            return false;
        }
        self.range = range;
        true
    }
    /// Sets the range of the [`Slice`]
    ///
    /// Panics if the range exceeds the bounds of the inner buf
    pub fn set_range(&mut self, range: Range<usize>) {
        assert!(self.try_set_range(range), "range out of bounds")
    }

    pub fn try_sub_slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        let begin = match range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        if begin > self.range.end {
            return None;
        }

        let end = match range.end_bound() {
            Bound::Included(&n) => n.checked_add(1).expect("out of range"),
            Bound::Excluded(&n) => n,
            Bound::Unbounded => self.init_len(),
        };
        if end > self.range.end {
            return None;
        }

        let begin = begin.checked_add(self.range.start)?;
        let end = end.checked_add(self.range.start)?;

        self.into_inner().try_slice(begin..end)
    }
}

unsafe impl<T: IoBuf> IoBuf for Slice<T> {
    fn stable_init_ptr(&self) -> *const u8 {
        self.deref().as_ptr()
    }

    fn init_len(&self) -> usize {
        self.range.len()
    }

    fn total_len(&self) -> usize {
        self.range.len()
    }
}

unsafe impl<T: IoBufMut> IoBufMut for Slice<T> {
    fn stable_mut_ptr(&mut self) -> NonNull<MaybeUninit<u8>> {
        unsafe { NonNull::new_unchecked(self.deref_mut().as_ptr() as *mut MaybeUninit<u8>) }
    }

    unsafe fn set_init(&mut self, len: usize) {
        self.inner.set_init(self.range.start + len)
    }
}

impl<T: IoBuf> Deref for Slice<T> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        let slice = deref(&self.inner);
        &slice[self.range.clone()]
    }
}

impl<T: IoBufMut> DerefMut for Slice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let slice = deref_mut(&mut self.inner);
        &mut slice[self.range.clone()]
    }
}
