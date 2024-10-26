use std::{mem::MaybeUninit, ptr::NonNull};

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
    fn stable_mut_ptr(&self) -> NonNull<MaybeUninit<u8>>;
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
    fn stable_mut_ptr(&self) -> NonNull<MaybeUninit<u8>> {
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
