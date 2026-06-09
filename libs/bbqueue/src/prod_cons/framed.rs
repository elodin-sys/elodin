//! Framed byte queue interfaces
//!
//! Useful for sending data that has coherent frames, e.g. network packets

use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::traits::{
    bbqhdl::BbqHandle,
    coordination::{Coord, ReadGrantError, WriteGrantError},
    notifier::{AsyncNotifier, Notifier},
    storage::Storage,
};

/// A trait that can be used as the "header" for separating framed storage.
///
/// Framed interfaces use a `u16` by default, which only requires two bytes
/// for the header, with the limitation that the largest grant allowed is
/// 64KiB at a time. You can also use a `usize` allowing the maximum platform
/// available size.
///
/// You should not have to implement this trait.
///
/// # Safety
///
/// Do it right
pub unsafe trait LenHeader: Into<usize> + Copy + Ord {
    /// Should be `[u8; size_of::<Self>()]`
    type Bytes;
    /// Convert Self into Self::Bytes, in little endian order
    fn to_le_bytes(&self) -> Self::Bytes;
    /// Convert Self::Bytes (which is in little endian order) to Self.
    fn from_le_bytes(by: Self::Bytes) -> Self;
}

/// A producer handle that can be used to write framed chunks
pub struct FramedProducer<Q, H = u16>
where
    Q: BbqHandle,
    H: LenHeader,
{
    pub(crate) bbq: Q::Target,
    pub(crate) pd: PhantomData<H>,
}

/// A consumer handle that can be used to read framed chunks
pub struct FramedConsumer<Q, H = u16>
where
    Q: BbqHandle,
    H: LenHeader,
{
    pub(crate) bbq: Q::Target,
    pub(crate) pd: PhantomData<H>,
}

/// A writing grant into the storage buffer
///
/// Grants implement Deref/DerefMut to access the contained storage.
#[must_use = "Write Grants must be committed to be effective"]
pub struct FramedGrantW<Q, H = u16>
where
    Q: BbqHandle,
    H: LenHeader,
{
    bbq: Q::Target,
    base_ptr: NonNull<u8>,
    hdr: H,
}

/// A reading grant into the storage buffer
///
/// Grants implement Deref/DerefMut to access the contained storage.
///
/// Write access is provided for read grants in case it is necessary to mutate
/// the storage in-place for decoding.
#[must_use = "Read Grants must be released to free space"]
pub struct FramedGrantR<Q, H = u16>
where
    Q: BbqHandle,
    H: LenHeader,
{
    bbq: Q::Target,
    body_ptr: NonNull<u8>,
    hdr: H,
}

// ---- impls ----

// ---- impl LenHeader ----

unsafe impl LenHeader for u16 {
    type Bytes = [u8; 2];

    #[inline(always)]
    fn to_le_bytes(&self) -> Self::Bytes {
        u16::to_le_bytes(*self)
    }

    #[inline(always)]
    fn from_le_bytes(by: Self::Bytes) -> Self {
        u16::from_le_bytes(by)
    }
}

unsafe impl LenHeader for usize {
    type Bytes = [u8; core::mem::size_of::<usize>()];

    #[inline(always)]
    fn to_le_bytes(&self) -> Self::Bytes {
        usize::to_le_bytes(*self)
    }

    #[inline(always)]
    fn from_le_bytes(by: Self::Bytes) -> Self {
        usize::from_le_bytes(by)
    }
}

// ---- impl FramedProducer ----

impl<Q, H> FramedProducer<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    /// Attempt to obtain a write grant of the given (max) size
    ///
    /// The returned grant can be used to write up to `sz` bytes, though
    /// a smaller size may be committed. Dropping the grant without calling
    /// commit means that no data will be made visible to the consumer.
    pub fn grant(&self, sz: H) -> Result<FramedGrantW<Q, H>, WriteGrantError> {
        let (ptr, cap) = unsafe { self.bbq.sto.ptr_len() };
        let needed = sz.into() + core::mem::size_of::<H>();

        let offset = self.bbq.cor.grant_exact(cap, needed)?;

        let base_ptr = unsafe {
            let p = ptr.as_ptr().byte_add(offset);
            NonNull::new_unchecked(p)
        };
        Ok(FramedGrantW {
            bbq: self.bbq.clone(),
            base_ptr,
            hdr: sz,
        })
    }

    /// Get the total capacity of the buffer, e.g. how much space is present in [`Storage`]
    pub fn capacity(&self) -> usize {
        self.bbq.capacity()
    }
}

impl<Q, H> FramedProducer<Q, H>
where
    Q: BbqHandle,
    Q::Notifier: AsyncNotifier,
    H: LenHeader,
{
    /// Wait for the given write grant to become available
    ///
    /// If `sz` is larger than the storage buffer, this method will never
    /// return.
    ///
    /// The returned grant can be used to write up to `sz` bytes, though
    /// a smaller size may be committed. Dropping the grant without calling
    /// commit means that no data will be made visible to the consumer.
    pub async fn wait_grant(&self, sz: H) -> FramedGrantW<Q, H> {
        self.bbq.not.wait_for_not_full(|| self.grant(sz).ok()).await
    }
}

// ---- impl FramedConsumer ----

impl<Q, H> FramedConsumer<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    /// Attempt to receive a single frame
    ///
    /// The FramedConsumer has no control over the size of the read grant,
    /// we see whatever size was written by the FramedProducer.
    ///
    /// The returned grant must be released to free the space in the buffer.
    pub fn read(&self) -> Result<FramedGrantR<Q, H>, ReadGrantError> {
        let (ptr, _cap) = unsafe { self.bbq.sto.ptr_len() };
        let (offset, grant_len) = self.bbq.cor.read()?;

        // Calculate the size so we can figure out where the body
        // starts in the grant
        let hdr_sz = const { core::mem::size_of::<H>() };
        if hdr_sz > grant_len {
            // This means that we got a read grant that doesn't even
            // cover the size of a header - this should only be possible
            // if you used a stream producer to create a grant, this is
            // not compatible. We need to release the read grant, and
            // return an error
            self.bbq.cor.release_inner(0);
            return Err(ReadGrantError::InconsistentFrameHeader);
        }

        // Ptr is the base of (HDR, Body)
        let ptr = unsafe { ptr.as_ptr().byte_add(offset) };
        // Read the potentially unaligned header
        let hdr: H = unsafe { ptr.cast::<H>().read_unaligned() };
        if (hdr_sz + hdr.into()) > grant_len {
            // Again, the header value + header size are larger than
            // the actual read grant, this means someone is doing
            // something sketch. We need to release the read grant,
            // and return an error
            self.bbq.cor.release_inner(0);
            return Err(ReadGrantError::InconsistentFrameHeader);
        }

        // Get the body, which is the base ptr offset by the header size
        let body_ptr = unsafe {
            let p = ptr.byte_add(hdr_sz);
            core::ptr::NonNull::new_unchecked(p)
        };
        Ok(FramedGrantR {
            bbq: self.bbq.clone(),
            body_ptr,
            hdr,
        })
    }

    /// Get the total capacity of the buffer, e.g. how much space is present in [`Storage`]
    pub fn capacity(&self) -> usize {
        self.bbq.capacity()
    }
}

impl<Q, H> FramedConsumer<Q, H>
where
    Q: BbqHandle,
    Q::Notifier: AsyncNotifier,
    H: LenHeader,
{
    /// Wait for a single frame
    ///
    /// The FramedConsumer has no control over the size of the read grant,
    /// we see whatever size was written by the FramedProducer.
    ///
    /// The returned grant must be released to free the space in the buffer.
    pub async fn wait_read(&self) -> FramedGrantR<Q, H> {
        self.bbq.not.wait_for_not_empty(|| self.read().ok()).await
    }
}

// ---- impl FramedGrantW ----

impl<Q, H> FramedGrantW<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    /// Commit `used` bytes of the grant to be visible.
    ///
    /// If `used` is greater than the `sz` used to create this grant, the
    /// amount will be clamped to `sz`.
    pub fn commit(self, used: H) {
        let (_ptr, cap) = unsafe { self.bbq.sto.ptr_len() };
        let hdrlen: usize = const { core::mem::size_of::<H>() };
        let grant_len = hdrlen + self.hdr.into();
        let clamp_hdr = self.hdr.min(used);
        let used_len: usize = hdrlen + clamp_hdr.into();

        unsafe {
            self.base_ptr
                .cast::<H>()
                .as_ptr()
                .write_unaligned(clamp_hdr);
        }

        self.bbq.cor.commit_inner(cap, grant_len, used_len);
        self.bbq.not.wake_one_consumer();
        core::mem::forget(self);
    }

    /// Aborts the grant, making no frame available to the consumer
    ///
    /// Can be used to silence "must_use" errors.
    pub fn abort(self) {
        // The default behavior is to abort - do nothing, let the
        // drop impl run
    }
}

impl<Q, H> Deref for FramedGrantW<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        let len = self.hdr.into();
        let body_ptr = unsafe {
            let hdr_sz = const { core::mem::size_of::<H>() };
            self.base_ptr.as_ptr().byte_add(hdr_sz)
        };
        unsafe { core::slice::from_raw_parts(body_ptr, len) }
    }
}

impl<Q, H> DerefMut for FramedGrantW<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.hdr.into();
        let body_ptr = unsafe {
            let hdr_sz = const { core::mem::size_of::<H>() };
            self.base_ptr.as_ptr().byte_add(hdr_sz)
        };
        unsafe { core::slice::from_raw_parts_mut(body_ptr, len) }
    }
}

impl<Q, H> Drop for FramedGrantW<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    fn drop(&mut self) {
        // Default drop performs an "abort"
        let (_ptr, cap) = unsafe { self.bbq.sto.ptr_len() };
        let hdrlen: usize = const { core::mem::size_of::<H>() };
        let grant_len = hdrlen + self.hdr.into();
        self.bbq.cor.commit_inner(cap, grant_len, 0);
    }
}

unsafe impl<Q, H> Send for FramedGrantW<Q, H>
where
    Q: BbqHandle,
    Q::Target: Send,
    H: LenHeader + Send,
{
}

// ---- impl FramedGrantR ----

impl<Q, H> FramedGrantR<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    /// Release the entire read grant
    ///
    /// It is not possible to partially release a framed read grant.
    pub fn release(self) {
        let len: usize = self.hdr.into();
        let hdrlen: usize = const { core::mem::size_of::<H>() };
        let used = len + hdrlen;
        self.bbq.cor.release_inner(used);
        self.bbq.not.wake_one_producer();
        core::mem::forget(self);
    }

    /// Drop the grant WITHOUT releasing the message from the queue.
    ///
    /// The next call to read will observe the same packet again.
    pub fn keep(self) {
        // Default behavior is "keep"
    }
}

impl<Q, H> Deref for FramedGrantR<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        let len: usize = self.hdr.into();
        unsafe { core::slice::from_raw_parts(self.body_ptr.as_ptr(), len) }
    }
}

impl<Q, H> DerefMut for FramedGrantR<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len: usize = self.hdr.into();
        unsafe { core::slice::from_raw_parts_mut(self.body_ptr.as_ptr(), len) }
    }
}

impl<Q, H> Drop for FramedGrantR<Q, H>
where
    Q: BbqHandle,
    H: LenHeader,
{
    fn drop(&mut self) {
        // Default behavior is "keep" - release zero bytes
        self.bbq.cor.release_inner(0);
    }
}

unsafe impl<Q, H> Send for FramedGrantR<Q, H>
where
    Q: BbqHandle,
    Q::Target: Send,
    H: LenHeader + Send,
{
}
