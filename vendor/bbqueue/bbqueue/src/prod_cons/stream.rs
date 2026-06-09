//! Stream byte queue interfaces
//!
//! Useful for sending stream-oriented data where the consumer doesn't care
//! about how the data was pushed, e.g. a serial port stream where multiple
//! writes from the software may be transferred out over DMA in a single
//! transfer.

use core::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::traits::{
    bbqhdl::BbqHandle,
    coordination::{Coord, ReadGrantError, WriteGrantError},
    notifier::{AsyncNotifier, Notifier},
    storage::Storage,
};

/// A producer handle that may write data into the buffer
pub struct StreamProducer<Q>
where
    Q: BbqHandle,
{
    pub(crate) bbq: Q::Target,
}

/// A consumer handle that may read data from the buffer
pub struct StreamConsumer<Q>
where
    Q: BbqHandle,
{
    pub(crate) bbq: Q::Target,
}

/// A writing grant into the storage buffer
///
/// Grants implement Deref/DerefMut to access the contained storage.
#[must_use = "Write Grants must be committed to be effective"]
pub struct StreamGrantW<Q>
where
    Q: BbqHandle,
{
    bbq: Q::Target,
    ptr: NonNull<u8>,
    len: usize,
    to_commit: usize,
}

/// A reading grant into the storage buffer
///
/// Grants implement Deref/DerefMut to access the contained storage.
///
/// Write access is provided for read grants in case it is necessary to mutate
/// the storage in-place for decoding.
pub struct StreamGrantR<Q>
where
    Q: BbqHandle,
{
    bbq: Q::Target,
    ptr: NonNull<u8>,
    len: usize,
    to_release: usize,
}

// ---- impls ----

// ---- StreamProducer ----

impl<Q> StreamProducer<Q>
where
    Q: BbqHandle,
{
    /// Obtain a grant UP TO the given `max` size.
    ///
    /// If we return a grant, it will have a nonzero amount of space.
    ///
    /// If the grant represents LESS than `max` size, this is due to either:
    ///
    /// * There is less than `max` free space available in the queue for writing
    /// * The grant represents the remaining space in the buffer that WOULDN'T cause
    ///   a wrap-around of the ring buffer
    ///
    /// This method will never cause an "early wraparound" of the ring buffer unless
    /// there is no capacity without wrapping around. There may still be available
    /// writing capacity in the buffer after commiting this write grant, so it may be
    /// useful to call `grant_max_remaining` in a loop until `Err(WriteGrantError::InsufficientSize)`
    /// is returned.
    pub fn grant_max_remaining(&self, max: usize) -> Result<StreamGrantW<Q>, WriteGrantError> {
        let (ptr, cap) = unsafe { self.bbq.sto.ptr_len() };
        let (offset, len) = self.bbq.cor.grant_max_remaining(cap, max)?;
        let ptr = unsafe {
            let p = ptr.as_ptr().byte_add(offset);
            NonNull::new_unchecked(p)
        };
        Ok(StreamGrantW {
            bbq: self.bbq.clone(),
            ptr,
            len,
            to_commit: 0,
        })
    }

    /// Obtain a grant with EXACTLY `sz` capacity
    ///
    /// Unlike `grant_max_remaining`, if there is insufficient size at the "tail" of
    /// the ring buffer, this method WILL cause a wrap-around to occur to attempt to
    /// find the requested write capacity.
    pub fn grant_exact(&self, sz: usize) -> Result<StreamGrantW<Q>, WriteGrantError> {
        let (ptr, cap) = unsafe { self.bbq.sto.ptr_len() };
        let offset = self.bbq.cor.grant_exact(cap, sz)?;
        let ptr = unsafe {
            let p = ptr.as_ptr().byte_add(offset);
            NonNull::new_unchecked(p)
        };
        Ok(StreamGrantW {
            bbq: self.bbq.clone(),
            ptr,
            len: sz,
            to_commit: 0,
        })
    }

    /// Get the total capacity of the buffer, e.g. how much space is present in [`Storage`]
    pub fn capacity(&self) -> usize {
        self.bbq.capacity()
    }
}

impl<Q> StreamProducer<Q>
where
    Q: BbqHandle,
    Q::Notifier: AsyncNotifier,
{
    /// Wait for a grant of any size, up to `max`, to become available
    pub async fn wait_grant_max_remaining(&self, max: usize) -> StreamGrantW<Q> {
        self.bbq
            .not
            .wait_for_not_full(|| self.grant_max_remaining(max).ok())
            .await
    }

    /// Wait for a grant of EXACTLY `sz` to become available.
    ///
    /// If `sz` exceeds the capacity of the buffer, this method will never return.
    pub async fn wait_grant_exact(&self, sz: usize) -> StreamGrantW<Q> {
        self.bbq
            .not
            .wait_for_not_full(|| self.grant_exact(sz).ok())
            .await
    }
}

unsafe impl<Q: BbqHandle + Send> Send for StreamProducer<Q> {}

// ---- StreamConsumer ----

impl<Q> StreamConsumer<Q>
where
    Q: BbqHandle,
{
    /// Obtain a chunk of readable data
    ///
    /// The returned chunk may NOT represent all available data if the available
    /// data wraps around the internal ring buffer. You may want to call `read`
    /// in a loop until `Err(ReadGrantError::Empty)` is returned if you want to
    /// drain the queue entirely.
    pub fn read(&self) -> Result<StreamGrantR<Q>, ReadGrantError> {
        let (ptr, _cap) = unsafe { self.bbq.sto.ptr_len() };
        let (offset, len) = self.bbq.cor.read()?;
        let ptr = unsafe {
            let p = ptr.as_ptr().byte_add(offset);
            NonNull::new_unchecked(p)
        };
        Ok(StreamGrantR {
            bbq: self.bbq.clone(),
            ptr,
            len,
            to_release: 0,
        })
    }

    /// Get the total capacity of the buffer, e.g. how much space is present in [`Storage`]
    pub fn capacity(&self) -> usize {
        self.bbq.capacity()
    }
}

impl<Q> StreamConsumer<Q>
where
    Q: BbqHandle,
    Q::Notifier: AsyncNotifier,
{
    /// Wait for any read data to become available
    pub async fn wait_read(&self) -> StreamGrantR<Q> {
        self.bbq.not.wait_for_not_empty(|| self.read().ok()).await
    }
}

unsafe impl<Q: BbqHandle + Send> Send for StreamConsumer<Q> {}

// ---- StreamGrantW ----

impl<Q> StreamGrantW<Q>
where
    Q: BbqHandle,
{
    /// Make `used` bytes available to be read.
    ///
    /// `used` is capped to the length of the grant.
    pub fn commit(self, used: usize) {
        let (_, cap) = unsafe { self.bbq.sto.ptr_len() };
        let used = used.min(self.len);
        self.bbq.cor.commit_inner(cap, self.len, used);
        if used != 0 {
            self.bbq.not.wake_one_consumer();
        }
        core::mem::forget(self);
    }
}

impl<Q> Deref for StreamGrantW<Q>
where
    Q: BbqHandle,
{
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<Q> DerefMut for StreamGrantW<Q>
where
    Q: BbqHandle,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<Q> Drop for StreamGrantW<Q>
where
    Q: BbqHandle,
{
    fn drop(&mut self) {
        let StreamGrantW {
            bbq,
            ptr: _,
            len,
            to_commit,
        } = self;
        let (_, cap) = unsafe { bbq.sto.ptr_len() };
        let len = *len;
        let used = (*to_commit).min(len);
        bbq.cor.commit_inner(cap, len, used);
        if used != 0 {
            bbq.not.wake_one_consumer();
        }
    }
}

unsafe impl<Q: BbqHandle + Send> Send for StreamGrantW<Q> {}

// ---- StreamGrantR ----

impl<Q> StreamGrantR<Q>
where
    Q: BbqHandle,
{
    /// Make `used` bytes available for writing.
    ///
    /// `used` is capped to the length of the grant
    pub fn release(self, used: usize) {
        let used = used.min(self.len);
        self.bbq.cor.release_inner(used);
        if used != 0 {
            self.bbq.not.wake_one_producer();
        }
        core::mem::forget(self);
    }
}

impl<Q> Deref for StreamGrantR<Q>
where
    Q: BbqHandle,
{
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<Q> DerefMut for StreamGrantR<Q>
where
    Q: BbqHandle,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<Q> Drop for StreamGrantR<Q>
where
    Q: BbqHandle,
{
    fn drop(&mut self) {
        let StreamGrantR {
            bbq,
            ptr: _,
            len,
            to_release,
        } = self;
        let len = *len;
        let used = (*to_release).min(len);
        bbq.cor.release_inner(used);
        if used != 0 {
            bbq.not.wake_one_producer();
        }
    }
}

unsafe impl<Q: BbqHandle + Send> Send for StreamGrantR<Q> {}
