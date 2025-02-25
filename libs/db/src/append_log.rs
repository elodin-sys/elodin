use memmap2::MmapRaw;
use zerocopy::{Immutable, IntoBytes};

use crate::Error;
use std::{
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write as _},
    marker::PhantomData,
    os::fd::AsRawFd,
    path::Path,
    slice::{self, SliceIndex},
    sync::{
        Arc,
        atomic::{self, AtomicU64},
    },
};

/// [`AppendLog`] is a memory-mapped append-only time-series data file.
/// It works by mapping a large amount of memory ~8g to a sparse file. The file contains a
/// a `Header`, a series of committed data, and finally the uncommitted `head` data.
///
/// ```ignore
/// | Header | Committed Data | Head |
/// ```
///
/// When a `AppendLog` is created or opened you get access to both a [`TimeSeries`] and an associated [`TimeSeriesWriter`]. [`TimeSeries`] is a read only view into the time series,
/// giving you access to only the committed data. [`AppendLogWriter`] provides write access to the `head` of the [`TimeSeries`] with [`TimeSeriesWriter::write_head`]. It also provides the ability to commit the current head using
/// [`AppendLogWriter::commit_head_copy`]
pub struct AppendLog<E> {
    map: Arc<memmap2::MmapRaw>,
    header_extra: PhantomData<E>,
}

impl<E> Clone for AppendLog<E> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            header_extra: PhantomData,
        }
    }
}

#[repr(C)]
struct Header<E> {
    pub committed_len: AtomicU64,
    pub head_len: AtomicU64,
    pub extra: E,
}

impl<E: IntoBytes + Immutable> AppendLog<E> {
    pub fn create(path: impl AsRef<Path>, extra: E) -> Result<(Self, AppendLogWriter<E>), Error> {
        const FILE_SIZE: u64 = 1024 * 1024 * 1024 * 8; // 8gb
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(path)?;
        file.seek(SeekFrom::Start(FILE_SIZE))?;
        file.write_all(&[0])?;
        let map = Arc::new(memmap2::MmapRaw::map_raw(file.as_raw_fd())?);
        let map = Self {
            map,
            header_extra: PhantomData,
        };
        unsafe {
            let map = map.map.as_mut_ptr().add(size_of::<AtomicU64>() * 2);
            let extra_buf = std::slice::from_raw_parts_mut(map, size_of::<E>());
            extra_buf.copy_from_slice(extra.as_bytes());
        }
        map.committed_len()
            .store(size_of::<Header<E>>() as u64, atomic::Ordering::SeqCst);
        map.head_len().store(0, atomic::Ordering::SeqCst);
        let writer = AppendLogWriter { inner: map.clone() };
        Ok((map, writer))
    }

    pub fn open(path: impl AsRef<Path>) -> Result<(Self, AppendLogWriter<E>), Error> {
        let file = OpenOptions::new().write(true).read(true).open(path)?;
        let map = Arc::new(memmap2::MmapRaw::map_raw(file.as_raw_fd())?);
        let map = Self {
            map,
            header_extra: PhantomData,
        };
        let writer = AppendLogWriter { inner: map.clone() };
        Ok((map, writer))
    }

    /// Returns a slice of data offset into the committed data region of the [`AppendLog`]
    pub fn get(&self, range: impl SliceIndex<[u8], Output = [u8]>) -> Option<&'_ [u8]> {
        self.data().get(range)
    }

    fn header(&self) -> &Header<E> {
        let ptr = self.map.as_mut_ptr();
        unsafe { &*(ptr as *const Header<E>) }
    }

    /// Returns a reference to the [`AppendLog`]'s committed_len
    ///
    /// Since this is an `AtomicU64` we can use `committed_len` to push the committed length forward
    ///
    /// NOTE: `committed_len` represents the region including the header. So even when zero data is in the time-series
    /// `committed_len` will be `HEADER_SIZE`
    fn committed_len(&self) -> &AtomicU64 {
        &self.header().committed_len
    }

    /// Returns a reference to the [`AppendLog`]'s head length. This is the length of the current "head" of the writer -- aka the data waiting to be committed.
    fn head_len(&self) -> &AtomicU64 {
        &self.header().head_len
    }

    /// The extra data stored in the header
    pub fn extra(&self) -> &E {
        &self.header().extra
    }

    /// The current committed length, excluding the `HEADER_SIZE`
    pub fn len(&self) -> u64 {
        self.committed_len().load(atomic::Ordering::Acquire) - size_of::<Header<E>>() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data(&self) -> &[u8] {
        let slice: &[u8] = unsafe { slice::from_raw_parts(self.map.as_mut_ptr(), self.map.len()) };
        let end = self.committed_len().load(atomic::Ordering::Acquire) as usize;
        &slice[size_of::<Header<E>>()..end]
    }

    pub(crate) fn raw_mmap(&self) -> &Arc<MmapRaw> {
        &self.map
    }
}

pub struct AppendLogWriter<E> {
    pub inner: AppendLog<E>,
}

impl<E: IntoBytes + Immutable> AppendLogWriter<E> {
    /// Grant's access to a mutable buffer of `AppendLog` that can be written into
    pub fn head_mut(&mut self, len: usize) -> Result<&mut [u8], Error> {
        let slice: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(self.inner.map.as_mut_ptr(), self.inner.map.len()) };

        let end = self.inner.committed_len().load(atomic::Ordering::Acquire) as usize;
        let head_end = end.checked_add(len).ok_or(Error::MapOverflow)?;
        if head_end > slice.len() {
            return Err(Error::MapOverflow);
        }
        let slice = slice.get_mut(end..head_end).ok_or(Error::MapOverflow)?;
        self.inner
            .head_len()
            .store(len as u64, atomic::Ordering::Release);
        Ok(slice)
    }

    pub fn head(&self) -> Result<&[u8], Error> {
        let len = self.inner.head_len().load(atomic::Ordering::Acquire) as usize;

        let slice: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(self.inner.map.as_mut_ptr(), self.inner.map.len()) };

        let end = self.inner.committed_len().load(atomic::Ordering::Acquire) as usize;
        let head_end = end.checked_add(len).ok_or(Error::MapOverflow)?;
        if head_end > slice.len() {
            return Err(Error::MapOverflow);
        }
        let slice = slice.get_mut(end..head_end).ok_or(Error::MapOverflow)?;
        Ok(slice)
    }

    /// Commits the existing head into the committed data region, by advancing the `committed_len` value.
    /// The current head value is also copied into the new head region.
    pub fn commit_head_copy(&mut self) -> Result<(), Error> {
        let slice: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(self.inner.map.as_mut_ptr(), self.inner.map.len()) };
        let end = self.inner.committed_len().load(atomic::Ordering::Acquire) as usize;
        let head_len = self.inner.head_len().load(atomic::Ordering::Acquire) as usize;
        let head_end = end.checked_add(head_len).ok_or(Error::MapOverflow)?;

        let new_head_end = head_end.checked_add(head_len).ok_or(Error::MapOverflow)?;
        let heads = slice.get_mut(end..new_head_end).ok_or(Error::MapOverflow)?;
        let (head, new_head) = heads.split_at_mut(head_len);
        new_head.copy_from_slice(head);
        self.inner
            .committed_len()
            .store(head_end as u64, atomic::Ordering::Release);

        Ok(())
    }

    /// Commits the existing head into the committed data region, by advancing the `committed_len` value.
    /// The current head value is also copied into the new head region.
    pub fn commit_head(&mut self, head: &[u8]) -> Result<(), Error> {
        let slice: &mut [u8] =
            unsafe { slice::from_raw_parts_mut(self.inner.map.as_mut_ptr(), self.inner.map.len()) };
        let end = self.inner.committed_len().load(atomic::Ordering::Acquire) as usize;
        let head_len = self.inner.head_len().load(atomic::Ordering::Acquire) as usize;
        let head_end = end.checked_add(head_len).ok_or(Error::MapOverflow)?;

        let new_head_end = head_end.checked_add(head.len()).ok_or(Error::MapOverflow)?;
        let heads = slice.get_mut(end..new_head_end).ok_or(Error::MapOverflow)?;
        let (_, new_head) = heads.split_at_mut(head_len);
        new_head.copy_from_slice(head);
        self.inner
            .head_len()
            .store(head.len() as u64, atomic::Ordering::Release);
        self.inner
            .committed_len()
            .store(head_end as u64, atomic::Ordering::Release);

        Ok(())
    }
}
