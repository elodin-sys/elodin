use crate::Error;
use std::{
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write as _},
    os::fd::AsRawFd,
    path::Path,
    slice::{self, SliceIndex},
    sync::{
        atomic::{self, AtomicU64},
        Arc,
    },
};

const HEADER_SIZE: usize = size_of::<Header>();

/// [`TimeSeries`] is a memory-mapped append-only time-series data file.
/// It works by mapping a large amount of memory ~8g to a sparse file. The file contains a
/// a `Header`, a series of committed data, and finally the uncommitted `head` data.
///
/// ```ignore
/// | Header | Committed Data | Head |
/// ```
///
/// When a `TimeSeries` is created or opened you get access to both a [`TimeSeries`] and an associated [`TimeSeriesWriter`]. [`TimeSeries`] is a read only view into the time series,
/// giving you access to only the committed data. [`TimeSeriesWriter`] provides write access to the `head` of the [`TimeSeries`] with [`TimeSeriesWriter::write_head`]. It also provides the ability to commit the current head using
/// [`TimeSeriesWriter::commit_head_copy`]
#[derive(Clone)]
pub struct TimeSeries {
    map: Arc<memmap2::MmapRaw>,
}

#[repr(C)]
struct Header {
    pub committed_len: AtomicU64,
    pub head_len: AtomicU64,
    pub start_tick: u64,
}

impl TimeSeries {
    pub fn create(
        path: impl AsRef<Path>,
        start_tick: u64,
    ) -> Result<(Self, TimeSeriesWriter), Error> {
        const FILE_SIZE: u64 = 1024 * 1024 * 1024 * 8; // 8gb
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .read(true)
            .open(path)?;
        file.seek(SeekFrom::Start(FILE_SIZE))?;
        file.write_all(&[0])?;
        let map = Arc::new(memmap2::MmapRaw::map_raw(file.as_raw_fd())?);
        let map = Self { map };
        map.committed_len()
            .store(HEADER_SIZE as u64, atomic::Ordering::SeqCst);
        map.head_len().store(0, atomic::Ordering::SeqCst);
        unsafe {
            ((&raw const map.header().start_tick) as *mut u64).write(start_tick);
        }
        let writer = TimeSeriesWriter { inner: map.clone() };
        Ok((map, writer))
    }

    pub fn open(path: impl AsRef<Path>) -> Result<(Self, TimeSeriesWriter), Error> {
        let file = OpenOptions::new().write(true).read(true).open(path)?;
        let map = Arc::new(memmap2::MmapRaw::map_raw(file.as_raw_fd())?);
        let map = Self { map };
        let writer = TimeSeriesWriter { inner: map.clone() };
        Ok((map, writer))
    }

    /// Returns a slice of data offset into the committed data region of the [`TimeSeries`]
    pub fn get(&self, range: impl SliceIndex<[u8], Output = [u8]>) -> Option<&'_ [u8]> {
        let slice: &[u8] = unsafe { slice::from_raw_parts(self.map.as_mut_ptr(), self.map.len()) };
        let end = self.committed_len().load(atomic::Ordering::Acquire) as usize;
        slice.get(HEADER_SIZE..end)?.get(range)
    }

    fn header(&self) -> &Header {
        let ptr = self.map.as_mut_ptr();
        unsafe { &*(ptr as *const Header) }
    }

    /// Returns a reference to the [`TimeSeries`]'s committed_len
    ///
    /// Since this is an `AtomicU64` we can use `committed_len` to push the committed length forward
    ///
    /// NOTE: `committed_len` represents the region including the header. So even when zero data is in the time-series
    /// `committed_len` will be `HEADER_SIZE`
    fn committed_len(&self) -> &AtomicU64 {
        &self.header().committed_len
    }

    /// Returns a reference to the [`TimeSeries`]'s head length. This is the length of the current "head" of the writer -- aka the data waiting to be committed.
    fn head_len(&self) -> &AtomicU64 {
        &self.header().head_len
    }

    /// The starting tick of the time-series
    pub fn start_tick(&self) -> u64 {
        self.header().start_tick
    }

    /// The current committed length, excluding the `HEADER_SIZE`
    pub fn len(&self) -> u64 {
        self.committed_len().load(atomic::Ordering::Acquire) - HEADER_SIZE as u64
    }
}

pub struct TimeSeriesWriter {
    inner: TimeSeries,
}

impl TimeSeriesWriter {
    /// Grant's access to a mutable buffer of `TimeSeries` that can be written into
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
}
