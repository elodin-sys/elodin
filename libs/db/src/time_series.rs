use std::{ops::Range, path::Path, sync::Arc, sync::RwLock};

use impeller2::types::Timestamp;
use stellarator::sync::WaitQueue;
use tracing::warn;
use zerocopy::FromBytes;

use crate::{Error, append_log::AppendLog};

#[derive(Clone)]
pub struct TimeSeries {
    name: Arc<RwLock<String>>,
    index: AppendLog<Timestamp>,
    data: AppendLog<u64>,
    data_waker: Arc<WaitQueue>,
}

impl TimeSeries {
    pub fn create(
        path: impl AsRef<Path>,
        name: String,
        start_timestamp: Timestamp,
        element_size: u64,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let index = AppendLog::create(path.join("index"), start_timestamp)?;
        let data = AppendLog::create(path.join("data"), element_size)?;
        let data_waker = Arc::new(WaitQueue::new());
        let time_series = Self {
            name: Arc::new(RwLock::new(name)),
            index,
            data,
            data_waker: data_waker.clone(),
        };
        Ok(time_series)
    }

    pub fn open(path: impl AsRef<Path>, name: String) -> Result<Self, Error> {
        let path = path.as_ref();
        let index = AppendLog::open(path.join("index"))?;
        let data = AppendLog::open(path.join("data"))?;
        let data_waker = Arc::new(WaitQueue::new());
        let time_series = Self {
            name: Arc::new(RwLock::new(name)),
            index,
            data,
            data_waker: data_waker.clone(),
        };
        Ok(time_series)
    }

    pub fn start_timestamp(&self) -> Timestamp {
        let index_ts = *self.index.extra();
        match self.timestamps().first() {
            Some(first_ts) => index_ts.min(*first_ts),
            None => index_ts,
        }
    }

    /// Returns the raw `extra` value from the index AppendLog header.
    /// This is the start_timestamp set at creation time.
    pub fn index_extra(&self) -> &Timestamp {
        self.index.extra()
    }

    /// Overwrite the start_timestamp stored in the index file header.
    /// Used by the follower to match the source's start_timestamp exactly.
    pub fn set_start_timestamp(&self, ts: Timestamp) {
        self.index.set_extra(ts);
    }

    fn timestamps(&self) -> &[Timestamp] {
        <[Timestamp]>::ref_from_bytes(self.index.get(..).expect("couldn't get full range"))
            .expect("mmep unaligned")
    }

    pub fn element_size(&self) -> usize {
        *self.data.extra() as usize
    }

    pub fn get(&self, timestamp: Timestamp) -> Option<&[u8]> {
        let timestamps = self.timestamps();
        let index = timestamps.binary_search(&timestamp).ok()?;
        let element_size = self.element_size();
        let i = index * element_size;
        self.data.get(i..i + element_size)
    }

    pub fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, &[u8])> {
        let timestamps = self.timestamps();
        let index = match timestamps.binary_search(&timestamp) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let element_size = self.element_size();
        let timestamp = timestamps.get(index)?;
        let i = index * element_size;
        let buf = self.data.get(i..i + element_size)?;
        Some((*timestamp, buf))
    }

    pub fn get_range(&self, range: &Range<Timestamp>) -> Option<(&[Timestamp], &[u8])> {
        let timestamps = self.timestamps();

        let start = range.start;
        let end = range.end;
        let start_index = match timestamps.binary_search(&start) {
            Ok(i) => i,
            Err(i) => i,
        };

        let end_index = match timestamps.binary_search(&end) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        let timestamps = timestamps.get(start_index..=end_index)?;
        let element_size = self.element_size();
        let data = self
            .data
            .get(start_index * element_size..end_index.saturating_add(1) * element_size)?;

        Some((timestamps, data))
    }

    pub async fn wait(&self) {
        let _ = self.data_waker.wait().await;
    }

    pub fn waiter(&self) -> Arc<WaitQueue> {
        self.data_waker.clone()
    }

    pub fn latest(&self) -> Option<(&Timestamp, &[u8])> {
        if self.index.is_empty() {
            return None;
        }
        let index = self.index.len() as usize / size_of::<Timestamp>() - 1;
        let element_size = self.element_size();
        let i = index * element_size;
        let data = self.data.get(i..i + element_size)?;
        let timestamp = self.timestamps().get(index)?;
        Some((timestamp, data))
    }

    pub(crate) fn data(&self) -> &AppendLog<u64> {
        &self.data
    }

    pub(crate) fn index(&self) -> &AppendLog<Timestamp> {
        &self.index
    }

    pub fn sync_all(&self) -> Result<(), Error> {
        self.index.sync_all()?;
        self.data.sync_all()?;
        Ok(())
    }

    /// Update the human-readable name for this time series.
    ///
    /// This is used to provide better context in warning messages.
    pub fn set_name(&self, name: String) {
        if let Ok(mut guard) = self.name.write() {
            *guard = name;
        }
    }

    /// Truncate the time series, clearing all data while preserving the schema.
    ///
    /// This resets both the index and data append logs, effectively removing
    /// all stored time-series data without deallocating the underlying files.
    pub fn truncate(&self) {
        self.index.truncate();
        self.data.truncate();
    }

    pub fn push_buf(&self, timestamp: Timestamp, buf: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "profile")]
        let _pb_start = std::time::Instant::now();

        let _span = tracing::trace_span!("push_buf").entered();
        let len = self.index.len() as usize;

        #[cfg(feature = "profile")]
        let _ts_check_start = std::time::Instant::now();

        if len > 0 {
            let last_timestamp = self
                .index
                .get(len - size_of::<i64>()..len)
                .expect("couldn't find last timestamp");
            let last_timestamp = Timestamp::from_le_bytes(
                last_timestamp
                    .try_into()
                    .expect("last_timestamp was wrong size"),
            );
            if last_timestamp > timestamp {
                let component_name = self.name.read().map(|g| g.clone()).unwrap_or_default();
                warn!(component = %component_name, ?last_timestamp, ?timestamp, "time travel");
                return Err(Error::TimeTravel);
            }
        }

        #[cfg(feature = "profile")]
        {
            use crate::profile_stats;
            profile_stats::PUSH_BUF_TS_CHECK_NS.fetch_add(
                _ts_check_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        #[cfg(feature = "profile")]
        let _dw_start = std::time::Instant::now();

        self.data.write(buf)?;

        #[cfg(feature = "profile")]
        {
            use crate::profile_stats;
            profile_stats::PUSH_BUF_DATA_WRITE_NS.fetch_add(
                _dw_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        #[cfg(feature = "profile")]
        let _iw_start = std::time::Instant::now();

        self.index.write(&timestamp.to_le_bytes())?;

        #[cfg(feature = "profile")]
        {
            use crate::profile_stats;
            profile_stats::PUSH_BUF_INDEX_WRITE_NS.fetch_add(
                _iw_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        #[cfg(feature = "profile")]
        let _wake_start = std::time::Instant::now();

        self.data_waker.wake_all();

        #[cfg(feature = "profile")]
        {
            use crate::profile_stats;
            profile_stats::WAKE_ALL_DATA_WAKER_NS.fetch_add(
                _wake_start.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            profile_stats::WAKE_ALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let pb_ns = _pb_start.elapsed().as_nanos() as u64;
            profile_stats::PUSH_BUF_NS.fetch_add(pb_ns, std::sync::atomic::Ordering::Relaxed);
            profile_stats::PUSH_BUF_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            profile_stats::PUSH_BUF_MAX_NS.fetch_max(pb_ns, std::sync::atomic::Ordering::Relaxed);
        }

        Ok(())
    }

    /// Returns the number of samples currently stored.
    pub fn sample_count(&self) -> usize {
        self.index.len() as usize / size_of::<i64>()
    }
}
