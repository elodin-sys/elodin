use std::{ops::Range, path::Path, sync::Arc};

use impeller2::types::Timestamp;
use stellarator::sync::WaitQueue;
use tracing::warn;
use zerocopy::FromBytes;

use crate::{Error, append_log::AppendLog};

#[derive(Clone)]
pub struct TimeSeries {
    index: AppendLog<Timestamp>,
    data: AppendLog<u64>,
    data_waker: Arc<WaitQueue>,
}

impl TimeSeries {
    pub fn create(
        path: impl AsRef<Path>,
        start_timestamp: Timestamp,
        element_size: u64,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let index = AppendLog::create(path.join("index"), start_timestamp)?;
        let data = AppendLog::create(path.join("data"), element_size)?;
        let data_waker = Arc::new(WaitQueue::new());
        let time_series = Self {
            index,
            data,
            data_waker: data_waker.clone(),
        };
        Ok(time_series)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let index = AppendLog::open(path.join("index"))?;
        let data = AppendLog::open(path.join("data"))?;
        let data_waker = Arc::new(WaitQueue::new());
        let time_series = Self {
            index,
            data,
            data_waker: data_waker.clone(),
        };
        Ok(time_series)
    }

    pub fn start_timestamp(&self) -> Timestamp {
        *self.index.extra()
    }

    fn timestamps(&self) -> &[Timestamp] {
        <[Timestamp]>::ref_from_bytes(self.index.get(..).expect("couldn't get full range"))
            .expect("mmep unaligned")
    }

    fn element_size(&self) -> usize {
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

    pub fn get_range(&self, range: Range<Timestamp>) -> Option<(&[Timestamp], &[u8])> {
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

    pub fn push_buf(&self, timestamp: Timestamp, buf: &[u8]) -> Result<(), Error> {
        let len = self.index.len() as usize;

        // check if timestamp is greater than the last timestamp
        // to ensure index is ordered
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
                warn!(?last_timestamp, ?timestamp, "time travel");
                return Err(Error::TimeTravel);
            }
        }

        // write new data to head of data writer
        self.data.write(buf)?;

        // always write index last so we get consistent reads
        self.index.write(&timestamp.to_le_bytes())?;

        self.data_waker.wake_all();
        Ok(())
    }
}
