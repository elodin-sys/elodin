use std::{
    borrow::Cow,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use impeller2::{buf::UmbraBuf, types::Timestamp};
use impeller2_wkt::MsgMetadata;
use stellarator::sync::WaitQueue;
use zerocopy::{FromBytes, IntoBytes};

use crate::{Error, MetadataExt, append_log::AppendLog};

const DATA_LOG_SEGMENT_LIMIT: u64 = u32::MAX as u64;

#[derive(Clone)]
pub struct MsgLog {
    timestamps: AppendLog<()>,
    bufs: BufLog,
    waker: Arc<WaitQueue>,
    metadata: Option<MsgMetadata>,
    path: PathBuf,
}

impl MsgLog {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::create_with_segment_limit(path, DATA_LOG_SEGMENT_LIMIT)
    }

    pub(crate) fn create_with_segment_limit(
        path: impl AsRef<Path>,
        segment_limit: u64,
    ) -> Result<Self, Error> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let timestamps = AppendLog::create(path.join("timestamps"), ())?;
        let bufs = BufLog::create(path, segment_limit)?;
        let waker = Arc::new(WaitQueue::new());
        let time_series = Self {
            waker,
            timestamps,
            bufs,
            metadata: None,
            path: path.to_path_buf(),
        };
        Ok(time_series)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let timestamps = AppendLog::open(path.join("timestamps"))?;
        let bufs = BufLog::open(path, DATA_LOG_SEGMENT_LIMIT)?;
        let waker = Arc::new(WaitQueue::new());
        let metadata_path = path.join("metadata");
        let metadata = if metadata_path.exists() {
            Some(MsgMetadata::read(metadata_path)?)
        } else {
            None
        };
        let time_series = Self {
            waker,
            timestamps,
            bufs,
            path: path.to_path_buf(),
            metadata,
        };
        Ok(time_series)
    }

    pub fn push(&self, timestamp: Timestamp, msg: &[u8]) -> Result<(), Error> {
        self.bufs.insert_msg(msg)?;
        self.timestamps.write(&timestamp.to_le_bytes())?;
        self.waker.wake_all();
        Ok(())
    }

    pub fn timestamps(&self) -> &[Timestamp] {
        <[Timestamp]>::ref_from_bytes(self.timestamps.get(..).expect("couldn't get full range"))
            .expect("mmep unaligned")
    }

    pub fn get(&self, timestamp: Timestamp) -> Option<Cow<'_, [u8]>> {
        let timestamps = self.timestamps();
        let i = timestamps.binary_search(&timestamp).ok()?;
        self.bufs.get_msg(i)
    }

    pub fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, Cow<'_, [u8]>)> {
        let timestamps = self.timestamps();
        let i = match timestamps.binary_search(&timestamp) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let timestamp = timestamps.get(i)?;
        let buf = self.bufs.get_msg(i)?;
        Some((*timestamp, buf))
    }

    pub fn latest(&self) -> Option<(Timestamp, Cow<'_, [u8]>)> {
        let timestamps = self.timestamps();
        let i = timestamps.len().saturating_sub(1);
        let timestamp = timestamps.get(i)?;
        let buf = self.bufs.get_msg(i)?;
        Some((*timestamp, buf))
    }

    pub fn get_index(&self, index: usize) -> Option<(Timestamp, Cow<'_, [u8]>)> {
        let timestamp = self.timestamps().get(index)?;
        let buf = self.bufs.get_msg(index)?;
        Some((*timestamp, buf))
    }

    pub fn get_range(
        &self,
        range: &std::ops::Range<Timestamp>,
    ) -> impl Iterator<Item = (Timestamp, Cow<'_, [u8]>)> {
        let timestamps = self.timestamps();
        let start_index = match timestamps.binary_search(&range.start) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let end_index = match timestamps.binary_search(&range.end) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        (start_index..=end_index).flat_map(|i| {
            let timestamp = timestamps.get(i)?;
            let buf = self.bufs.get_msg(i)?;
            Some((*timestamp, buf))
        })
    }

    pub async fn wait(&self) {
        let _ = self.waker.wait().await;
    }

    pub fn waiter(&self) -> Arc<WaitQueue> {
        self.waker.clone()
    }

    pub fn sync_all(&self) -> Result<(), Error> {
        self.timestamps.sync_all()?;
        self.bufs.sync_all()?;
        let metadata_path = self.path.join("metadata");
        if metadata_path.exists() {
            File::open(&metadata_path)?.sync_all()?;
        }
        File::open(&self.path)?.sync_all()?;
        Ok(())
    }

    /// Truncate the message log, clearing all messages while preserving metadata.
    ///
    /// Extra `data_log.N` segments are dropped so the next write starts at
    /// `data_log` again.
    pub fn truncate(&self) {
        self.timestamps.truncate();
        self.bufs.truncate();
    }

    pub fn set_metadata(&mut self, metadata: MsgMetadata) -> Result<(), Error> {
        let metadata = self.metadata.insert(metadata);
        let metadata_path = self.path.join("metadata");
        metadata.write(&metadata_path)?;
        Ok(())
    }

    pub fn metadata(&self) -> Option<&MsgMetadata> {
        self.metadata.as_ref()
    }
}

#[derive(Clone)]
struct BufLog {
    offsets: AppendLog<()>,
    data_logs: Arc<DataLogSegment>,
    path: Arc<PathBuf>,
    segment_limit: u64,
    write_lock: Arc<Mutex<()>>,
}

struct DataLogSegment {
    log: AppendLog<()>,
    next: Mutex<Option<Arc<DataLogSegment>>>,
}

impl DataLogSegment {
    fn new(log: AppendLog<()>) -> Self {
        Self {
            log,
            next: Mutex::new(None),
        }
    }
}

impl BufLog {
    fn create(path: &Path, segment_limit: u64) -> Result<Self, Error> {
        if segment_limit == 0 || segment_limit > DATA_LOG_SEGMENT_LIMIT {
            return Err(Error::MapOverflow);
        }
        Ok(Self {
            offsets: AppendLog::create(path.join("offsets"), ())?,
            data_logs: Arc::new(DataLogSegment::new(AppendLog::create(
                data_log_path(path, 0),
                (),
            )?)),
            path: Arc::new(path.to_path_buf()),
            segment_limit,
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    fn open(path: &Path, segment_limit: u64) -> Result<Self, Error> {
        let data_logs = Arc::new(DataLogSegment::new(AppendLog::open(data_log_path(
            path, 0,
        ))?));
        let mut current = data_logs.clone();
        let mut index = 1u32;
        loop {
            let segment_path = data_log_path(path, index);
            if !segment_path.exists() {
                break;
            }
            let next = Arc::new(DataLogSegment::new(AppendLog::open(segment_path)?));
            *current.next.lock().unwrap() = Some(next.clone());
            current = next;
            index = index.checked_add(1).ok_or(Error::MapOverflow)?;
        }
        Ok(Self {
            offsets: AppendLog::open(path.join("offsets"))?,
            data_logs,
            path: Arc::new(path.to_path_buf()),
            segment_limit,
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn bufs(&self) -> &[UmbraBuf] {
        <[UmbraBuf]>::ref_from_bytes(self.offsets.data()).expect("offsets buf invalid")
    }

    /// Clone each link so a concurrent `truncate` cannot drop a segment
    /// still being read. The mmap stays alive until this `Arc` is dropped.
    fn segment(&self, index: u32) -> Option<Arc<DataLogSegment>> {
        let mut segment = self.data_logs.clone();
        for _ in 0..index {
            let next = segment.next.lock().unwrap().clone()?;
            segment = next;
        }
        Some(segment)
    }

    fn last_segment(&self) -> (u32, Arc<DataLogSegment>) {
        let mut index = 0u32;
        let mut segment = self.data_logs.clone();
        loop {
            let next = segment.next.lock().unwrap().clone();
            let Some(next) = next else {
                return (index, segment);
            };
            index += 1;
            segment = next;
        }
    }

    fn sync_all(&self) -> Result<(), Error> {
        self.offsets.sync_all()?;
        let mut segment = Some(self.data_logs.clone());
        while let Some(current) = segment {
            current.log.sync_all()?;
            segment = current.next.lock().unwrap().clone();
        }
        Ok(())
    }

    fn truncate(&self) {
        let _guard = self.write_lock.lock().unwrap();
        self.offsets.truncate();
        self.data_logs.log.truncate();
        let mut current = self.data_logs.next.lock().unwrap().take();
        let mut index = 1u32;
        while let Some(segment) = current {
            current = segment.next.lock().unwrap().take();
            drop(segment);
            let _ = std::fs::remove_file(data_log_path(&self.path, index));
            index = index.saturating_add(1);
        }
    }

    pub fn get_msg(&self, index: usize) -> Option<Cow<'_, [u8]>> {
        let buf = self.bufs().get(index)?;
        match buf.len as usize {
            len @ ..=12 => {
                let inline = unsafe { &buf.data.inline[..len] };
                Some(Cow::Owned(inline.to_vec()))
            }
            len => {
                let segment_index = buf.segment_index()?;
                let offset = buf.offset()? as usize;
                let prefix = buf.prefix()?;
                self.long_payload(segment_index, offset, len, prefix)
                    .map(Cow::Owned)
            }
        }
    }

    fn long_payload(
        &self,
        segment_index: u32,
        offset: usize,
        len: usize,
        prefix: [u8; 4],
    ) -> Option<Vec<u8>> {
        let copy = |segment: &DataLogSegment, start: usize| {
            segment
                .log
                .get(start..start + len)
                .filter(|data| data.get(..4) == Some(&prefix[..]))
                .map(|data| data.to_vec())
        };
        if let Some(data) = self
            .segment(segment_index)
            .and_then(|segment| copy(&segment, offset))
        {
            return Some(data);
        }
        // Single-segment logs that stored the file offset in the middle word
        // (legacy `_index` slot) instead of the last word.
        if segment_index != 0 {
            self.segment(0)
                .and_then(|segment| copy(&segment, segment_index as usize))
        } else {
            None
        }
    }

    pub fn insert_msg(&self, msg: &[u8]) -> Result<(), Error> {
        let len = u32::try_from(msg.len()).map_err(|_| Error::MapOverflow)?;
        let buf = if len > 12 {
            let _guard = self.write_lock.lock().unwrap();
            if msg.len() as u64 > self.segment_limit {
                return Err(Error::MapOverflow);
            }
            let (mut segment_index, mut segment) = self.last_segment();
            if segment.log.len() + msg.len() as u64 > self.segment_limit {
                segment_index = segment_index.checked_add(1).ok_or(Error::MapOverflow)?;
                {
                    let mut next = segment.next.lock().unwrap();
                    if next.is_none() {
                        *next = Some(Arc::new(DataLogSegment::new(AppendLog::create(
                            data_log_path(&self.path, segment_index),
                            (),
                        )?)));
                    }
                }
                segment = self.segment(segment_index).ok_or(Error::BadMessage)?;
            }
            let prefix = msg[..4].try_into().expect("trivial cast failed");
            let offset = u32::try_from(segment.log.write(msg)?).map_err(|_| Error::MapOverflow)?;
            UmbraBuf::with_segment_offset(len, prefix, segment_index, offset)
        } else {
            let mut inline = [0u8; 12];
            inline[..msg.len()].copy_from_slice(msg);
            UmbraBuf::with_inline(len, inline)
        };
        self.offsets.write(buf.as_bytes())?;
        Ok(())
    }
}

fn data_log_path(path: &Path, segment_index: u32) -> PathBuf {
    if segment_index == 0 {
        path.join("data_log")
    } else {
        path.join(format!("data_log.{segment_index}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn rotates_and_reopens_segmented_payloads() {
        let temp = TempDir::new().unwrap();
        let payloads = [[1u8; 20], [2u8; 20], [3u8; 20]];
        let log = MsgLog::create_with_segment_limit(temp.path(), 32).unwrap();

        for (index, payload) in payloads.iter().enumerate() {
            log.push(Timestamp(index as i64), payload).unwrap();
        }

        assert!(temp.path().join("data_log.1").exists());
        assert!(temp.path().join("data_log.2").exists());
        for (index, payload) in payloads.iter().enumerate() {
            assert_eq!(log.get_index(index).unwrap().1.as_ref(), payload);
        }

        drop(log);
        let reopened = MsgLog::open(temp.path()).unwrap();
        for (index, payload) in payloads.iter().enumerate() {
            assert_eq!(reopened.get_index(index).unwrap().1.as_ref(), payload);
        }
    }

    #[test]
    fn truncate_clears_every_segment() {
        let temp = TempDir::new().unwrap();
        let log = MsgLog::create_with_segment_limit(temp.path(), 32).unwrap();
        log.push(Timestamp(1), &[1u8; 20]).unwrap();
        log.push(Timestamp(2), &[2u8; 20]).unwrap();
        assert!(temp.path().join("data_log.1").exists());

        log.truncate();
        assert!(log.timestamps().is_empty());
        assert!(!temp.path().join("data_log.1").exists());
        log.push(Timestamp(3), &[3u8; 20]).unwrap();
        assert_eq!(log.get_index(0).unwrap().0, Timestamp(3));
        assert_eq!(log.get_index(0).unwrap().1.as_ref(), &[3u8; 20]);
        assert!(!temp.path().join("data_log.1").exists());

        drop(log);
        let reopened = MsgLog::open(temp.path()).unwrap();
        assert_eq!(reopened.get_index(0).unwrap().0, Timestamp(3));
        assert_eq!(reopened.get_index(0).unwrap().1.as_ref(), &[3u8; 20]);
        assert!(!temp.path().join("data_log.1").exists());
    }

    #[test]
    fn get_survives_concurrent_truncate() {
        let temp = TempDir::new().unwrap();
        let log = MsgLog::create_with_segment_limit(temp.path(), 32).unwrap();
        for index in 0..8 {
            log.push(Timestamp(index), &[index as u8; 20]).unwrap();
        }
        assert!(temp.path().join("data_log.1").exists());

        let log = Arc::new(log);
        let reader = {
            let log = log.clone();
            std::thread::spawn(move || {
                for _ in 0..8_000 {
                    let _ = log.get_index(0);
                    let _ = log.get_index(7);
                    let _ = log.latest();
                    let _ = log.get_range(&(Timestamp(0)..Timestamp(100))).count();
                }
            })
        };
        std::thread::yield_now();
        log.truncate();
        reader.join().unwrap();
        assert!(log.timestamps().is_empty());
        assert!(!temp.path().join("data_log.1").exists());
    }
}
