use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use impeller2::{buf::UmbraBuf, types::Timestamp};
use impeller2_wkt::MsgMetadata;
use stellarator::sync::WaitQueue;
use zerocopy::{FromBytes, IntoBytes};

use crate::{Error, MetadataExt, append_log::AppendLog};

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
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let timestamps = AppendLog::create(path.join("timestamps"), ())?;
        let offsets = AppendLog::create(path.join("offsets"), ())?;
        let data_log = AppendLog::create(path.join("data_log"), ())?;
        let waker = Arc::new(WaitQueue::new());
        let time_series = Self {
            waker,
            timestamps,
            bufs: BufLog { offsets, data_log },
            metadata: None,
            path: path.to_path_buf(),
        };
        Ok(time_series)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let timestamps = AppendLog::open(path.join("timestamps"))?;
        let offsets = AppendLog::open(path.join("offsets"))?;
        let data_log = AppendLog::open(path.join("data_log"))?;
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
            bufs: BufLog { offsets, data_log },
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

    pub fn get(&self, timestamp: Timestamp) -> Option<&[u8]> {
        let timestamps = self.timestamps();
        let i = timestamps.binary_search(&timestamp).ok()?;
        self.bufs.get_msg(i)
    }

    pub fn get_nearest(&self, timestamp: Timestamp) -> Option<(Timestamp, &[u8])> {
        let timestamps = self.timestamps();
        let i = match timestamps.binary_search(&timestamp) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let timestamp = timestamps.get(i)?;
        let buf = self.bufs.get_msg(i)?;
        Some((*timestamp, buf))
    }

    pub fn latest(&self) -> Option<(Timestamp, &[u8])> {
        let timestamps = self.timestamps();
        let i = timestamps.len().saturating_sub(1);
        let timestamp = timestamps.get(i)?;
        let buf = self.bufs.get_msg(i)?;
        Some((*timestamp, buf))
    }

    pub fn get_range(
        &self,
        range: std::ops::Range<Timestamp>,
    ) -> impl Iterator<Item = (Timestamp, &[u8])> {
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
    data_log: AppendLog<()>,
}

impl BufLog {
    pub fn bufs(&self) -> &[UmbraBuf] {
        <[UmbraBuf]>::ref_from_bytes(self.offsets.data()).expect("offsets buf invalid")
    }

    pub fn get_msg(&self, index: usize) -> Option<&[u8]> {
        let buf = self.bufs().get(index)?;
        let data = match buf.len as usize {
            len @ ..=12 => unsafe { &buf.data.inline[..len] },
            len => {
                let offset = unsafe { buf.data.offset.offset } as usize;
                self.data_log.get(offset..offset + len)?
            }
        };
        Some(data)
    }

    pub fn insert_msg(&self, msg: &[u8]) -> Result<(), Error> {
        let len = msg.len() as u32;
        let buf = if len > 12 {
            let prefix = msg[..4].try_into().expect("trivial cast failed");
            let offset = self.data_log.write(msg)?;
            UmbraBuf::with_offset(len, prefix, offset as u32)
        } else {
            let mut inline = [0u8; 12];
            inline[..msg.len()].copy_from_slice(msg);
            UmbraBuf::with_inline(len, inline)
        };
        self.offsets.write(buf.as_bytes())?;
        Ok(())
    }
}
