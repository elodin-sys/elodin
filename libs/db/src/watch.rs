//! Change cursors: a reader registers the series it mirrors once, and every append by anyone
//! else marks them, so the reader drains what changed instead of polling each series.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};

use stellarator::sync::WaitQueue;

/// Series one reader watches, by slot. An append marks its slot until the reader drains it;
/// repeated appends before a drain coalesce into one mark.
pub struct WatchCursor {
    marks: Box<[AtomicU64]>,
    pending: AtomicBool,
    changed: WaitQueue,
    coalesced: AtomicU64,
}

impl WatchCursor {
    /// A cursor with `slots` unmarked slots, assigned with `TimeSeries::watch`.
    pub fn new(slots: usize) -> Self {
        Self {
            marks: (0..slots.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            pending: AtomicBool::new(false),
            changed: WaitQueue::new(),
            coalesced: AtomicU64::new(0),
        }
    }

    fn mark(&self, slot: usize) {
        let bit = 1u64 << (slot % 64);
        if self.marks[slot / 64].fetch_or(bit, Ordering::AcqRel) & bit != 0 {
            self.coalesced.fetch_add(1, Ordering::Relaxed);
        }
        if !self.pending.swap(true, Ordering::AcqRel) {
            self.changed.wake_all();
        }
    }

    /// Calls `changed` with every slot marked since the last drain, clears the marks, and
    /// returns how many there were. Touches no mark when nothing changed.
    pub fn drain(&self, mut changed: impl FnMut(usize)) -> usize {
        if !self.pending.swap(false, Ordering::AcqRel) {
            return 0;
        }
        let mut drained = 0;
        for (word, marks) in self.marks.iter().enumerate() {
            let mut bits = marks.swap(0, Ordering::AcqRel);
            while bits != 0 {
                changed(word * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
                drained += 1;
            }
        }
        drained
    }

    /// Whether `slot` changed since the last drain.
    pub fn is_marked(&self, slot: usize) -> bool {
        self.marks[slot / 64].load(Ordering::Acquire) & (1 << (slot % 64)) != 0
    }

    /// Waits until a slot is marked.
    pub async fn changed(&self) {
        while !self.pending.load(Ordering::Acquire) {
            let wait = self.changed.wait();
            if self.pending.load(Ordering::Acquire) {
                break;
            }
            let _ = wait.await;
        }
    }

    /// Appends since the last call that found their slot already marked.
    pub fn take_coalesced(&self) -> u64 {
        self.coalesced.swap(0, Ordering::Relaxed)
    }
}

/// The cursors watching one series, each with the slot it gave the series.
#[derive(Clone, Default)]
pub(crate) struct Watchers(Arc<RwLock<Vec<CursorSlot>>>);

type CursorSlot = (Weak<WatchCursor>, usize);

impl Watchers {
    pub(crate) fn add(&self, cursor: &Arc<WatchCursor>, slot: usize) {
        let mut watchers = self.0.write().unwrap();
        watchers.retain(|(cursor, _)| cursor.strong_count() > 0);
        watchers.push((Arc::downgrade(cursor), slot));
    }

    /// Marks every cursor but `writer`'s own.
    pub(crate) fn mark(&self, writer: Option<&WatchCursor>) {
        for (cursor, slot) in self.0.read().unwrap().iter() {
            if writer.is_some_and(|writer| std::ptr::eq(writer, cursor.as_ptr())) {
                continue;
            }
            if let Some(cursor) = cursor.upgrade() {
                cursor.mark(*slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;
    use impeller::types::Timestamp;

    #[test]
    fn marks_appends_by_others_once_per_drain() {
        let dir = tempfile::tempdir().unwrap();
        let series = TimeSeries::create(dir.path(), "test".into(), Timestamp(0), 8).unwrap();
        let cursor = Arc::new(WatchCursor::new(130));
        series.watch(&cursor, 129);
        assert_eq!(cursor.drain(|_| panic!("nothing appended")), 0);

        series.push_buf(Timestamp(1), &1u64.to_le_bytes()).unwrap();
        series.push_buf(Timestamp(2), &2u64.to_le_bytes()).unwrap();
        assert!(cursor.is_marked(129));
        let mut drained = Vec::new();
        assert_eq!(cursor.drain(|slot| drained.push(slot)), 1);
        assert_eq!((drained, cursor.take_coalesced()), (vec![129], 1));

        series
            .push_buf_from(Some(&cursor), Timestamp(3), &3u64.to_le_bytes())
            .unwrap();
        assert_eq!(cursor.drain(|_| panic!("own append")), 0);
    }
}
