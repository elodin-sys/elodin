//! Source PTS → DB timestamp mapping.
//!
//! Mirrors `elodinsink`'s scheme (`fsw/gstreamer/src/lib.rs`): anchor to the
//! DB's `last_updated` at the first frame, add the per-frame PTS delta, and
//! enforce **strict monotonicity** because `MsgLog` binary-searches timestamps.
//! All values are microseconds (Elodin `Timestamp` is `i64` microseconds of
//! simulation time, not wall clock).

/// Maps stream presentation timestamps onto the DB timeline.
#[derive(Clone, Debug)]
pub struct ClockMapper {
    base_us: i64,
    first_pts_us: Option<i64>,
    last_written_us: Option<i64>,
}

impl ClockMapper {
    /// Creates a mapper anchored to `base_us` (the DB's `last_updated` at the
    /// moment the first frame is about to be written).
    pub fn new(base_us: i64) -> Self {
        Self {
            base_us,
            first_pts_us: None,
            last_written_us: None,
        }
    }

    /// Maps a source PTS (microseconds) to a strictly-increasing DB timestamp
    /// (microseconds). The first call establishes the PTS origin and returns
    /// `base_us`; later calls return `base_us + (pts - first_pts)`, bumped by
    /// one microsecond if needed to stay strictly above the last value.
    pub fn map(&mut self, pts_us: i64) -> i64 {
        let first = *self.first_pts_us.get_or_insert(pts_us);
        let offset = pts_us.saturating_sub(first).max(0);
        let mut ts = self.base_us.saturating_add(offset);
        if let Some(last) = self.last_written_us
            && ts <= last
        {
            ts = last.saturating_add(1);
        }
        self.last_written_us = Some(ts);
        ts
    }

    /// Re-anchors after a reconnect: refresh the base and reset the PTS origin,
    /// while preserving monotonicity against already-written timestamps (the new
    /// base + a fresh PTS origin can otherwise overlap earlier frames).
    pub fn reanchor(&mut self, base_us: i64) {
        self.base_us = base_us;
        self.first_pts_us = None;
    }

    /// The most recent timestamp returned by [`map`](Self::map), if any.
    pub fn last_written_us(&self) -> Option<i64> {
        self.last_written_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_frame_lands_on_base() {
        let mut c = ClockMapper::new(1_000_000);
        assert_eq!(c.map(50_000), 1_000_000);
    }

    #[test]
    fn subsequent_frames_add_pts_delta() {
        let mut c = ClockMapper::new(1_000_000);
        assert_eq!(c.map(50_000), 1_000_000);
        assert_eq!(c.map(83_333), 1_033_333);
        assert_eq!(c.map(116_666), 1_066_666);
    }

    #[test]
    fn equal_pts_is_bumped_for_strict_monotonicity() {
        let mut c = ClockMapper::new(1_000_000);
        assert_eq!(c.map(50_000), 1_000_000);
        assert_eq!(c.map(50_000), 1_000_001);
        assert_eq!(c.map(50_000), 1_000_002);
    }

    #[test]
    fn decreasing_pts_is_bumped_not_rewound() {
        let mut c = ClockMapper::new(1_000_000);
        assert_eq!(c.map(100_000), 1_000_000);
        assert_eq!(c.map(40_000), 1_000_001);
    }

    #[test]
    fn reanchor_resets_origin_but_keeps_monotonic() {
        let mut c = ClockMapper::new(1_000_000);
        assert_eq!(c.map(50_000), 1_000_000);
        assert_eq!(c.map(83_333), 1_033_333);

        // Reconnect: DB advanced its clock; new frames start from a fresh PTS.
        c.reanchor(2_000_000);
        assert_eq!(c.map(500_000), 2_000_000);
        assert_eq!(c.map(533_333), 2_033_333);
    }

    #[test]
    fn reanchor_below_last_written_still_advances() {
        let mut c = ClockMapper::new(5_000_000);
        assert_eq!(c.map(0), 5_000_000);
        // A reconnect that anchors *behind* the last written timestamp must not
        // emit a non-increasing value.
        c.reanchor(1_000_000);
        assert_eq!(c.map(0), 5_000_001);
    }
}
