//! Lock-free coordination based on Compare and Swap atomics

use const_init::ConstInit;

use super::{Coord, ReadGrantError, WriteGrantError};
use core::{
    cmp::min,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Coordination using CAS atomics
pub struct AtomicCoord {
    /// Where the next byte will be written
    write: AtomicUsize,

    /// Where the next byte will be read from
    read: AtomicUsize,

    /// Used in the inverted case to mark the end of the
    /// readable streak. Otherwise will == sizeof::<self.buf>().
    /// Writer is responsible for placing this at the correct
    /// place when entering an inverted condition, and Reader
    /// is responsible for moving it back to sizeof::<self.buf>()
    /// when exiting the inverted condition
    last: AtomicUsize,

    /// Used by the Writer to remember what bytes are currently
    /// allowed to be written to, but are not yet ready to be
    /// read from
    reserve: AtomicUsize,

    /// Is there an active read grant?
    read_in_progress: AtomicBool,

    /// Is there an active write grant?
    write_in_progress: AtomicBool,
}

impl AtomicCoord {
    /// Create a new coordination structure that uses atomic CAS operations
    pub const fn new() -> Self {
        Self {
            write: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
            last: AtomicUsize::new(0),
            reserve: AtomicUsize::new(0),
            read_in_progress: AtomicBool::new(false),
            write_in_progress: AtomicBool::new(false),
        }
    }
}

impl Default for AtomicCoord {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstInit for AtomicCoord {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self::new();
}

unsafe impl Coord for AtomicCoord {
    fn reset(&self) {
        // Re-initialize the buffer (not totally needed, but nice to do)
        self.write.store(0, Ordering::Release);
        self.read.store(0, Ordering::Release);
        self.reserve.store(0, Ordering::Release);
        self.last.store(0, Ordering::Release);
    }

    fn grant_max_remaining(
        &self,
        capacity: usize,
        mut sz: usize,
    ) -> Result<(usize, usize), WriteGrantError> {
        if self.write_in_progress.swap(true, Ordering::AcqRel) {
            return Err(WriteGrantError::GrantInProgress);
        }

        // Writer component. Must never write to `read`,
        // be careful writing to `load`
        let write = self.write.load(Ordering::Acquire);
        let read = self.read.load(Ordering::Acquire);
        let max = capacity;

        let already_inverted = write < read;

        let start = if already_inverted {
            // In inverted case, read is always > write
            let remain = read - write - 1;

            if remain != 0 {
                sz = min(remain, sz);
                write
            } else {
                // Inverted, no room is available
                self.write_in_progress.store(false, Ordering::Release);
                return Err(WriteGrantError::InsufficientSize);
            }
        } else {
            #[allow(clippy::collapsible_if)]
            if write != max {
                // Some (or all) room remaining in un-inverted case
                sz = min(max - write, sz);
                write
            } else {
                // Not inverted, but need to go inverted

                // NOTE: We check read > 1, NOT read >= 1, because
                // write must never == read in an inverted condition, since
                // we will then not be able to tell if we are inverted or not
                if read > 1 {
                    sz = min(read - 1, sz);
                    0
                } else {
                    // Not invertible, no space
                    self.write_in_progress.store(false, Ordering::Release);
                    return Err(WriteGrantError::InsufficientSize);
                }
            }
        };

        // Safe write, only viewed by this task
        self.reserve.store(start + sz, Ordering::Release);

        Ok((start, sz))
    }

    fn grant_exact(&self, capacity: usize, sz: usize) -> Result<usize, WriteGrantError> {
        if self.write_in_progress.swap(true, Ordering::AcqRel) {
            return Err(WriteGrantError::GrantInProgress);
        }

        // Writer component. Must never write to `read`,
        // be careful writing to `load`
        let write = self.write.load(Ordering::Acquire);
        let read = self.read.load(Ordering::Acquire);
        let max = capacity;
        let already_inverted = write < read;

        let start = if already_inverted {
            if (write + sz) < read {
                // Inverted, room is still available
                write
            } else {
                // Inverted, no room is available
                self.write_in_progress.store(false, Ordering::Release);
                return Err(WriteGrantError::InsufficientSize);
            }
        } else {
            #[allow(clippy::collapsible_if)]
            if write + sz <= max {
                // Non inverted condition
                write
            } else {
                // Not inverted, but need to go inverted

                // NOTE: We check sz < read, NOT <=, because
                // write must never == read in an inverted condition, since
                // we will then not be able to tell if we are inverted or not
                if sz < read {
                    // Invertible situation
                    0
                } else {
                    // Not invertible, no space
                    self.write_in_progress.store(false, Ordering::Release);
                    return Err(WriteGrantError::InsufficientSize);
                }
            }
        };

        // Safe write, only viewed by this task
        self.reserve.store(start + sz, Ordering::Release);

        Ok(start)
    }

    fn read(&self) -> Result<(usize, usize), ReadGrantError> {
        if self.read_in_progress.swap(true, Ordering::AcqRel) {
            return Err(ReadGrantError::GrantInProgress);
        }

        let write = self.write.load(Ordering::Acquire);
        let last = self.last.load(Ordering::Acquire);
        let mut read = self.read.load(Ordering::Acquire);

        // Resolve the inverted case or end of read
        if (read == last) && (write < read) {
            read = 0;
            // This has some room for error, the other thread reads this
            // Impact to Grant:
            //   Grant checks if read < write to see if inverted. If not inverted, but
            //     no space left, Grant will initiate an inversion, but will not trigger it
            // Impact to Commit:
            //   Commit does not check read, but if Grant has started an inversion,
            //   grant could move Last to the prior write position
            // MOVING READ BACKWARDS!
            self.read.store(0, Ordering::Release);
        }

        let sz = if write < read {
            // Inverted, only believe last
            last
        } else {
            // Not inverted, only believe write
            write
        } - read;

        if sz == 0 {
            self.read_in_progress.store(false, Ordering::Release);
            return Err(ReadGrantError::Empty);
        }

        Ok((read, sz))
    }

    fn commit_inner(&self, capacity: usize, grant_len: usize, used: usize) {
        // If there is no grant in progress, return early. This
        // generally means we are dropping the grant within a
        // wrapper structure
        if !self.write_in_progress.load(Ordering::Acquire) {
            return;
        }

        // Writer component. Must never write to READ,
        // be careful writing to LAST

        // Saturate the grant commit
        let len = grant_len;
        let used = min(len, used);

        let write = self.write.load(Ordering::Acquire);
        self.reserve.fetch_sub(len - used, Ordering::AcqRel);

        let max = capacity;
        let last = self.last.load(Ordering::Acquire);
        let new_write = self.reserve.load(Ordering::Acquire);

        if (new_write < write) && (write != max) {
            // We have already wrapped, but we are skipping some bytes at the end of the ring.
            // Mark `last` where the write pointer used to be to hold the line here
            self.last.store(write, Ordering::Release);
        } else if new_write > last {
            // We're about to pass the last pointer, which was previously the artificial
            // end of the ring. Now that we've passed it, we can "unlock" the section
            // that was previously skipped.
            //
            // Since new_write is strictly larger than last, it is safe to move this as
            // the other thread will still be halted by the (about to be updated) write
            // value
            self.last.store(max, Ordering::Release);
        }
        // else: If new_write == last, either:
        // * last == max, so no need to write, OR
        // * If we write in the end chunk again, we'll update last to max next time
        // * If we write to the start chunk in a wrap, we'll update last when we
        //     move write backwards

        // Write must be updated AFTER last, otherwise read could think it was
        // time to invert early!
        self.write.store(new_write, Ordering::Release);

        // Allow subsequent grants
        self.write_in_progress.store(false, Ordering::Release);
    }

    fn release_inner(&self, used: usize) {
        // If there is no grant in progress, return early. This
        // generally means we are dropping the grant within a
        // wrapper structure
        if !self.read_in_progress.load(Ordering::Acquire) {
            return;
        }

        // // This should always be checked by the public interfaces
        // debug_assert!(used <= self.buf.len());

        // This should be fine, purely incrementing
        let _ = self.read.fetch_add(used, Ordering::Release);

        self.read_in_progress.store(false, Ordering::Release);
    }
}
