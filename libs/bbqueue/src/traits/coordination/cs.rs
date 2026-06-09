//! Mutex/Critical section based coordination
//!
//! This is provided so `bbqueue` is usable on bare metal targets that don't
//! have CAS atomics, like `cortex-m0`/`thumbv6m` targets.

use const_init::ConstInit;

use super::{Coord, ReadGrantError, WriteGrantError};
use core::{
    cmp::min,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

/// Coordination that uses a critical section to perform coordination operations
///
/// The critical section is only taken for a short time to obtain or release grants,
/// not for the entire duration of the grant.
pub struct CsCoord {
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

impl CsCoord {
    /// Create a new critical-section based coordination
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

impl Default for CsCoord {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstInit for CsCoord {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self::new();
}

unsafe impl Coord for CsCoord {
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
        critical_section::with(|_cs| {
            if self.write_in_progress.load(Ordering::Relaxed) {
                return Err(WriteGrantError::GrantInProgress);
            }
            self.write_in_progress.store(true, Ordering::Relaxed);

            // Writer component. Must never write to `read`,
            // be careful writing to `load`
            let write = self.write.load(Ordering::Relaxed);
            let read = self.read.load(Ordering::Relaxed);
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
                    self.write_in_progress.store(false, Ordering::Relaxed);
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
                        self.write_in_progress.store(false, Ordering::Relaxed);
                        return Err(WriteGrantError::InsufficientSize);
                    }
                }
            };

            // Safe write, only viewed by this task
            self.reserve.store(start + sz, Ordering::Relaxed);

            Ok((start, sz))
        })
    }

    fn grant_exact(&self, capacity: usize, sz: usize) -> Result<usize, WriteGrantError> {
        critical_section::with(|_cs| {
            if self.write_in_progress.load(Ordering::Relaxed) {
                return Err(WriteGrantError::GrantInProgress);
            }
            self.write_in_progress.store(true, Ordering::Relaxed);

            // Writer component. Must never write to `read`,
            // be careful writing to `load`
            let write = self.write.load(Ordering::Relaxed);
            let read = self.read.load(Ordering::Relaxed);
            let max = capacity;
            let already_inverted = write < read;

            let start = if already_inverted {
                if (write + sz) < read {
                    // Inverted, room is still available
                    write
                } else {
                    // Inverted, no room is available
                    self.write_in_progress.store(false, Ordering::Relaxed);
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
                        self.write_in_progress.store(false, Ordering::Relaxed);
                        return Err(WriteGrantError::InsufficientSize);
                    }
                }
            };

            // Safe write, only viewed by this task
            self.reserve.store(start + sz, Ordering::Relaxed);

            Ok(start)
        })
    }

    fn read(&self) -> Result<(usize, usize), ReadGrantError> {
        critical_section::with(|_cs| {
            if self.read_in_progress.load(Ordering::Relaxed) {
                return Err(ReadGrantError::GrantInProgress);
            }
            self.read_in_progress.store(true, Ordering::Relaxed);

            let write = self.write.load(Ordering::Relaxed);
            let last = self.last.load(Ordering::Relaxed);
            let mut read = self.read.load(Ordering::Relaxed);

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
                self.read.store(0, Ordering::Relaxed);
            }

            let sz = if write < read {
                // Inverted, only believe last
                last
            } else {
                // Not inverted, only believe write
                write
            } - read;

            if sz == 0 {
                self.read_in_progress.store(false, Ordering::Relaxed);
                return Err(ReadGrantError::Empty);
            }

            Ok((read, sz))
        })
    }

    fn commit_inner(&self, capacity: usize, grant_len: usize, used: usize) {
        critical_section::with(|_cs| {
            // If there is no grant in progress, return early. This
            // generally means we are dropping the grant within a
            // wrapper structure
            if !self.write_in_progress.load(Ordering::Relaxed) {
                return;
            }

            // Writer component. Must never write to READ,
            // be careful writing to LAST

            // Saturate the grant commit
            let len = grant_len;
            let used = min(len, used);

            let write = self.write.load(Ordering::Relaxed);
            let old_reserve = self.reserve.load(Ordering::Relaxed);
            self.reserve
                .store(old_reserve - (len - used), Ordering::Relaxed);

            let max = capacity;
            let last = self.last.load(Ordering::Relaxed);
            let new_write = self.reserve.load(Ordering::Relaxed);

            if (new_write < write) && (write != max) {
                // We have already wrapped, but we are skipping some bytes at the end of the ring.
                // Mark `last` where the write pointer used to be to hold the line here
                self.last.store(write, Ordering::Relaxed);
            } else if new_write > last {
                // We're about to pass the last pointer, which was previously the artificial
                // end of the ring. Now that we've passed it, we can "unlock" the section
                // that was previously skipped.
                //
                // Since new_write is strictly larger than last, it is safe to move this as
                // the other thread will still be halted by the (about to be updated) write
                // value
                self.last.store(max, Ordering::Relaxed);
            }
            // else: If new_write == last, either:
            // * last == max, so no need to write, OR
            // * If we write in the end chunk again, we'll update last to max next time
            // * If we write to the start chunk in a wrap, we'll update last when we
            //     move write backwards

            // Write must be updated AFTER last, otherwise read could think it was
            // time to invert early!
            self.write.store(new_write, Ordering::Relaxed);

            // Allow subsequent grants
            self.write_in_progress.store(false, Ordering::Relaxed);
        })
    }

    fn release_inner(&self, used: usize) {
        critical_section::with(|_cs| {
            // If there is no grant in progress, return early. This
            // generally means we are dropping the grant within a
            // wrapper structure
            if !self.read_in_progress.load(Ordering::Acquire) {
                return;
            }

            // // This should always be checked by the public interfaces
            // debug_assert!(used <= self.buf.len());

            // This should be fine, purely incrementing
            let old_read = self.read.load(Ordering::Relaxed);
            self.read.store(used + old_read, Ordering::Relaxed);
            self.read_in_progress.store(false, Ordering::Relaxed);
        })
    }
}
