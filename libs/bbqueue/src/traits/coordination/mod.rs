//! "Coordination" functionality
//!
//! This trait is used to coordinate between Producers and Consumers.
//!
//! Unless you are on an embedded target without Compare and Swap atomics, e.g.
//! `cortex-m0`/`thumbv6m`, you almost certainly want to use the [`cas`] version
//! of coordination.
//!
//! The `cas` module is toggled automatically based on `#[cfg(target_has_atomic = "ptr")]`.

use const_init::ConstInit;

#[cfg(target_has_atomic = "ptr")]
pub mod cas;

#[cfg(feature = "critical-section")]
pub mod cs;

/// Errors associated with obtaining a write grant
#[derive(PartialEq, Debug)]
pub enum WriteGrantError {
    /// Unable to create write grant due to not enough room in the buffer
    InsufficientSize,
    /// Unable to create write grant due to existing write grant
    GrantInProgress,
}

/// Errors associated with obtaining a read grant
#[derive(PartialEq, Debug)]
pub enum ReadGrantError {
    /// Unable to create read grant due to no available bytes
    Empty,
    /// Unable to create read grant due to existing read grant
    GrantInProgress,
    /// We observed a frame header that did not make sense. This should only
    /// occur if a stream producer was used on one end and a frame consumer was
    /// used on the other end. Don't do that.
    ///
    /// If you see this error and you are NOT doing that, please report it, as it
    /// is a bug.
    InconsistentFrameHeader,
}

/// Coordination Handler
///
/// The coordination handler is responsible for arbitrating access to the storage.
///
/// # Safety
///
/// you must implement these correctly, or UB could happen
pub unsafe trait Coord: ConstInit {
    /// Reset all values back to the initial empty state
    fn reset(&self);

    // Write Grants

    /// Obtain a non-zero length grant UP TO `sz` of available writing space,
    /// e.g. `0 < len <= sz`.
    ///
    /// On success, the producer will have exclusive access to this region.
    ///
    /// If a non-zero length grant is available, a tuple is returned that contains:
    ///
    /// * `.0`: the offset in bytes from the base storage pointer
    /// * `.1`: the length in bytes of the region
    ///
    /// The returned grant must remain valid until `commit_inner` is called.
    fn grant_max_remaining(
        &self,
        capacity: usize,
        sz: usize,
    ) -> Result<(usize, usize), WriteGrantError>;

    /// Obtain a grant of EXACTLY `sz` bytes.
    ///
    /// On success, the producer will have exclusive access to this region, and the
    /// offset in bytes from the base storage pointer will be returned.
    ///
    /// The returned grant must remain valid until `commit_inner` is called.
    fn grant_exact(&self, capacity: usize, sz: usize) -> Result<usize, WriteGrantError>;

    /// Make `used` bytes available
    fn commit_inner(&self, capacity: usize, grant_len: usize, used: usize);

    // Read Grants

    /// Attempt to obtain a read grant.
    fn read(&self) -> Result<(usize, usize), ReadGrantError>;

    /// Mark `used` bytes as available for writing
    fn release_inner(&self, used: usize);
}
