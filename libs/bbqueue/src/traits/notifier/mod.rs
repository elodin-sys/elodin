//! "Notification" functionality
//!
//! This functionality allows (or doesn't allow) for awaiting a read/write grant

use const_init::ConstInit;

#[cfg(feature = "maitake-sync-0_3")]
pub mod maitake;

pub mod polling;

/// Non-async notifications
pub trait Notifier: ConstInit {
    /// Inform one consumer that there is data available to read
    fn wake_one_consumer(&self);
    /// Inform one producer that there is room available to write
    fn wake_one_producer(&self);
}

/// Async notifications
#[allow(async_fn_in_trait)]
pub trait AsyncNotifier: Notifier {
    /// Wait until the queue is NOT empty, e.g. data is available to read
    async fn wait_for_not_empty<T, F: FnMut() -> Option<T>>(&self, f: F) -> T;
    /// Wait until the queue is NOT full, e.g. room is available to write
    async fn wait_for_not_full<T, F: FnMut() -> Option<T>>(&self, f: F) -> T;
}
