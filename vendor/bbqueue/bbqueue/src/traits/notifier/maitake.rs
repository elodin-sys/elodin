//! A Maitake-Sync based notifications

use const_init::ConstInit;
use maitake_sync::WaitCell;

use super::{AsyncNotifier, Notifier};

/// A Maitake-Sync based SPSC notifier
///
/// Usable for async context. Should not be used with multiple consumers or multiple producers
/// at the same time.
pub struct MaiNotSpsc {
    not_empty: WaitCell,
    not_full: WaitCell,
}

impl MaiNotSpsc {
    /// Create a new Maitake-Sync based notifier
    pub const fn new() -> Self {
        Self::INIT
    }
}

impl Default for MaiNotSpsc {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstInit for MaiNotSpsc {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = Self {
        not_empty: WaitCell::new(),
        not_full: WaitCell::new(),
    };
}

impl Notifier for MaiNotSpsc {
    fn wake_one_consumer(&self) {
        _ = self.not_empty.wake();
    }

    fn wake_one_producer(&self) {
        _ = self.not_full.wake();
    }
}

impl AsyncNotifier for MaiNotSpsc {
    async fn wait_for_not_empty<T, F: FnMut() -> Option<T>>(&self, f: F) -> T {
        self.not_empty.wait_for_value(f).await.unwrap()
    }

    async fn wait_for_not_full<T, F: FnMut() -> Option<T>>(&self, f: F) -> T {
        self.not_full.wait_for_value(f).await.unwrap()
    }
}
