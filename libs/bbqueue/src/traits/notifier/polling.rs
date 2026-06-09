//! Polling Notifier
//!
//! The "Polling" notifier is the simplest notifier: it does NO notification!

use super::Notifier;
use const_init::ConstInit;

/// A Blocking/Polling coordination
///
/// This performs no notifiication
pub struct Polling;

// Blocking performs no notification
impl Notifier for Polling {
    fn wake_one_consumer(&self) {}
    fn wake_one_producer(&self) {}
}

impl ConstInit for Polling {
    const INIT: Self = Polling;
}
