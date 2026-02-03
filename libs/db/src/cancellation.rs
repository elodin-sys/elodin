//! Signal handling and cancellation support for CLI tools.
//!
//! This module provides graceful shutdown capabilities by registering signal handlers
//! for SIGINT and SIGTERM, allowing long-running operations to check for cancellation
//! and exit cleanly.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::Error;

/// Global termination flag set by signal handlers.
///
/// This is set to `true` when SIGINT (Ctrl+C) or SIGTERM is received.
static TERMINATED: AtomicBool = AtomicBool::new(false);

/// Install signal handlers for SIGINT and SIGTERM.
///
/// After calling this function, receiving SIGINT or SIGTERM will set the
/// `TERMINATED` flag to `true`. Use `check_cancelled()` to check this flag
/// and return an error if cancellation was requested.
///
/// This function should be called once at the start of `main()` for CLI tools.
///
/// # Example
///
/// ```ignore
/// fn main() -> Result<(), Error> {
///     elodin_db::cancellation::install_signal_handlers();
///     
///     // Long-running operation with periodic checks
///     for item in items {
///         elodin_db::cancellation::check_cancelled()?;
///         process(item)?;
///     }
///     
///     Ok(())
/// }
/// ```
#[cfg(unix)]
pub fn install_signal_handlers() {
    use signal_hook::consts::signal::{SIGINT, SIGTERM};
    use signal_hook::low_level;

    // Use low-level API to register a handler that sets our static flag
    unsafe {
        // Register SIGINT (Ctrl+C)
        if let Err(e) = low_level::register(SIGINT, || {
            TERMINATED.store(true, Ordering::SeqCst);
        }) {
            tracing::warn!("Failed to register SIGINT handler: {}", e);
        }

        // Register SIGTERM
        if let Err(e) = low_level::register(SIGTERM, || {
            TERMINATED.store(true, Ordering::SeqCst);
        }) {
            tracing::warn!("Failed to register SIGTERM handler: {}", e);
        }
    }
}

/// Install signal handlers (no-op on non-Unix platforms).
#[cfg(not(unix))]
pub fn install_signal_handlers() {
    // Windows and other platforms: no-op for now
    // Could be extended to use SetConsoleCtrlHandler on Windows
}

/// Check if cancellation has been requested.
///
/// Returns `Ok(())` if the operation should continue, or `Err(Error::Cancelled)`
/// if a termination signal has been received.
///
/// This check is extremely cheap (single atomic load with relaxed ordering),
/// so it can be called frequently in hot loops without performance impact.
#[inline]
pub fn check_cancelled() -> Result<(), Error> {
    if TERMINATED.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

/// Check if cancellation has been requested (returns bool).
///
/// This is useful when you need to check cancellation but handle it differently
/// than returning an error.
#[inline]
pub fn is_cancelled() -> bool {
    TERMINATED.load(Ordering::Relaxed)
}

/// Reset the termination flag.
///
/// This is primarily useful for testing. In normal operation, once terminated
/// the process should exit.
#[cfg(test)]
pub fn reset() {
    TERMINATED.store(false, Ordering::Relaxed);
}
