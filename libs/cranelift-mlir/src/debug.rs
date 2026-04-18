//! Debug / diagnostics gate for the cranelift-mlir backend.
//!
//! The single environment variable `ELODIN_CRANELIFT_DEBUG_DIR` drives
//! every report, profile, and dump this crate produces. When set, every
//! instrumentation path fires and every file artifact lands flat under
//! the given directory; when unset, the machine code is bit-identical
//! to a plain build and there is zero stderr chatter beyond the compile
//! banner.
//!
//! Outputs written under `$ELODIN_CRANELIFT_DEBUG_DIR/` at runtime:
//! - `stablehlo.mlir`, `compile_context.json` — lowered IR + input meta.
//! - `input_<i>.bin`, `cranelift_output_<i>.bin`, `xla_output_<i>.bin`,
//!   `checkpoint.json` — first-tick correctness checkpoint.
//! - `profile.json` — structured runtime profile report.
//!
//! Stderr outputs (human-readable, not file artifacts): per-function
//! profile summary, IR instruction counts, const-fold histogram,
//! inliner trace, slot-pool hit rate.

use std::path::PathBuf;
use std::sync::OnceLock;

/// Returns the configured debug directory if
/// `ELODIN_CRANELIFT_DEBUG_DIR` is set, else `None`. Cached on first
/// read for the life of the process.
pub fn dir() -> Option<&'static PathBuf> {
    static CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            std::env::var("ELODIN_CRANELIFT_DEBUG_DIR")
                .ok()
                .map(PathBuf::from)
        })
        .as_ref()
}

/// True when the debug dir is configured. Use this as the single gate
/// for profile probe emission, instr reports, fold histogram, inliner
/// trace, etc.
pub fn enabled() -> bool {
    dir().is_some()
}
