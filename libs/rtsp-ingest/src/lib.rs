//! RTSP H.264 ingest glue for Elodin-DB.
//!
//! This crate holds the *pure* logic needed to turn an RTSP H.264 stream into
//! the exact on-disk shape `elodinsink` produces (timestamped Annex-B access
//! units in `MsgLog`):
//!
//! - [`annexb`]: AVC (length-prefixed) → Annex-B framing with in-band SPS/PPS
//!   injection ahead of IDR access units.
//! - [`clock`]: source PTS → strictly-increasing DB microsecond timestamps,
//!   anchored to the DB's `last_updated` (mirroring `elodinsink`).
//!
//! These modules are deliberately free of `retina`, `tokio`, and the DB so they
//! can be unit-tested without any network or hardware. The `retina` session
//! manager that drives them lives behind a separate (future) feature.

pub mod annexb;
pub mod clock;

/// Errors produced while reframing RTSP H.264 into the DB storage contract.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("access unit contains an IDR slice but no SPS/PPS are available to inject")]
    MissingParameterSets,
    #[error("AVC NAL length prefix is truncated")]
    TruncatedNal,
    #[error("AVC NAL has a declared length of zero")]
    ZeroLengthNal,
    #[error("access unit contains no NAL units")]
    EmptyAccessUnit,
    #[error("invalid NAL length size: {0} (must be in 1..=4)")]
    InvalidNalLengthSize(usize),
}
