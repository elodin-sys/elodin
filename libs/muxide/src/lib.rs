//! # Muxide
//!
//! **Minimal-dependency pure-Rust MP4 muxer for recording applications.**
//!
//! ## Core Invariant
//!
//! > Muxide guarantees that any **correctly-timestamped**, **already-encoded** audio/video
//! > stream can be turned into a **standards-compliant**, **immediately-playable** MP4
//! > **without external tooling**.
//!
//! ## What Muxide Does
//!
//! - Accepts encoded H.264/H.265/AV1/VP9 video frames with timestamps
//! - Accepts encoded AAC/Opus audio frames with timestamps  
//! - Outputs MP4 files with fast-start (moov before mdat) for instant web playback
//! - Supports B-frames via explicit PTS/DTS
//! - Supports fragmented MP4 (fMP4) for DASH/HLS streaming
//!
//! ## What Muxide Does NOT Do
//!
//! - ❌ Encode or decode video/audio (use openh264, x264, etc.)
//! - ❌ Read or demux MP4 files
//! - ❌ Fix bad timestamps (rejects invalid input)
//! - ❌ DRM, encryption, or content protection
//! - ❌ MKV, WebM, or other container formats
//!
//! See `docs/charter.md` and `docs/contract.md` for full invariants.
//!
//! # Example
//!
//! ```no_run
//! use muxide::api::{MuxerBuilder, VideoCodec};
//! use std::fs::File;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file = File::create("out.mp4")?;
//! let mut muxer = MuxerBuilder::new(file)
//!     .video(VideoCodec::H264, 1920, 1080, 30.0)
//!     .build()?;
//!
//! // Write frames (encoded elsewhere).
//! // muxer.write_video(pts_secs, annex_b_bytes, is_keyframe)?;
//!
//! let _stats = muxer.finish_with_stats()?;
//! # Ok(())
//! # }
//! ```

mod muxer;

// Re-export the API module so users can simply `use muxide::api::...`.
pub mod api;

// Fragmented MP4 support for streaming applications
pub mod fragmented;

// Codec configuration extraction (minimal bitstream parsing)
pub mod codec;

// Input validation utilities for dry-run functionality
pub mod validation;

// Invariant PPT testing framework
pub mod invariant_ppt;
