//! Codec configuration extraction for container muxing.
//!
//! This module provides minimal bitstream parsing required to build codec
//! configuration boxes (avcC, hvcC, av1C, dOps). It does NOT perform decoding,
//! transcoding, or frame reconstruction.
//!
//! # Supported Codecs
//!
//! - **H.264/AVC**: Extract SPS/PPS from Annex B NAL units
//! - **H.265/HEVC**: Extract VPS/SPS/PPS from Annex B NAL units
//! - **VP9**: Extract frame headers and configuration from compressed frames
//! - **Opus**: Parse TOC for frame duration, build dOps config
//! - **AV1**: Parse OBU headers for sequence configuration
//!
//! # Input Format
//!
//! All video input is expected in **Annex B** format (start code delimited):
//! - 4-byte start code: `0x00 0x00 0x00 0x01`
//! - 3-byte start code: `0x00 0x00 0x01`
//!
//! The muxer internally converts to length-prefixed format (AVCC/HVCC) for MP4.

pub mod av1;
pub mod common;
pub mod h264;
pub mod h265;
pub mod opus;
pub mod vp9;

pub use common::{find_start_code, AnnexBNalIter};
pub use h264::{annexb_to_avcc, extract_avc_config, is_h264_keyframe, AvcConfig};
pub use h265::{extract_hevc_config, hevc_annexb_to_hvcc, is_hevc_keyframe, HevcConfig};
pub use opus::{is_valid_opus_packet, opus_packet_samples, OpusConfig, OPUS_SAMPLE_RATE};
