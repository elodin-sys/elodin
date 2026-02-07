//! Opus codec support for MP4 muxing.
//!
//! This module provides utilities for working with Opus audio in MP4 containers.
//! Opus in MP4 follows the ISO/IEC 14496-3 Amendment 4 specification, using the
//! `Opus` sample entry and `dOps` (Opus Decoder Configuration) box.
//!
//! # Opus in MP4
//!
//! Key characteristics:
//! - Sample rate is always 48000 Hz (per Opus spec, internal rate is 48kHz)
//! - Timescale should be 48000 for proper timing
//! - Pre-skip samples must be signaled in dOps
//! - Variable frame duration (2.5ms to 60ms)
//!
//! # Frame Duration
//!
//! Opus packets encode their duration in the TOC (Table of Contents) byte.
//! This module can infer frame duration from the TOC or accept user-provided duration.

/// Default Opus sample rate (48kHz, per Opus specification)
pub const OPUS_SAMPLE_RATE: u32 = 48000;

/// Opus decoder configuration for the dOps box.
#[derive(Debug, Clone)]
pub struct OpusConfig {
    /// Opus version (should be 0)
    pub version: u8,
    /// Number of output channels (1-8)
    pub output_channel_count: u8,
    /// Pre-skip samples (samples to discard at start for encoder/decoder delay)
    pub pre_skip: u16,
    /// Original sample rate (for informational purposes only, Opus always decodes at 48kHz)
    pub input_sample_rate: u32,
    /// Output gain in dB (Q7.8 fixed point: value / 256.0 = dB)
    pub output_gain: i16,
    /// Channel mapping family (0 = mono/stereo, 1 = Vorbis order, 2+ = application-defined)
    pub channel_mapping_family: u8,
    /// Stream count (for mapping family >= 1)
    pub stream_count: Option<u8>,
    /// Coupled stream count (for mapping family >= 1)
    pub coupled_count: Option<u8>,
    /// Channel mapping table (for mapping family >= 1)
    pub channel_mapping: Option<Vec<u8>>,
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self {
            version: 0,
            output_channel_count: 2,
            pre_skip: 312, // Common encoder delay
            input_sample_rate: 48000,
            output_gain: 0,
            channel_mapping_family: 0,
            stream_count: None,
            coupled_count: None,
            channel_mapping: None,
        }
    }
}

impl OpusConfig {
    /// Create a mono Opus configuration.
    pub fn mono() -> Self {
        Self {
            output_channel_count: 1,
            ..Default::default()
        }
    }

    /// Create a stereo Opus configuration.
    pub fn stereo() -> Self {
        Self {
            output_channel_count: 2,
            ..Default::default()
        }
    }

    /// Create configuration with custom pre-skip.
    pub fn with_pre_skip(mut self, pre_skip: u16) -> Self {
        self.pre_skip = pre_skip;
        self
    }

    /// Create configuration with custom channel count.
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.output_channel_count = channels;
        if channels > 2 {
            // For > 2 channels, need mapping family 1 or higher
            self.channel_mapping_family = 1;
        }
        self
    }
}

/// Opus frame duration in samples at 48kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusFrameDuration {
    /// 2.5ms = 120 samples
    Ms2_5,
    /// 5ms = 240 samples
    Ms5,
    /// 10ms = 480 samples
    Ms10,
    /// 20ms = 960 samples
    Ms20,
    /// 40ms = 1920 samples
    Ms40,
    /// 60ms = 2880 samples
    Ms60,
}

impl OpusFrameDuration {
    /// Get the duration in samples at 48kHz.
    pub fn samples(self) -> u32 {
        match self {
            OpusFrameDuration::Ms2_5 => 120,
            OpusFrameDuration::Ms5 => 240,
            OpusFrameDuration::Ms10 => 480,
            OpusFrameDuration::Ms20 => 960,
            OpusFrameDuration::Ms40 => 1920,
            OpusFrameDuration::Ms60 => 2880,
        }
    }

    /// Get the duration in seconds.
    pub fn seconds(self) -> f64 {
        self.samples() as f64 / OPUS_SAMPLE_RATE as f64
    }
}

/// Extract frame duration from the Opus TOC byte.
///
/// The TOC byte encodes the frame configuration:
/// - Bits 0-4: Frame count configuration
/// - Bits 5-7: Bandwidth/mode/config
///
/// Returns the frame duration for a single frame in the packet.
pub fn opus_frame_duration_from_toc(toc: u8) -> Option<OpusFrameDuration> {
    // Extract config bits (bits 3-7)
    let config = (toc >> 3) & 0x1F;

    if config > 31 {
        return None;
    }

    // Frame size depends on config value
    // See RFC 6716 Section 3.1
    match config {
        // SILK-only modes
        0..=3 => Some(OpusFrameDuration::Ms10),
        4..=7 => Some(OpusFrameDuration::Ms20),
        8..=11 => Some(OpusFrameDuration::Ms40),
        12..=15 => Some(OpusFrameDuration::Ms60),
        // Hybrid modes
        16..=19 => Some(OpusFrameDuration::Ms10),
        20..=23 => Some(OpusFrameDuration::Ms20),
        // CELT-only modes
        24..=27 => Some(OpusFrameDuration::Ms2_5),
        28..=31 => Some(OpusFrameDuration::Ms5),
        _ => None,
    }
}

/// Extract the frame count from the Opus packet.
///
/// Opus packets can contain 1, 2, or a variable number of frames.
/// Returns (frame_count, is_vbr) where is_vbr indicates variable bitrate.
pub fn opus_frame_count(packet: &[u8]) -> Option<(u8, bool)> {
    if packet.is_empty() {
        return None;
    }

    let toc = packet[0];
    let code = toc & 0x03;

    if code > 3 {
        return None;
    }

    match code {
        0 => Some((1, false)), // 1 frame
        1 => Some((2, false)), // 2 frames, equal size
        2 => Some((2, true)),  // 2 frames, different sizes
        3 => {
            // Code 3: arbitrary number of frames
            if packet.len() < 2 {
                return None;
            }

            let frame_count_byte = packet[1];
            let is_vbr = (frame_count_byte & 0x80) != 0;
            let count = frame_count_byte & 0x3F;

            if count == 0 {
                return None;
            }

            Some((count, is_vbr))
        }
        _ => None,
    }
}

/// Calculate total sample duration for an Opus packet.
///
/// Returns the total number of samples (at 48kHz) in the packet.
pub fn opus_packet_samples(packet: &[u8]) -> Option<u32> {
    if packet.is_empty() {
        return None;
    }

    let frame_duration = opus_frame_duration_from_toc(packet[0])?;
    let (frame_count, _) = opus_frame_count(packet)?;

    if !(1..=63).contains(&frame_count) {
        return None;
    }

    let samples = frame_duration.samples() * frame_count as u32;

    if samples == 0 {
        return None;
    }

    Some(samples)
}

/// Validate an Opus packet for basic structural correctness.
///
/// Returns true if the packet appears to be a valid Opus packet.
pub fn is_valid_opus_packet(packet: &[u8]) -> bool {
    if packet.is_empty() {
        return false;
    }

    // Check if we can parse the TOC and frame count
    opus_packet_samples(packet).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_config_default() {
        let config = OpusConfig::default();
        assert_eq!(config.version, 0);
        assert_eq!(config.output_channel_count, 2);
        assert_eq!(config.pre_skip, 312);
        assert_eq!(config.input_sample_rate, 48000);
        assert_eq!(config.output_gain, 0);
        assert_eq!(config.channel_mapping_family, 0);
    }

    #[test]
    fn test_opus_config_mono() {
        let config = OpusConfig::mono();
        assert_eq!(config.output_channel_count, 1);
    }

    #[test]
    fn test_opus_config_stereo() {
        let config = OpusConfig::stereo();
        assert_eq!(config.output_channel_count, 2);
    }

    #[test]
    fn test_opus_frame_duration_samples() {
        assert_eq!(OpusFrameDuration::Ms2_5.samples(), 120);
        assert_eq!(OpusFrameDuration::Ms5.samples(), 240);
        assert_eq!(OpusFrameDuration::Ms10.samples(), 480);
        assert_eq!(OpusFrameDuration::Ms20.samples(), 960);
        assert_eq!(OpusFrameDuration::Ms40.samples(), 1920);
        assert_eq!(OpusFrameDuration::Ms60.samples(), 2880);
    }

    #[test]
    fn test_opus_frame_duration_from_toc_silk() {
        // SILK 10ms (config 0-3)
        assert_eq!(
            opus_frame_duration_from_toc(0b0000_0000),
            Some(OpusFrameDuration::Ms10)
        );
        // SILK 20ms (config 4-7)
        assert_eq!(
            opus_frame_duration_from_toc(0b0010_0000),
            Some(OpusFrameDuration::Ms20)
        );
        // SILK 40ms (config 8-11)
        assert_eq!(
            opus_frame_duration_from_toc(0b0100_0000),
            Some(OpusFrameDuration::Ms40)
        );
        // SILK 60ms (config 12-15)
        assert_eq!(
            opus_frame_duration_from_toc(0b0110_0000),
            Some(OpusFrameDuration::Ms60)
        );
    }

    #[test]
    fn test_opus_frame_duration_from_toc_celt() {
        // CELT 2.5ms (config 24-27)
        assert_eq!(
            opus_frame_duration_from_toc(0b1100_0000),
            Some(OpusFrameDuration::Ms2_5)
        );
        // CELT 5ms (config 28-31)
        assert_eq!(
            opus_frame_duration_from_toc(0b1110_0000),
            Some(OpusFrameDuration::Ms5)
        );
    }

    #[test]
    fn test_opus_frame_count_single() {
        // TOC with code 0 = 1 frame
        let packet = vec![0b0000_0000, 0x01, 0x02, 0x03];
        assert_eq!(opus_frame_count(&packet), Some((1, false)));
    }

    #[test]
    fn test_opus_frame_count_double_equal() {
        // TOC with code 1 = 2 frames, equal size
        let packet = vec![0b0000_0001, 0x01, 0x02, 0x03];
        assert_eq!(opus_frame_count(&packet), Some((2, false)));
    }

    #[test]
    fn test_opus_frame_count_double_different() {
        // TOC with code 2 = 2 frames, different sizes
        let packet = vec![0b0000_0010, 0x01, 0x02, 0x03];
        assert_eq!(opus_frame_count(&packet), Some((2, true)));
    }

    #[test]
    fn test_opus_frame_count_arbitrary() {
        // TOC with code 3 = N frames, count in second byte
        let packet = vec![0b0000_0011, 0b0000_0100]; // 4 frames, CBR
        assert_eq!(opus_frame_count(&packet), Some((4, false)));

        let packet_vbr = vec![0b0000_0011, 0b1000_0100]; // 4 frames, VBR
        assert_eq!(opus_frame_count(&packet_vbr), Some((4, true)));
    }

    #[test]
    fn test_opus_packet_samples() {
        // SILK 20ms frame (config=4), 1 frame (code=0)
        // TOC: config=4 (bits 3-7 = 0b00100), s=0, c=0
        // Binary: 0b00100_0_00 = 0x20 = 32
        let packet = vec![0x20, 0x01, 0x02, 0x03];
        assert_eq!(opus_packet_samples(&packet), Some(960));

        // SILK 20ms frame (config=4), 2 frames (code=1)
        // TOC: config=4 (bits 3-7 = 0b00100), s=0, c=1
        // Binary: 0b00100_0_01 = 0x21 = 33
        let packet2 = vec![0x21, 0x01, 0x02, 0x03];
        assert_eq!(opus_packet_samples(&packet2), Some(1920));
    }

    #[test]
    fn test_opus_functions_handle_bad_input_gracefully() {
        // Empty packet should return None, not panic
        assert_eq!(opus_frame_count(&[]), None);
        assert_eq!(opus_packet_samples(&[]), None);

        // Valid TOC config should work
        let valid_toc = vec![0xFF]; // config=31, valid
        assert_eq!(
            opus_frame_duration_from_toc(valid_toc[0]),
            Some(OpusFrameDuration::Ms5)
        );
        // But packet is too short for frame count
        assert_eq!(opus_packet_samples(&valid_toc), None);

        // Code 3 with count == 0 should return None
        let invalid_code3 = vec![0x03, 0x00]; // code=3, count=0
        assert_eq!(opus_frame_count(&invalid_code3), None);
        assert_eq!(opus_packet_samples(&invalid_code3), None);
    }

    #[test]
    fn test_is_valid_opus_packet() {
        // Valid: config=4 (SILK 20ms), code=0 (1 frame)
        assert!(is_valid_opus_packet(&[0x20, 0x01, 0x02]));
        assert!(!is_valid_opus_packet(&[]));
    }
}
