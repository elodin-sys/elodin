//! H.264/AVC codec configuration extraction.
//!
//! Provides minimal NAL unit parsing to extract SPS (Sequence Parameter Set)
//! and PPS (Picture Parameter Set) for building the avcC configuration box.
//!
//! # NAL Unit Types
//!
//! | Type | Name | Purpose |
//! |------|------|---------|
//! | 1 | Non-IDR slice | P/B frame data |
//! | 5 | IDR slice | Keyframe (I-frame) |
//! | 7 | SPS | Sequence Parameter Set |
//! | 8 | PPS | Picture Parameter Set |
//!
//! # Input Format
//!
//! Input must be in Annex B format with start codes:
//! ```text
//! [0x00 0x00 0x00 0x01][NAL unit][0x00 0x00 0x00 0x01][NAL unit]...
//! ```

use super::common::AnnexBNalIter;
use crate::assert_invariant;

/// H.264 NAL unit type constants.
pub mod nal_type {
    /// Non-IDR coded slice (P/B frame)
    pub const NON_IDR_SLICE: u8 = 1;
    /// IDR coded slice (keyframe)
    pub const IDR_SLICE: u8 = 5;
    /// Sequence Parameter Set
    pub const SPS: u8 = 7;
    /// Picture Parameter Set
    pub const PPS: u8 = 8;
}

/// AVC (H.264) codec configuration extracted from NAL units.
///
/// Contains the raw SPS and PPS NAL units needed to build the
/// avcC box in MP4 containers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvcConfig {
    /// Sequence Parameter Set NAL unit (without start code)
    pub sps: Vec<u8>,
    /// Picture Parameter Set NAL unit (without start code)
    pub pps: Vec<u8>,
}

impl AvcConfig {
    /// Create a new AVC configuration from SPS and PPS data.
    pub fn new(sps: Vec<u8>, pps: Vec<u8>) -> Self {
        Self { sps, pps }
    }

    /// Extract profile_idc from the SPS.
    ///
    /// Profile indicates the feature set (Baseline=66, Main=77, High=100).
    pub fn profile_idc(&self) -> u8 {
        self.sps.get(1).copied().unwrap_or(66)
    }

    /// Extract profile_compatibility flags from the SPS.
    pub fn profile_compatibility(&self) -> u8 {
        self.sps.get(2).copied().unwrap_or(0)
    }

    /// Extract level_idc from the SPS.
    ///
    /// Level indicates max bitrate/resolution (31 = level 3.1 = 720p30).
    pub fn level_idc(&self) -> u8 {
        self.sps.get(3).copied().unwrap_or(31)
    }
}

/// Default SPS for 640x480 @ Baseline Profile, Level 3.0.
///
/// Used as fallback when no SPS is provided in the stream.
/// Matches the original Muxide default for backwards compatibility.
pub const DEFAULT_SPS: &[u8] = &[0x67, 0x42, 0x00, 0x1e, 0xda, 0x02, 0x80, 0x2d, 0x8b, 0x11];

/// Default PPS.
///
/// Used as fallback when no PPS is provided in the stream.
/// Matches the original Muxide default for backwards compatibility.
pub const DEFAULT_PPS: &[u8] = &[0x68, 0xce, 0x38, 0x80];

/// Extract AVC configuration (SPS/PPS) from an Annex B keyframe.
///
/// Scans the NAL units in the provided data and extracts the first
/// SPS (type 7) and PPS (type 8) found.
///
/// # Arguments
///
/// * `data` - Annex B formatted data containing at least one keyframe
///
/// # Returns
///
/// - `Some(AvcConfig)` if both SPS and PPS are found
/// - `None` if either SPS or PPS is missing
///
/// # Example
///
/// ```
/// use muxide::codec::h264::extract_avc_config;
///
/// let keyframe = [
///     0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f,  // SPS
///     0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xe3, 0xcb,  // PPS
///     0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00,  // IDR slice
/// ];
///
/// let config = extract_avc_config(&keyframe).expect("should have config");
/// assert_eq!(config.sps[0] & 0x1f, 7);  // SPS NAL type
/// assert_eq!(config.pps[0] & 0x1f, 8);  // PPS NAL type
/// ```
pub fn extract_avc_config(data: &[u8]) -> Option<AvcConfig> {
    if data.is_empty() {
        return None;
    }

    let mut sps: Option<&[u8]> = None;
    let mut pps: Option<&[u8]> = None;

    for nal in AnnexBNalIter::new(data) {
        if nal.is_empty() {
            continue;
        }

        let nal_type = nal[0] & 0x1f;

        // INV-301: NAL type must be valid
        assert_invariant!(
            nal_type <= 31,
            "INV-301: H.264 NAL type must be valid (0-31)",
            "codec::h264::extract_avc_config"
        );

        if nal_type == nal_type::SPS && sps.is_none() {
            sps = Some(nal);
        } else if nal_type == nal_type::PPS && pps.is_none() {
            pps = Some(nal);
        }

        // Early exit once we have both
        if sps.is_some() && pps.is_some() {
            break;
        }
    }

    // INV-302: Both SPS and PPS must be found for valid config
    if let (Some(sps_data), Some(pps_data)) = (sps, pps) {
        assert_invariant!(
            !sps_data.is_empty() && !pps_data.is_empty(),
            "INV-302: H.264 SPS and PPS must be non-empty",
            "codec::h264::extract_avc_config"
        );

        Some(AvcConfig {
            sps: sps_data.to_vec(),
            pps: pps_data.to_vec(),
        })
    } else {
        None
    }
}

/// Create a default AVC configuration for testing/fallback.
///
/// Returns a valid configuration for 1080p @ High Profile, Level 4.0.
pub fn default_avc_config() -> AvcConfig {
    AvcConfig {
        sps: DEFAULT_SPS.to_vec(),
        pps: DEFAULT_PPS.to_vec(),
    }
}

/// Convert Annex B formatted data to AVCC (length-prefixed) format.
///
/// MP4 containers use AVCC format where each NAL unit is prefixed with
/// its length as a 4-byte big-endian integer, rather than start codes.
///
/// # Arguments
///
/// * `data` - Annex B formatted data with start codes
///
/// # Returns
///
/// AVCC formatted data with 4-byte length prefixes.
///
/// # Example
///
/// ```
/// use muxide::codec::h264::annexb_to_avcc;
///
/// let annexb = [
///     0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84,  // NAL with start code
/// ];
///
/// let avcc = annexb_to_avcc(&annexb);
/// // Result: [0x00, 0x00, 0x00, 0x03, 0x65, 0x88, 0x84]
/// //          ^--- 4-byte length (3)     ^--- NAL data
/// ```
pub fn annexb_to_avcc(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();

    for nal in AnnexBNalIter::new(data) {
        if nal.is_empty() {
            continue;
        }
        let len = nal.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(nal);
    }

    // Fallback: if no start codes found, treat entire input as single NAL
    if out.is_empty() && !data.is_empty() {
        let len = data.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(data);
    }

    out
}

/// Check if the given Annex B data represents a keyframe (IDR slice).
///
/// A keyframe is identified by the presence of an IDR slice NAL unit (type 5).
///
/// # Arguments
///
/// * `data` - Annex B formatted frame data
///
/// # Returns
///
/// `true` if the data contains an IDR slice, `false` otherwise.
pub fn is_h264_keyframe(data: &[u8]) -> bool {
    for nal in AnnexBNalIter::new(data) {
        if nal.is_empty() {
            continue;
        }
        let nal_type = nal[0] & 0x1f;
        if nal_type == nal_type::IDR_SLICE {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_avc_config_success() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f, // SPS (type 7)
            0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xe3, 0xcb, // PPS (type 8)
            0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, // IDR (type 5)
        ];

        let config = extract_avc_config(&data).unwrap();
        assert_eq!(config.sps, &[0x67, 0x64, 0x00, 0x1f]);
        assert_eq!(config.pps, &[0x68, 0xeb, 0xe3, 0xcb]);
    }

    #[test]
    fn test_extract_avc_config_missing_sps() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, 0xe3, 0xcb, // PPS only
        ];
        assert!(extract_avc_config(&data).is_none());
    }

    #[test]
    fn test_extract_avc_config_missing_pps() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x1f, // SPS only
        ];
        assert!(extract_avc_config(&data).is_none());
    }

    #[test]
    fn test_avc_config_accessors() {
        let config = AvcConfig::new(
            vec![0x67, 0x64, 0x00, 0x28], // High profile, level 4.0
            vec![0x68, 0xeb],
        );

        assert_eq!(config.profile_idc(), 0x64); // 100 = High
        assert_eq!(config.profile_compatibility(), 0x00);
        assert_eq!(config.level_idc(), 0x28); // 40 = Level 4.0
    }

    #[test]
    fn test_annexb_to_avcc() {
        let annexb = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, // SPS (3 bytes)
            0x00, 0x00, 0x00, 0x01, 0x68, 0xeb, // PPS (2 bytes)
        ];

        let avcc = annexb_to_avcc(&annexb);

        // First NAL: length 3 + data
        assert_eq!(&avcc[0..4], &[0x00, 0x00, 0x00, 0x03]);
        assert_eq!(&avcc[4..7], &[0x67, 0x64, 0x00]);

        // Second NAL: length 2 + data
        assert_eq!(&avcc[7..11], &[0x00, 0x00, 0x00, 0x02]);
        assert_eq!(&avcc[11..13], &[0x68, 0xeb]);
    }

    #[test]
    fn test_annexb_to_avcc_no_start_codes() {
        // Data without start codes - treated as single NAL
        let data = [0x65, 0x88, 0x84];
        let avcc = annexb_to_avcc(&data);

        assert_eq!(&avcc[0..4], &[0x00, 0x00, 0x00, 0x03]);
        assert_eq!(&avcc[4..7], &[0x65, 0x88, 0x84]);
    }

    #[test]
    fn test_is_keyframe_idr() {
        let idr_frame = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, // SPS
            0x00, 0x00, 0x00, 0x01, 0x65, 0x88, // IDR (type 5)
        ];
        assert!(is_h264_keyframe(&idr_frame));
    }

    #[test]
    fn test_is_keyframe_non_idr() {
        let p_frame = [
            0x00, 0x00, 0x00, 0x01, 0x41, 0x9a, // Non-IDR (type 1)
        ];
        assert!(!is_h264_keyframe(&p_frame));
    }

    #[test]
    fn test_is_keyframe_empty() {
        assert!(!is_h264_keyframe(&[]));
    }

    #[test]
    fn test_default_avc_config() {
        let config = default_avc_config();
        assert!(!config.sps.is_empty());
        assert!(!config.pps.is_empty());
        assert_eq!(config.sps[0] & 0x1f, nal_type::SPS);
        assert_eq!(config.pps[0] & 0x1f, nal_type::PPS);
    }
}
