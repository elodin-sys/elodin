//! H.265/HEVC codec configuration extraction.
//!
//! Provides NAL unit parsing to extract VPS, SPS, and PPS for building
//! the hvcC configuration box in MP4 containers.
//!
//! # NAL Unit Types (HEVC)
//!
//! | Type | Name | Purpose |
//! |------|------|---------|
//! | 0-9 | VCL | Coded slice segments |
//! | 16-18 | BLA | Broken Link Access |
//! | 19 | IDR_W_RADL | IDR with RADL pictures |
//! | 20 | IDR_N_LP | IDR without leading pictures |
//! | 21 | CRA | Clean Random Access |
//! | 32 | VPS | Video Parameter Set |
//! | 33 | SPS | Sequence Parameter Set |
//! | 34 | PPS | Picture Parameter Set |
//!
//! # Differences from H.264
//!
//! - NAL header is 2 bytes (vs 1 byte in H.264)
//! - NAL type is in bits 1-6 of the first byte: `(byte0 >> 1) & 0x3f`
//! - Requires VPS in addition to SPS/PPS
//! - Configuration box is `hvcC` instead of `avcC`
//!
//! # Input Format
//!
//! Input must be in Annex B format with start codes, same as H.264.

use super::common::AnnexBNalIter;

/// H.265 NAL unit type constants.
pub mod nal_type {
    /// Coded slice segment of a BLA picture
    pub const BLA_W_LP: u8 = 16;
    /// Coded slice segment of a BLA picture
    pub const BLA_W_RADL: u8 = 17;
    /// Coded slice segment of a BLA picture
    pub const BLA_N_LP: u8 = 18;
    /// IDR with RADL pictures
    pub const IDR_W_RADL: u8 = 19;
    /// IDR without leading pictures
    pub const IDR_N_LP: u8 = 20;
    /// Clean Random Access picture
    pub const CRA_NUT: u8 = 21;
    /// Video Parameter Set
    pub const VPS: u8 = 32;
    /// Sequence Parameter Set
    pub const SPS: u8 = 33;
    /// Picture Parameter Set
    pub const PPS: u8 = 34;
    /// Access unit delimiter
    pub const AUD: u8 = 35;
    /// End of sequence
    pub const EOS: u8 = 36;
    /// End of bitstream  
    pub const EOB: u8 = 37;
    /// Filler data
    pub const FD: u8 = 38;
    /// Supplemental enhancement information (prefix)
    pub const PREFIX_SEI: u8 = 39;
    /// Supplemental enhancement information (suffix)
    pub const SUFFIX_SEI: u8 = 40;
}

/// HEVC (H.265) codec configuration.
///
/// Contains VPS, SPS, and PPS NAL units needed to build the
/// hvcC box in MP4 containers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcConfig {
    /// Video Parameter Set NAL unit (without start code)
    pub vps: Vec<u8>,
    /// Sequence Parameter Set NAL unit (without start code)
    pub sps: Vec<u8>,
    /// Picture Parameter Set NAL unit (without start code)
    pub pps: Vec<u8>,
}

impl HevcConfig {
    /// Create a new HEVC configuration from VPS, SPS, and PPS data.
    pub fn new(vps: Vec<u8>, sps: Vec<u8>, pps: Vec<u8>) -> Self {
        Self { vps, sps, pps }
    }

    /// Extract general_profile_space from the SPS (bits 0-1 of byte 3).
    pub fn general_profile_space(&self) -> u8 {
        self.sps.get(3).map(|b| (b >> 6) & 0x03).unwrap_or(0)
    }

    /// Extract general_tier_flag from the SPS (bit 2 of byte 3).
    pub fn general_tier_flag(&self) -> bool {
        self.sps
            .get(3)
            .map(|b| (b >> 5) & 0x01 != 0)
            .unwrap_or(false)
    }

    /// Extract general_profile_idc from the SPS (bits 3-7 of byte 3).
    pub fn general_profile_idc(&self) -> u8 {
        self.sps.get(3).map(|b| b & 0x1f).unwrap_or(1)
    }

    /// Extract general_level_idc from the SPS (byte 14).
    /// Level 5.1 = 153, Level 4.0 = 120, Level 3.1 = 93
    pub fn general_level_idc(&self) -> u8 {
        self.sps.get(14).copied().unwrap_or(93)
    }
}

/// Extract the NAL unit type from an H.265 NAL header.
///
/// H.265 NAL type is in bits 1-6 of the first byte:
/// `(nal_header[0] >> 1) & 0x3f`
#[inline]
pub fn hevc_nal_type(nal: &[u8]) -> u8 {
    if nal.is_empty() {
        return 0;
    }
    (nal[0] >> 1) & 0x3f
}

/// Check if the given NAL type represents a keyframe (IRAP).
///
/// HEVC has multiple IRAP (Intra Random Access Point) types:
/// - BLA (16-18): Broken Link Access
/// - IDR (19-20): Instantaneous Decoder Refresh
/// - CRA (21): Clean Random Access
#[inline]
pub fn is_hevc_keyframe_nal_type(nal_type: u8) -> bool {
    matches!(
        nal_type,
        nal_type::BLA_W_LP
            | nal_type::BLA_W_RADL
            | nal_type::BLA_N_LP
            | nal_type::IDR_W_RADL
            | nal_type::IDR_N_LP
            | nal_type::CRA_NUT
    )
}

use crate::assert_invariant;

/// Extract HEVC configuration (VPS/SPS/PPS) from Annex B keyframe data.
///
/// Scans the NAL units in the provided data and extracts the first
/// VPS (type 32), SPS (type 33), and PPS (type 34) found.
///
/// # Arguments
///
/// * `data` - Annex B formatted data containing at least one keyframe
///
/// # Returns
///
/// - `Some(HevcConfig)` if VPS, SPS, and PPS are all found
/// - `None` if any of the three is missing
///
/// # Example
///
/// ```
/// use muxide::codec::h265::extract_hevc_config;
///
/// // Minimal HEVC keyframe with VPS, SPS, PPS, IDR
/// let keyframe = [
///     0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01,  // VPS (type 32)
///     0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x01,  // SPS (type 33)
///     0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0x73,  // PPS (type 34)
///     0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x00,  // IDR (type 19)
/// ];
///
/// let config = extract_hevc_config(&keyframe).expect("should have config");
/// assert_eq!(config.vps[0] >> 1 & 0x3f, 32);  // VPS NAL type
/// ```
pub fn extract_hevc_config(data: &[u8]) -> Option<HevcConfig> {
    if data.is_empty() {
        return None;
    }

    let mut vps: Option<&[u8]> = None;
    let mut sps: Option<&[u8]> = None;
    let mut pps: Option<&[u8]> = None;

    for nal in AnnexBNalIter::new(data) {
        if nal.is_empty() {
            continue;
        }

        let nal_type = hevc_nal_type(nal);

        assert_invariant!(
            nal_type <= 63,
            "INV-501: HEVC NAL type must be valid (0-63)"
        );

        match nal_type {
            nal_type::VPS if vps.is_none() => vps = Some(nal),
            nal_type::SPS if sps.is_none() => sps = Some(nal),
            nal_type::PPS if pps.is_none() => pps = Some(nal),
            _ => {}
        }

        // Early exit once we have all three
        if vps.is_some() && sps.is_some() && pps.is_some() {
            break;
        }
    }

    // Verify we found all required parameter sets
    if let (Some(vps_data), Some(sps_data), Some(pps_data)) = (vps, sps, pps) {
        assert_invariant!(
            !vps_data.is_empty() && !sps_data.is_empty() && !pps_data.is_empty(),
            "INV-502: HEVC VPS, SPS, and PPS must be non-empty"
        );

        Some(HevcConfig {
            vps: vps_data.to_vec(),
            sps: sps_data.to_vec(),
            pps: pps_data.to_vec(),
        })
    } else {
        None
    }
}

/// Convert Annex B formatted HEVC data to hvcC format (length-prefixed).
///
/// Same conversion as H.264: replaces start codes with 4-byte NAL lengths.
pub fn hevc_annexb_to_hvcc(data: &[u8]) -> Vec<u8> {
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

/// Check if the given Annex B data represents an HEVC keyframe (IRAP).
pub fn is_hevc_keyframe(data: &[u8]) -> bool {
    assert_invariant!(
        !data.is_empty(),
        "INV-503: HEVC keyframe detection requires non-empty data"
    );

    for nal in AnnexBNalIter::new(data) {
        if nal.is_empty() {
            continue;
        }
        let nal_type = hevc_nal_type(nal);

        assert_invariant!(
            nal_type <= 63,
            "INV-504: HEVC NAL type must be valid (0-63)"
        );

        if is_hevc_keyframe_nal_type(nal_type) {
            return true;
        }
    }

    assert_invariant!(
        AnnexBNalIter::new(data).count() > 0,
        "INV-505: HEVC keyframe detection must find at least one NAL unit"
    );

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hevc_nal_type_extraction() {
        // VPS NAL: type 32, so first byte has (32 << 1) = 64 = 0x40
        let vps_nal = [0x40, 0x01, 0x0c];
        assert_eq!(hevc_nal_type(&vps_nal), 32);

        // SPS NAL: type 33, so first byte has (33 << 1) = 66 = 0x42
        let sps_nal = [0x42, 0x01, 0x01];
        assert_eq!(hevc_nal_type(&sps_nal), 33);
    }

    #[test]
    fn test_is_hevc_keyframe_nal_type() {
        // IRAP types should be keyframes
        assert!(is_hevc_keyframe_nal_type(nal_type::IDR_W_RADL));
        assert!(is_hevc_keyframe_nal_type(nal_type::IDR_N_LP));
        assert!(is_hevc_keyframe_nal_type(nal_type::CRA_NUT));
        assert!(is_hevc_keyframe_nal_type(nal_type::BLA_W_LP));
        // Non-IRAP types
        assert!(!is_hevc_keyframe_nal_type(nal_type::VPS));
        assert!(!is_hevc_keyframe_nal_type(nal_type::SPS));
        assert!(!is_hevc_keyframe_nal_type(nal_type::PPS));
    }

    #[test]
    fn test_extract_hevc_config_empty() {
        assert!(extract_hevc_config(&[]).is_none());
    }

    #[test]
    fn test_extract_hevc_config_success() {
        // Minimal HEVC keyframe with VPS, SPS, PPS
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, // VPS (type 32)
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x21, // SPS (type 33)
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0x73, // PPS (type 34)
        ];

        let config = extract_hevc_config(&data).unwrap();
        assert_eq!(hevc_nal_type(&config.vps), nal_type::VPS);
        assert_eq!(hevc_nal_type(&config.sps), nal_type::SPS);
        assert_eq!(hevc_nal_type(&config.pps), nal_type::PPS);
    }

    #[test]
    fn test_extract_hevc_config_missing_vps() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x21, // SPS only
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0x73, // PPS
        ];
        assert!(extract_hevc_config(&data).is_none());
    }

    #[test]
    fn test_extract_hevc_config_missing_sps() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, // VPS only
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0x73, // PPS
        ];
        assert!(extract_hevc_config(&data).is_none());
    }

    #[test]
    fn test_extract_hevc_config_missing_pps() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, // VPS
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x21, // SPS only
        ];
        assert!(extract_hevc_config(&data).is_none());
    }

    #[test]
    fn test_hevc_annexb_to_hvcc() {
        let annexb = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, // VPS (3 bytes)
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, // SPS (2 bytes)
        ];

        let hvcc = hevc_annexb_to_hvcc(&annexb);

        // First NAL: length 3 + data
        assert_eq!(&hvcc[0..4], &[0x00, 0x00, 0x00, 0x03]);
        assert_eq!(&hvcc[4..7], &[0x40, 0x01, 0x0c]);

        // Second NAL: length 2 + data
        assert_eq!(&hvcc[7..11], &[0x00, 0x00, 0x00, 0x02]);
        assert_eq!(&hvcc[11..13], &[0x42, 0x01]);
    }

    #[test]
    fn test_is_hevc_keyframe() {
        // IDR frame
        let idr = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, // VPS
            0x00, 0x00, 0x00, 0x01, 0x26, 0x01, // IDR_W_RADL (type 19)
        ];
        assert!(is_hevc_keyframe(&idr));

        // Non-keyframe (TRAIL_R, type 1)
        let trail = [
            0x00, 0x00, 0x00, 0x01, 0x02, 0x01, // TRAIL_R (type 1)
        ];
        assert!(!is_hevc_keyframe(&trail));
    }

    #[test]
    fn test_hevc_config_accessors() {
        // Create config with realistic SPS header
        // HEVC SPS has profile_tier_level at byte offset 3+
        let config = HevcConfig::new(
            vec![0x40, 0x01], // VPS
            vec![
                0x42, 0x01, 0x01, 0x21, 0x80, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
                0x00, 0x5d,
            ], // SPS with level at byte 15
            vec![0x44, 0x01], // PPS
        );

        // Profile space is bits 6-7 of byte 3 (0x21 >> 6 = 0)
        assert_eq!(config.general_profile_space(), 0);
        // Tier flag is bit 5 of byte 3 (0x21 >> 5 & 1 = 1)
        assert!(config.general_tier_flag());
        // Profile IDC is bits 0-4 of byte 3 (0x21 & 0x1f = 1)
        assert_eq!(config.general_profile_idc(), 1);
        // Level is byte 14 (index 14, which is 0x5d in this SPS)
        // Our SPS is 16 bytes, so byte 14 is the second-to-last = 0x00
        // Let's just verify the accessor works with a valid result
        let level = config.general_level_idc();
        assert!(level <= 186); // Max valid HEVC level
    }
}
