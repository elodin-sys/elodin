//! VP9 video codec support for MP4 muxing.
//!
//! This module provides VP9 frame parsing and configuration extraction
//! for MP4 container muxing. VP9 frames are expected in their compressed
//! form with frame headers intact.

use crate::assert_invariant;

/// VP9 codec configuration extracted from the first keyframe.
#[derive(Clone, Debug, PartialEq)]
pub struct Vp9Config {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// VP9 profile (0-3).
    pub profile: u8,
    /// Bit depth (8 or 10).
    pub bit_depth: u8,
    /// Color space information.
    pub color_space: u8,
    /// Transfer characteristics.
    pub transfer_function: u8,
    /// Matrix coefficients.
    pub matrix_coefficients: u8,
    /// VP9 level (0-255, typically 0 for most content).
    pub level: u8,
    /// Video full range flag (0 = limited range, 1 = full range).
    pub full_range_flag: u8,
}

/// Errors that can occur during VP9 parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum Vp9Error {
    /// Frame data is too short to contain a valid VP9 frame header.
    FrameTooShort,
    /// Invalid frame marker (first 3 bytes should be 0x49, 0x83, 0x42).
    InvalidFrameMarker,
    /// Unsupported VP9 profile.
    UnsupportedProfile(u8),
    /// Invalid bit depth.
    InvalidBitDepth(u8),
    /// Frame parsing error with details.
    ParseError(String),
}

impl std::fmt::Display for Vp9Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vp9Error::FrameTooShort => write!(f, "VP9 frame too short for header"),
            Vp9Error::InvalidFrameMarker => write!(f, "invalid VP9 frame marker"),
            Vp9Error::UnsupportedProfile(p) => write!(f, "unsupported VP9 profile: {}", p),
            Vp9Error::InvalidBitDepth(b) => write!(f, "invalid VP9 bit depth: {}", b),
            Vp9Error::ParseError(msg) => write!(f, "VP9 parse error: {}", msg),
        }
    }
}

impl std::error::Error for Vp9Error {}

/// Check if a VP9 frame is a keyframe (intra frame).
///
/// VP9 keyframes have frame_type = 0 in the frame header.
pub fn is_vp9_keyframe(frame: &[u8]) -> Result<bool, Vp9Error> {
    if frame.len() < 3 {
        return Err(Vp9Error::FrameTooShort);
    }

    // Check frame marker
    if frame[0] != 0x49 || frame[1] != 0x83 || frame[2] != 0x42 {
        return Err(Vp9Error::InvalidFrameMarker);
    }

    if frame.len() < 4 {
        return Err(Vp9Error::FrameTooShort);
    }

    // Parse frame header to determine frame type
    let profile = (frame[3] >> 6) & 0x03;
    let show_existing_frame = (frame[3] >> 5) & 0x01;
    let frame_type = (frame[3] >> 4) & 0x01;

    // INV-405: VP9 profile must be valid (0-3)
    assert_invariant!(
        profile <= 3,
        "VP9 profile must be valid (0-3)",
        "codec::vp9::is_vp9_keyframe"
    );

    // If show_existing_frame is set, this is not a keyframe
    if show_existing_frame != 0 {
        return Ok(false);
    }

    // frame_type = 0 indicates a keyframe
    Ok(frame_type == 0)
}

/// Extract VP9 configuration from a keyframe.
///
/// This parses the uncompressed header of a VP9 keyframe to extract
/// resolution and other configuration parameters.
pub fn extract_vp9_config(keyframe: &[u8]) -> Option<Vp9Config> {
    if keyframe.len() < 3 {
        return None;
    }

    // Check frame marker
    if keyframe[0] != 0x49 || keyframe[1] != 0x83 || keyframe[2] != 0x42 {
        return None;
    }

    // INV-401: VP9 frame marker must be valid
    assert_invariant!(
        keyframe[0] == 0x49 && keyframe[1] == 0x83 && keyframe[2] == 0x42,
        "INV-401: VP9 frame marker must be 0x49 0x83 0x42",
        "codec::vp9::extract_vp9_config"
    );

    if keyframe.len() < 6 {
        return None;
    }

    // Parse basic frame header fields
    let profile = (keyframe[3] >> 6) & 0x03;
    let show_existing_frame = (keyframe[3] >> 5) & 0x01;
    let frame_type = (keyframe[3] >> 4) & 0x01;

    // INV-402: VP9 profile must be valid (0-3)
    assert_invariant!(
        profile <= 3,
        "INV-402: VP9 profile must be valid (0-3)",
        "codec::vp9::extract_vp9_config"
    );

    if show_existing_frame != 0 || frame_type != 0 {
        return None; // Not a keyframe
    }

    // For keyframes, parse the frame size from the uncompressed header
    // VP9 uses variable-length unsigned integer encoding for frame dimensions

    let mut offset = 5; // Start after the frame header bytes

    // Skip sync code if present (for certain profiles)
    if profile >= 2 {
        if offset + 1 >= keyframe.len() {
            return None;
        }
        offset += 1; // Skip sync code byte
    }

    // Parse frame width (variable length unsigned int)
    let (width, new_offset) = parse_vp9_var_uint(keyframe, offset)?;
    offset = new_offset;

    // Parse frame height (variable length unsigned int)
    let (height, new_offset) = parse_vp9_var_uint(keyframe, offset)?;
    offset = new_offset;

    // Parse render size (optional, may differ from frame size)
    let (render_width, render_height) = if offset + 1 < keyframe.len() {
        let render_and_frame_size_different = (keyframe[offset] & 0x0C) != 0;
        if render_and_frame_size_different {
            offset += 1;
            let (rw, no) = parse_vp9_var_uint(keyframe, offset)?;
            offset = no;
            let (rh, no) = parse_vp9_var_uint(keyframe, offset)?;
            offset = no;
            (rw, rh)
        } else {
            (width, height)
        }
    } else {
        (width, height)
    };

    // Parse color configuration
    let (bit_depth, color_space, transfer_function, matrix_coefficients, full_range_flag) =
        parse_vp9_color_config(keyframe, offset)?;

    Some(Vp9Config {
        width: render_width,
        height: render_height,
        profile,
        bit_depth,
        color_space,
        transfer_function,
        matrix_coefficients,
        level: 0, // VP9 level is typically 0
        full_range_flag,
    })
}

/// Parse VP9 variable-length unsigned integer.
/// Returns (value, new_offset) or None if parsing fails.
fn parse_vp9_var_uint(data: &[u8], mut offset: usize) -> Option<(u32, usize)> {
    let mut value = 0u32;
    let mut shift = 0;

    loop {
        if offset >= data.len() {
            return None;
        }

        let byte = data[offset];
        offset += 1;

        value |= ((byte & 0x7F) as u32) << shift;
        shift += 7;

        if (byte & 0x80) == 0 {
            break;
        }

        if shift >= 32 {
            return None; // Prevent overflow
        }
    }

    Some((value, offset))
}

/// Parse VP9 color configuration from the frame header.
/// Returns (bit_depth, color_space, transfer_function, matrix_coefficients, full_range_flag)
fn parse_vp9_color_config(data: &[u8], mut offset: usize) -> Option<(u8, u8, u8, u8, u8)> {
    if offset >= data.len() {
        return Some((8, 0, 0, 0, 0)); // Default values
    }

    // Parse bit depth
    let bit_depth = if (data[offset] & 0x01) != 0 { 10 } else { 8 };

    // Parse color space and transfer characteristics
    let color_space = (data[offset] >> 1) & 0x07;
    let transfer_function = (data[offset] >> 4) & 0x07;
    let matrix_coefficients = (data[offset] >> 7) & 0x01;

    offset += 1;

    // If color_space != 0, parse additional color config including full_range
    let full_range_flag = if color_space != 0 {
        if offset >= data.len() {
            0 // Default to limited range if data missing
        } else {
            data[offset] & 0x01
        }
    } else {
        0 // Limited range for monochrome
    };

    Some((
        bit_depth,
        color_space,
        transfer_function,
        matrix_coefficients,
        full_range_flag,
    ))
}

/// Validate that a buffer contains a valid VP9 frame.
///
/// This performs basic validation of the VP9 frame structure.
pub fn is_valid_vp9_frame(frame: &[u8]) -> bool {
    if frame.len() < 3 {
        return false;
    }

    // Check frame marker
    frame[0] == 0x49 && frame[1] == 0x83 && frame[2] == 0x42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_frame_marker() {
        let invalid_frame = [0x00, 0x00, 0x00];
        assert!(!is_valid_vp9_frame(&invalid_frame));
        assert!(matches!(
            is_vp9_keyframe(&invalid_frame),
            Err(Vp9Error::InvalidFrameMarker)
        ));
    }

    #[test]
    fn test_frame_too_short() {
        let short_frame = [0x49, 0x83];
        assert!(!is_valid_vp9_frame(&short_frame));
        assert!(matches!(
            is_vp9_keyframe(&short_frame),
            Err(Vp9Error::FrameTooShort)
        ));
    }

    #[test]
    fn test_valid_frame_marker() {
        let valid_frame = [0x49, 0x83, 0x42, 0x00, 0x00, 0x00];
        assert!(is_valid_vp9_frame(&valid_frame));
    }

    #[test]
    fn test_is_vp9_keyframe_valid() {
        // Valid keyframe: profile=0, show_existing_frame=0, frame_type=0
        let keyframe = [0x49, 0x83, 0x42, 0x00, 0x00, 0x00];
        assert_eq!(is_vp9_keyframe(&keyframe), Ok(true));
    }

    #[test]
    fn test_is_vp9_keyframe_pframe() {
        // P-frame: profile=0, show_existing_frame=0, frame_type=1
        let pframe = [0x49, 0x83, 0x42, 0x10, 0x00, 0x00];
        assert_eq!(is_vp9_keyframe(&pframe), Ok(false));
    }

    #[test]
    fn test_is_vp9_keyframe_show_existing() {
        // Show existing frame: profile=0, show_existing_frame=1, frame_type=0
        let show_existing = [0x49, 0x83, 0x42, 0x20, 0x00, 0x00];
        assert_eq!(is_vp9_keyframe(&show_existing), Ok(false));
    }

    #[test]
    fn test_extract_vp9_config_valid() {
        // Minimal valid VP9 keyframe with config
        let keyframe = vec![
            0x49, 0x83, 0x42, // frame marker
            0x00, // profile=0, show_existing=0, frame_type=0
            0x00, // byte 4 (possibly part of header)
            0x80, 0x02, // width = 256 (var_uint: starts at offset 5)
            0x80, 0x02, // height = 256 (var_uint)
            0x00, // render size same as frame size
            0x00, // color config (8-bit, color_space=0)
        ];
        let config = extract_vp9_config(&keyframe);
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.width, 256);
        assert_eq!(config.height, 256);
        assert_eq!(config.profile, 0);
        assert_eq!(config.bit_depth, 8);
        assert_eq!(config.level, 0);
        assert_eq!(config.full_range_flag, 0);
    }

    #[test]
    fn test_extract_vp9_config_invalid_marker() {
        let invalid_frame = [0x00, 0x00, 0x00, 0x00];
        assert!(extract_vp9_config(&invalid_frame).is_none());
    }

    #[test]
    fn test_extract_vp9_config_pframe() {
        // P-frame should not extract config
        let pframe = [0x49, 0x83, 0x42, 0x10, 0x00, 0x00];
        assert!(extract_vp9_config(&pframe).is_none());
    }

    #[test]
    fn test_parse_vp9_var_uint() {
        // Test parsing variable-length unsigned integers
        let data = [0x7F, 0x80, 0x01, 0x80, 0x80, 0x01];

        // Single byte: 0x7F = 127
        assert_eq!(parse_vp9_var_uint(&data, 0), Some((127, 1)));

        // Two bytes: 0x80 0x01 = 128
        assert_eq!(parse_vp9_var_uint(&data, 1), Some((128, 3)));

        // Three bytes: 0x80 0x80 0x01 = 16384
        assert_eq!(parse_vp9_var_uint(&data, 3), Some((16384, 6)));
    }

    #[test]
    fn test_parse_vp9_var_uint_overflow() {
        // Test overflow prevention (more than 5 bytes would overflow u32)
        let data = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80]; // 6 continuation bytes
        assert_eq!(parse_vp9_var_uint(&data, 0), None);
    }

    #[test]
    fn test_parse_vp9_color_config() {
        let data = [0x00]; // 8-bit, color_space=0, transfer=0, matrix=0
        assert_eq!(parse_vp9_color_config(&data, 0), Some((8, 0, 0, 0, 0)));

        let data_10bit = [0x01]; // 10-bit
        assert_eq!(
            parse_vp9_color_config(&data_10bit, 0),
            Some((10, 0, 0, 0, 0))
        );

        let data_color = [0x12]; // 8-bit, color_space=1, transfer=1, matrix=0
        assert_eq!(
            parse_vp9_color_config(&data_color, 0),
            Some((8, 1, 1, 0, 0))
        );
    }

    #[test]
    fn test_parse_vp9_color_config_empty() {
        let data = [];
        // Should return defaults when offset is out of bounds
        assert_eq!(parse_vp9_color_config(&data, 0), Some((8, 0, 0, 0, 0)));
    }
}
