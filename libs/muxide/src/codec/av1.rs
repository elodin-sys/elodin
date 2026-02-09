//! AV1 codec configuration extraction.
//!
//! This module provides parsing for AV1 OBU (Open Bitstream Unit) streams
//! to extract configuration needed for MP4 muxing.
//!
//! # Overview
//!
//! AV1 uses OBU (Open Bitstream Unit) framing instead of NAL units.
//! Configuration requires extracting the Sequence Header OBU.
//!
//! # OBU Types
//!
//! | Type | Name | Purpose |
//! |------|------|---------|
//! | 1 | OBU_SEQUENCE_HEADER | Sequence configuration |
//! | 3 | OBU_FRAME_HEADER | Frame metadata |
//! | 6 | OBU_FRAME | Complete frame |
//!
//! # Key Differences from H.264/H.265
//!
//! - No start codes; uses length-prefixed OBUs
//! - OBU header is 1-2 bytes (has_extension flag)
//! - Sizes use LEB128 variable-length encoding
//! - Configuration box is `av1C`
//! - Keyframes are identified by `frame_type == KEY_FRAME` in header

use crate::assert_invariant;

/// AV1 OBU type constants.
pub mod obu_type {
    /// Sequence Header OBU
    pub const SEQUENCE_HEADER: u8 = 1;
    /// Temporal Delimiter OBU
    pub const TEMPORAL_DELIMITER: u8 = 2;
    /// Frame Header OBU
    pub const FRAME_HEADER: u8 = 3;
    /// Tile Group OBU
    pub const TILE_GROUP: u8 = 4;
    /// Metadata OBU
    pub const METADATA: u8 = 5;
    /// Frame OBU (contains header + tile data)
    pub const FRAME: u8 = 6;
    /// Redundant Frame Header OBU
    pub const REDUNDANT_FRAME_HEADER: u8 = 7;
    /// Tile List OBU
    pub const TILE_LIST: u8 = 8;
    /// Padding OBU
    pub const PADDING: u8 = 15;
}

/// AV1 frame types.
pub mod frame_type {
    /// Key frame
    pub const KEY_FRAME: u8 = 0;
    /// Inter frame
    pub const INTER_FRAME: u8 = 1;
    /// Intra-only frame
    pub const INTRA_ONLY_FRAME: u8 = 2;
    /// Switch frame
    pub const SWITCH_FRAME: u8 = 3;
}

/// AV1 codec configuration.
///
/// Contains the Sequence Header OBU and derived configuration
/// needed to build the av1C box in MP4 containers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1Config {
    /// Raw Sequence Header OBU bytes (including OBU header)
    pub sequence_header: Vec<u8>,
    /// seq_profile (0-2)
    pub seq_profile: u8,
    /// seq_level_idx
    pub seq_level_idx: u8,
    /// seq_tier (0 or 1)
    pub seq_tier: u8,
    /// high_bitdepth flag
    pub high_bitdepth: bool,
    /// twelve_bit flag (only valid when high_bitdepth is true)
    pub twelve_bit: bool,
    /// monochrome flag
    pub monochrome: bool,
    /// chroma_subsampling_x
    pub chroma_subsampling_x: bool,
    /// chroma_subsampling_y
    pub chroma_subsampling_y: bool,
    /// chroma_sample_position
    pub chroma_sample_position: u8,
}

impl Default for Av1Config {
    fn default() -> Self {
        Self {
            sequence_header: Vec::new(),
            seq_profile: 0,
            seq_level_idx: 0,
            seq_tier: 0,
            high_bitdepth: false,
            twelve_bit: false,
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: true,
            chroma_sample_position: 0,
        }
    }
}

/// Extract the OBU type from an OBU header byte.
///
/// OBU type is in bits 3-6 of the first byte:
/// `(obu_header >> 3) & 0x0f`
#[inline]
pub fn obu_type(header_byte: u8) -> u8 {
    (header_byte >> 3) & 0x0f
}

/// Check if the OBU header has an extension byte.
#[inline]
pub fn obu_has_extension(header_byte: u8) -> bool {
    (header_byte & 0x04) != 0
}

/// Check if the OBU has a size field.
#[inline]
pub fn obu_has_size(header_byte: u8) -> bool {
    (header_byte & 0x02) != 0
}

/// Read a LEB128 (Little Endian Base 128) encoded unsigned integer.
///
/// Returns (value, bytes_consumed) or None if invalid.
pub fn read_leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().take(8).enumerate() {
        value |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Some((value, i + 1));
        }
        shift += 7;
    }
    None
}

/// Information about a parsed OBU.
#[derive(Debug, Clone)]
pub struct ObuInfo {
    /// OBU type
    pub obu_type: u8,
    /// Has extension header
    pub has_extension: bool,
    /// Size of header (1 or 2 bytes)
    pub header_size: usize,
    /// Size of payload (excluding header)
    pub payload_size: usize,
    /// Total size (header + payload)
    pub total_size: usize,
}

/// Parse an OBU header and return info about the OBU.
///
/// Returns (ObuInfo, header_end_offset) or None if invalid.
pub fn parse_obu_header(data: &[u8]) -> Option<ObuInfo> {
    if data.is_empty() {
        return None;
    }

    let header_byte = data[0];

    // Check forbidden bit (must be 0)
    if (header_byte & 0x80) != 0 {
        return None;
    }

    let obu_type_val = obu_type(header_byte);
    let has_extension = obu_has_extension(header_byte);
    let has_size = obu_has_size(header_byte);

    let mut header_size = 1;

    // Skip extension byte if present
    if has_extension {
        if data.len() < 2 {
            return None;
        }
        header_size = 2;
    }

    // Parse size if present
    let payload_size = if has_size {
        if data.len() <= header_size {
            return None;
        }
        let (size, leb_len) = read_leb128(&data[header_size..])?;
        header_size += leb_len;
        size as usize
    } else {
        // Without size field, OBU extends to end of data
        data.len().saturating_sub(header_size)
    };

    Some(ObuInfo {
        obu_type: obu_type_val,
        has_extension,
        header_size,
        payload_size,
        total_size: header_size + payload_size,
    })
}

/// Iterator over OBUs in AV1 bitstream data.
pub struct ObuIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ObuIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'a> Iterator for ObuIter<'a> {
    type Item = (ObuInfo, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let remaining = &self.data[self.pos..];
        let info = parse_obu_header(remaining)?;

        if self.pos + info.total_size > self.data.len() {
            return None;
        }

        let obu_data = &remaining[..info.total_size];
        self.pos += info.total_size;
        Some((info, obu_data))
    }
}

/// Extract AV1 configuration from bitstream data.
///
/// Searches for the Sequence Header OBU and extracts configuration.
pub fn extract_av1_config(data: &[u8]) -> Option<Av1Config> {
    if data.is_empty() {
        return None;
    }

    for (info, obu_data) in ObuIter::new(data) {
        // INV-201: OBU type must be valid
        assert_invariant!(
            info.obu_type <= 15,
            "INV-201: AV1 OBU type must be valid (0-15)",
            "codec::av1::extract_av1_config"
        );

        if info.obu_type == obu_type::SEQUENCE_HEADER {
            return parse_sequence_header(obu_data, info.header_size);
        }
    }
    None
}

/// Parse the Sequence Header OBU payload to extract configuration.
fn parse_sequence_header(obu_data: &[u8], header_size: usize) -> Option<Av1Config> {
    // INV-202: Header size must be valid
    assert_invariant!(
        header_size <= obu_data.len(),
        "AV1 header size must not exceed OBU data length",
        "codec::av1::parse_sequence_header"
    );

    let payload = &obu_data[header_size..];
    if payload.is_empty() {
        return None;
    }

    // INV-203: Sequence header payload must be non-empty
    assert_invariant!(
        !payload.is_empty(),
        "AV1 sequence header payload must be non-empty",
        "codec::av1::parse_sequence_header"
    );

    // Create a bit reader for the payload
    let mut reader = BitReader::new(payload);

    // seq_profile: 3 bits
    let seq_profile = reader.read_bits(3)? as u8;

    // INV-204: Sequence profile must be valid (0-3)
    assert_invariant!(
        seq_profile <= 3,
        "AV1 sequence profile must be valid (0-3)",
        "codec::av1::parse_sequence_header"
    );

    // still_picture: 1 bit
    let _still_picture = reader.read_bit()?;

    // reduced_still_picture_header: 1 bit
    let reduced_still_picture_header = reader.read_bit()?;

    let (seq_level_idx, seq_tier) = if reduced_still_picture_header {
        // seq_level_idx[0]: 5 bits
        (reader.read_bits(5)? as u8, 0u8)
    } else {
        // timing_info_present_flag: 1 bit
        let timing_info_present = reader.read_bit()?;
        if timing_info_present {
            // Skip timing_info
            reader.skip_bits(32)?; // num_units_in_display_tick
            reader.skip_bits(32)?; // time_scale
            let equal_picture_interval = reader.read_bit()?;
            if equal_picture_interval {
                // Skip num_ticks_per_picture_minus_1 (uvlc)
                skip_uvlc(&mut reader)?;
            }
        }

        // decoder_model_info_present_flag: 1 bit
        let decoder_model_info_present = reader.read_bit()?;
        let mut buffer_delay_length = 0;
        if decoder_model_info_present {
            buffer_delay_length = reader.read_bits(5)? as u8 + 1;
            reader.skip_bits(32)?; // num_units_in_decoding_tick
            reader.skip_bits(5)?; // buffer_removal_time_length
            reader.skip_bits(5)?; // frame_presentation_time_length
        }

        // initial_display_delay_present_flag: 1 bit
        let initial_display_delay_present = reader.read_bit()?;

        // operating_points_cnt_minus_1: 5 bits
        let op_cnt = reader.read_bits(5)? as usize + 1;

        let mut first_seq_level_idx = 0u8;
        let mut first_seq_tier = 0u8;

        for i in 0..op_cnt {
            reader.skip_bits(12)?; // operating_point_idc
            let level_idx = reader.read_bits(5)? as u8;
            let tier = if level_idx > 7 {
                reader.read_bit()? as u8
            } else {
                0
            };
            if i == 0 {
                first_seq_level_idx = level_idx;
                first_seq_tier = tier;
            }
            if decoder_model_info_present {
                let decoder_model_present = reader.read_bit()?;
                if decoder_model_present {
                    reader.skip_bits(buffer_delay_length as usize)?; // decoder_buffer_delay
                    reader.skip_bits(buffer_delay_length as usize)?; // encoder_buffer_delay
                    reader.read_bit()?; // low_delay_mode_flag
                }
            }
            if initial_display_delay_present {
                let display_delay_present = reader.read_bit()?;
                if display_delay_present {
                    reader.skip_bits(4)?; // initial_display_delay_minus_1
                }
            }
        }
        (first_seq_level_idx, first_seq_tier)
    };

    // frame_width_bits_minus_1: 4 bits
    let frame_width_bits = reader.read_bits(4)? as usize + 1;
    // frame_height_bits_minus_1: 4 bits
    let frame_height_bits = reader.read_bits(4)? as usize + 1;
    // max_frame_width_minus_1
    let _max_width = reader.read_bits(frame_width_bits)? + 1;
    // max_frame_height_minus_1
    let _max_height = reader.read_bits(frame_height_bits)? + 1;

    // For reduced_still_picture_header, frame_id is not present
    if !reduced_still_picture_header {
        let frame_id_numbers_present = reader.read_bit()?;
        if frame_id_numbers_present {
            let delta_frame_id_length = reader.read_bits(4)? as usize + 2;
            let _frame_id_length = reader.read_bits(3)? as usize + delta_frame_id_length + 1;
        }
    }

    // use_128x128_superblock: 1 bit
    reader.read_bit()?;
    // enable_filter_intra: 1 bit
    reader.read_bit()?;
    // enable_intra_edge_filter: 1 bit
    reader.read_bit()?;

    // More flags for non-reduced headers
    if !reduced_still_picture_header {
        // enable_interintra_compound: 1 bit
        reader.read_bit()?;
        // enable_masked_compound: 1 bit
        reader.read_bit()?;
        // enable_warped_motion: 1 bit
        reader.read_bit()?;
        // enable_dual_filter: 1 bit
        reader.read_bit()?;
        // enable_order_hint: 1 bit
        let enable_order_hint = reader.read_bit()?;
        if enable_order_hint {
            // enable_jnt_comp: 1 bit
            reader.read_bit()?;
            // enable_ref_frame_mvs: 1 bit
            reader.read_bit()?;
        }
        // seq_choose_screen_content_tools: 1 bit
        let seq_choose_screen_content_tools = reader.read_bit()?;
        let seq_force_screen_content_tools = if seq_choose_screen_content_tools {
            2 // SELECT_SCREEN_CONTENT_TOOLS
        } else {
            reader.read_bit()? as u8
        };
        if seq_force_screen_content_tools > 0 {
            // seq_choose_integer_mv: 1 bit
            let seq_choose_integer_mv = reader.read_bit()?;
            if !seq_choose_integer_mv {
                // seq_force_integer_mv: 1 bit
                reader.read_bit()?;
            }
        }
        if enable_order_hint {
            // order_hint_bits_minus_1: 3 bits
            reader.skip_bits(3)?;
        }
    }

    // enable_superres: 1 bit
    reader.read_bit()?;
    // enable_cdef: 1 bit
    reader.read_bit()?;
    // enable_restoration: 1 bit
    reader.read_bit()?;

    // color_config
    let (
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x,
        chroma_subsampling_y,
        chroma_sample_position,
    ) = parse_color_config(&mut reader, seq_profile)?;

    // film_grain_params_present: 1 bit
    reader.read_bit()?;

    Some(Av1Config {
        sequence_header: obu_data.to_vec(),
        seq_profile,
        seq_level_idx,
        seq_tier,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x,
        chroma_subsampling_y,
        chroma_sample_position,
    })
}

/// Parse color_config from sequence header.
fn parse_color_config(
    reader: &mut BitReader,
    seq_profile: u8,
) -> Option<(bool, bool, bool, bool, bool, u8)> {
    // high_bitdepth: 1 bit
    let high_bitdepth = reader.read_bit()?;

    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        reader.read_bit()?
    } else {
        false
    };

    let bit_depth = if seq_profile == 2 && twelve_bit {
        12
    } else if high_bitdepth {
        10
    } else {
        8
    };

    let monochrome = if seq_profile == 1 {
        false
    } else {
        reader.read_bit()?
    };

    // color_description_present_flag: 1 bit
    let color_description_present = reader.read_bit()?;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present {
            let cp = reader.read_bits(8)? as u8;
            let tc = reader.read_bits(8)? as u8;
            let mc = reader.read_bits(8)? as u8;
            (cp, tc, mc)
        } else {
            (2, 2, 2) // Unspecified
        };

    let (chroma_subsampling_x, chroma_subsampling_y, _chroma_sample_position) = if monochrome {
        // color_range: 1 bit
        reader.read_bit()?;
        (true, true, 0)
    } else if color_primaries == 1 && transfer_characteristics == 13 && matrix_coefficients == 0 {
        // sRGB/sYCC
        (false, false, 0)
    } else {
        // color_range: 1 bit
        reader.read_bit()?;

        if seq_profile == 0 {
            (true, true, 0)
        } else if seq_profile == 1 {
            (false, false, 0)
        } else if bit_depth == 12 {
            let subsampling_x = reader.read_bit()?;
            let subsampling_y = if subsampling_x {
                reader.read_bit()?
            } else {
                false
            };
            (subsampling_x, subsampling_y, 0)
        } else {
            (true, false, 0)
        }
    };

    let chroma_sample_position = if chroma_subsampling_x && chroma_subsampling_y {
        reader.read_bits(2)? as u8
    } else {
        0
    };

    // separate_uv_delta_q: 1 bit (if not monochrome)
    if !monochrome {
        reader.read_bit()?;
    }

    Some((
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x,
        chroma_subsampling_y,
        chroma_sample_position,
    ))
}

/// Skip a uvlc (unsigned variable length code) value.
fn skip_uvlc(reader: &mut BitReader) -> Option<()> {
    let mut leading_zeros = 0;
    while !reader.read_bit()? {
        leading_zeros += 1;
        if leading_zeros > 32 {
            return None;
        }
    }
    if leading_zeros > 0 {
        reader.skip_bits(leading_zeros)?;
    }
    Some(())
}

/// Simple bit reader for parsing AV1 bitstream.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit != 0)
    }

    fn read_bits(&mut self, count: usize) -> Option<u64> {
        if count > 64 {
            return None;
        }
        let mut value = 0u64;
        for _ in 0..count {
            value = (value << 1) | (self.read_bit()? as u64);
        }
        Some(value)
    }

    fn skip_bits(&mut self, count: usize) -> Option<()> {
        for _ in 0..count {
            self.read_bit()?;
        }
        Some(())
    }
}

/// Check if the given data contains an AV1 keyframe.
///
/// An AV1 keyframe is identified by a Frame or Frame Header OBU
/// with frame_type == KEY_FRAME.
pub fn is_av1_keyframe(data: &[u8]) -> bool {
    for (info, obu_data) in ObuIter::new(data) {
        if info.obu_type == obu_type::FRAME || info.obu_type == obu_type::FRAME_HEADER {
            // Parse frame header to check frame_type
            // First bit after header is show_existing_frame
            let payload = &obu_data[info.header_size..];
            if !payload.is_empty() {
                let mut reader = BitReader::new(payload);
                // show_existing_frame: 1 bit
                if let Some(show_existing) = reader.read_bit() {
                    if show_existing {
                        // This shows an existing frame, not a new keyframe
                        continue;
                    }
                    // frame_type: 2 bits
                    if let Some(frame_type_val) = reader.read_bits(2) {
                        if frame_type_val as u8 == frame_type::KEY_FRAME {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_to_bytes(bits: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc = 0u8;
        let mut n = 0;
        for ch in bits.chars() {
            if ch != '0' && ch != '1' {
                continue;
            }
            acc <<= 1;
            if ch == '1' {
                acc |= 1;
            }
            n += 1;
            if n == 8 {
                out.push(acc);
                acc = 0;
                n = 0;
            }
        }
        if n != 0 {
            acc <<= 8 - n;
            out.push(acc);
        }
        out
    }

    #[test]
    fn test_obu_type_extraction() {
        // OBU type 1 (Sequence Header): (1 << 3) = 8 = 0x08
        let seq_header = 0x08;
        assert_eq!(obu_type(seq_header), 1);

        // OBU type 6 (Frame): (6 << 3) = 48 = 0x30
        let frame = 0x30;
        assert_eq!(obu_type(frame), 6);
    }

    #[test]
    fn test_obu_flags() {
        // OBU with extension: bit 2 set
        let with_ext = 0x04;
        assert!(obu_has_extension(with_ext));
        assert!(!obu_has_extension(0x00));

        // OBU with size: bit 1 set
        let with_size = 0x02;
        assert!(obu_has_size(with_size));
        assert!(!obu_has_size(0x00));
    }

    #[test]
    fn test_read_leb128() {
        // Single byte: 0x00 = 0
        assert_eq!(read_leb128(&[0x00]), Some((0, 1)));

        // Single byte: 0x7F = 127
        assert_eq!(read_leb128(&[0x7F]), Some((127, 1)));

        // Two bytes: 0x80 0x01 = 128
        assert_eq!(read_leb128(&[0x80, 0x01]), Some((128, 2)));

        // Two bytes: 0xFF 0x01 = 255
        assert_eq!(read_leb128(&[0xFF, 0x01]), Some((255, 2)));
    }

    #[test]
    fn test_parse_obu_header() {
        // Sequence Header with size = 5
        // Header: 0x0A = OBU type 1, has_size=1
        // Size: 0x05 (LEB128 = 5)
        let data = [0x0A, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00];
        let info = parse_obu_header(&data).unwrap();
        assert_eq!(info.obu_type, 1);
        assert!(!info.has_extension);
        assert_eq!(info.header_size, 2); // 1 byte header + 1 byte size
        assert_eq!(info.payload_size, 5);
        assert_eq!(info.total_size, 7);
    }

    #[test]
    fn test_obu_iterator() {
        // Two OBUs: Temporal Delimiter (empty) + Sequence Header (3 bytes)
        // TD: 0x12 (type=2, has_size=1), 0x00 (size=0)
        // SH: 0x0A (type=1, has_size=1), 0x03 (size=3), 0xAA, 0xBB, 0xCC
        let data = [0x12, 0x00, 0x0A, 0x03, 0xAA, 0xBB, 0xCC];
        let obus: Vec<_> = ObuIter::new(&data).collect();

        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].0.obu_type, 2); // Temporal Delimiter
        assert_eq!(obus[1].0.obu_type, 1); // Sequence Header
    }

    #[test]
    fn test_is_av1_keyframe() {
        // Frame OBU with keyframe
        // Header: 0x32 (type=6, has_size=1)
        // Size: 0x02
        // Payload: show_existing_frame=0, frame_type=0 (KEY_FRAME)
        // First byte of payload: 0b0_00_xxxxx = 0x00 (show_existing=0, frame_type=00)
        let keyframe = [0x32, 0x02, 0x00, 0x00];
        assert!(is_av1_keyframe(&keyframe));

        // Frame OBU with inter frame
        // Payload: show_existing_frame=0, frame_type=1 (INTER_FRAME)
        // First byte: 0b0_01_xxxxx = 0x20
        let interframe = [0x32, 0x02, 0x20, 0x00];
        assert!(!is_av1_keyframe(&interframe));
    }

    #[test]
    fn test_is_av1_keyframe_skips_show_existing_frame() {
        // Frame Header OBU with show_existing_frame=1 (not a new keyframe),
        // followed by a Frame Header OBU with show_existing_frame=0 and keyframe type.
        // Header byte: type=3 (FRAME_HEADER) with has_size=1 => (3<<3)|0x02 = 0x1A.
        let show_existing = [0x1A, 0x02, 0x80, 0x00];
        let keyframe = [0x1A, 0x02, 0x00, 0x00];
        let mut stream = Vec::new();
        stream.extend_from_slice(&show_existing);
        stream.extend_from_slice(&keyframe);

        assert!(is_av1_keyframe(&stream));
    }

    #[test]
    fn test_av1_config_default_values() {
        let cfg = Av1Config::default();
        assert!(cfg.sequence_header.is_empty());
        assert_eq!(cfg.seq_profile, 0);
        assert_eq!(cfg.seq_level_idx, 0);
        assert_eq!(cfg.seq_tier, 0);
        assert!(!cfg.high_bitdepth);
        assert!(!cfg.twelve_bit);
        assert!(!cfg.monochrome);
        assert!(cfg.chroma_subsampling_x);
        assert!(cfg.chroma_subsampling_y);
        assert_eq!(cfg.chroma_sample_position, 0);
    }

    #[test]
    fn test_read_leb128_rejects_overlong_encoding() {
        // 8 bytes with continuation bit set should be rejected (no terminator within 8).
        let bytes = [0x80u8; 8];
        assert!(read_leb128(&bytes).is_none());
    }

    #[test]
    fn test_parse_obu_header_rejects_forbidden_bit() {
        // forbidden bit is MSB, must be 0.
        assert!(parse_obu_header(&[0x80]).is_none());
    }

    #[test]
    fn test_parse_obu_header_extension_requires_second_byte() {
        // has_extension=1 but missing extension byte.
        // type=1, extension=1 => (1<<3)|0x04 = 0x0C
        assert!(parse_obu_header(&[0x0C]).is_none());
    }

    #[test]
    fn test_parse_obu_header_size_requires_bytes() {
        // has_size=1 but no bytes to read size from.
        // type=1, has_size=1 => (1<<3)|0x02 = 0x0A
        assert!(parse_obu_header(&[0x0A]).is_none());

        // header byte + size byte present: parse succeeds and reports total_size.
        let info = parse_obu_header(&[0x0A, 0x01]).expect("OBU header should parse");
        assert_eq!(info.header_size, 2);
        assert_eq!(info.payload_size, 1);
        assert_eq!(info.total_size, 3);
    }

    #[test]
    fn test_bit_reader_read_bits_over_64_returns_none() {
        let data = [0u8; 16];
        let mut reader = BitReader::new(&data);
        assert!(reader.read_bits(65).is_none());
    }

    #[test]
    fn test_skip_uvlc_rejects_too_many_leading_zeros() {
        // Provide > 32 leading zeros with no terminating '1' bit.
        let data = [0u8; 8]; // 64 zero bits
        let mut reader = BitReader::new(&data);
        assert!(skip_uvlc(&mut reader).is_none());
    }

    #[test]
    fn test_parse_color_config_monochrome_twelve_bit_path() {
        // seq_profile=2, high_bitdepth=1, twelve_bit=1, monochrome=1,
        // color_description_present=0, color_range=1, chroma_sample_position=01.
        let bytes = bits_to_bytes("1 1 1 0 1 01");
        let mut reader = BitReader::new(&bytes);
        let parsed = parse_color_config(&mut reader, 2).unwrap();
        assert!(parsed.0); // high_bitdepth
        assert!(parsed.1); // twelve_bit
        assert!(parsed.2); // monochrome
        assert!(parsed.3); // chroma_subsampling_x
        assert!(parsed.4); // chroma_subsampling_y
        assert_eq!(parsed.5, 1); // chroma_sample_position
    }

    #[test]
    fn test_parse_color_config_srgb_path() {
        // sRGB/sYCC path: monochrome=0, color_description_present=1,
        // cp=1, tc=13, mc=0 => (false,false,0) subsampling and chroma_sample_position=0.
        let bytes = bits_to_bytes("0 0 1 00000001 00001101 00000000 0");
        let mut reader = BitReader::new(&bytes);
        let parsed = parse_color_config(&mut reader, 0).unwrap();
        assert!(!parsed.2); // monochrome
        assert!(!parsed.3); // chroma_subsampling_x
        assert!(!parsed.4); // chroma_subsampling_y
        assert_eq!(parsed.5, 0); // chroma_sample_position
    }

    #[test]
    fn test_extract_av1_config_empty() {
        assert!(extract_av1_config(&[]).is_none());
    }

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b01100011];
        let mut reader = BitReader::new(&data);

        // Read first 4 bits: 1011 = 11
        assert_eq!(reader.read_bits(4), Some(11));
        // Read next 4 bits: 0100 = 4
        assert_eq!(reader.read_bits(4), Some(4));
        // Read next 2 bits: 01 = 1
        assert_eq!(reader.read_bits(2), Some(1));
    }

    #[test]
    fn test_parse_sequence_header_basic() {
        // Minimal sequence header OBU
        // OBU header: type=1 (SEQUENCE_HEADER), has_size=1 => 0x0A
        // Size: 0x10 (16 bytes)
        // Sequence header payload (simplified)
        let seq_header = [
            0x0A, 0x10, // OBU header + size
            0x00, 0x00, 0x00,
            0x00, // seq_profile=0, still_picture=0, reduced_still_picture_header=0
            0x00, 0x00, 0x00, 0x00, // timing_info_present=0, decoder_model_info_present=0
            0x00, 0x00, 0x00,
            0x00, // initial_display_delay_present=0, operating_points_cnt_minus_1=0
            0x00, 0x00, // operating_point_idc[0]=0, seq_level_idx[0]=0
            0x00, // seq_tier[0]=0
        ];

        let config = parse_sequence_header(&seq_header[2..], 2);
        assert!(config.is_some());
        let cfg = config.unwrap();
        assert_eq!(cfg.seq_profile, 0);
        assert_eq!(cfg.seq_level_idx, 0);
        assert_eq!(cfg.seq_tier, 0);
        assert!(!cfg.high_bitdepth);
        assert!(!cfg.twelve_bit);
        assert!(!cfg.monochrome);
    }
}
