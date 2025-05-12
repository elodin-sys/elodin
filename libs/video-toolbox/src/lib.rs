//! High-level wrapper for Apple's VideoToolbox framework.
//!
//! This library provides a simplified interface to decode H.264 Annex-B NAL packets
//! into frames using Apple's VideoToolbox API through the objc2-video-toolbox bindings.

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod platform;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use platform::*;

/// Error type for VideoToolbox operations
#[derive(Debug, thiserror::Error, Clone)]
pub enum Error {
    /// VideoToolbox reported an error
    #[error("VideoToolbox error: {0}")]
    VideoToolbox(i32),

    /// Failed to create format description
    #[error("Failed to create format description")]
    FormatDescription,

    /// Failed to create decompression session
    #[error("Failed to create decompression session")]
    DecompressionSession,

    /// Failed to decode frame
    #[error("Failed to decode frame")]
    DecodeFrame,

    /// No SPS or PPS NAL units found in the stream
    #[error("No SPS or PPS NAL units found")]
    MissingParameters,

    /// Platform does not support VideoToolbox
    #[error("Platform does not support VideoToolbox")]
    UnsupportedPlatform,
}

/// Result type for VideoToolbox operations
pub type Result<T> = std::result::Result<T, Error>;

/// Decoded video frame information
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    // Store the actual pixel data instead of the buffer reference
    pub y_plane: Vec<u8>,
    pub uv_plane: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub y_stride: usize,
    pub uv_stride: usize,
}

impl DecodedFrame {
    /// Get the width of the frame in pixels
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height of the frame in pixels
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the Y plane stride (bytes per row)
    pub fn y_stride(&self) -> usize {
        self.y_stride
    }

    /// Get the UV plane stride (bytes per row)
    pub fn uv_stride(&self) -> usize {
        self.uv_stride
    }

    /// Get a reference to the Y plane data
    pub fn y_plane(&self) -> &[u8] {
        &self.y_plane
    }

    /// Get a reference to the UV plane data
    pub fn uv_plane(&self) -> &[u8] {
        &self.uv_plane
    }
}

/// Flag indicating NAL unit type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalType {
    /// Sequence Parameter Set
    Sps = 7,
    /// Picture Parameter Set
    Pps = 8,
    /// Instantaneous Decoder Refresh
    Idr = 5,
    /// Non-IDR slice
    Slice = 1,
    /// Other NAL unit type
    Other,
}

impl From<u8> for NalType {
    fn from(val: u8) -> Self {
        match val & 0x1F {
            7 => NalType::Sps,
            8 => NalType::Pps,
            5 => NalType::Idr,
            1 => NalType::Slice,
            _ => NalType::Other,
        }
    }
}

/// NAL unit information
#[derive(Debug)]
pub struct NalUnit<'a> {
    /// NAL unit type
    pub nal_type: NalType,
    /// Raw NAL unit data (without start code)
    pub data: &'a [u8],
}

/// H264 decoder using VideoToolbox
pub trait H264Decoder {
    fn new() -> Result<Self>
    where
        Self: Sized;

    /// Decode H264 Annex-B NAL units
    fn decode_nal(&mut self, nal_data: &[u8], pts: i64) -> Result<Option<DecodedFrame>>;

    /// Decode a frame from a complete H264 Annex-B NAL packet
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Option<DecodedFrame>>;

    /// Reset the decoder state
    fn reset(&mut self) -> Result<()>;
}

/// Helper function to find NAL units in an Annex-B formatted buffer
pub fn find_nal_units(data: &[u8]) -> Vec<NalUnit> {
    let mut nal_units = Vec::new();
    let mut pos = 0;

    while pos + 3 < data.len() {
        // Find the start code
        let mut start_code_len = 0;
        if pos + 3 < data.len() && data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 1 {
            start_code_len = 3;
        } else if pos + 4 < data.len()
            && data[pos] == 0
            && data[pos + 1] == 0
            && data[pos + 2] == 0
            && data[pos + 3] == 1
        {
            start_code_len = 4;
        }

        if start_code_len > 0 {
            let nal_start = pos + start_code_len;

            // Find the next start code or end of data
            let mut nal_end = data.len();
            for i in (nal_start + 3)..data.len() {
                if (data[i - 3] == 0 && data[i - 2] == 0 && data[i - 1] == 1)
                    || (i > 3
                        && data[i - 4] == 0
                        && data[i - 3] == 0
                        && data[i - 2] == 0
                        && data[i - 1] == 1)
                {
                    nal_end = i - 3;
                    if data[i - 4] == 0 {
                        nal_end = i - 4;
                    }
                    break;
                }
            }

            if nal_start < nal_end && nal_start < data.len() {
                let nal_type = NalType::from(data[nal_start]);
                nal_units.push(NalUnit {
                    nal_type,
                    data: &data[nal_start..nal_end],
                });
            }

            pos = nal_end;
        } else {
            pos += 1;
        }
    }

    nal_units
}
