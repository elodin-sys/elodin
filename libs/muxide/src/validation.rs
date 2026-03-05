//! Input validation utilities for pre-muxing checks.
//!
//! This module provides functions to validate encoded frames and muxing
//! parameters before creating output files. Useful for implementing
//! "dry-run" functionality in recording applications.

use crate::api::{AudioCodec, VideoCodec};
use crate::codec::av1::is_av1_keyframe;
use crate::codec::h264::is_h264_keyframe;
use crate::codec::h265::is_hevc_keyframe;
use crate::codec::opus::is_valid_opus_packet;
use crate::codec::vp9::is_vp9_keyframe;

/// Result of input validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    /// Whether the input is valid for muxing.
    pub is_valid: bool,
    /// Human-readable validation messages.
    pub messages: Vec<String>,
    /// Detailed error information if invalid.
    pub errors: Vec<String>,
}

impl ValidationResult {
    /// Create a successful validation result.
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            messages: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Create a failed validation result with errors.
    pub fn invalid(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            messages: Vec::new(),
            errors,
        }
    }

    /// Add an informational message.
    pub fn with_message(mut self, message: String) -> Self {
        self.messages.push(message);
        self
    }

    /// Add an error.
    pub fn with_error(mut self, error: String) -> Self {
        self.errors.push(error);
        self.is_valid = false;
        self
    }
}

/// Configuration for video validation.
#[derive(Debug, Clone)]
pub struct VideoValidationConfig {
    pub codec: Option<VideoCodec>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub framerate: Option<f64>,
    pub sample_frame: Option<(Vec<u8>, bool)>,
}

/// Configuration for audio validation.
#[derive(Debug, Clone)]
pub struct AudioValidationConfig {
    pub codec: Option<AudioCodec>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub sample_frame: Option<Vec<u8>>,
}

/// Validate video codec parameters for muxing.
///
/// Checks that the provided dimensions, framerate, and codec are supported
/// and within reasonable limits.
pub fn validate_video_config(
    codec: VideoCodec,
    width: u32,
    height: u32,
    framerate: f64,
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check codec support
    match codec {
        VideoCodec::H264 | VideoCodec::H265 | VideoCodec::Av1 | VideoCodec::Vp9 => {
            result = result.with_message(format!("✓ Video codec {} is supported", codec));
        }
    }

    // Check dimensions
    if width == 0 || height == 0 {
        result = result.with_error("Video width and height must be positive".to_string());
    } else if width > 4096 || height > 2160 {
        result = result.with_error(format!(
            "Video dimensions {}x{} exceed maximum supported size (4096x2160)",
            width, height
        ));
    } else if width < 320 || height < 240 {
        result = result.with_error(format!(
            "Video dimensions {}x{} below minimum supported size (320x240)",
            width, height
        ));
    } else {
        result = result.with_message(format!("✓ Video dimensions {}x{} are valid", width, height));
    }

    // Check framerate
    if framerate <= 0.0 {
        result = result.with_error("Video framerate must be positive".to_string());
    } else if framerate > 120.0 {
        result = result.with_error(format!(
            "Video framerate {:.1} exceeds maximum supported rate (120 fps)",
            framerate
        ));
    } else {
        result = result.with_message(format!("✓ Video framerate {:.1} fps is valid", framerate));
    }

    result
}

/// Validate audio codec parameters for muxing.
pub fn validate_audio_config(
    codec: AudioCodec,
    sample_rate: u32,
    channels: u8,
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Check codec support
    match codec {
        AudioCodec::Aac(_) | AudioCodec::Opus => {
            result = result.with_message(format!("✓ Audio codec {} is supported", codec));
        }
        AudioCodec::None => {
            result = result.with_message("✓ No audio configured".to_string());
            return result;
        }
    }

    // Check sample rate
    if sample_rate == 0 {
        result = result.with_error("Audio sample rate must be positive".to_string());
    } else if sample_rate > 192000 {
        result = result.with_error(format!(
            "Audio sample rate {} exceeds maximum supported rate (192000 Hz)",
            sample_rate
        ));
    } else {
        result = result.with_message(format!("✓ Audio sample rate {} Hz is valid", sample_rate));
    }

    // Check channels
    if channels == 0 {
        result = result.with_error("Audio channels must be positive".to_string());
    } else if channels > 8 {
        result = result.with_error(format!(
            "Audio channels {} exceeds maximum supported count (8)",
            channels
        ));
    } else {
        result = result.with_message(format!("✓ Audio channels {} are valid", channels));
    }

    result
}

/// Validate a single video frame for the given codec.
///
/// Checks that the frame data is properly formatted and contains
/// necessary headers for the specified codec.
pub fn validate_video_frame(
    codec: VideoCodec,
    frame_data: &[u8],
    is_keyframe: bool,
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    if frame_data.is_empty() {
        return result.with_error("Video frame data cannot be empty".to_string());
    }

    // Check keyframe detection
    let detected_keyframe = match codec {
        VideoCodec::H264 => is_h264_keyframe(frame_data),
        VideoCodec::H265 => is_hevc_keyframe(frame_data),
        VideoCodec::Av1 => is_av1_keyframe(frame_data),
        VideoCodec::Vp9 => is_vp9_keyframe(frame_data).unwrap_or(false),
    };

    if is_keyframe && !detected_keyframe {
        result = result.with_error(format!(
            "Frame marked as keyframe but {} codec detection indicates it's not a keyframe",
            codec
        ));
    } else if !is_keyframe && detected_keyframe {
        result = result.with_message(format!(
            "⚠ Frame not marked as keyframe but {} codec detection indicates it is a keyframe",
            codec
        ));
    } else {
        result = result.with_message(format!(
            "✓ Frame keyframe flag matches {} codec detection",
            codec
        ));
    }

    result
}

/// Validate a single audio frame for the given codec.
pub fn validate_audio_frame(codec: AudioCodec, frame_data: &[u8]) -> ValidationResult {
    let mut result = ValidationResult::valid();

    if frame_data.is_empty() {
        return result.with_error("Audio frame data cannot be empty".to_string());
    }

    match codec {
        AudioCodec::Aac(_) => {
            // Basic ADTS header check
            if frame_data.len() < 7 {
                result = result.with_error("AAC frame too short for ADTS header".to_string());
            } else if frame_data[0] != 0xFF || (frame_data[1] & 0xF0) != 0xF0 {
                result = result.with_error("Invalid AAC ADTS syncword".to_string());
            } else {
                result = result.with_message("✓ AAC frame has valid ADTS header".to_string());
            }
        }
        AudioCodec::Opus => {
            if !is_valid_opus_packet(frame_data) {
                result = result.with_error("Invalid Opus packet structure".to_string());
            } else {
                result = result.with_message("✓ Opus packet has valid structure".to_string());
            }
        }
        AudioCodec::None => {
            result = result.with_error("Cannot validate audio frame for None codec".to_string());
        }
    }

    result
}

/// Comprehensive validation for a complete muxing configuration.
///
/// Validates all parameters and performs basic checks on sample frames
/// to ensure they can be successfully muxed.
pub fn validate_muxing_config(
    video_config: VideoValidationConfig,
    audio_config: AudioValidationConfig,
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    // Validate video config if provided
    if let (Some(vc), Some(w), Some(h), Some(fps)) = (
        video_config.codec,
        video_config.width,
        video_config.height,
        video_config.framerate,
    ) {
        let video_result = validate_video_config(vc, w, h, fps);
        result.is_valid &= video_result.is_valid;
        result.messages.extend(video_result.messages);
        result.errors.extend(video_result.errors);

        // Validate sample video frame if provided
        if let Some((frame_data, is_keyframe)) = &video_config.sample_frame {
            let frame_result = validate_video_frame(vc, frame_data, *is_keyframe);
            result.is_valid &= frame_result.is_valid;
            result.messages.extend(frame_result.messages);
            result.errors.extend(frame_result.errors);
        }
    } else if video_config.codec.is_some() {
        result = result.with_error(
            "Video codec specified but missing width, height, or framerate".to_string(),
        );
    }

    // Validate audio config if provided
    if let (Some(ac), Some(sr), Some(ch)) = (
        audio_config.codec,
        audio_config.sample_rate,
        audio_config.channels,
    ) {
        let audio_result = validate_audio_config(ac, sr, ch);
        result.is_valid &= audio_result.is_valid;
        result.messages.extend(audio_result.messages);
        result.errors.extend(audio_result.errors);

        // Validate sample audio frame if provided
        if let Some(frame_data) = &audio_config.sample_frame {
            let frame_result = validate_audio_frame(ac, frame_data);
            result.is_valid &= frame_result.is_valid;
            result.messages.extend(frame_result.messages);
            result.errors.extend(frame_result.errors);
        }
    } else if audio_config.codec.is_some() && audio_config.codec != Some(AudioCodec::None) {
        result = result
            .with_error("Audio codec specified but missing sample rate or channels".to_string());
    }

    // Require at least one media stream
    if video_config.codec.is_none()
        && (audio_config.codec.is_none() || audio_config.codec == Some(AudioCodec::None))
    {
        result = result.with_error("At least one of video or audio must be configured".to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{AacProfile, AudioCodec, VideoCodec};

    #[test]
    fn test_validate_video_config_valid() {
        let result = validate_video_config(VideoCodec::H264, 1920, 1080, 30.0);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        assert!(result.messages.len() >= 3); // codec, dimensions, framerate
    }

    #[test]
    fn test_validate_video_config_invalid_dimensions() {
        let result = validate_video_config(VideoCodec::H264, 0, 1080, 30.0);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_audio_config_valid() {
        let result = validate_audio_config(AudioCodec::Aac(AacProfile::Lc), 44100, 2);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_muxing_config_minimal() {
        let video_config = VideoValidationConfig {
            codec: Some(VideoCodec::H264),
            width: Some(1920),
            height: Some(1080),
            framerate: Some(30.0),
            sample_frame: None,
        };
        let audio_config = AudioValidationConfig {
            codec: None,
            sample_rate: None,
            channels: None,
            sample_frame: None,
        };
        let result = validate_muxing_config(video_config, audio_config);
        assert!(result.is_valid);
    }
}
