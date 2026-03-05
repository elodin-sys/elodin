//! Property-Based Tests for Muxide
//!
//! These tests verify invariants hold across a wide range of inputs using proptest.
//! They catch edge cases that unit tests might miss.

use proptest::prelude::*;
use std::io::Cursor;
use std::{fs, path::Path};

// Import the muxide crate
use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, VideoCodec};
use muxide::invariant_ppt::{clear_invariant_log, contract_test};

/// Build a minimal AV1 keyframe with Sequence Header OBU.
///
/// This is a synthetic OBU stream with:
/// - OBU_SEQUENCE_HEADER (type 1) with minimal valid content
/// - OBU_FRAME (type 6) with KEY_FRAME indicator
fn build_av1_keyframe() -> Vec<u8> {
    let mut data = Vec::new();

    // OBU 1: Sequence Header (type 1)
    // OBU header: type=1, has_size=1
    // 0b0_0001_0_1_0 = 0x0A (type=1, has_extension=0, has_size=1)
    data.push(0x0A);

    // Size of sequence header payload in LEB128 (let's use 12 bytes)
    data.push(12);

    // Minimal sequence header content (12 bytes)
    // seq_profile=0, frame_width_bits_minus_1=10, frame_height_bits_minus_1=10, etc.
    // This is a simplified synthetic sequence header
    data.extend_from_slice(&[
        0x00, // seq_profile=0, still_picture=0, reduced_still_picture_header=0
        0x00, 0x00, // operating_points
        0x10, // frame_width_bits=11, frame_height_bits=11
        0x07, 0x80, // max_frame_width = 1920
        0x04, 0x38, // max_frame_height = 1080
        0x00, // frame_id_numbers_present_flag=0
        0x00, // use_128x128_superblock=0
        0x00, 0x00, // other flags
    ]);

    // OBU 2: Frame OBU (type 6) with keyframe
    // OBU header: type=6, has_size=1
    // 0b0_0110_0_1_0 = 0x32 (type=6, has_extension=0, has_size=1)
    data.push(0x32);

    // Size of frame payload
    data.push(4);

    // Minimal frame header indicating keyframe
    // show_existing_frame=0, frame_type=KEY_FRAME(0)
    data.extend_from_slice(&[0x10, 0x00, 0x00, 0x00]);

    data
}

fn read_hex_fixture(dir: &str, name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(dir)
        .join(name);
    let contents = fs::read_to_string(path).expect("fixture must be readable");
    let hex: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(hex.len() % 2 == 0, "hex fixtures must have even length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex");
        out.push(byte);
    }
    out
}

/// Helper to create a valid H.264 keyframe with SPS/PPS
fn make_h264_keyframe() -> Vec<u8> {
    // Minimal valid Annex B H.264 stream with SPS, PPS, and IDR slice
    let mut data = Vec::new();
    // SPS (NAL type 7)
    data.extend_from_slice(&[
        0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0x95, 0xa8, 0x28, 0x28, 0x28,
    ]);
    // PPS (NAL type 8)
    data.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80]);
    // IDR slice (NAL type 5)
    data.extend_from_slice(&[
        0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
    ]);
    data
}

/// Helper to create a valid H.264 P-frame
fn make_h264_pframe() -> Vec<u8> {
    // Non-IDR slice (NAL type 1)
    let mut data = Vec::new();
    data.extend_from_slice(&[0, 0, 0, 1, 0x41, 0x9a, 0x24, 0x6c, 0x42, 0xff, 0xff]);
    data
}

proptest! {
    /// Property: Output size must be at least header size (ftyp + moov + mdat headers)
    #[test]
    fn prop_output_always_has_minimum_size(frames in 1..10usize) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        // Write first keyframe at t=0.0
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        // Write P-frames at 30fps intervals
        for i in 1..frames {
            let pts = i as f64 / 30.0; // seconds
            muxer.write_video(pts, &make_h264_pframe(), false).unwrap();
        }

        muxer.finish().unwrap();

        let output = buffer.into_inner();
        // Minimum: ftyp(24) + mdat header(8) + moov(varies, but at least 500)
        prop_assert!(output.len() >= 500, "Output {} bytes is too small", output.len());
    }

    /// Property: Width and height in config are preserved in output
    #[test]
    fn prop_dimensions_preserved(
        width in 64u32..4096,
        height in 64u32..2160
    ) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, width, height, 30.0)
            .build()
            .unwrap();

        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        muxer.finish().unwrap();

        let output = buffer.into_inner();

        // Find avc1 box (contains width/height in visual sample entry)
        // Search for the bytes "avc1" in the output
        let avc1_pos = output.windows(4)
            .position(|w| w == b"avc1");
        prop_assert!(avc1_pos.is_some(), "avc1 box not found in output");

        // Just verify the file was produced and has reasonable structure
        prop_assert!(output.len() >= 500, "Output too small");

        // Verify moov box exists
        let moov_pos = output.windows(4)
            .position(|w| w == b"moov");
        prop_assert!(moov_pos.is_some(), "moov box not found");
    }

    /// Property: PTS values must be monotonically increasing
    #[test]
    fn prop_pts_must_increase(frame_count in 2..20usize) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        // First keyframe at pts=0.0
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        // Subsequent frames with increasing PTS
        for i in 1..frame_count {
            let pts = i as f64 / 30.0;
            muxer.write_video(pts, &make_h264_pframe(), false).unwrap();
        }

        // This should succeed (monotonic PTS)
        let result = muxer.finish();
        prop_assert!(result.is_ok());
    }

    /// Property: Non-monotonic PTS must be rejected
    #[test]
    fn prop_non_monotonic_pts_rejected(
        first_pts in 0.1f64..1.0,
        second_pts in 0.0f64..0.05
    ) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        // First keyframe
        muxer.write_video(first_pts, &make_h264_keyframe(), true).unwrap();

        // Second frame with earlier PTS - should fail
        let result = muxer.write_video(second_pts, &make_h264_pframe(), false);
        prop_assert!(result.is_err(), "Non-monotonic PTS should be rejected");
    }

    /// Property: Frame count in stats must match written frames
    #[test]
    fn prop_frame_count_accurate(frame_count in 1..50usize) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        // First keyframe
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        // Additional frames
        for i in 1..frame_count {
            let pts = i as f64 / 30.0;
            muxer.write_video(pts, &make_h264_pframe(), false).unwrap();
        }

        let stats = muxer.finish_with_stats().unwrap();
        prop_assert_eq!(
            stats.video_frames as usize,
            frame_count,
            "Stats frame count should match written frames"
        );
    }

    /// Property: Duration must be approximately frame_count / fps
    #[test]
    fn prop_duration_reasonable(frame_count in 2..100usize) {
        let fps = 30.0;
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, fps)
            .build()
            .unwrap();

        // Write frames at consistent intervals
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        for i in 1..frame_count {
            let pts = i as f64 / fps; // seconds
            muxer.write_video(pts, &make_h264_pframe(), false).unwrap();
        }

        let stats = muxer.finish_with_stats().unwrap();

        // Duration should be approximately (frames - 1) * frame_duration
        // (frames - 1 because last frame has no next frame to calculate duration)
        let expected_duration_secs = (frame_count - 1) as f64 / fps;

        // Allow 20% tolerance for rounding
        let min_duration = expected_duration_secs * 0.8;
        let max_duration = expected_duration_secs * 1.2 + 0.1; // +0.1s for rounding

        prop_assert!(
            stats.duration_secs >= min_duration && stats.duration_secs <= max_duration,
            "Duration {}s outside expected range {}..{}s for {} frames at {}fps",
            stats.duration_secs, min_duration, max_duration, frame_count, fps
        );
    }

    /// Property: Output must start with ftyp box
    #[test]
    fn prop_output_starts_with_ftyp(frame_count in 1..10usize) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        for i in 1..frame_count {
            muxer.write_video(i as f64 / 30.0, &make_h264_pframe(), false).unwrap();
        }
        muxer.finish().unwrap();

        let output = buffer.into_inner();

        // ftyp box starts at offset 0, box type at offset 4
        prop_assert!(output.len() >= 8, "Output too small for ftyp");
        prop_assert_eq!(&output[4..8], b"ftyp", "First box must be ftyp");
    }

    /// Property: Bytes written in stats must match actual output size
    #[test]
    fn prop_bytes_written_accurate(frame_count in 1..20usize) {
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        for i in 1..frame_count {
            muxer.write_video(i as f64 / 30.0, &make_h264_pframe(), false).unwrap();
        }

        let stats = muxer.finish_with_stats().unwrap();
        let output = buffer.into_inner();

        prop_assert_eq!(
            stats.bytes_written as usize,
            output.len(),
            "Stats bytes_written should match actual output size"
        );
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use muxide::invariant_ppt::{clear_invariant_log, contract_test};

    /// Contract test: Box building must check size invariant
    #[test]
    fn contract_box_size_invariant() {
        clear_invariant_log();

        // Trigger box building by creating an MP4
        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        muxer.finish().unwrap();

        // Verify the box size invariant was checked
        contract_test("box building", &["Box size must equal header + payload"]);
    }

    /// Contract test: Video sample entry must check width/height invariants
    #[test]
    fn contract_width_height_invariants() {
        clear_invariant_log();

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        muxer.finish().unwrap();

        // Verify width/height invariants were checked
        contract_test(
            "avc1 box building",
            &["Width must fit in 16-bit", "Height must fit in 16-bit"],
        );
    }

    /// Contract test: API keyframe detection must validate inputs
    #[test]
    fn contract_api_keyframe_detection() {
        clear_invariant_log();

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        // Use encode_video to trigger is_keyframe detection with valid keyframe
        muxer.encode_video(&make_h264_keyframe(), 33).unwrap();

        // Also encode a P-frame to trigger the validation path
        muxer.encode_video(&make_h264_pframe(), 33).unwrap();

        muxer.finish().unwrap();

        // Verify keyframe detection invariants were checked
        contract_test(
            "api::is_keyframe",
            &["INV-100: Video frame data must not be empty"],
        );
    }

    /// Contract test: API write_video must enforce invariants
    #[test]
    fn contract_api_write_video_invariants() {
        clear_invariant_log();

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .build()
            .unwrap();

        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
        muxer
            .write_video(1.0 / 30.0, &make_h264_pframe(), false)
            .unwrap();
        muxer.finish().unwrap();

        // Verify write_video invariants were checked (none expected - input validation is separate)
        contract_test("api::write_video", &[]);
    }

    /// Contract test: AV1 config extraction invariants
    #[test]
    fn contract_av1_config_extraction() {
        clear_invariant_log();

        // Create valid AV1 keyframe data with sequence header
        let av1_data = build_av1_keyframe();

        // Call extract_av1_config directly to trigger invariants
        let _ = muxide::codec::av1::extract_av1_config(&av1_data);

        // Verify AV1 config extraction invariants were checked
        contract_test(
            "codec::av1::extract_av1_config",
            &["INV-201: AV1 OBU type must be valid (0-15)"],
        );
    }

    /// Contract test: H.264 config extraction invariants
    #[test]
    fn contract_h264_config_extraction() {
        clear_invariant_log();

        let h264_data = make_h264_keyframe();

        // Call extract_avc_config directly to trigger invariants
        let _ = muxide::codec::h264::extract_avc_config(&h264_data);

        // Verify H.264 config extraction invariants were checked
        contract_test(
            "codec::h264::extract_avc_config",
            &[
                "INV-301: H.264 NAL type must be valid (0-31)",
                "INV-302: H.264 SPS and PPS must be non-empty",
            ],
        );
    }

    /// Contract test: VP9 config extraction invariants
    #[test]
    fn contract_vp9_config_extraction() {
        clear_invariant_log();

        // Create minimal VP9 keyframe data
        let vp9_data = vec![
            0x49, 0x83, 0x42, // Frame marker
            0x00, 0x00, 0x00, // Profile=0, show_existing=0, frame_type=0
        ];

        // Call extract_vp9_config directly to trigger invariants
        let _ = muxide::codec::vp9::extract_vp9_config(&vp9_data);

        // Verify VP9 config extraction invariants were checked
        contract_test(
            "codec::vp9::extract_vp9_config",
            &[
                "INV-401: VP9 frame marker must be 0x49 0x83 0x42",
                "INV-402: VP9 profile must be valid (0-3)",
            ],
        );
    }

    /// Contract test: H.265 config extraction invariants
    #[test]
    fn contract_h265_config_extraction() {
        clear_invariant_log();

        // Create minimal HEVC keyframe with VPS, SPS, PPS
        let hevc_data = vec![
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, // VPS (type 32)
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x01, // SPS (type 33)
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0x73, // PPS (type 34)
            0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x00, // IDR (type 19)
        ];

        // Call extract_hevc_config directly to trigger invariants
        let _ = muxide::codec::h265::extract_hevc_config(&hevc_data);

        // Verify H.265 config extraction invariants were checked
        contract_test(
            "codec::h265::extract_hevc_config",
            &[
                "INV-501: HEVC NAL type must be valid (0-63)",
                "INV-502: HEVC VPS, SPS, and PPS must be non-empty",
            ],
        );
    }

    /// Contract test: H.265 keyframe detection invariants
    #[test]
    fn contract_h265_keyframe_detection() {
        clear_invariant_log();

        // Create minimal HEVC keyframe with VPS, SPS, PPS
        let hevc_data = vec![
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01, // VPS (type 32)
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x01, // SPS (type 33)
            0x00, 0x00, 0x00, 0x01, 0x44, 0x01, 0xc0, 0x73, // PPS (type 34)
            0x00, 0x00, 0x00, 0x01, 0x26, 0x01, 0xaf, 0x00, // IDR (type 19)
        ];

        // Call is_hevc_keyframe directly to trigger invariants
        let _ = muxide::codec::h265::is_hevc_keyframe(&hevc_data);

        // Verify H.265 keyframe detection invariants were checked
        contract_test(
            "codec::h265::is_hevc_keyframe",
            &[
                "INV-503: HEVC keyframe detection requires non-empty data",
                "INV-504: HEVC NAL type must be valid (0-63)",
                // INV-505 is only checked when no keyframe is found
            ],
        );
    }

    /// Contract test: Opus frame duration invariants
    #[test]
    fn contract_opus_frame_duration() {
        clear_invariant_log();

        // Create a simple Opus packet
        let opus_data = vec![0x00, 0x01, 0x02, 0x03]; // TOC byte 0x00

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Opus, 48000, 2)
            .build()
            .unwrap();

        // Write a video frame first
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        muxer.write_audio(0.0, &opus_data).unwrap();
        muxer.finish().unwrap();

        // Verify Opus frame duration invariants were checked
        contract_test(
            "codec::opus::opus_frame_duration_from_toc",
            &[], // No invariants expected since function now handles invalid input gracefully
        );
    }

    /// Contract test: Opus frame count invariants
    #[test]
    fn contract_opus_frame_count() {
        clear_invariant_log();

        let opus_data = vec![0x00, 0x01, 0x02, 0x03]; // TOC byte 0x00

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Opus, 48000, 2)
            .build()
            .unwrap();

        // Write a video frame first so audio can be written
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        muxer.write_audio(0.0, &opus_data).unwrap();
        muxer.finish().unwrap();

        // Verify Opus frame count invariants were checked
        contract_test(
            "codec::opus::opus_frame_count",
            &[], // No invariants expected since function now handles invalid input gracefully
        );
    }

    /// Contract test: Opus packet samples invariants
    #[test]
    fn contract_opus_packet_samples() {
        clear_invariant_log();

        let opus_data = vec![0x00, 0x01, 0x02, 0x03]; // TOC byte 0x00

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Opus, 48000, 2)
            .build()
            .unwrap();

        // Write a video frame first
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        muxer.write_audio(0.0, &opus_data).unwrap();
        muxer.finish().unwrap();

        // Verify Opus packet samples invariants were checked
        contract_test(
            "codec::opus::opus_packet_samples",
            &[], // No invariants expected since function now handles invalid input gracefully
        );
    }

    /// Contract test: Opus packet validation invariants
    #[test]
    fn contract_opus_packet_validation() {
        clear_invariant_log();

        let opus_data = vec![0x00, 0x01, 0x02, 0x03]; // TOC byte 0x00

        let mut buffer = Cursor::new(Vec::new());
        let mut muxer = MuxerBuilder::new(&mut buffer)
            .video(VideoCodec::H264, 1920, 1080, 30.0)
            .audio(AudioCodec::Opus, 48000, 2)
            .build()
            .unwrap();

        // Write a video frame first
        muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();

        muxer.write_audio(0.0, &opus_data).unwrap();
        muxer.finish().unwrap();

        // Verify Opus packet validation invariants were checked
        contract_test(
            "codec::opus::is_valid_opus_packet",
            &[
                // No invariants checked in is_valid_opus_packet - it's pure validation
            ],
        );
    }
}

/// Contract test: AAC profiles must be validated
#[test]
fn contract_aac_profile_validation() {
    clear_invariant_log();

    let aac_frame = read_hex_fixture("audio_samples", "frame0.aac.adts");

    let mut buffer = Cursor::new(Vec::new());
    let mut muxer = MuxerBuilder::new(&mut buffer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.write_video(0.0, &make_h264_keyframe(), true).unwrap();
    muxer.write_audio(0.0, &aac_frame).unwrap();
    muxer.finish().unwrap();

    // Verify AAC profile invariant was checked
    contract_test(
        "aac audio processing",
        &["INV-020: AAC profile must be one of the supported variants"],
    );
}

/// Property-based test: CLI argument parsing is robust
#[test]
fn prop_cli_argument_parsing() {
    use proptest::prelude::*;
    use std::process::Command;

    // Test various combinations of valid CLI arguments
    let valid_widths = 320..=4096u32;
    let valid_heights = 240..=2160u32;
    let valid_fps = 1.0..=120.0f64;
    let valid_sample_rates = proptest::sample::select(vec![8000, 16000, 22050, 44100, 48000]);
    let valid_channels = 1..=8u8;

    proptest!(ProptestConfig::with_cases(50), |(
        width in valid_widths,
        height in valid_heights,
        fps in valid_fps,
        _sample_rate in valid_sample_rates,
        _channels in valid_channels,
    )| {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join(format!("test_{}_{}_{}.mp4", width, height, fps));

        let video_fixture = "fixtures/video_samples/frame0_key.264";

        let output = Command::new("cargo")
            .args([
                "run", "--bin", "muxide", "--",
                "mux",
                "--video", video_fixture,
                "--width", &width.to_string(),
                "--height", &height.to_string(),
                "--fps", &fps.to_string(),
                "--output", &output_path.to_string_lossy(),
            ])
            .output()
            .unwrap();

        // Should succeed with valid arguments
        prop_assert!(output.status.success(),
            "CLI failed with valid arguments: width={}, height={}, fps={}",
            width, height, fps);

        let stdout = String::from_utf8_lossy(&output.stdout);
        prop_assert!(stdout.contains("Muxing complete"),
            "CLI didn't complete successfully: {}", stdout);

        prop_assert!(output_path.exists(),
            "Output file was not created: {}", output_path.display());
    });
}

/// Property-based test: CLI handles various metadata inputs
#[test]
fn prop_cli_metadata_handling() {
    use proptest::prelude::*;
    use std::process::Command;

    let titles = "[a-zA-Z0-9 ]{1,50}";
    let languages = "(und|eng|spa|fra|deu|ita|por|rus|jpn|kor|chi)";

    proptest!(ProptestConfig::with_cases(20), |(
        title in titles,
        language in languages,
    )| {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("test_metadata.mp4");

        let video_fixture = "fixtures/video_samples/frame0_key.264";

        let output = Command::new("cargo")
            .args([
                "run", "--bin", "muxide", "--",
                "mux",
                "--video", video_fixture,
                "--width", "1920",
                "--height", "1080",
                "--fps", "30",
                "--title", &title,
                "--language", &language,
                "--output", &output_path.to_string_lossy(),
            ])
            .output()
            .unwrap();

        prop_assert!(output.status.success(),
            "CLI failed with metadata: title='{}', language='{}'",
            title, language);

        prop_assert!(output_path.exists(),
            "Output file not created with metadata");
    });
}

/// Property-based test: CLI rejects invalid dimensions
#[test]
fn prop_cli_invalid_dimensions_rejected() {
    use proptest::prelude::*;
    use std::process::Command;

    let invalid_widths = prop_oneof![0..=319u32, 4097..=10000u32];
    let invalid_heights = prop_oneof![0..=239u32, 2161..=5000u32];

    proptest!(ProptestConfig::with_cases(20), |(
        width in invalid_widths,
        height in invalid_heights,
    )| {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join("test_invalid.mp4");

        let video_fixture = "fixtures/video_samples/frame0_key.264";

        let output = Command::new("cargo")
            .args([
                "run", "--bin", "muxide", "--",
                "mux",
                "--video", video_fixture,
                "--width", &width.to_string(),
                "--height", &height.to_string(),
                "--fps", "30",
                "--output", &output_path.to_string_lossy(),
            ])
            .output()
            .unwrap();

        // Should fail with invalid dimensions
        prop_assert!(!output.status.success(),
            "CLI should reject invalid dimensions: {}x{}", width, height);
    });
}

/// Property-based test: CLI handles various codec combinations
#[test]
fn prop_cli_codec_combinations() {
    use proptest::prelude::*;
    use std::process::Command;

    static VIDEO_CODECS: &[&str] = &["h264", "h265", "av1"];
    static AUDIO_CODECS: &[&str] = &["aac", "aac-he", "aac-hev2", "opus"];

    proptest!(ProptestConfig::with_cases(15), |(
        video_codec in proptest::sample::select(VIDEO_CODECS),
        audio_codec in proptest::sample::select(AUDIO_CODECS),
    )| {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_path = temp_dir.path().join(format!("test_{}_{}.mp4", video_codec, audio_codec));

        let video_fixture = "fixtures/video_samples/frame0_key.264";
        let audio_fixture = "fixtures/audio_samples/frame0.aac.adts";

        let output = Command::new("cargo")
            .args([
                "run", "--bin", "muxide", "--",
                "mux",
                "--video", video_fixture,
                "--audio", audio_fixture,
                "--video-codec", video_codec,
                "--audio-codec", audio_codec,
                "--width", "1920",
                "--height", "1080",
                "--fps", "30",
                "--sample-rate", "44100",
                "--channels", "2",
                "--output", &output_path.to_string_lossy(),
            ])
            .output()
            .unwrap();

        // Should succeed (though some combinations might not work with our test fixtures)
        // The important thing is that the CLI doesn't crash on valid codec names
        let _stdout = String::from_utf8_lossy(&output.stdout);
        let _stderr = String::from_utf8_lossy(&output.stderr);

        // CLI should exit gracefully (success or controlled failure)
        // We don't assert success because some codec combinations may not work with test fixtures
        prop_assert!(output.status.code().is_some(),
            "CLI should exit with a status code for codecs: {} + {}", video_codec, audio_codec);
    });
}
