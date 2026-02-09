//! Test the new API methods

use muxide::api::{MuxerBuilder, VideoCodec};
use muxide::codec::h264::default_avc_config;
use muxide::codec::vp9::Vp9Config;
use muxide::fragmented::FragmentedMuxer;

#[test]
fn test_new_with_fragment_creates_fragmented_muxer() {
    let writer: Vec<u8> = Vec::new();
    let config = default_avc_config();
    let result = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .with_sps(config.sps)
        .with_pps(config.pps)
        .new_with_fragment();

    assert!(result.is_ok());
    let mut muxer: FragmentedMuxer = result.unwrap();

    // Verify init segment is available
    let init_segment = muxer.init_segment();
    assert!(!init_segment.is_empty());
    // Check that it contains "ftyp" box (may not be at the start due to box structure)
    assert!(init_segment.windows(4).any(|w| w == b"ftyp"));
}

#[test]
fn test_new_with_fragment_h265_requires_vps_sps_pps() {
    let writer: Vec<u8> = Vec::new();

    // Test H.265 requires VPS
    let result = MuxerBuilder::new(writer.clone())
        .video(VideoCodec::H265, 1920, 1080, 30.0)
        .with_sps(vec![0x42, 0x01, 0x01])
        .with_pps(vec![0x44, 0x01, 0xc0])
        .new_with_fragment();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("VPS must be provided"));

    // Test H.265 requires SPS
    let result = MuxerBuilder::new(writer.clone())
        .video(VideoCodec::H265, 1920, 1080, 30.0)
        .with_vps(vec![0x40, 0x01, 0x0c])
        .with_pps(vec![0x44, 0x01, 0xc0])
        .new_with_fragment();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("SPS must be provided"));

    // Test H.265 requires PPS
    let result = MuxerBuilder::new(writer.clone())
        .video(VideoCodec::H265, 1920, 1080, 30.0)
        .with_vps(vec![0x40, 0x01, 0x0c])
        .with_sps(vec![0x42, 0x01, 0x01])
        .new_with_fragment();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("PPS must be provided"));
}

#[test]
fn test_new_with_fragment_vp9_requires_config() {
    let writer: Vec<u8> = Vec::new();

    // Test VP9 requires config
    let result = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 1920, 1080, 30.0)
        .new_with_fragment();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("VP9 config must be provided"));
}

#[test]
fn test_new_with_fragment_h265_success() {
    let writer: Vec<u8> = Vec::new();
    let result = MuxerBuilder::new(writer)
        .video(VideoCodec::H265, 1920, 1080, 30.0)
        .with_vps(vec![0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00])
        .with_sps(vec![
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00,
        ])
        .with_pps(vec![0x44, 0x01, 0xc0, 0x73, 0xc0, 0x4c, 0x90])
        .new_with_fragment();

    assert!(result.is_ok());
    let mut muxer: FragmentedMuxer = result.unwrap();

    // Verify init segment contains hvc1
    let init = muxer.init_segment();
    assert!(init.windows(4).any(|w| w == b"hvc1"));
}

#[test]
fn test_new_with_fragment_av1_success() {
    let writer: Vec<u8> = Vec::new();
    // Minimal AV1 sequence header OBU from the codec tests
    let seq_header = vec![
        0x0A, 0x10, // OBU header + size
        0x00, 0x00, 0x00,
        0x00, // seq_profile=0, still_picture=0, reduced_still_picture_header=0
        0x00, 0x00, 0x00, 0x00, // timing_info_present=0, decoder_model_info_present=0
        0x00, 0x00, 0x00,
        0x00, // initial_display_delay_present=0, operating_points_cnt_minus_1=0
        0x00, 0x00, // operating_point_idc[0]=0, seq_level_idx[0]=0
        0x00, // seq_tier[0]=0
    ];

    let result = MuxerBuilder::new(writer)
        .video(VideoCodec::Av1, 1920, 1080, 30.0)
        .with_av1_sequence_header(seq_header)
        .new_with_fragment();

    assert!(result.is_ok());
    let mut muxer: FragmentedMuxer = result.unwrap();

    // Verify init segment contains av01
    let init = muxer.init_segment();
    assert!(init.windows(4).any(|w| w == b"av01"));
}

#[test]
fn test_new_with_fragment_vp9_success() {
    let writer: Vec<u8> = Vec::new();
    let vp9_config = Vp9Config {
        width: 1920,
        height: 1080,
        profile: 0,
        bit_depth: 8,
        color_space: 0,
        transfer_function: 0,
        matrix_coefficients: 0,
        level: 0,
        full_range_flag: 0,
    };

    let result = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 1920, 1080, 30.0)
        .with_vp9_config(vp9_config)
        .new_with_fragment();

    assert!(result.is_ok());
    let mut muxer: FragmentedMuxer = result.unwrap();

    // Verify init segment contains vp09
    let init = muxer.init_segment();
    assert!(init.windows(4).any(|w| w == b"vp09"));
}

#[test]
fn test_flush_alias_for_finish() {
    let writer: Vec<u8> = Vec::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .build()
        .unwrap();

    // Write a minimal video frame
    let sps_pps = vec![
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0A, 0xF8, 0x41, 0xA2, // SPS
        0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80, // PPS
    ];
    let keyframe = vec![
        0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x20, // IDR slice
    ];
    let mut frame_data = sps_pps;
    frame_data.extend(keyframe);

    muxer.write_video(0.0, &frame_data, true).unwrap();

    // Test that flush() works as an alias for finish()
    let result = muxer.flush();
    assert!(result.is_ok());
}
