//! Tests for fragmented MP4 muxing

use muxide::codec::vp9::Vp9Config;
use muxide::fragmented::{FragmentConfig, FragmentedError, FragmentedMuxer};

#[test]
fn test_fragmented_dts_must_be_monotonic() {
    let config = FragmentConfig {
        width: 1920,
        height: 1080,
        timescale: 90000,
        fragment_duration_ms: 2000,
        sps: vec![0x00, 0x00, 0x00, 0x01, 0x67], // Fake SPS
        pps: vec![0x00, 0x00, 0x00, 0x01, 0x68], // Fake PPS
        vps: None,
        av1_sequence_header: None,
        vp9_config: None,
    };

    let mut muxer = FragmentedMuxer::new(config);

    // First sample OK
    let data = vec![0x00, 0x00, 0x00, 0x04, 0x65, 0x01, 0x02, 0x03];
    assert!(muxer.write_video(0, 0, &data, true).is_ok());

    // Same DTS OK
    assert!(muxer.write_video(3000, 0, &data, false).is_ok());

    // Increasing DTS OK
    assert!(muxer.write_video(6000, 3000, &data, false).is_ok());

    // Decreasing DTS should fail
    let result = muxer.write_video(9000, 1000, &data, false);
    assert!(matches!(
        result,
        Err(FragmentedError::NonMonotonicDts {
            prev_dts: 3000,
            curr_dts: 1000
        })
    ));
}

#[test]
fn test_fragmented_basic_functionality() {
    let config = FragmentConfig {
        width: 1920,
        height: 1080,
        timescale: 90000,
        fragment_duration_ms: 2000,
        sps: vec![0x00, 0x00, 0x00, 0x01, 0x67], // Fake SPS
        pps: vec![0x00, 0x00, 0x00, 0x01, 0x68], // Fake PPS
        vps: None,
        av1_sequence_header: None,
        vp9_config: None,
    };

    let mut muxer = FragmentedMuxer::new(config);

    // Get init segment
    let init = muxer.init_segment();
    assert!(!init.is_empty());
    assert!(init.windows(4).any(|w| w == b"ftyp"));
    assert!(init.windows(4).any(|w| w == b"moov"));

    // Add samples
    let data = vec![0x00, 0x00, 0x00, 0x04, 0x65, 0x01, 0x02, 0x03];
    muxer.write_video(0, 0, &data, true).unwrap();
    muxer.write_video(3000, 3000, &data, false).unwrap();

    // Not ready to flush yet (duration < 2000ms)
    assert!(!muxer.ready_to_flush());

    // Add more samples to reach duration (2000ms at 90000 timescale)
    // 2000ms * 90000 / 1000 = 180000 ticks
    // At 3000 ticks per frame, need about 60 frames
    for i in 2..65 {
        muxer.write_video(i * 3000, i * 3000, &data, false).unwrap();
    }

    // Now ready
    assert!(muxer.ready_to_flush());

    // Flush segment
    let segment = muxer.flush_segment().unwrap();
    assert!(!segment.is_empty());
    assert!(segment.windows(4).any(|w| w == b"moof"));
    assert!(segment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn test_fragmented_h265_basic() {
    let config = FragmentConfig {
        width: 1920,
        height: 1080,
        timescale: 90000,
        fragment_duration_ms: 2000,
        sps: vec![
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00,
        ], // Fake H.265 SPS
        pps: vec![0x44, 0x01, 0xc0, 0x73, 0xc0, 0x4c, 0x90], // Fake H.265 PPS
        vps: Some(vec![0x40, 0x01, 0x0c, 0x01, 0xff, 0xff, 0x01, 0x60, 0x00]), // Fake H.265 VPS
        av1_sequence_header: None,
        vp9_config: None,
    };

    let mut muxer = FragmentedMuxer::new(config);

    // Get init segment - should contain hvc1 sample entry
    let init = muxer.init_segment();
    assert!(!init.is_empty());
    assert!(init.windows(4).any(|w| w == b"ftyp"));
    assert!(init.windows(4).any(|w| w == b"moov"));
    // Should contain hvc1 box for H.265
    assert!(init.windows(4).any(|w| w == b"hvc1"));

    // Add samples
    let data = vec![0x00, 0x00, 0x00, 0x04, 0x26, 0x01, 0xaf, 0x06]; // Fake H.265 IDR frame
    muxer.write_video(0, 0, &data, true).unwrap();
    muxer.write_video(3000, 3000, &data, false).unwrap();

    // Flush segment
    let segment = muxer.flush_segment().unwrap();
    assert!(!segment.is_empty());
    assert!(segment.windows(4).any(|w| w == b"moof"));
    assert!(segment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn test_fragmented_av1_basic() {
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

    let config = FragmentConfig {
        width: 1920,
        height: 1080,
        timescale: 90000,
        fragment_duration_ms: 2000,
        sps: vec![], // Not used for AV1
        pps: vec![], // Not used for AV1
        vps: None,   // Not used for AV1
        av1_sequence_header: Some(seq_header),
        vp9_config: None,
    };

    let mut muxer = FragmentedMuxer::new(config);

    // Get init segment - should contain av01 sample entry
    let init = muxer.init_segment();
    assert!(!init.is_empty());
    assert!(init.windows(4).any(|w| w == b"ftyp"));
    assert!(init.windows(4).any(|w| w == b"moov"));
    // Should contain av01 box for AV1
    assert!(init.windows(4).any(|w| w == b"av01"));

    // Add samples
    let data = vec![0x12, 0x00, 0x32, 0x02, 0x00, 0x00]; // Fake AV1 keyframe (TD + Frame OBU)
    muxer.write_video(0, 0, &data, true).unwrap();
    muxer.write_video(3000, 3000, &data, false).unwrap();

    // Flush segment
    let segment = muxer.flush_segment().unwrap();
    assert!(!segment.is_empty());
    assert!(segment.windows(4).any(|w| w == b"moof"));
    assert!(segment.windows(4).any(|w| w == b"mdat"));
}

#[test]
fn test_fragmented_vp9_basic() {
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

    let config = FragmentConfig {
        width: 1920,
        height: 1080,
        timescale: 90000,
        fragment_duration_ms: 2000,
        sps: vec![], // Not used for VP9
        pps: vec![], // Not used for VP9
        vps: None,   // Not used for VP9
        av1_sequence_header: None,
        vp9_config: Some(vp9_config),
    };

    let mut muxer = FragmentedMuxer::new(config);

    // Get init segment - should contain vp09 sample entry
    let init = muxer.init_segment();
    assert!(!init.is_empty());
    assert!(init.windows(4).any(|w| w == b"ftyp"));
    assert!(init.windows(4).any(|w| w == b"moov"));
    // Should contain vp09 box for VP9
    assert!(init.windows(4).any(|w| w == b"vp09"));

    // Add samples
    let data = vec![0x49, 0x83, 0x42, 0x00, 0x00, 0x00]; // Fake VP9 keyframe (frame marker + minimal header)
    muxer.write_video(0, 0, &data, true).unwrap();
    muxer.write_video(3000, 3000, &data, false).unwrap();

    // Flush segment
    let segment = muxer.flush_segment().unwrap();
    assert!(!segment.is_empty());
    assert!(segment.windows(4).any(|w| w == b"moof"));
    assert!(segment.windows(4).any(|w| w == b"mdat"));
}
