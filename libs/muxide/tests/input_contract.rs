//! Slice 1: Input contract enforcement tests
//!
//! These tests verify that Muxide rejects invalid input with descriptive errors
//! rather than producing corrupt output or panicking.

mod support;

use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, MuxerError, VideoCodec};
use std::{fs, path::Path};
use support::SharedBuffer;

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

/// Valid ADTS frame for testing (7 byte header + minimal payload)
fn valid_adts_frame() -> Vec<u8> {
    vec![0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb]
}

// =============================================================================
// Video PTS contract tests
// =============================================================================

#[test]
fn video_pts_negative_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    let err = muxer.write_video(-0.001, &frame, true).unwrap_err();

    assert!(
        matches!(err, MuxerError::NegativeVideoPts { pts, frame_index } 
        if pts < 0.0 && frame_index == 0)
    );

    let msg = err.to_string();
    assert!(
        msg.contains("-0.001") || msg.contains("negative"),
        "Error should mention the negative value: {}",
        msg
    );
    assert!(
        msg.contains("frame 0"),
        "Error should include frame index: {}",
        msg
    );
}

#[test]
fn video_pts_nan_inf_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");

    // Test NaN
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    let err = muxer.write_video(f64::NAN, &frame, true).unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidVideoPts { pts, frame_index }
        if pts.is_nan() && frame_index == 0)
    );

    // Test positive infinity
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    let err = muxer.write_video(f64::INFINITY, &frame, true).unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidVideoPts { pts, frame_index }
        if pts.is_infinite() && pts > 0.0 && frame_index == 0)
    );

    // Test negative infinity
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    let err = muxer
        .write_video(f64::NEG_INFINITY, &frame, true)
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidVideoPts { pts, frame_index }
        if pts.is_infinite() && pts < 0.0 && frame_index == 0)
    );
}

#[test]
fn video_pts_non_increasing_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    muxer.write_video(0.0, &frame, true).unwrap();
    muxer.write_video(0.033, &frame, false).unwrap();

    // Same timestamp as previous
    let err = muxer.write_video(0.033, &frame, false).unwrap_err();

    assert!(
        matches!(err, MuxerError::NonIncreasingVideoPts { prev_pts, curr_pts, frame_index }
        if (prev_pts - 0.033).abs() < 0.001 && (curr_pts - 0.033).abs() < 0.001 && frame_index == 2)
    );

    let msg = err.to_string();
    assert!(
        msg.contains("frame 2"),
        "Error should include frame index: {}",
        msg
    );
    assert!(
        msg.contains("increase") || msg.contains("greater"),
        "Error should explain timestamps must increase: {}",
        msg
    );
}

#[test]
fn video_pts_decreasing_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    muxer.write_video(0.0, &frame, true).unwrap();
    muxer.write_video(0.066, &frame, false).unwrap();

    // Decreasing timestamp
    let err = muxer.write_video(0.033, &frame, false).unwrap_err();

    assert!(matches!(err, MuxerError::NonIncreasingVideoPts { .. }));

    let msg = err.to_string();
    assert!(
        msg.contains("0.033") || msg.contains("0.066"),
        "Error should show the timestamp values: {}",
        msg
    );
}

// =============================================================================
// DTS contract tests (B-frames)
// =============================================================================

#[test]
fn dts_non_increasing_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    // I-frame at DTS=0
    muxer.write_video_with_dts(0.0, 0.0, &frame, true).unwrap();
    // P-frame at DTS=0.033
    muxer
        .write_video_with_dts(0.1, 0.033, &frame, false)
        .unwrap();

    // Try to write with DTS <= previous DTS
    let err = muxer
        .write_video_with_dts(0.066, 0.033, &frame, false)
        .unwrap_err();

    assert!(matches!(err, MuxerError::NonIncreasingDts { .. }));

    let msg = err.to_string();
    assert!(msg.contains("DTS"), "Error should mention DTS: {}", msg);
    assert!(
        msg.contains("increase"),
        "Error should explain DTS must increase: {}",
        msg
    );
}

#[test]
fn dts_negative_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    let err = muxer
        .write_video_with_dts(0.0, -0.001, &frame, true)
        .unwrap_err();

    // Negative DTS treated as negative DTS error
    assert!(matches!(err, MuxerError::NegativeVideoDts { .. }));
}

#[test]
fn dts_nan_inf_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");

    // Test NaN
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    let err = muxer
        .write_video_with_dts(0.0, f64::NAN, &frame, true)
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidVideoDts { dts, frame_index }
        if dts.is_nan() && frame_index == 0)
    );

    // Test positive infinity
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    let err = muxer
        .write_video_with_dts(0.0, f64::INFINITY, &frame, true)
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidVideoDts { dts, frame_index }
        if dts.is_infinite() && dts > 0.0 && frame_index == 0)
    );

    // Test negative infinity
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();
    let err = muxer
        .write_video_with_dts(0.0, f64::NEG_INFINITY, &frame, true)
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidVideoDts { dts, frame_index }
        if dts.is_infinite() && dts < 0.0 && frame_index == 0)
    );
}

// =============================================================================
// Audio contract tests
// =============================================================================

#[test]
fn audio_pts_negative_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.write_video(0.0, &frame, true).unwrap();

    let err = muxer.write_audio(-0.001, &valid_adts_frame()).unwrap_err();

    assert!(
        matches!(err, MuxerError::NegativeAudioPts { pts, frame_index }
        if pts < 0.0 && frame_index == 0)
    );
}

#[test]
fn audio_pts_nan_inf_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");

    // Test NaN
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();
    muxer.write_video(0.0, &frame, true).unwrap();
    let err = muxer
        .write_audio(f64::NAN, &valid_adts_frame())
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidAudioPts { pts, frame_index }
        if pts.is_nan() && frame_index == 0)
    );

    // Test positive infinity
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();
    muxer.write_video(0.0, &frame, true).unwrap();
    let err = muxer
        .write_audio(f64::INFINITY, &valid_adts_frame())
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidAudioPts { pts, frame_index }
        if pts.is_infinite() && pts > 0.0 && frame_index == 0)
    );

    // Test negative infinity
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();
    muxer.write_video(0.0, &frame, true).unwrap();
    let err = muxer
        .write_audio(f64::NEG_INFINITY, &valid_adts_frame())
        .unwrap_err();
    assert!(
        matches!(err, MuxerError::InvalidAudioPts { pts, frame_index }
        if pts.is_infinite() && pts < 0.0 && frame_index == 0)
    );
}

#[test]
fn audio_pts_decreasing_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.write_video(0.0, &frame, true).unwrap();
    muxer.write_audio(0.0, &valid_adts_frame()).unwrap();
    muxer.write_audio(0.023, &valid_adts_frame()).unwrap();

    // Decreasing audio timestamp
    let err = muxer.write_audio(0.010, &valid_adts_frame()).unwrap_err();

    assert!(
        matches!(err, MuxerError::DecreasingAudioPts { prev_pts, curr_pts, frame_index }
        if (prev_pts - 0.023).abs() < 0.001 && (curr_pts - 0.010).abs() < 0.001 && frame_index == 2)
    );
}

#[test]
fn audio_before_first_video_is_rejected() {
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    // No video written yet
    let err = muxer.write_audio(0.0, &valid_adts_frame()).unwrap_err();

    assert!(matches!(
        err,
        MuxerError::AudioBeforeFirstVideo {
            first_video_pts: None,
            ..
        }
    ));

    let msg = err.to_string();
    assert!(msg.contains("video"), "Error should mention video: {}", msg);
}

#[test]
fn audio_pts_before_first_video_pts_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    // Video starts at 1.0 second
    muxer.write_video(1.0, &frame, true).unwrap();

    // Audio at 0.5 seconds (before video)
    let err = muxer.write_audio(0.5, &valid_adts_frame()).unwrap_err();

    assert!(matches!(
        err,
        MuxerError::AudioBeforeFirstVideo {
            audio_pts,
            first_video_pts: Some(video_pts)
        } if (audio_pts - 0.5).abs() < 0.001 && (video_pts - 1.0).abs() < 0.001
    ));
}

#[test]
fn audio_empty_frame_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.write_video(0.0, &frame, true).unwrap();

    let err = muxer.write_audio(0.0, &[]).unwrap_err();

    assert!(matches!(
        err,
        MuxerError::EmptyAudioFrame { frame_index: 0 }
    ));
}

#[test]
fn audio_invalid_adts_is_rejected() {
    let frame = read_hex_fixture("video_samples", "frame0_key.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48000, 2)
        .build()
        .unwrap();

    muxer.write_video(0.0, &frame, true).unwrap();

    // Invalid data (doesn't start with 0xFFF sync word)
    let err = muxer
        .write_audio(0.0, &[0x00, 0x01, 0x02, 0x03])
        .unwrap_err();

    assert!(matches!(
        err,
        MuxerError::InvalidAdtsDetailed { frame_index: 0, .. }
    ));

    let msg = err.to_string();
    assert!(
        msg.contains("ADTS") || msg.contains("sync"),
        "Error should mention ADTS format: {}",
        msg
    );
}

// =============================================================================
// First frame contract tests
// =============================================================================

#[test]
fn first_video_frame_must_be_keyframe() {
    let p_frame = read_hex_fixture("video_samples", "frame1_p.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    let err = muxer.write_video(0.0, &p_frame, false).unwrap_err();

    assert!(matches!(err, MuxerError::FirstVideoFrameMustBeKeyframe));

    let msg = err.to_string();
    assert!(
        msg.contains("keyframe") || msg.contains("IDR"),
        "Error should explain first frame must be keyframe: {}",
        msg
    );
}

#[test]
fn first_keyframe_must_contain_sps_pps() {
    // A frame marked as keyframe but without SPS/PPS
    let p_frame = read_hex_fixture("video_samples", "frame1_p.264");
    let (writer, _) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()
        .unwrap();

    let err = muxer.write_video(0.0, &p_frame, true).unwrap_err();

    assert!(matches!(err, MuxerError::FirstVideoFrameMissingSpsPps));

    let msg = err.to_string();
    assert!(
        msg.contains("SPS") && msg.contains("PPS"),
        "Error should mention SPS and PPS: {}",
        msg
    );
}

// =============================================================================
// Error message quality tests
// =============================================================================

#[test]
fn error_messages_are_educational() {
    // All error messages should contain guidance on how to fix the issue
    let frame = read_hex_fixture("video_samples", "frame0_key.264");

    // Test NonIncreasingVideoPts suggests write_video_with_dts for B-frames
    {
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .build()
            .unwrap();
        muxer.write_video(0.0, &frame, true).unwrap();
        let err = muxer.write_video(0.0, &frame, false).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("write_video_with_dts"),
            "NonIncreasingVideoPts should suggest write_video_with_dts: {}",
            msg
        );
    }

    // Test FirstVideoFrameMissingSpsPps explains what to do
    {
        let p_frame = read_hex_fixture("video_samples", "frame1_p.264");
        let (writer, _) = SharedBuffer::new();
        let mut muxer = MuxerBuilder::new(writer)
            .video(VideoCodec::H264, 640, 480, 30.0)
            .build()
            .unwrap();
        let err = muxer.write_video(0.0, &p_frame, true).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("prepend") || msg.contains("NAL type"),
            "FirstVideoFrameMissingSpsPps should explain how to fix: {}",
            msg
        );
    }
}
