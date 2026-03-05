//! Integration tests for AV1 muxing.
//!
//! These tests verify that AV1 video produces:
//! - av01 sample entry box
//! - av1C configuration box with Sequence Header OBU

mod support;

use muxide::api::{MuxerBuilder, VideoCodec};
use support::SharedBuffer;

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

/// Build an AV1 frame without Sequence Header (for error testing).
fn build_av1_frame_no_seq_header() -> Vec<u8> {
    let mut data = Vec::new();

    // Only a Frame OBU, no Sequence Header
    // OBU header: type=6, has_size=1
    data.push(0x32);
    data.push(4);
    data.extend_from_slice(&[0x10, 0x00, 0x00, 0x00]);

    data
}

/// Recursively search for a 4CC in an MP4 container by pattern matching
fn contains_box(data: &[u8], fourcc: &[u8; 4]) -> bool {
    data.windows(4).any(|window| window == fourcc)
}

#[test]
fn av1_muxer_produces_av01_sample_entry() {
    let (writer, buffer) = SharedBuffer::new();

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Av1, 1920, 1080, 30.0)
        .build()
        .expect("build should succeed");

    // Write a keyframe with Sequence Header
    let keyframe = build_av1_keyframe();
    muxer
        .write_video(0.0, &keyframe, true)
        .expect("write_video should succeed");

    muxer.finish().expect("finish should succeed");

    let produced = buffer.lock().unwrap();

    // Find the av01 box
    assert!(
        contains_box(&produced, b"av01"),
        "Output should contain av01 sample entry"
    );

    // Find the av1C box
    assert!(
        contains_box(&produced, b"av1C"),
        "Output should contain av1C configuration box"
    );
}

#[test]
fn av1_first_frame_must_be_keyframe() {
    let (writer, _buffer) = SharedBuffer::new();

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Av1, 1920, 1080, 30.0)
        .build()
        .expect("build should succeed");

    // Build a frame without marking it as keyframe
    let frame = build_av1_keyframe();

    // First frame as non-keyframe should fail
    let result = muxer.write_video(0.0, &frame, false);
    assert!(result.is_err(), "first frame must be keyframe");

    let err = result.unwrap_err();
    assert!(
        format!("{}", err).contains("keyframe"),
        "error message should mention keyframe: {}",
        err
    );
}

#[test]
fn av1_keyframe_must_have_sequence_header() {
    let (writer, _buffer) = SharedBuffer::new();

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Av1, 1920, 1080, 30.0)
        .build()
        .expect("build should succeed");

    // Keyframe without Sequence Header should fail
    let frame_no_seq = build_av1_frame_no_seq_header();
    let result = muxer.write_video(0.0, &frame_no_seq, true);
    assert!(
        result.is_err(),
        "keyframe without Sequence Header should fail"
    );

    let err = result.unwrap_err();
    assert!(
        format!("{}", err).contains("Sequence Header"),
        "error message should mention Sequence Header: {}",
        err
    );
}
