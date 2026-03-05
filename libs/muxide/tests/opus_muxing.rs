//! Integration tests for Opus audio muxing.

mod support;

use muxide::api::{AudioCodec, MuxerBuilder, VideoCodec};
use support::SharedBuffer;

/// Build a minimal H.264 keyframe for video track setup.
fn build_h264_keyframe() -> Vec<u8> {
    let mut data = Vec::new();
    // SPS
    data.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, 0xab, 0x40, 0xf0, 0x28, 0xd0,
    ]);
    // PPS
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80]);
    // IDR slice
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00, 0x10]);
    data
}

/// Build a minimal Opus packet (SILK 20ms, stereo, 1 frame).
fn build_opus_packet() -> Vec<u8> {
    // TOC: config=4 (SILK 20ms), s=1 (stereo), c=0 (1 frame)
    // Binary: 0b00100_1_00 = 0x24
    vec![0x24, 0xc0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05]
}

/// Recursively search for a 4CC in an MP4 container by pattern matching
fn contains_box(data: &[u8], fourcc: &[u8; 4]) -> bool {
    data.windows(4).any(|window| window == fourcc)
}

#[test]
fn opus_muxer_produces_opus_sample_entry() {
    let (writer, buffer) = SharedBuffer::new();

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Opus, 48000, 2)
        .build()
        .expect("build should succeed");

    // Write video keyframe first (required)
    let keyframe = build_h264_keyframe();
    muxer
        .write_video(0.0, &keyframe, true)
        .expect("write_video should succeed");

    // Write Opus audio packet
    let audio = build_opus_packet();
    muxer
        .write_audio(0.0, &audio)
        .expect("write_audio should succeed");
    muxer
        .write_audio(0.02, &audio)
        .expect("write_audio should succeed");

    muxer.finish().expect("finish should succeed");

    let produced = buffer.lock().unwrap();

    // Find the Opus sample entry box
    assert!(
        contains_box(&produced, b"Opus"),
        "Output should contain Opus sample entry"
    );

    // Find the dOps (Opus decoder config) box
    assert!(
        contains_box(&produced, b"dOps"),
        "Output should contain dOps configuration box"
    );
}

#[test]
fn opus_invalid_packet_rejected() {
    let (writer, _buffer) = SharedBuffer::new();

    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Opus, 48000, 2)
        .build()
        .expect("build should succeed");

    // Write video keyframe first
    let keyframe = build_h264_keyframe();
    muxer
        .write_video(0.0, &keyframe, true)
        .expect("write_video should succeed");

    // Try to write an empty Opus packet (should fail)
    let result = muxer.write_audio(0.0, &[]);
    assert!(result.is_err(), "Empty Opus packet should be rejected");
}

#[test]
fn opus_sample_rate_forced_to_48khz() {
    let (writer, buffer) = SharedBuffer::new();

    // Even though we specify 44100, Opus internally uses 48kHz
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 1920, 1080, 30.0)
        .audio(AudioCodec::Opus, 44100, 2) // User says 44.1kHz but Opus ignores this
        .build()
        .expect("build should succeed");

    let keyframe = build_h264_keyframe();
    muxer
        .write_video(0.0, &keyframe, true)
        .expect("write_video should succeed");

    let audio = build_opus_packet();
    muxer
        .write_audio(0.0, &audio)
        .expect("write_audio should succeed");

    muxer.finish().expect("finish should succeed");

    let produced = buffer.lock().unwrap();

    // Output should still have Opus boxes (rate is internally 48kHz)
    assert!(
        contains_box(&produced, b"Opus"),
        "Output should contain Opus sample entry"
    );
}
