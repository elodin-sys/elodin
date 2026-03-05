mod support;

use muxide::api::{MuxerBuilder, VideoCodec};
use support::SharedBuffer;

/// Build a minimal VP9 keyframe with valid frame header.
///
/// This creates a synthetic VP9 frame with:
/// - Valid frame marker (0x49 0x83 0x42)
/// - Profile 0, keyframe, 100x100 resolution
/// - 8-bit color depth, BT.709 color space
fn build_vp9_keyframe() -> Vec<u8> {
    let mut data = Vec::new();

    // VP9 frame marker
    data.extend_from_slice(&[0x49, 0x83, 0x42]);

    // Frame header byte 0: profile=0, show_existing_frame=0, frame_type=0 (keyframe)
    data.push(0x00);

    // Frame header byte 1: show_frame=1, error_resilient_mode=0
    data.push(0x80);

    // Frame width (100) as LEB128
    // 100 = 0x64, LEB128: [0x64]
    data.push(0x64);

    // Frame height (100) as LEB128
    // 100 = 0x64, LEB128: [0x64]
    data.push(0x64);

    // Render size same as frame size (bit 2 = 0)
    // Color config: bit_depth=8, color_space=1 (BT.709), transfer_function=1, matrix_coefficients=1
    data.push(0x12); // 0001 0010: bit_depth=8, color_space=1, transfer=0, matrix=0

    // Minimal frame data (just enough to make it a valid frame)
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    data
}

/// Build a minimal VP9 P-frame (inter frame).
///
/// This creates a synthetic VP9 P-frame with:
/// - Valid frame marker
/// - Frame type = 1 (inter frame)
/// - References the previous keyframe
fn build_vp9_pframe() -> Vec<u8> {
    let mut data = Vec::new();

    // VP9 frame marker
    data.extend_from_slice(&[0x49, 0x83, 0x42]);

    // Frame header byte 0: profile=0, show_existing_frame=0, frame_type=1 (inter)
    data.push(0x10);

    // Frame header byte 1: show_frame=1, error_resilient_mode=0
    data.push(0x80);

    // Minimal frame data for P-frame
    data.extend_from_slice(&[0x00, 0x00]);

    data
}

/// Recursively search for a 4CC in an MP4 container by pattern matching
fn contains_box(data: &[u8], fourcc: &[u8; 4]) -> bool {
    data.windows(4).any(|window| window == fourcc)
}

#[test]
fn vp9_first_frame_must_be_keyframe() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Try to write a P-frame first - should fail
    let pframe = build_vp9_pframe();
    let result = muxer.write_video(0.0, &pframe, false);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("first video frame must be a keyframe"));
}

#[test]
fn vp9_keyframe_must_have_valid_config() {
    let (writer, _buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Try to write an invalid VP9 frame (wrong marker)
    let invalid_frame = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = muxer.write_video(0.0, &invalid_frame, true);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("first VP9 frame must contain sequence parameters"));
}

#[test]
fn vp9_muxer_produces_vp09_sample_entry() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Write keyframe
    let keyframe = build_vp9_keyframe();
    muxer.write_video(0.0, &keyframe, true).unwrap();

    // Write P-frame
    let pframe = build_vp9_pframe();
    muxer.write_video(1.0 / 30.0, &pframe, false).unwrap();

    // Finalize
    muxer.finish().unwrap();

    // Check output contains vp09 sample entry
    let output = buffer.lock().unwrap();
    assert!(contains_box(&output, b"vp09"));
}

#[test]
fn vp9_muxer_produces_vpcc_config_box() {
    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::Vp9, 100, 100, 30.0)
        .build()
        .unwrap();

    // Write keyframe
    let keyframe = build_vp9_keyframe();
    muxer.write_video(0.0, &keyframe, true).unwrap();

    // Finalize
    muxer.finish().unwrap();

    // Check output contains vpcC configuration box
    let output = buffer.lock().unwrap();
    assert!(contains_box(&output, b"vpcC"));
}

#[test]
fn vp9_config_extraction_works() {
    use muxide::codec::vp9::extract_vp9_config;

    let keyframe = build_vp9_keyframe();
    let config = extract_vp9_config(&keyframe).unwrap();

    assert_eq!(config.width, 100);
    assert_eq!(config.height, 100);
    assert_eq!(config.profile, 0);
    assert_eq!(config.bit_depth, 8);
    assert_eq!(config.color_space, 1); // BT.709
}

#[test]
fn vp9_keyframe_detection_works() {
    use muxide::codec::vp9::is_vp9_keyframe;

    let keyframe = build_vp9_keyframe();
    assert!(is_vp9_keyframe(&keyframe).unwrap());

    let pframe = build_vp9_pframe();
    assert!(!is_vp9_keyframe(&pframe).unwrap());
}
