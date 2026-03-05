mod support;

use muxide::api::{MuxerBuilder, MuxerConfig, MuxerStats, VideoCodec};
use std::{fs, path::Path};
use support::SharedBuffer;

fn read_hex_fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("video_samples")
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

#[test]
fn finish_with_stats_reports_frames_duration_and_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("frame0_key.264");
    let frame1 = read_hex_fixture("frame1_p.264");
    let frame2 = read_hex_fixture("frame2_p.264");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    muxer.write_video(1.0 / 30.0, &frame1, false)?;
    muxer.write_video(2.0 / 30.0, &frame2, false)?;

    let stats: MuxerStats = muxer.finish_with_stats()?;

    let produced_len = buffer.lock().unwrap().len() as u64;
    assert_eq!(stats.video_frames, 3);
    assert_eq!(stats.bytes_written, produced_len);

    // 3 frames at 30fps => end time is (2/30 + 1/30) = 0.1s.
    let expected = 0.1_f64;
    assert!((stats.duration_secs - expected).abs() < 1e-9);

    Ok(())
}

#[test]
fn muxer_builder_creates_muxer_for_video_only() -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("frame0_key.264");

    let (writer, buffer) = SharedBuffer::new();
    let config = MuxerConfig::new(640, 480, 30.0);
    let mut muxer = MuxerBuilder::new(writer)
        .video(
            VideoCodec::H264,
            config.width,
            config.height,
            config.framerate,
        )
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    let _stats = muxer.finish_with_stats()?;

    assert!(!buffer.lock().unwrap().is_empty());
    Ok(())
}
