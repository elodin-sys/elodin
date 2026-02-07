mod support;

use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, VideoCodec};
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

#[test]
fn finish_in_place_errors_on_double_finish_and_blocks_writes(
) -> Result<(), Box<dyn std::error::Error>> {
    let frame0 = read_hex_fixture("video_samples", "frame0_key.264");

    let (writer, buffer) = SharedBuffer::new();
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    muxer.finish_in_place()?;

    assert!(muxer.finish_in_place().is_err());
    assert!(muxer.write_video(0.033, &frame0, false).is_err());

    drop(muxer);
    assert!(!buffer.lock().unwrap().is_empty());
    Ok(())
}

#[test]
fn finish_is_deterministic_for_same_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let v0 = read_hex_fixture("video_samples", "frame0_key.264");
    let v1 = read_hex_fixture("video_samples", "frame1_p.264");
    let v2 = read_hex_fixture("video_samples", "frame2_p.264");

    let a0 = read_hex_fixture("audio_samples", "frame0.aac.adts");
    let a1 = read_hex_fixture("audio_samples", "frame1.aac.adts");
    let a2 = read_hex_fixture("audio_samples", "frame2.aac.adts");

    let (w1, b1) = SharedBuffer::new();
    let mut m1 = MuxerBuilder::new(w1)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build()?;

    m1.write_video(0.0, &v0, true)?;
    m1.write_audio(0.0, &a0)?;
    m1.write_audio(0.021, &a1)?;
    m1.write_video(0.033, &v1, false)?;
    m1.write_audio(0.042, &a2)?;
    m1.write_video(0.066, &v2, false)?;
    m1.finish()?;

    let (w2, b2) = SharedBuffer::new();
    let mut m2 = MuxerBuilder::new(w2)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build()?;

    m2.write_video(0.0, &v0, true)?;
    m2.write_audio(0.0, &a0)?;
    m2.write_audio(0.021, &a1)?;
    m2.write_video(0.033, &v1, false)?;
    m2.write_audio(0.042, &a2)?;
    m2.write_video(0.066, &v2, false)?;
    m2.finish()?;

    assert_eq!(*b1.lock().unwrap(), *b2.lock().unwrap());
    Ok(())
}
