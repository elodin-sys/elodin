use muxide::api::{MuxerBuilder, MuxerConfig, MuxerStats, VideoCodec};
use std::{env, fs::File, io::Write, path::PathBuf};

fn read_hex_bytes(contents: &str) -> Vec<u8> {
    let hex: String = contents.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(hex.len() % 2 == 0, "hex must have even length");

    let mut out = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16).expect("valid hex");
        out.push(byte);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Writes a tiny MP4 using the repository's test fixtures.
    // Usage: cargo run --example write_fixture_video -- out.mp4

    let out_path: PathBuf = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out.mp4"));

    let frame0 = read_hex_bytes(include_str!("../fixtures/video_samples/frame0_key.264"));
    let frame1 = read_hex_bytes(include_str!("../fixtures/video_samples/frame1_p.264"));
    let frame2 = read_hex_bytes(include_str!("../fixtures/video_samples/frame2_p.264"));

    let file = File::create(&out_path)?;
    let config = MuxerConfig::new(640, 480, 30.0);
    let mut muxer = MuxerBuilder::new(file)
        .video(
            VideoCodec::H264,
            config.width,
            config.height,
            config.framerate,
        )
        .build()?;

    muxer.write_video(0.0, &frame0, true)?;
    muxer.write_video(1.0 / 30.0, &frame1, false)?;
    muxer.write_video(2.0 / 30.0, &frame2, false)?;

    let stats: MuxerStats = muxer.finish_with_stats()?;

    let mut stderr = std::io::stderr();
    writeln!(
        &mut stderr,
        "wrote {} video frames, {:.3}s, {} bytes -> {}",
        stats.video_frames,
        stats.duration_secs,
        stats.bytes_written,
        out_path.display()
    )?;

    Ok(())
}
