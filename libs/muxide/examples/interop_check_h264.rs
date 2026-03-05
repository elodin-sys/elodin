use muxide::api::{MuxerBuilder, MuxerConfig, VideoCodec};
use muxide::codec::h264::{annexb_to_avcc, extract_avc_config};
use muxide::fragmented::{FragmentConfig, FragmentedMuxer};
use std::{
    env,
    fs::{self, File},
    io::{self, Write},
    path::PathBuf,
};

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
    // Interop sanity check against ffmpeg/ffprobe.
    // Usage: cargo run --example interop_check_h264 -- <out_dir>

    let out_dir: PathBuf = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/interop"));

    fs::create_dir_all(&out_dir)?;

    let frame0 = read_hex_bytes(include_str!("../fixtures/video_samples/frame0_key.264"));
    let frame1 = read_hex_bytes(include_str!("../fixtures/video_samples/frame1_p.264"));
    let frame2 = read_hex_bytes(include_str!("../fixtures/video_samples/frame2_p.264"));

    // Write an elementary Annex B stream for ffmpeg reference muxing.
    let stream_h264_path = out_dir.join("stream.h264");
    {
        let mut f = File::create(&stream_h264_path)?;
        f.write_all(&frame0)?;
        f.write_all(&frame1)?;
        f.write_all(&frame2)?;
        f.flush()?;
    }

    // 1) Produce a regular MP4 using muxide (non-fragmented).
    let muxide_mp4_path = out_dir.join("muxide.mp4");
    {
        let file = File::create(&muxide_mp4_path)?;
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
        muxer.finish()?;
    }

    // 2) Produce a fragmented MP4 (init + one media segment), concatenated.
    let avc = extract_avc_config(&frame0).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "fixture keyframe missing SPS/PPS",
        )
    })?;

    let cfg = FragmentConfig {
        width: 640,
        height: 480,
        timescale: 90_000,
        // Make it very easy to flush a segment with just a few frames.
        fragment_duration_ms: 1,
        sps: avc.sps,
        pps: avc.pps,
        vps: None,
        av1_sequence_header: None,
        vp9_config: None,
    };

    let mut fmux = FragmentedMuxer::new(cfg);
    let init = fmux.init_segment();

    let f0 = annexb_to_avcc(&frame0);
    let f1 = annexb_to_avcc(&frame1);
    let f2 = annexb_to_avcc(&frame2);

    // 30fps in a 90kHz timescale.
    let dt = 3000u64;
    fmux.write_video(0, 0, &f0, true)?;
    fmux.write_video(dt, dt, &f1, false)?;
    fmux.write_video(2 * dt, 2 * dt, &f2, false)?;

    let seg = fmux
        .flush_segment()
        .ok_or_else(|| io::Error::other("fragment did not flush"))?;

    let muxide_fmp4_combined_path = out_dir.join("muxide_fmp4_combined.mp4");
    {
        let mut f = File::create(&muxide_fmp4_combined_path)?;
        f.write_all(&init)?;
        f.write_all(&seg)?;
        f.flush()?;
    }

    eprintln!("Wrote: {}", stream_h264_path.display());
    eprintln!("Wrote: {}", muxide_mp4_path.display());
    eprintln!("Wrote: {}", muxide_fmp4_combined_path.display());

    Ok(())
}
