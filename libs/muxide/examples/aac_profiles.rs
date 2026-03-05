use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, MuxerConfig, VideoCodec};
use std::{env, fs::File, path::PathBuf};

/// Generate a simple AAC ADTS frame for testing
/// In real usage, this would come from an audio encoder or microphone
fn generate_aac_frame(_profile: AacProfile, _frame_index: usize) -> Vec<u8> {
    // Valid ADTS frame for testing (7 byte header + minimal payload)
    vec![0xff, 0xf1, 0x4c, 0x80, 0x01, 0x3f, 0xfc, 0xaa, 0xbb]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Demonstrates AAC profile support with real muxing
    // Usage: cargo run --example aac_profiles -- <profile> <output.mp4>
    // Example: cargo run --example aac_profiles -- lc output.mp4

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <profile> <output.mp4>", args[0]);
        eprintln!("Profiles: lc, main, ssr, ltp, he, hev2");
        std::process::exit(1);
    }

    let profile_str = &args[1];
    let out_path = PathBuf::from(&args[2]);

    let profile = match profile_str.as_str() {
        "lc" => AacProfile::Lc,
        "main" => AacProfile::Main,
        "ssr" => AacProfile::Ssr,
        "ltp" => AacProfile::Ltp,
        "he" => AacProfile::He,
        "hev2" => AacProfile::Hev2,
        _ => {
            eprintln!("Invalid profile. Use: lc, main, ssr, ltp, he, hev2");
            std::process::exit(1);
        }
    };

    println!(
        "Testing AAC {} profile muxing...",
        profile_str.to_uppercase()
    );

    // Create test video frames (minimal H.264)
    let video_frame = vec![
        0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1e, 0x95, 0xa8, 0x28, 0x28, 0x28, // SPS
        0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80, // PPS
        0, 0, 0, 1, 0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, // IDR
    ];

    let file = File::create(&out_path)?;
    let config = MuxerConfig {
        width: 640,
        height: 480,
        framerate: 30.0,
        audio: Some(muxide::api::AudioTrackConfig {
            codec: AudioCodec::Aac(profile),
            sample_rate: 48000,
            channels: 2,
        }),
        metadata: None,
        fast_start: true,
    };

    let mut muxer = MuxerBuilder::new(file)
        .video(
            VideoCodec::H264,
            config.width,
            config.height,
            config.framerate,
        )
        .audio(
            config.audio.as_ref().unwrap().codec,
            config.audio.as_ref().unwrap().sample_rate,
            config.audio.as_ref().unwrap().channels,
        )
        .build()?;

    // Write video keyframe
    muxer.write_video(0.0, &video_frame, true)?;

    // Write several AAC frames (simulating ~1 second of audio at 48kHz)
    for i in 0..46 {
        // ~46 frames per second for AAC
        let pts = i as f64 * (1024.0 / 48000.0); // AAC frame duration
        let aac_frame = generate_aac_frame(profile, i);
        muxer.write_audio(pts, &aac_frame)?;
    }

    let stats = muxer.finish_with_stats()?;

    println!(
        "✅ Successfully created AAC {} MP4 file!",
        profile_str.to_uppercase()
    );
    println!(
        "📊 Stats: {} video frames, {} audio frames, {:.3}s duration, {} bytes",
        stats.video_frames, stats.audio_frames, stats.duration_secs, stats.bytes_written
    );
    println!("🎵 Output: {}", out_path.display());

    // Verification instructions
    println!("\n🔍 To verify the AAC audio:");
    println!(
        "1. Play with: ffplay {}  (or any MP4 player)",
        out_path.display()
    );
    println!(
        "2. Check streams: ffprobe -i {} -show_streams",
        out_path.display()
    );
    println!(
        "3. Extract audio: ffmpeg -i {} -vn -acodec copy audio.aac",
        out_path.display()
    );

    Ok(())
}
