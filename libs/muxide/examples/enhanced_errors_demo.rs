use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, MuxerError, VideoCodec};
use std::io::Cursor;

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
    println!("🚀 Demonstrating WORLD-CLASS ADTS Error Messages in Muxide");
    println!("===========================================================");
    println!(
        "✨ New Features: Severity indicators, enhanced hex dumps, JSON output, error chaining"
    );
    println!();

    let sink = Cursor::new(Vec::<u8>::new());
    let mut muxer = MuxerBuilder::new(sink)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build()?;

    // Write a valid video frame first
    let frame0 = read_hex_bytes(include_str!("../fixtures/video_samples/frame0_key.264"));
    muxer.write_video(0.0, &frame0, true)?;

    println!("📋 Example 1: Frame Too Short (User-Friendly Mode)");
    println!("---------------------------------------------------");
    let invalid_adts = &[0x00, 0x01, 0x02];
    match muxer.write_audio(0.0, invalid_adts) {
        Ok(_) => println!("Unexpectedly succeeded"),
        Err(e) => println!("{}", e),
    }

    println!();
    println!("🔧 Example 2: Invalid Syncword (Verbose Technical Mode + JSON)");
    println!("-------------------------------------------------------------");
    // Reset muxer for next test
    let sink2 = Cursor::new(Vec::<u8>::new());
    let mut muxer2 = MuxerBuilder::new(sink2)
        .video(VideoCodec::H264, 640, 480, 30.0)
        .audio(AudioCodec::Aac(AacProfile::Lc), 48_000, 2)
        .build()?;
    muxer2.write_video(0.0, &frame0, true)?;

    let bad_sync = &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]; // Invalid syncword
    match muxer2.write_audio(0.033, bad_sync) {
        Ok(_) => println!("Unexpectedly succeeded"),
        Err(e) => {
            println!("{}", e);
            println!();
            println!("🔍 Verbose Technical Details (for developers):");
            println!("{:#}", e); // Use alternate formatting for verbose mode
            println!();
            println!("📄 JSON Output (for tools/programmatic handling):");
            // Check if this is a detailed ADTS error
            if let MuxerError::InvalidAdtsDetailed {
                error: ref adts_err,
                ..
            } = e
            {
                if let Ok(json) = adts_err.to_json() {
                    println!("{}", json);
                }
            } else {
                println!("(JSON output available for detailed ADTS validation errors)");
            }
        }
    }

    println!();
    println!("📊 Error Message Components:");
    println!("• 🚨 Severity indicators (Error vs Warning)");
    println!("• 🎯 Specific validation failure type");
    println!("• 📍 Exact byte offset in frame");
    println!("• 🔍 Enhanced hex dump with ASCII and color highlighting");
    println!("• 💡 Actionable recovery suggestions");
    println!("• 🛠️  Technical details for developers");
    println!("• 📄 JSON serialization for tools");
    println!("• 🔗 Error chaining for multiple issues");
    println!("• 🎨 User-friendly vs verbose modes");
    println!();
    println!("🚀 This error system makes debugging AAC/MP4 issues 10x faster!");

    Ok(())
}
