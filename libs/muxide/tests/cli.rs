use std::path::PathBuf;
use std::process::Command;

/// Test CLI help output
#[test]
fn cli_help_works() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "muxide", "--", "--help"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Muxide"));
    assert!(stdout.contains("mux"));
    assert!(stdout.contains("validate"));
    assert!(stdout.contains("info"));
}

/// Test CLI mux command help
#[test]
fn cli_mux_help_works() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "muxide", "--", "mux", "--help"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Mux encoded frames into MP4"));
    assert!(stdout.contains("--video"));
    assert!(stdout.contains("--audio"));
    assert!(stdout.contains("--output"));
}

/// Test CLI mux with video only
#[test]
fn cli_mux_video_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_video.mp4");

    // Use the fixture video frame
    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Muxing complete"));
    assert!(stdout.contains("Video frames: 1"));
    assert!(output_path.exists());
}

/// Test CLI mux with video and audio
#[test]
fn cli_mux_video_and_audio() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_av.mp4");

    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");
    let audio_fixture = PathBuf::from("fixtures/audio_samples/frame0.aac.adts");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--audio",
            audio_fixture.to_str().unwrap(),
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--sample-rate",
            "44100",
            "--channels",
            "2",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Muxing complete"));
    assert!(stdout.contains("Video frames: 1"));
    assert!(stdout.contains("Audio frames: 1"));
    assert!(output_path.exists());
}

/// Test CLI mux with metadata
#[test]
fn cli_mux_with_metadata() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_metadata.mp4");

    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--title",
            "Test Recording",
            "--language",
            "eng",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Muxing complete"));
    assert!(output_path.exists());
}

/// Test CLI mux with different video codecs
#[test]
fn cli_mux_different_video_codecs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_h265.mp4");

    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "--verbose",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--video-codec",
            "h264",
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Configured video: H.264"));
    assert!(output_path.exists());
}

/// Test CLI mux with different audio codecs
#[test]
fn cli_mux_different_audio_codecs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_aac_he.mp4");

    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");
    let audio_fixture = PathBuf::from("fixtures/audio_samples/frame0.aac.adts");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "--verbose",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--audio",
            audio_fixture.to_str().unwrap(),
            "--video-codec",
            "h264",
            "--audio-codec",
            "aac-he",
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--sample-rate",
            "44100",
            "--channels",
            "2",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Configured audio: AAC-HE"));
    assert!(output_path.exists());
}

/// Test CLI validate command with missing inputs
#[test]
fn cli_validate_command() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "muxide", "--", "validate"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success()); // Command succeeds but reports validation failure
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("At least one of video or audio input must be specified"));
}

/// Test CLI info command with nonexistent file
#[test]
fn cli_info_command() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "muxide", "--", "info", "nonexistent.mp4"])
        .output()
        .expect("Failed to run CLI");

    assert!(!output.status.success()); // Should fail for nonexistent file
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Input file does not exist"));
}

/// Test CLI info command with valid MP4 file
#[test]
fn cli_info_command_valid_file() {
    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "info",
            "fixtures/minimal.mp4",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Valid MP4: Yes"));
    assert!(stdout.contains("ftyp"));
    assert!(stdout.contains("moov"));
    assert!(stdout.contains("mdat"));
}

/// Test CLI error handling - missing required parameters
#[test]
fn cli_error_missing_video_params() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_error.mp4");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "mux",
            "--video",
            "fixtures/video_samples/frame0_key.264",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Video parameters must be complete when video input is provided"));
}

/// Test CLI error handling - no video or audio
#[test]
fn cli_error_no_inputs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_error.mp4");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "mux",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("At least one of --video or --audio must be specified"));
}

/// Test CLI JSON output
#[test]
fn cli_json_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_json.mp4");

    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "--json",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should be valid JSON
    let json: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(json.is_object());
    assert!(json.get("video_frames").is_some());
    assert!(json.get("audio_frames").is_some());
    assert!(json.get("total_bytes").is_some());
}

/// Test CLI verbose output
#[test]
fn cli_verbose_output() {
    let temp_dir = tempfile::tempdir().unwrap();
    let output_path = temp_dir.path().join("test_verbose.mp4");

    let video_fixture = PathBuf::from("fixtures/video_samples/frame0_key.264");

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "muxide",
            "--",
            "--verbose",
            "mux",
            "--video",
            video_fixture.to_str().unwrap(),
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "30",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Muxide v"));
    assert!(stderr.contains("Setting up muxer"));
    assert!(stderr.contains("Configured video"));
    assert!(stderr.contains("Processing video frames"));
    assert!(stderr.contains("Finalizing MP4"));
}
