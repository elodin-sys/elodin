//! Export video message logs to MP4 files.
//!
//! Reads H.264 Annex B NAL units from `State::msg_logs`, parses SPS for
//! resolution, and muxes frames into MP4 via muxide.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor};
use std::path::PathBuf;

use glob::Pattern;
use impeller2::types::{PacketId, msg_id};
use impeller2_wkt::SensorCameraConfig;
use muxide::api::{MuxerBuilder, VideoCodec};
use muxide::codec::h264::is_h264_keyframe;
use openh264::OpenH264API;
use openh264::encoder::{
    BitRate, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, RateControlMode, SpsPpsStrategy,
};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use scuffle_h264::Sps;

use crate::msg_log::MsgLog;
use crate::{DB, Error};

/// Annex B start code (4-byte).
const START_CODE_4: &[u8] = &[0x00, 0x00, 0x00, 0x01];
/// Annex B start code (3-byte).
const START_CODE_3: &[u8] = &[0x00, 0x00, 0x01];
/// SPS NAL unit type.
const NAL_TYPE_SPS: u8 = 7;

fn invalid_data(message: impl Into<String>) -> Error {
    Error::Io(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

fn safe_output_name(name: &str) -> String {
    name.replace([std::path::MAIN_SEPARATOR, '/'], "_")
}

/// Find the first SPS NAL unit in Annex B payload.
/// Returns the NAL unit bytes (including the NAL header byte), without the start code.
fn find_sps_nal(payload: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i < payload.len() {
        let rest = &payload[i..];
        let (start_len, nal_start) = if rest.starts_with(START_CODE_4) {
            (4, i + 4)
        } else if rest.starts_with(START_CODE_3) {
            (3, i + 3)
        } else {
            i += 1;
            continue;
        };
        i += start_len;
        if nal_start >= payload.len() {
            break;
        }
        let nal_header = payload[nal_start];
        let nal_type = nal_header & 0x1F;
        if nal_type == NAL_TYPE_SPS {
            let search_from = nal_start + 1;
            let end = payload[search_from..]
                .windows(START_CODE_4.len())
                .position(|w| w == START_CODE_4)
                .map(|p| search_from + p)
                .or_else(|| {
                    payload[search_from..]
                        .windows(START_CODE_3.len())
                        .position(|w| w == START_CODE_3)
                        .map(|p| search_from + p)
                })
                .unwrap_or(payload.len());
            return Some(&payload[nal_start..end]);
        }
        // Skip to next start code
        while i < payload.len() {
            if payload[i..].starts_with(START_CODE_4) || payload[i..].starts_with(START_CODE_3) {
                break;
            }
            i += 1;
        }
    }
    None
}

/// Convert one packed RGBA frame into the I420 layout expected by OpenH264.
fn rgba_to_i420(payload: &[u8], width: usize, height: usize) -> Result<YUVBuffer, Error> {
    if width % 2 != 0 || height % 2 != 0 {
        return Err(invalid_data(format!(
            "sensor camera dimensions must be even for I420: {}x{}",
            width, height
        )));
    }
    let expected_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| invalid_data("sensor camera frame dimensions overflow"))?;
    if payload.len() != expected_bytes {
        return Err(invalid_data(format!(
            "unexpected sensor camera frame size {} (expected {})",
            payload.len(),
            expected_bytes
        )));
    }
    let rgba = RgbaSliceU8::new(payload, (width, height));
    Ok(YUVBuffer::from_rgb_source(rgba))
}

struct SensorEncoder {
    encoder: Encoder,
}

impl SensorEncoder {
    fn new(width: u32, height: u32, fps: u32) -> Result<Self, Error> {
        let fps = fps.max(1);
        let target_bitrate = (width as u64)
            .saturating_mul(height as u64)
            .saturating_mul(fps as u64)
            .saturating_mul(3)
            .clamp(300_000, 12_000_000) as u32;
        let config = EncoderConfig::new()
            .bitrate(BitRate::from_bps(target_bitrate))
            .skip_frames(false)
            .max_frame_rate(FrameRate::from_hz(fps as f32))
            .rate_control_mode(RateControlMode::Off)
            .sps_pps_strategy(SpsPpsStrategy::ConstantId)
            .intra_frame_period(IntraFramePeriod::from_num_frames(fps.saturating_mul(2)));
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| invalid_data(format!("openh264 encoder init: {}", e)))?;
        Ok(Self { encoder })
    }

    fn encode_frame(&mut self, yuv: &YUVBuffer) -> Result<Vec<u8>, Error> {
        self.encoder
            .encode(yuv)
            .map(|bitstream| bitstream.to_vec())
            .map_err(|e| invalid_data(format!("openh264 encode: {}", e)))
    }
}

/// Export a single H.264 Annex B message log to an MP4 file.
fn export_one_h264(
    msg_log: &MsgLog,
    name: &str,
    output_path: &std::path::Path,
    default_fps: u32,
) -> Result<bool, Error> {
    let timestamps = msg_log.timestamps();
    if timestamps.is_empty() {
        eprintln!("  {}: no frames, skipping", name);
        return Ok(false);
    }

    let mut sps_bytes: Option<&[u8]> = None;
    for &ts in timestamps.iter().take(20) {
        if let Some(payload) = msg_log.get(ts)
            && let Some(sps) = find_sps_nal(payload)
        {
            sps_bytes = Some(sps);
            break;
        }
    }
    let sps_bytes = match sps_bytes {
        Some(b) => b,
        None => {
            eprintln!("  {}: no SPS NAL found in first frames, skipping", name);
            return Ok(false);
        }
    };

    let sps = Sps::parse_with_emulation_prevention(Cursor::new(sps_bytes)).map_err(|e| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("SPS parse error: {}", e),
        ))
    })?;
    let width = sps.width() as u32;
    let height = sps.height() as u32;
    let fps = sps.frame_rate().unwrap_or(default_fps as f64);

    let safe_name = safe_output_name(name);
    let mp4_path = output_path.join(format!("{}.mp4", safe_name));
    let file = File::create(&mp4_path)?;
    let mut muxer = MuxerBuilder::new(file)
        .video(VideoCodec::H264, width, height, fps)
        .with_fast_start(true)
        .build()
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("muxide: {}", e),
            ))
        })?;

    let first_ts = timestamps[0];
    for &ts in timestamps {
        let payload = match msg_log.get(ts) {
            Some(p) => p,
            None => continue,
        };
        let pts_secs = (ts.0 - first_ts.0) as f64 / 1_000_000.0;
        let keyframe = is_h264_keyframe(payload);
        muxer
            .write_video(pts_secs, payload, keyframe)
            .map_err(|e| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("muxide write_video: {}", e),
                ))
            })?;
    }

    let _stats = muxer.finish_with_stats().map_err(|e| {
        Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("muxide finish: {}", e),
        ))
    })?;

    let frame_count = timestamps.len();
    let duration_secs = if frame_count > 0 {
        (timestamps[frame_count - 1].0 - first_ts.0) as f64 / 1_000_000.0
    } else {
        0.0
    };
    println!("  {}: {} frames, {:.1}s", name, frame_count, duration_secs);
    println!(
        "    Detected resolution: {}x{} from SPS (frame rate: {} fps)",
        width, height, fps
    );
    println!(
        "    Exported {} ({} frames, {} fps, fast-start)",
        mp4_path.display(),
        frame_count,
        fps
    );
    Ok(true)
}

/// Export a raw RGBA sensor-camera message log by encoding it to H.264 first.
fn export_one_sensor(
    msg_log: &MsgLog,
    name: &str,
    output_path: &std::path::Path,
    cfg: &SensorCameraConfig,
    default_fps: u32,
) -> Result<bool, Error> {
    let timestamps = msg_log.timestamps();
    if timestamps.is_empty() {
        eprintln!("  {}: no frames, skipping", name);
        return Ok(false);
    }

    let width = cfg.width;
    let height = cfg.height;
    let fps = default_fps.max(1) as f64;
    let safe_name = safe_output_name(name);
    let mp4_path = output_path.join(format!("{}.mp4", safe_name));
    let file = File::create(&mp4_path)?;
    let mut encoder = SensorEncoder::new(width, height, default_fps)?;
    let mut muxer = MuxerBuilder::new(file)
        .video(VideoCodec::H264, width, height, fps)
        .with_fast_start(true)
        .build()
        .map_err(|e| invalid_data(format!("muxide: {}", e)))?;

    let first_ts = timestamps[0];
    let mut encoded_count = 0usize;
    let mut skipped_count = 0usize;
    for &ts in timestamps {
        let Some(payload) = msg_log.get(ts) else {
            continue;
        };
        let yuv = match rgba_to_i420(payload, width as usize, height as usize) {
            Ok(yuv) => yuv,
            Err(e) => {
                eprintln!("  {}: skipping frame at {}: {}", name, ts.0, e);
                skipped_count += 1;
                continue;
            }
        };
        let annexb = encoder.encode_frame(&yuv)?;
        if annexb.is_empty() {
            continue;
        }
        let pts_secs = (ts.0 - first_ts.0) as f64 / 1_000_000.0;
        let keyframe = is_h264_keyframe(&annexb);
        muxer
            .write_video(pts_secs, &annexb, keyframe)
            .map_err(|e| invalid_data(format!("muxide write_video: {}", e)))?;
        encoded_count += 1;
    }

    if encoded_count == 0 {
        eprintln!("  {}: no valid sensor_camera frames, skipping", name);
        drop(muxer);
        let _ = std::fs::remove_file(&mp4_path);
        return Ok(false);
    }

    let _stats = muxer
        .finish_with_stats()
        .map_err(|e| invalid_data(format!("muxide finish: {}", e)))?;

    let frame_count = timestamps.len();
    let duration_secs = if frame_count > 0 {
        (timestamps[frame_count - 1].0 - first_ts.0) as f64 / 1_000_000.0
    } else {
        0.0
    };
    println!(
        "  {}: {} frames, {:.1}s, {}x{} (sensor_camera, encoded)",
        name, encoded_count, duration_secs, width, height
    );
    if skipped_count > 0 {
        println!("    Skipped {} malformed frame(s)", skipped_count);
    }
    println!(
        "    Exported {} ({} frames, {} fps, fast-start)",
        mp4_path.display(),
        encoded_count,
        fps
    );
    Ok(true)
}

fn export_one(
    msg_log: &MsgLog,
    name: &str,
    output_path: &std::path::Path,
    default_fps: u32,
    sensor_cfg: Option<&SensorCameraConfig>,
) -> Result<bool, Error> {
    match sensor_cfg {
        Some(cfg) => export_one_sensor(msg_log, name, output_path, cfg, default_fps),
        None => export_one_h264(msg_log, name, output_path, default_fps),
    }
}

/// Build a mapping from msg PacketId to friendly name by scanning the
/// schematic KDL for `video_stream "name"` entries and computing `msg_id(name)`.
fn video_name_map_from_schematic(schematic: &str) -> HashMap<PacketId, String> {
    let mut map = HashMap::new();
    // Parse KDL entries like:  video_stream "test-video" name="Test Pattern"
    // The first string argument after video_stream is the msg_name.
    for line in schematic.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("video_stream") {
            let rest = rest.trim();
            // Extract the first quoted string argument (the msg_name)
            if let Some(start) = rest.find('"')
                && let Some(end) = rest[start + 1..].find('"')
            {
                let msg_name = &rest[start + 1..start + 1 + end];
                if !msg_name.is_empty() {
                    let id = msg_id(msg_name);
                    map.insert(id, msg_name.to_string());
                }
            }
        }
    }
    map
}

/// Export video message logs to MP4 files.
///
/// * `db_path` - Path to the database directory
/// * `output_path` - Directory to write MP4 files
/// * `pattern` - Optional glob pattern to filter message logs by name
/// * `fps` - Default frame rate when SPS has no timing_info (default 30)
pub fn run(
    db_path: PathBuf,
    output_path: PathBuf,
    pattern: Option<String>,
    fps: u32,
) -> Result<(), Error> {
    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }
    let db_state_path = db_path.join("db_state");
    if !db_state_path.exists() {
        return Err(Error::MissingDbState(db_state_path));
    }

    let glob_pattern = pattern
        .as_ref()
        .map(|p| Pattern::new(p))
        .transpose()
        .map_err(|e| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid glob pattern: {}", e),
            ))
        })?;

    println!("Opening database: {}", db_path.display());
    let db = DB::open(db_path)?;
    std::fs::create_dir_all(&output_path)?;
    println!("Exporting videos to: {}", output_path.display());
    if let Some(ref p) = pattern {
        println!("Filter pattern: {}", p);
    }
    println!();

    let exported = db.with_state(|state| {
        // Build a fallback name map from the schematic's video_stream entries
        let schematic_names = state
            .db_config
            .schematic_content()
            .map(video_name_map_from_schematic)
            .unwrap_or_default();

        let sensor_cameras: Vec<SensorCameraConfig> =
            match state.db_config.metadata.get("sensor_cameras") {
                Some(json) => match serde_json::from_str(json) {
                    Ok(cameras) => cameras,
                    Err(e) => {
                        eprintln!("Warning: failed to parse sensor_cameras metadata: {}", e);
                        Vec::new()
                    }
                },
                None => Vec::new(),
            };
        let sensor_by_msg_id: HashMap<PacketId, SensorCameraConfig> = sensor_cameras
            .into_iter()
            .map(|camera| (msg_id(&camera.camera_name), camera))
            .collect();

        let mut video_logs: Vec<(PacketId, String, &MsgLog)> = Vec::new();
        for (packet_id, msg_log) in state.msg_logs.iter() {
            let name: String = msg_log
                .metadata()
                .map(|m| m.name.clone())
                .or_else(|| {
                    sensor_by_msg_id
                        .get(packet_id)
                        .map(|camera| camera.camera_name.clone())
                })
                .or_else(|| schematic_names.get(packet_id).cloned())
                .unwrap_or_else(|| format!("msg-{}", u16::from_le_bytes(*packet_id)));
            if let Some(ref pat) = glob_pattern
                && !pat.matches(&name)
            {
                continue;
            }
            video_logs.push((*packet_id, name, msg_log));
        }
        println!("Found {} video stream(s)", video_logs.len());
        let mut count = 0;
        for (packet_id, name, msg_log) in video_logs {
            let sensor_cfg = sensor_by_msg_id.get(&packet_id);
            match export_one(msg_log, &name, &output_path, fps, sensor_cfg) {
                Ok(true) => {
                    count += 1;
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!("  {}: error: {}", name, e);
                }
            }
        }
        count
    });
    println!();
    println!("Export complete: {} video(s) exported", exported);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openh264::formats::YUVSource;

    #[test]
    fn rgba_to_i420_known_colors() {
        let cases = [
            ([0, 0, 0, 255], 16, 128, 128),
            ([255, 255, 255, 255], 235, 128, 128),
            ([255, 0, 0, 255], 81, 90, 239),
            ([0, 255, 0, 255], 144, 54, 34),
            ([0, 0, 255, 255], 40, 239, 110),
        ];
        for (pixel, y, u, v) in cases {
            let frame = pixel.repeat(4);
            let yuv = rgba_to_i420(&frame, 2, 2).expect("rgba_to_i420");
            assert_eq!(yuv.y(), [y, y, y, y]);
            assert_eq!(yuv.u(), [u]);
            assert_eq!(yuv.v(), [v]);
        }
    }

    #[test]
    fn rgba_to_i420_dimensions() {
        let width = 6;
        let height = 4;
        let frame = vec![128; width * height * 4];
        let yuv = rgba_to_i420(&frame, width, height).expect("rgba_to_i420");
        assert_eq!(yuv.dimensions(), (width, height));
        assert_eq!(yuv.y().len(), width * height);
        assert_eq!(yuv.u().len(), width * height / 4);
        assert_eq!(yuv.v().len(), width * height / 4);
    }
}
