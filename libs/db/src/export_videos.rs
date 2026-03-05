//! Export video message logs to MP4 files.
//!
//! Reads H.264 Annex B NAL units from `State::msg_logs`, parses SPS for
//! resolution, and muxes frames into MP4 via muxide.

use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::PathBuf;

use glob::Pattern;
use impeller2::types::{PacketId, msg_id};
use muxide::api::{MuxerBuilder, VideoCodec};
use muxide::codec::h264::is_h264_keyframe;
use scuffle_h264::Sps;

use crate::msg_log::MsgLog;
use crate::{DB, Error};

/// Annex B start code (4-byte).
const START_CODE_4: &[u8] = &[0x00, 0x00, 0x00, 0x01];
/// Annex B start code (3-byte).
const START_CODE_3: &[u8] = &[0x00, 0x00, 0x01];
/// SPS NAL unit type.
const NAL_TYPE_SPS: u8 = 7;

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

/// Export a single message log to an MP4 file.
fn export_one(
    msg_log: &MsgLog,
    name: &str,
    output_path: &std::path::Path,
    default_fps: u32,
) -> Result<(), Error> {
    let timestamps = msg_log.timestamps();
    if timestamps.is_empty() {
        eprintln!("  {}: no frames, skipping", name);
        return Ok(());
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
            return Ok(());
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

    let safe_name = name.replace([std::path::MAIN_SEPARATOR, '/'], "_");
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
    Ok(())
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

        let mut video_logs: Vec<(String, &MsgLog)> = Vec::new();
        for (packet_id, msg_log) in state.msg_logs.iter() {
            let name: String = msg_log
                .metadata()
                .map(|m| m.name.clone())
                .or_else(|| schematic_names.get(packet_id).cloned())
                .unwrap_or_else(|| format!("msg-{}", u16::from_le_bytes(*packet_id)));
            if let Some(ref pat) = glob_pattern
                && !pat.matches(&name)
            {
                continue;
            }
            video_logs.push((name, msg_log));
        }
        println!("Found {} video stream(s)", video_logs.len());
        let mut count = 0;
        for (name, msg_log) in video_logs {
            if let Err(e) = export_one(msg_log, &name, &output_path, fps) {
                eprintln!("  {}: error: {}", name, e);
            } else {
                count += 1;
            }
        }
        count
    });
    println!();
    println!("Export complete: {} video(s) exported", exported);
    Ok(())
}
