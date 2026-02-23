//! Database trim tool
//!
//! Removes component and message log data outside a specified time range,
//! producing a trimmed database that can be played, followed, and replayed
//! correctly.
//!
//! Timestamps are **relative**:
//! - `--from-start X` removes the first X microseconds from the start (keeps data >= start+X)
//! - `--from-end Y` removes the last Y microseconds from the end (keeps data <= end-Y)
//! - Together they trim both ends of the recording

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};

use impeller2::buf::UmbraBuf;
use impeller2_wkt::DbConfig;
use zerocopy::FromBytes;

use crate::{Error, MetadataExt, copy_file_native, sync_dir};

const HEADER_SIZE: usize = 24;

/// Trim a database to only include data within the specified time range.
///
/// - `from_start`: remove the first N microseconds from the start
/// - `from_end`: remove the last N microseconds from the end
///
/// At least one must be provided. If `output` is `None`, modifies in place.
pub fn run(
    db_path: PathBuf,
    from_start: Option<i64>,
    from_end: Option<i64>,
    output: Option<PathBuf>,
    dry_run: bool,
    auto_confirm: bool,
) -> Result<(), Error> {
    if from_start.is_none() && from_end.is_none() {
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Must specify at least one of --from-start or --from-end",
        )));
    }

    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }

    let db_state_path = db_path.join("db_state");
    if !db_state_path.exists() {
        return Err(Error::MissingDbState(db_state_path));
    }

    let in_place = output.is_none();
    let final_output = output.unwrap_or_else(|| db_path.clone());

    if !in_place && final_output.exists() {
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "Output directory already exists: {}",
                final_output.display()
            ),
        )));
    }

    let info = analyze_database(&db_path)?;

    let config = DbConfig::read(db_state_path)?;
    let recording_start = config
        .time_start_timestamp_micros()
        .unwrap_or_else(|| info.time_range.map_or(0, |(start, _)| start));
    let recording_end = info.time_range.map_or(i64::MAX, |(_, end)| end);
    let abs_start = from_start.map(|s| recording_start.saturating_add(s));
    let abs_end = from_end.map(|e| recording_end.saturating_sub(e));

    let kept_range = info.time_range.map(|(ds, de)| {
        let kept_start = abs_start.map_or(ds, |s| s.max(ds));
        let kept_end = abs_end.map_or(de, |e| e.min(de));
        (kept_start, kept_end)
    });

    if let (Some(start), Some(end)) = (abs_start, abs_end)
        && start > end
    {
        let duration = info.time_range.map_or(0, |(s, e)| e - s);
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Trim amounts exceed recording duration: \
                 --from-start {} + --from-end {} = {} us > {} us duration",
                from_start.unwrap_or(0),
                from_end.unwrap_or(0),
                from_start.unwrap_or(0) + from_end.unwrap_or(0),
                duration
            ),
        )));
    }

    println!("Trimming database: {}", db_path.display());
    print_trim_plan(
        &info,
        from_start,
        from_end,
        kept_range,
        &final_output,
        in_place,
    );

    if dry_run {
        println!("\nDRY RUN - no changes will be made");
        return Ok(());
    }

    if !auto_confirm {
        print!("\nProceed? [y/N] ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    let parent_dir = final_output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&parent_dir)?;

    let tmp_output = parent_dir.join(format!(
        "{}.trim-tmp",
        final_output
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    ));
    if tmp_output.exists() {
        fs::remove_dir_all(&tmp_output)?;
    }
    fs::create_dir_all(&tmp_output)?;

    let keep_start = abs_start.unwrap_or(i64::MIN);
    let keep_end = abs_end.unwrap_or(i64::MAX);

    let result = do_trim(&db_path, &tmp_output, keep_start, keep_end);

    if result.is_err() {
        eprintln!("Trim failed, cleaning up temporary directory...");
        if let Err(cleanup_err) = fs::remove_dir_all(&tmp_output) {
            eprintln!(
                "Warning: Failed to clean up temporary directory {}: {}",
                tmp_output.display(),
                cleanup_err
            );
        }
        return result;
    }

    result?;

    sync_dir(&tmp_output)?;

    if in_place {
        let backup = parent_dir.join(format!(
            "{}.trim-old",
            db_path.file_name().unwrap_or_default().to_string_lossy()
        ));
        if backup.exists() {
            fs::remove_dir_all(&backup)?;
        }
        fs::rename(&db_path, &backup)?;
        fs::rename(&tmp_output, &final_output)?;
        fs::remove_dir_all(&backup)?;
    } else {
        fs::rename(&tmp_output, &final_output)?;
    }
    sync_dir(&parent_dir)?;

    println!(
        "\nSuccessfully trimmed database to {}",
        final_output.display()
    );
    Ok(())
}

struct TrimInfo {
    component_count: usize,
    msg_log_count: usize,
    time_range: Option<(i64, i64)>,
}

fn analyze_database(db_path: &Path) -> Result<TrimInfo, Error> {
    let mut component_count = 0;
    let mut msg_log_count = 0;
    let mut min_timestamp = i64::MAX;
    let mut max_timestamp = i64::MIN;

    for entry in fs::read_dir(db_path)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
        if dir_name == "msgs" {
            if let Ok(msgs_dir) = fs::read_dir(&path) {
                for msg_entry in msgs_dir.flatten() {
                    if msg_entry.path().is_dir() {
                        msg_log_count += 1;
                    }
                }
            }
            continue;
        }
        if dir_name.parse::<u64>().is_err() {
            continue;
        }
        if path.join("schema").exists() {
            component_count += 1;
            if let Ok(Some((start, end))) = read_timestamp_range(&path.join("index")) {
                min_timestamp = min_timestamp.min(start);
                max_timestamp = max_timestamp.max(end);
            }
        }
    }

    let time_range = if min_timestamp <= max_timestamp && min_timestamp != i64::MAX {
        Some((min_timestamp, max_timestamp))
    } else {
        None
    };

    Ok(TrimInfo {
        component_count,
        msg_log_count,
        time_range,
    })
}

fn format_duration(us: i64) -> String {
    let secs = us as f64 / 1_000_000.0;
    if secs >= 60.0 {
        let mins = (secs / 60.0).floor();
        let rem = secs - mins * 60.0;
        format!("{:.0}m {:.1}s", mins, rem)
    } else {
        format!("{:.3}s", secs)
    }
}

fn print_trim_plan(
    info: &TrimInfo,
    from_start: Option<i64>,
    from_end: Option<i64>,
    kept_range: Option<(i64, i64)>,
    output: &Path,
    in_place: bool,
) {
    println!("  Components: {}", info.component_count);
    if info.msg_log_count > 0 {
        println!("  Message logs: {}", info.msg_log_count);
    }
    if let Some((start, end)) = info.time_range {
        let duration = end - start;
        println!(
            "  Data duration: {} ({} components)",
            format_duration(duration),
            info.component_count
        );
    }
    println!();

    println!("Trim parameters:");
    if let Some(s) = from_start {
        println!(
            "  Remove first {} from start (--from-start {})",
            format_duration(s),
            s
        );
    }
    if let Some(e) = from_end {
        println!(
            "  Remove last {} from end (--from-end {})",
            format_duration(e),
            e
        );
    }

    if let Some((kept_start, kept_end)) = kept_range {
        if kept_start <= kept_end {
            let new_duration = kept_end - kept_start;
            let original = info.time_range.map_or(0, |(s, e)| e - s);
            println!(
                "  Resulting duration: {} (from {})",
                format_duration(new_duration),
                format_duration(original),
            );
        } else {
            println!("  Warning: trim amounts exceed recording duration -- result will be empty");
        }
    }

    if in_place {
        println!("\n  Output: {} (in-place)", output.display());
    } else {
        println!("\n  Output: {}", output.display());
    }
}

fn do_trim(db_path: &Path, tmp_output: &Path, keep_start: i64, keep_end: i64) -> Result<(), Error> {
    let mut trimmed_min = i64::MAX;
    let mut trimmed_max = i64::MIN;
    let mut components_trimmed = 0usize;
    let mut msg_logs_trimmed = 0usize;

    for entry in fs::read_dir(db_path)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();

        if dir_name == "msgs" {
            let target_msgs = tmp_output.join("msgs");
            fs::create_dir_all(&target_msgs)?;
            if let Ok(msgs_dir) = fs::read_dir(&path) {
                for msg_entry in msgs_dir.flatten() {
                    let msg_path = msg_entry.path();
                    if !msg_path.is_dir() {
                        continue;
                    }
                    let msg_id = msg_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let target_msg_dir = target_msgs.join(&msg_id);

                    let (msg_min, msg_max) =
                        trim_msg_log(&msg_path, &target_msg_dir, keep_start, keep_end)?;

                    if msg_min <= msg_max {
                        trimmed_min = trimmed_min.min(msg_min);
                        trimmed_max = trimmed_max.max(msg_max);
                        msg_logs_trimmed += 1;
                    }
                }
            }
            sync_dir(&target_msgs)?;
            continue;
        }

        if dir_name.parse::<u64>().is_err() {
            continue;
        }

        if !path.join("schema").exists() {
            continue;
        }

        let target_component = tmp_output.join(dir_name.as_ref());
        let (comp_min, comp_max) = trim_component(&path, &target_component, keep_start, keep_end)?;

        if comp_min <= comp_max {
            trimmed_min = trimmed_min.min(comp_min);
            trimmed_max = trimmed_max.max(comp_max);
        }
        components_trimmed += 1;
    }

    print!("Writing db_state... ");
    io::stdout().flush()?;

    let effective_start = if trimmed_min != i64::MAX {
        Some(trimmed_min)
    } else {
        None
    };
    write_trimmed_db_state(db_path, tmp_output, effective_start)?;
    println!("done");

    println!(
        "Trimmed {} components, {} message logs",
        components_trimmed, msg_logs_trimmed
    );

    Ok(())
}

/// Trim a single component directory, keeping only data in [keep_start, keep_end].
///
/// Returns (min_kept_timestamp, max_kept_timestamp). If no data is kept,
/// returns (i64::MAX, i64::MIN).
fn trim_component(
    src_dir: &Path,
    dst_dir: &Path,
    keep_start: i64,
    keep_end: i64,
) -> Result<(i64, i64), Error> {
    fs::create_dir_all(dst_dir)?;

    let schema_src = src_dir.join("schema");
    if schema_src.exists() {
        copy_file_native(&schema_src, &dst_dir.join("schema"))?;
    }

    let metadata_src = src_dir.join("metadata");
    if metadata_src.exists() {
        copy_file_native(&metadata_src, &dst_dir.join("metadata"))?;
    }

    let index_src = src_dir.join("index");
    let data_src = src_dir.join("data");

    if !index_src.exists() || !data_src.exists() {
        write_empty_index(&dst_dir.join("index"), 0)?;
        write_empty_data(&dst_dir.join("data"), 8)?;
        sync_dir(dst_dir)?;
        return Ok((i64::MAX, i64::MIN));
    }

    let (src_header, timestamps) = read_index_timestamps(&index_src)?;
    let element_size = read_data_element_size(&data_src)?;

    if timestamps.is_empty() {
        write_empty_index(&dst_dir.join("index"), src_header.start_ts)?;
        write_empty_data(&dst_dir.join("data"), element_size)?;
        sync_dir(dst_dir)?;
        return Ok((i64::MAX, i64::MIN));
    }

    let start_idx = timestamps.partition_point(|&ts| ts < keep_start);
    let end_idx = timestamps.partition_point(|&ts| ts <= keep_end);

    if start_idx >= end_idx {
        write_empty_index(&dst_dir.join("index"), src_header.start_ts)?;
        write_empty_data(&dst_dir.join("data"), element_size)?;
        sync_dir(dst_dir)?;
        return Ok((i64::MAX, i64::MIN));
    }

    let kept_timestamps = &timestamps[start_idx..end_idx];
    let min_kept = kept_timestamps[0];
    let max_kept = *kept_timestamps.last().unwrap();

    let new_start_ts = min_kept.min(src_header.start_ts.max(keep_start));
    write_trimmed_index(&dst_dir.join("index"), new_start_ts, kept_timestamps)?;

    write_trimmed_data(
        &data_src,
        &dst_dir.join("data"),
        element_size,
        start_idx,
        end_idx - 1,
    )?;

    sync_dir(dst_dir)?;
    Ok((min_kept, max_kept))
}

/// Trim a single message log directory, keeping only messages in [keep_start, keep_end].
///
/// Returns (min_kept_timestamp, max_kept_timestamp). If no messages are kept,
/// returns (i64::MAX, i64::MIN).
fn trim_msg_log(
    src_dir: &Path,
    dst_dir: &Path,
    keep_start: i64,
    keep_end: i64,
) -> Result<(i64, i64), Error> {
    fs::create_dir_all(dst_dir)?;

    let metadata_src = src_dir.join("metadata");
    if metadata_src.exists() {
        copy_file_native(&metadata_src, &dst_dir.join("metadata"))?;
    }

    let timestamps_src = src_dir.join("timestamps");
    let offsets_src = src_dir.join("offsets");
    let data_log_src = src_dir.join("data_log");

    if !timestamps_src.exists() || !offsets_src.exists() {
        write_empty_appendlog(&dst_dir.join("timestamps"))?;
        write_empty_appendlog(&dst_dir.join("offsets"))?;
        write_empty_appendlog(&dst_dir.join("data_log"))?;
        sync_dir(dst_dir)?;
        return Ok((i64::MAX, i64::MIN));
    }

    let msg_timestamps = read_msg_timestamps(&timestamps_src)?;

    if msg_timestamps.is_empty() {
        write_empty_appendlog(&dst_dir.join("timestamps"))?;
        write_empty_appendlog(&dst_dir.join("offsets"))?;
        write_empty_appendlog(&dst_dir.join("data_log"))?;
        sync_dir(dst_dir)?;
        return Ok((i64::MAX, i64::MIN));
    }

    let start_idx = msg_timestamps.partition_point(|&ts| ts < keep_start);
    let end_idx = msg_timestamps.partition_point(|&ts| ts <= keep_end);

    if start_idx >= end_idx {
        write_empty_appendlog(&dst_dir.join("timestamps"))?;
        write_empty_appendlog(&dst_dir.join("offsets"))?;
        write_empty_appendlog(&dst_dir.join("data_log"))?;
        sync_dir(dst_dir)?;
        return Ok((i64::MAX, i64::MIN));
    }

    let min_kept = msg_timestamps[start_idx];
    let max_kept = msg_timestamps[end_idx - 1];

    let src_offsets_data = read_appendlog_data(&offsets_src, MSG_HEADER_SIZE)?;
    let src_data_log_data = read_appendlog_data(&data_log_src, MSG_HEADER_SIZE)?;

    let umbra_buf_size = size_of::<UmbraBuf>();
    let num_bufs = src_offsets_data.len() / umbra_buf_size;

    let mut new_timestamps_data: Vec<u8> = Vec::new();
    let mut new_offsets_data: Vec<u8> = Vec::new();
    let mut new_data_log: Vec<u8> = Vec::new();

    for (src_idx, &ts) in msg_timestamps[start_idx..end_idx].iter().enumerate() {
        let abs_idx = start_idx + src_idx;
        if abs_idx >= num_bufs {
            break;
        }
        let buf_offset = abs_idx * umbra_buf_size;
        let buf_bytes = &src_offsets_data[buf_offset..buf_offset + umbra_buf_size];
        let umbra = <UmbraBuf>::ref_from_bytes(buf_bytes).expect("UmbraBuf alignment");

        let len = umbra.len as usize;
        if len <= 12 {
            new_timestamps_data.extend_from_slice(&ts.to_le_bytes());
            new_offsets_data.extend_from_slice(buf_bytes);
        } else {
            let old_offset = unsafe { umbra.data.offset.offset } as usize;
            if old_offset + len > src_data_log_data.len() {
                break;
            }
            let payload = &src_data_log_data[old_offset..old_offset + len];
            let prefix: [u8; 4] = payload[..4].try_into().expect("trivial cast");
            let new_offset = new_data_log.len() as u32;
            new_data_log.extend_from_slice(payload);
            let new_buf = UmbraBuf::with_offset(len as u32, prefix, new_offset);
            new_timestamps_data.extend_from_slice(&ts.to_le_bytes());
            new_offsets_data.extend_from_slice(zerocopy::IntoBytes::as_bytes(&new_buf));
        }
    }

    write_msg_appendlog(&dst_dir.join("timestamps"), &new_timestamps_data)?;
    write_msg_appendlog(&dst_dir.join("offsets"), &new_offsets_data)?;
    write_msg_appendlog(&dst_dir.join("data_log"), &new_data_log)?;

    sync_dir(dst_dir)?;
    Ok((min_kept, max_kept))
}

fn write_trimmed_db_state(
    source_db: &Path,
    target_db: &Path,
    effective_start: Option<i64>,
) -> Result<(), Error> {
    let mut config = DbConfig::read(source_db.join("db_state"))?;

    if let Some(start) = effective_start {
        config.set_time_start_timestamp_micros(start);
    }

    config.write(target_db.join("db_state"))?;
    Ok(())
}

// -- Index/Data file I/O helpers --

struct IndexHeader {
    start_ts: i64,
}

fn read_index_timestamps(index_path: &Path) -> Result<(IndexHeader, Vec<i64>), Error> {
    let mut file = File::open(index_path)?;
    let mut header = [0u8; HEADER_SIZE];
    let bytes_read = file.read(&mut header)?;

    if bytes_read < HEADER_SIZE {
        return Ok((IndexHeader { start_ts: 0 }, Vec::new()));
    }

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let committed_len = committed_len.max(HEADER_SIZE);
    let start_ts = i64::from_le_bytes(header[16..24].try_into().unwrap());

    let data_len = committed_len.saturating_sub(HEADER_SIZE);
    let num_timestamps = data_len / 8;

    if num_timestamps == 0 {
        return Ok((IndexHeader { start_ts }, Vec::new()));
    }

    let mut ts_data = vec![0u8; num_timestamps * 8];
    file.read_exact(&mut ts_data)?;

    let timestamps: Vec<i64> = ts_data
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok((IndexHeader { start_ts }, timestamps))
}

fn read_data_element_size(data_path: &Path) -> Result<u64, Error> {
    let mut file = File::open(data_path)?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;
    let element_size = u64::from_le_bytes(header[16..24].try_into().unwrap());
    Ok(element_size)
}

fn write_empty_index(path: &Path, start_ts: i64) -> Result<(), Error> {
    let mut data = [0u8; HEADER_SIZE];
    let committed_len = HEADER_SIZE as u64;
    data[0..8].copy_from_slice(&committed_len.to_le_bytes());
    data[16..24].copy_from_slice(&start_ts.to_le_bytes());
    let mut file = File::create(path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(())
}

fn write_empty_data(path: &Path, element_size: u64) -> Result<(), Error> {
    let mut data = [0u8; HEADER_SIZE];
    let committed_len = HEADER_SIZE as u64;
    data[0..8].copy_from_slice(&committed_len.to_le_bytes());
    data[16..24].copy_from_slice(&element_size.to_le_bytes());
    let mut file = File::create(path)?;
    file.write_all(&data)?;
    file.sync_all()?;
    Ok(())
}

fn write_trimmed_index(path: &Path, start_ts: i64, timestamps: &[i64]) -> Result<(), Error> {
    let data_len = timestamps.len() * 8;
    let committed_len = (HEADER_SIZE + data_len) as u64;

    let mut header = [0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(&committed_len.to_le_bytes());
    header[16..24].copy_from_slice(&start_ts.to_le_bytes());

    let mut file = File::create(path)?;
    file.write_all(&header)?;

    for ts in timestamps {
        file.write_all(&ts.to_le_bytes())?;
    }

    file.sync_all()?;
    if let Some(parent) = path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

fn write_trimmed_data(
    src_data: &Path,
    dst_data: &Path,
    element_size: u64,
    start_idx: usize,
    end_idx: usize,
) -> Result<(), Error> {
    let elem_size = element_size as usize;
    let num_kept = end_idx - start_idx + 1;
    let kept_data_len = num_kept * elem_size;
    let committed_len = (HEADER_SIZE + kept_data_len) as u64;

    let mut src_file = File::open(src_data)?;

    let mut header = [0u8; HEADER_SIZE];
    src_file.read_exact(&mut header)?;

    header[0..8].copy_from_slice(&committed_len.to_le_bytes());
    header[8..16].copy_from_slice(&0u64.to_le_bytes());

    let mut dst_file = File::create(dst_data)?;
    dst_file.write_all(&header)?;

    let skip_bytes = (start_idx * elem_size) as u64;
    src_file.seek(SeekFrom::Current(skip_bytes as i64))?;

    const CHUNK_SIZE: usize = 64 * 1024;
    let mut remaining = kept_data_len;
    let mut buf = vec![0u8; CHUNK_SIZE];

    while remaining > 0 {
        let to_read = remaining.min(CHUNK_SIZE);
        src_file.read_exact(&mut buf[..to_read])?;
        dst_file.write_all(&buf[..to_read])?;
        remaining -= to_read;
    }

    dst_file.sync_all()?;
    if let Some(parent) = dst_data.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

// -- Message log I/O helpers --

const MSG_HEADER_SIZE: usize = 16;

fn read_msg_timestamps(path: &Path) -> Result<Vec<i64>, Error> {
    let mut file = File::open(path)?;
    let mut header = [0u8; MSG_HEADER_SIZE];
    let bytes_read = file.read(&mut header)?;

    if bytes_read < MSG_HEADER_SIZE {
        return Ok(Vec::new());
    }

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let committed_len = committed_len.max(MSG_HEADER_SIZE);
    let data_len = committed_len.saturating_sub(MSG_HEADER_SIZE);
    let num_timestamps = data_len / 8;

    if num_timestamps == 0 {
        return Ok(Vec::new());
    }

    let mut ts_data = vec![0u8; num_timestamps * 8];
    file.read_exact(&mut ts_data)?;

    let timestamps: Vec<i64> = ts_data
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    Ok(timestamps)
}

fn read_appendlog_data(path: &Path, header_size: usize) -> Result<Vec<u8>, Error> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut file = File::open(path)?;
    let mut header_buf = vec![0u8; header_size];
    let bytes_read = file.read(&mut header_buf)?;

    if bytes_read < header_size {
        return Ok(Vec::new());
    }

    let committed_len = u64::from_le_bytes(header_buf[0..8].try_into().unwrap()) as usize;
    let committed_len = committed_len.max(header_size);
    let data_len = committed_len.saturating_sub(header_size);

    if data_len == 0 {
        return Ok(Vec::new());
    }

    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    Ok(data)
}

fn write_msg_appendlog(path: &Path, data: &[u8]) -> Result<(), Error> {
    let committed_len = (MSG_HEADER_SIZE + data.len()) as u64;
    let mut header = [0u8; MSG_HEADER_SIZE];
    header[0..8].copy_from_slice(&committed_len.to_le_bytes());

    let mut file = File::create(path)?;
    file.write_all(&header)?;
    file.write_all(data)?;
    file.sync_all()?;

    if let Some(parent) = path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

fn write_empty_appendlog(path: &Path) -> Result<(), Error> {
    write_msg_appendlog(path, &[])
}

fn read_timestamp_range(index_path: &Path) -> Result<Option<(i64, i64)>, Error> {
    if !index_path.exists() {
        return Ok(None);
    }

    let mut file = File::open(index_path)?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let data_len = committed_len.saturating_sub(HEADER_SIZE);
    if data_len < 8 {
        return Ok(None);
    }

    let num_timestamps = data_len / 8;

    let mut first_ts_bytes = [0u8; 8];
    file.read_exact(&mut first_ts_bytes)?;
    let first_ts = i64::from_le_bytes(first_ts_bytes);

    if num_timestamps == 1 {
        return Ok(Some((first_ts, first_ts)));
    }

    file.seek(SeekFrom::Start(
        (HEADER_SIZE + (num_timestamps - 1) * 8) as u64,
    ))?;
    let mut last_ts_bytes = [0u8; 8];
    file.read_exact(&mut last_ts_bytes)?;
    let last_ts = i64::from_le_bytes(last_ts_bytes);

    Ok(Some((first_ts, last_ts)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2::types::ComponentId;
    use impeller2_wkt::ComponentMetadata;
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_index_file(path: &Path, start_ts: i64, timestamps: &[i64]) {
        let data_len = timestamps.len() * 8;
        let committed_len = (HEADER_SIZE + data_len) as u64;

        let mut data = vec![0u8; HEADER_SIZE + data_len];
        data[0..8].copy_from_slice(&committed_len.to_le_bytes());
        data[16..24].copy_from_slice(&start_ts.to_le_bytes());

        for (i, ts) in timestamps.iter().enumerate() {
            let offset = HEADER_SIZE + i * 8;
            data[offset..offset + 8].copy_from_slice(&ts.to_le_bytes());
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    fn create_data_file(path: &Path, element_size: u64, num_entries: usize) {
        let data_len = num_entries * element_size as usize;
        let committed_len = (HEADER_SIZE + data_len) as u64;

        let mut data = vec![0u8; HEADER_SIZE + data_len];
        data[0..8].copy_from_slice(&committed_len.to_le_bytes());
        data[16..24].copy_from_slice(&element_size.to_le_bytes());

        for i in 0..num_entries {
            let offset = HEADER_SIZE + i * element_size as usize;
            let val = (i as f64) * 1.0;
            data[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    fn read_index_file(path: &Path) -> (i64, Vec<i64>) {
        let mut file = File::open(path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();

        let committed_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let start_ts = i64::from_le_bytes(data[16..24].try_into().unwrap());

        let data_len = committed_len.saturating_sub(HEADER_SIZE);
        let num_timestamps = data_len / 8;

        let mut timestamps = Vec::new();
        for i in 0..num_timestamps {
            let offset = HEADER_SIZE + i * 8;
            if offset + 8 <= data.len() {
                let ts = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                timestamps.push(ts);
            }
        }

        (start_ts, timestamps)
    }

    fn create_test_db(dir: &Path, components: &[(&str, i64, &[i64])]) -> Result<(), Error> {
        fs::create_dir_all(dir)?;

        let min_start = components
            .iter()
            .map(|(_, start, _)| *start)
            .min()
            .unwrap_or(0);
        let mut config = DbConfig {
            recording: false,
            default_stream_time_step: std::time::Duration::from_millis(10),
            metadata: HashMap::new(),
        };
        config.set_time_start_timestamp_micros(min_start);
        config.write(dir.join("db_state"))?;

        for (name, start_ts, timestamps) in components {
            let component_id = ComponentId::new(name);
            let component_dir = dir.join(component_id.to_string());
            fs::create_dir_all(&component_dir)?;

            let metadata = ComponentMetadata {
                component_id,
                name: name.to_string(),
                metadata: HashMap::new(),
            };
            metadata.write(component_dir.join("metadata"))?;

            let schema_data = [10u8, 0, 0, 0, 0, 0, 0, 0, 0];
            fs::write(component_dir.join("schema"), &schema_data)?;

            create_index_file(&component_dir.join("index"), *start_ts, timestamps);
            create_data_file(&component_dir.join("data"), 8, timestamps.len());
        }

        Ok(())
    }

    fn create_test_db_with_msgs(
        dir: &Path,
        components: &[(&str, i64, &[i64])],
        msg_logs: &[(&str, &[i64], &[&[u8]])],
    ) -> Result<(), Error> {
        create_test_db(dir, components)?;

        let msgs_dir = dir.join("msgs");
        fs::create_dir_all(&msgs_dir)?;

        for (msg_id, timestamps, payloads) in msg_logs {
            let msg_dir = msgs_dir.join(msg_id);
            fs::create_dir_all(&msg_dir)?;

            let mut ts_data = Vec::new();
            for ts in *timestamps {
                ts_data.extend_from_slice(&ts.to_le_bytes());
            }
            write_msg_appendlog(&msg_dir.join("timestamps"), &ts_data)?;

            let mut offsets_data = Vec::new();
            let mut data_log_data = Vec::new();

            for payload in *payloads {
                let len = payload.len() as u32;
                let buf = if len > 12 {
                    let prefix: [u8; 4] = payload[..4].try_into().unwrap();
                    let offset = data_log_data.len() as u32;
                    data_log_data.extend_from_slice(payload);
                    UmbraBuf::with_offset(len, prefix, offset)
                } else {
                    let mut inline = [0u8; 12];
                    inline[..payload.len()].copy_from_slice(payload);
                    UmbraBuf::with_inline(len, inline)
                };
                offsets_data.extend_from_slice(zerocopy::IntoBytes::as_bytes(&buf));
            }

            write_msg_appendlog(&msg_dir.join("offsets"), &offsets_data)?;
            write_msg_appendlog(&msg_dir.join("data_log"), &data_log_data)?;
        }

        Ok(())
    }

    // --from-start 3M: remove first 3s, keep data >= 3M → [3M,4M,5M]
    #[test]
    fn test_trim_before_only() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        create_test_db(
            &db_path,
            &[(
                "rocket.velocity",
                0,
                &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            )],
        )
        .unwrap();

        run(
            db_path,
            Some(3_000_000), // before=3M: remove data before 3M
            None,
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        assert!(output_path.exists());

        let comp_id = ComponentId::new("rocket.velocity");
        let (start_ts, timestamps) =
            read_index_file(&output_path.join(comp_id.to_string()).join("index"));

        assert_eq!(timestamps, vec![3_000_000, 4_000_000, 5_000_000]);
        assert_eq!(start_ts, 3_000_000);
    }

    // --from-end 2M: remove last 2 seconds from end.
    // Data [1M..5M] with end=5M → keep <= 5M-2M = 3M → [1M, 2M, 3M]
    #[test]
    fn test_trim_after_only() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        create_test_db(
            &db_path,
            &[(
                "rocket.velocity",
                0,
                &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            )],
        )
        .unwrap();

        run(
            db_path,
            None,
            Some(2_000_000), // remove last 2 seconds
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let comp_id = ComponentId::new("rocket.velocity");
        let (_start_ts, timestamps) =
            read_index_file(&output_path.join(comp_id.to_string()).join("index"));

        assert_eq!(timestamps, vec![1_000_000, 2_000_000, 3_000_000]);
    }

    // --from-start 2M --from-end 1M: trim 2s from start and 1s from end.
    // Data [1M..5M], start=0, end=5M → keep [0+2M, 5M-1M] = [2M, 4M]
    #[test]
    fn test_trim_both() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        create_test_db(
            &db_path,
            &[(
                "rocket.velocity",
                0,
                &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            )],
        )
        .unwrap();

        run(
            db_path,
            Some(2_000_000), // remove first 2s
            Some(1_000_000), // remove last 1s
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let comp_id = ComponentId::new("rocket.velocity");
        let (_start_ts, timestamps) =
            read_index_file(&output_path.join(comp_id.to_string()).join("index"));

        assert_eq!(timestamps, vec![2_000_000, 3_000_000, 4_000_000]);
    }

    #[test]
    fn test_trim_in_place() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        create_test_db(
            &db_path,
            &[(
                "rocket.velocity",
                0,
                &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            )],
        )
        .unwrap();

        run(
            db_path.clone(),
            Some(2_000_000), // remove first 2s
            Some(1_000_000), // remove last 1s
            None,            // in-place
            false,
            true,
        )
        .unwrap();

        let comp_id = ComponentId::new("rocket.velocity");
        let (_start_ts, timestamps) =
            read_index_file(&db_path.join(comp_id.to_string()).join("index"));

        assert_eq!(timestamps, vec![2_000_000, 3_000_000, 4_000_000]);
    }

    #[test]
    fn test_trim_msg_logs() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        let small_msg: &[u8] = b"hello";
        let large_msg: &[u8] = b"this is a longer message payload!";
        create_test_db_with_msgs(
            &db_path,
            &[("comp", 0, &[1_000_000, 2_000_000, 3_000_000])],
            &[(
                "42",
                &[1_000_000, 2_000_000, 3_000_000],
                &[small_msg, large_msg, small_msg],
            )],
        )
        .unwrap();

        // Data range: 1M-3M (3s duration, start=0, end=3M)
        // --from-start 1.5M → keep >= 1.5M, --from-end 0.5M → keep <= 3M-0.5M = 2.5M
        // Keeps only 2M
        run(
            db_path,
            Some(1_500_000),
            Some(500_000),
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let msg_timestamps = read_msg_timestamps(&output_path.join("msgs/42/timestamps")).unwrap();
        assert_eq!(msg_timestamps, vec![2_000_000]);
    }

    #[test]
    fn test_trim_updates_db_state() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        create_test_db(
            &db_path,
            &[(
                "rocket.velocity",
                0,
                &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            )],
        )
        .unwrap();

        // --from-start 2M → keep >= 2M, --from-end 1M → keep <= 5M-1M = 4M
        // Keeps [2M, 3M, 4M], earliest = 2M
        run(
            db_path,
            Some(2_000_000),
            Some(1_000_000),
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let config = DbConfig::read(output_path.join("db_state")).unwrap();
        assert_eq!(config.time_start_timestamp_micros(), Some(2_000_000));
    }

    #[test]
    fn test_trim_dry_run() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        create_test_db(&db_path, &[("test", 0, &[1_000_000, 2_000_000, 3_000_000])]).unwrap();

        run(
            db_path,
            None,
            Some(1_000_000), // remove last 1s
            Some(output_path.clone()),
            true, // dry_run
            true,
        )
        .unwrap();

        assert!(!output_path.exists());
    }

    #[test]
    fn test_trim_no_bounds_error() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        create_test_db(&db_path, &[("test", 0, &[1_000_000])]).unwrap();

        let result = run(db_path, None, None, None, false, true);
        assert!(result.is_err());
    }

    // Trim amounts that exceed the recording duration should error
    #[test]
    fn test_trim_exceeds_duration_error() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // 4s duration (1M to 5M, start=0)
        create_test_db(
            &db_path,
            &[(
                "test",
                0,
                &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000],
            )],
        )
        .unwrap();

        // --from-start 3M + --from-end 3M = trim 6s from a 5s recording → error
        let result = run(db_path, Some(3_000_000), Some(3_000_000), None, false, true);
        assert!(result.is_err());
    }

    // --from-start with non-zero base: remove first 2s of a recording starting at 100s
    #[test]
    fn test_trim_before_relative_to_recording_start() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        let base = 100_000_000i64;
        create_test_db(
            &db_path,
            &[(
                "sensor.data",
                base,
                &[
                    base + 0,
                    base + 1_000_000,
                    base + 2_000_000,
                    base + 3_000_000,
                    base + 4_000_000,
                    base + 5_000_000,
                ],
            )],
        )
        .unwrap();

        run(
            db_path,
            Some(2_000_000),
            None,
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let comp_id = ComponentId::new("sensor.data");
        let (_start_ts, timestamps) =
            read_index_file(&output_path.join(comp_id.to_string()).join("index"));

        assert_eq!(
            timestamps,
            vec![
                base + 2_000_000,
                base + 3_000_000,
                base + 4_000_000,
                base + 5_000_000,
            ]
        );
    }

    // --from-end with non-zero base: remove last 2s from a recording starting at 100s
    // Data: base..base+5M, end = base+5M → keep <= (base+5M) - 2M = base+3M
    #[test]
    fn test_trim_after_relative_to_recording_end() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        let base = 100_000_000i64;
        create_test_db(
            &db_path,
            &[(
                "sensor.data",
                base,
                &[
                    base + 0,
                    base + 1_000_000,
                    base + 2_000_000,
                    base + 3_000_000,
                    base + 4_000_000,
                    base + 5_000_000,
                ],
            )],
        )
        .unwrap();

        // --from-end 2M: remove last 2 seconds
        run(
            db_path,
            None,
            Some(2_000_000),
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let comp_id = ComponentId::new("sensor.data");
        let (_start_ts, timestamps) =
            read_index_file(&output_path.join(comp_id.to_string()).join("index"));

        assert_eq!(
            timestamps,
            vec![
                base + 0,
                base + 1_000_000,
                base + 2_000_000,
                base + 3_000_000,
            ]
        );
    }

    // --from-end with a value exceeding the recording duration should produce
    // an empty database (not incorrectly keep the first data point).
    #[test]
    fn test_trim_from_end_exceeds_duration_gives_empty() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        let output_path = temp.path().join("trimmed");

        create_test_db(
            &db_path,
            &[("sensor.data", 0, &[1_000_000, 2_000_000, 3_000_000])],
        )
        .unwrap();

        // --from-end 10M on a 3s recording (end=3M): keep <= 3M - 10M = -7M
        // All data is above -7M so this should keep everything... but the
        // resolved abs_end is negative, which is below all timestamps, so
        // the result should be empty.
        run(
            db_path,
            None,
            Some(10_000_000),
            Some(output_path.clone()),
            false,
            true,
        )
        .unwrap();

        let comp_id = ComponentId::new("sensor.data");
        let (_start_ts, timestamps) =
            read_index_file(&output_path.join(comp_id.to_string()).join("index"));

        assert_eq!(timestamps, Vec::<i64>::new());
    }
}
