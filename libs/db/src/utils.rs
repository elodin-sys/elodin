//! Shared utility functions for database operations.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::MetadataExt;
use impeller2::types::PrimType;
use impeller2_wkt::{ComponentMetadata, DbConfig};

/// Read a primitive value from a byte slice at the given offset and convert it to f64.
///
/// This is useful for converting time series data of various primitive types
/// into a common f64 format for visualization and downsampling.
///
/// # Arguments
/// * `prim_type` - The primitive type of the value
/// * `data` - The byte slice containing the data
/// * `offset` - The byte offset to read from
///
/// # Returns
/// The value as f64, or f64::NAN if the offset is out of bounds
pub fn read_prim_as_f64(prim_type: PrimType, data: &[u8], offset: usize) -> f64 {
    match prim_type {
        PrimType::F64 => {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]);
            f64::from_le_bytes(bytes)
        }
        PrimType::F32 => {
            let bytes: [u8; 4] = data
                .get(offset..offset + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 4]);
            f32::from_le_bytes(bytes) as f64
        }
        PrimType::U64 => {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]);
            u64::from_le_bytes(bytes) as f64
        }
        PrimType::U32 => {
            let bytes: [u8; 4] = data
                .get(offset..offset + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 4]);
            u32::from_le_bytes(bytes) as f64
        }
        PrimType::U16 => {
            let bytes: [u8; 2] = data
                .get(offset..offset + 2)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 2]);
            u16::from_le_bytes(bytes) as f64
        }
        PrimType::U8 => data.get(offset).copied().unwrap_or(0) as f64,
        PrimType::I64 => {
            let bytes: [u8; 8] = data
                .get(offset..offset + 8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]);
            i64::from_le_bytes(bytes) as f64
        }
        PrimType::I32 => {
            let bytes: [u8; 4] = data
                .get(offset..offset + 4)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 4]);
            i32::from_le_bytes(bytes) as f64
        }
        PrimType::I16 => {
            let bytes: [u8; 2] = data
                .get(offset..offset + 2)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 2]);
            i16::from_le_bytes(bytes) as f64
        }
        PrimType::I8 => data.get(offset).copied().unwrap_or(0) as i8 as f64,
        PrimType::Bool => {
            if data.get(offset).copied().unwrap_or(0) != 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Get a human-readable label for a component path.
///
/// Returns the component name from metadata if available, formatted as
/// "name (id)" if they differ, or just the directory name as a fallback.
pub fn component_label(path: &Path) -> String {
    let fallback = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>")
        .to_string();

    let metadata_path = path.join("metadata");
    let name = if metadata_path.exists() {
        read_component_name(&metadata_path).unwrap_or_else(|_| fallback.clone())
    } else {
        fallback.clone()
    };

    if name == fallback {
        name
    } else {
        format!("{name} ({fallback})")
    }
}

/// Read a component's name from its metadata file.
pub fn read_component_name(metadata_path: &Path) -> Result<String, std::io::Error> {
    let metadata = <ComponentMetadata as MetadataExt>::read(metadata_path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(metadata.name)
}

/// Check if a component directory contains a timestamp-source component.
///
/// Timestamp-source components store raw clock values used as timestamps for other
/// components, and should be excluded from time range calculations to avoid
/// wildly inflated duration estimates.
///
/// Returns `true` if the component's metadata has `is_timestamp_source` set,
/// `false` if metadata is missing or the flag is not set.
pub(crate) fn is_timestamp_source_component(component_path: &Path) -> bool {
    component_path
        .join("metadata")
        .exists()
        .then(|| ComponentMetadata::read(component_path.join("metadata")).ok())
        .flatten()
        .is_some_and(|m| m.is_timestamp_source())
}

/// Result of resolving a database's playback start timestamp.
#[derive(Debug, Clone, Copy)]
pub enum PlaybackStart {
    /// Start timestamp was found in the database config.
    FromConfig(i64),
    /// Start timestamp was derived from the earliest data in the database.
    FromData(i64),
}

impl PlaybackStart {
    /// Returns the timestamp value regardless of source.
    pub fn timestamp(self) -> i64 {
        match self {
            PlaybackStart::FromConfig(ts) => ts,
            PlaybackStart::FromData(ts) => ts,
        }
    }

    /// Returns true if the start was derived from data rather than config.
    pub fn is_from_data(self) -> bool {
        matches!(self, PlaybackStart::FromData(_))
    }
}

/// Resolve the playback start timestamp for a database.
///
/// First attempts to read `time_start_timestamp_micros` from the database config.
/// If not set, falls back to scanning the actual data for the earliest timestamp.
///
/// Returns `None` if the database has no config and no data to derive from.
pub fn resolve_playback_start(db_path: &Path) -> Result<Option<PlaybackStart>, std::io::Error> {
    let db_state_path = db_path.join("db_state");

    if db_state_path.exists() {
        let config = DbConfig::read(&db_state_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if let Some(ts) = config.time_start_timestamp_micros() {
            return Ok(Some(PlaybackStart::FromConfig(ts)));
        }
    }

    match derive_playback_start_from_data(db_path)? {
        Some(ts) => Ok(Some(PlaybackStart::FromData(ts))),
        None => Ok(None),
    }
}

/// Scan the database data to find the earliest timestamp.
///
/// This excludes timestamp-source components (which store raw clock values)
/// and scans both component indices and message logs.
fn derive_playback_start_from_data(db_path: &Path) -> Result<Option<i64>, std::io::Error> {
    let mut min_timestamp = i64::MAX;

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
                    let msg_path = msg_entry.path();
                    if msg_path.is_dir()
                        && let Ok(Some((start, _))) =
                            read_msg_timestamp_range(&msg_path.join("timestamps"))
                    {
                        min_timestamp = min_timestamp.min(start);
                    }
                }
            }
            continue;
        }

        if dir_name.parse::<u64>().is_err() {
            continue;
        }

        if path.join("schema").exists()
            && !is_timestamp_source_component(&path)
            && let Ok(Some((start, _))) = read_timestamp_range(&path.join("index"))
        {
            min_timestamp = min_timestamp.min(start);
        }
    }

    if min_timestamp == i64::MAX {
        Ok(None)
    } else {
        Ok(Some(min_timestamp))
    }
}

/// Header size for component index files (AppendLog with i64 start_timestamp).
/// Layout: committed_len (8) + head_len (8) + start_timestamp (8) = 24 bytes.
pub(crate) const INDEX_HEADER_SIZE: usize = 24;

/// Header size for message log files (AppendLog with () extra field).
/// Layout: committed_len (8) + head_len (8) = 16 bytes.
pub(crate) const MSG_HEADER_SIZE: usize = 16;

/// Read the first and last timestamp from a component index file (24-byte header).
///
/// Returns `Ok(None)` if the file doesn't exist, is truncated, or contains no timestamps.
pub(crate) fn read_timestamp_range(
    index_path: &Path,
) -> Result<Option<(i64, i64)>, std::io::Error> {
    if !index_path.exists() {
        return Ok(None);
    }

    let mut file = File::open(index_path)?;
    let mut header = [0u8; INDEX_HEADER_SIZE];
    let bytes_read = file.read(&mut header)?;
    if bytes_read < INDEX_HEADER_SIZE {
        return Ok(None);
    }

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let committed_len = committed_len.max(INDEX_HEADER_SIZE);
    let data_len = committed_len.saturating_sub(INDEX_HEADER_SIZE);
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
        (INDEX_HEADER_SIZE + (num_timestamps - 1) * 8) as u64,
    ))?;
    let mut last_ts_bytes = [0u8; 8];
    file.read_exact(&mut last_ts_bytes)?;
    let last_ts = i64::from_le_bytes(last_ts_bytes);

    Ok(Some((first_ts, last_ts)))
}

/// Read the first and last timestamp from a message log timestamps file (16-byte header).
///
/// Returns `Ok(None)` if the file doesn't exist, is truncated, or contains no timestamps.
pub(crate) fn read_msg_timestamp_range(
    timestamps_path: &Path,
) -> Result<Option<(i64, i64)>, std::io::Error> {
    if !timestamps_path.exists() {
        return Ok(None);
    }

    let mut file = File::open(timestamps_path)?;
    let mut header = [0u8; MSG_HEADER_SIZE];
    let bytes_read = file.read(&mut header)?;
    if bytes_read < MSG_HEADER_SIZE {
        return Ok(None);
    }

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let committed_len = committed_len.max(MSG_HEADER_SIZE);
    let data_len = committed_len.saturating_sub(MSG_HEADER_SIZE);
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
        (MSG_HEADER_SIZE + (num_timestamps - 1) * 8) as u64,
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
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_index(dir: &Path, timestamps: &[i64]) {
        let index_path = dir.join("index");
        let mut file = File::create(&index_path).unwrap();

        let data_len = timestamps.len() * 8;
        let committed_len = INDEX_HEADER_SIZE + data_len;

        let mut header = [0u8; INDEX_HEADER_SIZE];
        header[0..8].copy_from_slice(&(committed_len as u64).to_le_bytes());
        file.write_all(&header).unwrap();

        for ts in timestamps {
            file.write_all(&ts.to_le_bytes()).unwrap();
        }
    }

    fn create_test_component(db_dir: &Path, id: u64, timestamps: &[i64], is_ts_source: bool) {
        let comp_dir = db_dir.join(id.to_string());
        fs::create_dir_all(&comp_dir).unwrap();

        fs::write(comp_dir.join("schema"), b"test").unwrap();

        create_test_index(&comp_dir, timestamps);

        if is_ts_source {
            let mut metadata = ComponentMetadata {
                component_id: ComponentId::new(&format!("component{id}")),
                name: format!("Component{id}"),
                metadata: HashMap::new(),
            };
            metadata.set_timestamp_source(true);
            metadata.write(comp_dir.join("metadata")).unwrap();
        }
    }

    fn create_db_config(start_ts: Option<i64>) -> DbConfig {
        let mut config = DbConfig {
            recording: false,
            default_stream_time_step: std::time::Duration::from_millis(10),
            metadata: HashMap::new(),
        };
        if let Some(ts) = start_ts {
            config.set_time_start_timestamp_micros(ts);
        }
        config
    }

    #[test]
    fn test_resolve_playback_start_from_config() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path();

        let config = create_db_config(Some(1_000_000));
        config.write(db_path.join("db_state")).unwrap();

        create_test_component(db_path, 1, &[2_000_000, 3_000_000], false);

        let result = resolve_playback_start(db_path).unwrap().unwrap();
        assert!(matches!(result, PlaybackStart::FromConfig(1_000_000)));
        assert_eq!(result.timestamp(), 1_000_000);
        assert!(!result.is_from_data());
    }

    #[test]
    fn test_resolve_playback_start_from_data() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path();

        let config = create_db_config(None);
        config.write(db_path.join("db_state")).unwrap();

        create_test_component(db_path, 1, &[2_000_000, 3_000_000], false);
        create_test_component(db_path, 2, &[1_500_000, 2_500_000], false);

        let result = resolve_playback_start(db_path).unwrap().unwrap();
        assert!(matches!(result, PlaybackStart::FromData(1_500_000)));
        assert_eq!(result.timestamp(), 1_500_000);
        assert!(result.is_from_data());
    }

    #[test]
    fn test_resolve_playback_start_ignores_ts_source() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path();

        let config = create_db_config(None);
        config.write(db_path.join("db_state")).unwrap();

        create_test_component(db_path, 1, &[100, 200], true);
        create_test_component(db_path, 2, &[2_000_000, 3_000_000], false);

        let result = resolve_playback_start(db_path).unwrap().unwrap();
        assert!(matches!(result, PlaybackStart::FromData(2_000_000)));
    }

    #[test]
    fn test_resolve_playback_start_no_data() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path();

        let config = create_db_config(None);
        config.write(db_path.join("db_state")).unwrap();

        let result = resolve_playback_start(db_path).unwrap();
        assert!(result.is_none());
    }
}
