//! Database time alignment tool
//!
//! Shifts all timestamps in specified components to align their first timestamp
//! with a user-provided target timestamp. This solves the problem where components
//! written at the same real-world moment have different timestamp offsets.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};

use crate::utils::component_label;
use crate::{Error, MetadataExt};
use impeller2_wkt::ComponentMetadata;

const HEADER_SIZE: usize = 24; // committed_len (8) + head_len (8) + start_timestamp (8)

/// Information about a component for time alignment
#[derive(Debug)]
struct ComponentInfo {
    path: PathBuf,
    name: String,
    first_timestamp: i64,
    entry_count: usize,
}

/// Time-align components in an elodin-db database.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `target_timestamp` - Target timestamp (seconds) to align first sample to
/// * `align_all` - If true, align all components
/// * `component_name` - If provided, align only this specific component
/// * `dry_run` - If true, only show what would be changed without modifying
/// * `auto_confirm` - If true, skip the confirmation prompt
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(Error)` if the operation fails
pub fn run(
    db_path: PathBuf,
    target_timestamp: f64,
    align_all: bool,
    component_name: Option<String>,
    dry_run: bool,
    auto_confirm: bool,
) -> Result<(), Error> {
    // Validate arguments - must specify exactly one of --all or --component
    if align_all && component_name.is_some() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Cannot specify both --all and --component",
        )));
    }
    if !align_all && component_name.is_none() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Must specify either --all or --component",
        )));
    }

    if !target_timestamp.is_finite() {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "--timestamp must be a finite number, got: {}",
                target_timestamp
            ),
        )));
    }

    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }

    let db_state_path = db_path.join("db_state");
    if !db_state_path.exists() {
        return Err(Error::MissingDbState(db_state_path));
    }

    println!("Analyzing database: {}", db_path.display());
    if dry_run {
        println!("DRY RUN - no changes will be made");
    }
    println!();

    // Convert target timestamp to microseconds
    let target_timestamp_micros = (target_timestamp * 1_000_000.0) as i64;

    // Collect component information
    let components = collect_components(&db_path, component_name.as_deref())?;

    if components.is_empty() {
        if let Some(name) = &component_name {
            println!("Component \"{}\" not found in database.", name);
        } else {
            println!("No components found in database.");
        }
        return Ok(());
    }

    // Filter out empty components
    let components_to_align: Vec<_> = components
        .into_iter()
        .filter(|c| c.entry_count > 0)
        .collect();

    if components_to_align.is_empty() {
        println!("No components with data to align.");
        return Ok(());
    }

    // Display alignment plan
    println!(
        "Target timestamp: {:.3}s ({} us)",
        target_timestamp, target_timestamp_micros
    );
    println!();
    println!("Components to align:");

    for component in &components_to_align {
        let offset_micros = target_timestamp_micros - component.first_timestamp;
        let offset_secs = offset_micros as f64 / 1_000_000.0;
        let first_ts_secs = component.first_timestamp as f64 / 1_000_000.0;
        let new_first_ts_secs = first_ts_secs + offset_secs;

        println!(
            "  {}: {:.3}s -> {:.3}s (offset: {:+.3}s, {} entries)",
            component.name, first_ts_secs, new_first_ts_secs, offset_secs, component.entry_count
        );
    }
    println!();

    if dry_run {
        println!("Dry run complete. Run without --dry-run to apply changes.");
        return Ok(());
    }

    // Confirm
    if !auto_confirm {
        print!(
            "Align {} component{}? [y/N] ",
            components_to_align.len(),
            if components_to_align.len() == 1 {
                ""
            } else {
                "s"
            }
        );
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    println!("Aligning components...");

    let mut aligned_count = 0;
    for component in &components_to_align {
        let offset_micros = target_timestamp_micros - component.first_timestamp;

        match apply_time_shift(&component.path.join("index"), offset_micros) {
            Ok(()) => {
                println!(
                    "  Aligned: {} ({} entries)",
                    component.name, component.entry_count
                );
                aligned_count += 1;
            }
            Err(e) => {
                eprintln!("  Error aligning {}: {}", component.name, e);
            }
        }
    }

    println!();
    println!(
        "Done! Aligned {} component{}.",
        aligned_count,
        if aligned_count == 1 { "" } else { "s" }
    );

    Ok(())
}

/// Collect components from the database, optionally filtering by name.
///
/// If `component_name` is `Some`, returns only the matching component (if found).
/// If `component_name` is `None`, returns all components.
fn collect_components(
    db_path: &Path,
    component_name: Option<&str>,
) -> Result<Vec<ComponentInfo>, Error> {
    let mut components = Vec::new();

    for entry in fs::read_dir(db_path)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        // Skip non-component directories
        let dir_name = path.file_name().unwrap().to_string_lossy();
        if dir_name == "msgs" || dir_name == "db_state" || dir_name == "schematic" {
            continue;
        }

        // Check if it's a valid component (has index file)
        let index_path = path.join("index");
        if !index_path.exists() {
            continue;
        }

        // Get component name from metadata
        let metadata_path = path.join("metadata");
        let name = if metadata_path.exists() {
            match ComponentMetadata::read(&metadata_path) {
                Ok(metadata) => metadata.name,
                Err(_) => component_label(&path),
            }
        } else {
            component_label(&path)
        };

        // If filtering by name, skip non-matching components
        if let Some(target_name) = component_name
            && name != target_name
        {
            continue;
        }

        // Analyze the component
        match analyze_component(&index_path) {
            Ok((first_timestamp, entry_count)) => {
                components.push(ComponentInfo {
                    path,
                    name,
                    first_timestamp,
                    entry_count,
                });
            }
            Err(e) => {
                eprintln!("Warning: Failed to analyze {}: {}", path.display(), e);
            }
        }

        // If we're looking for a specific component and found it, we're done
        if component_name.is_some() && !components.is_empty() {
            break;
        }
    }

    Ok(components)
}

/// Analyze a component's index file to get the first timestamp and entry count.
fn analyze_component(index_path: &Path) -> Result<(i64, usize), std::io::Error> {
    let mut file = File::open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let data_len = committed_len.saturating_sub(HEADER_SIZE);
    let entry_count = data_len / 8; // Each timestamp is 8 bytes

    if entry_count == 0 {
        return Ok((i64::MAX, 0));
    }

    // Read first timestamp (it's the minimum since timestamps are ordered)
    let mut first_ts_bytes = [0u8; 8];
    file.read_exact(&mut first_ts_bytes)?;
    let first_timestamp = i64::from_le_bytes(first_ts_bytes);

    Ok((first_timestamp, entry_count))
}

/// Apply a time shift to an index file by adding offset to all timestamps.
///
/// This modifies the index file in-place:
/// 1. Updates the start_timestamp in the header
/// 2. Applies offset to all timestamp entries
fn apply_time_shift(index_path: &Path, offset: i64) -> Result<(), Error> {
    // If offset is zero, nothing to do
    if offset == 0 {
        return Ok(());
    }

    let mut file = OpenOptions::new().read(true).write(true).open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let data_len = committed_len.saturating_sub(HEADER_SIZE);

    // Validate data_len is multiple of 8
    if !data_len.is_multiple_of(8) {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Malformed index file: data length {} is not a multiple of 8 bytes",
                data_len
            ),
        )));
    }

    // Update start_timestamp in header (bytes 16-24)
    let start_ts = i64::from_le_bytes(header[16..24].try_into().unwrap());
    let new_start_ts = start_ts.saturating_add(offset);
    header[16..24].copy_from_slice(&new_start_ts.to_le_bytes());

    // Read all timestamps
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    // Apply offset to all timestamps
    for chunk in data.chunks_exact_mut(8) {
        let ts = i64::from_le_bytes(chunk.try_into().unwrap());
        let new_ts = ts.saturating_add(offset);
        chunk.copy_from_slice(&new_ts.to_le_bytes());
    }

    // Write back
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    file.sync_all()?;

    // Sync parent directory
    if let Some(parent) = index_path.parent() {
        crate::sync_dir(parent)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Helper to create an index file with timestamps
    fn create_index_file(path: &Path, start_ts: i64, timestamps: &[i64]) {
        let data_len = timestamps.len() * 8;
        let committed_len = (HEADER_SIZE + data_len) as u64;

        let mut data = vec![0u8; HEADER_SIZE + data_len];

        // Header: committed_len (8) + head_len (8) + start_timestamp (8)
        data[0..8].copy_from_slice(&committed_len.to_le_bytes());
        data[8..16].copy_from_slice(&0u64.to_le_bytes()); // head_len = 0
        data[16..24].copy_from_slice(&start_ts.to_le_bytes());

        // Timestamps
        for (i, ts) in timestamps.iter().enumerate() {
            let offset = HEADER_SIZE + i * 8;
            data[offset..offset + 8].copy_from_slice(&ts.to_le_bytes());
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    /// Helper to read timestamps from an index file
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

    #[test]
    fn test_analyze_component() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("index");

        // Create index with timestamps at 10s, 20s, 30s
        create_index_file(
            &index_path,
            10_000_000,
            &[10_000_000, 20_000_000, 30_000_000],
        );

        let (first_ts, count) = analyze_component(&index_path).unwrap();
        assert_eq!(first_ts, 10_000_000); // 10s in microseconds
        assert_eq!(count, 3);
    }

    #[test]
    fn test_analyze_empty_component() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("index");

        // Create empty index
        create_index_file(&index_path, 0, &[]);

        let (first_ts, count) = analyze_component(&index_path).unwrap();
        assert_eq!(first_ts, i64::MAX);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_apply_time_shift_positive() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("index");

        // Create index with timestamps starting at 10s
        let start_ts = 10_000_000i64;
        let timestamps = vec![10_000_000i64, 20_000_000, 30_000_000];
        create_index_file(&index_path, start_ts, &timestamps);

        // Shift forward by 5s
        let offset = 5_000_000i64;
        apply_time_shift(&index_path, offset).unwrap();

        // Verify
        let (new_start_ts, new_timestamps) = read_index_file(&index_path);
        assert_eq!(new_start_ts, 15_000_000); // 10s + 5s
        assert_eq!(new_timestamps, vec![15_000_000, 25_000_000, 35_000_000]);
    }

    #[test]
    fn test_apply_time_shift_negative() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("index");

        // Create index with timestamps starting at 15s
        let start_ts = 15_000_000i64;
        let timestamps = vec![15_000_000i64, 25_000_000, 35_000_000];
        create_index_file(&index_path, start_ts, &timestamps);

        // Shift backward by 15s (to align to 0)
        let offset = -15_000_000i64;
        apply_time_shift(&index_path, offset).unwrap();

        // Verify
        let (new_start_ts, new_timestamps) = read_index_file(&index_path);
        assert_eq!(new_start_ts, 0);
        assert_eq!(new_timestamps, vec![0, 10_000_000, 20_000_000]);
    }

    #[test]
    fn test_apply_time_shift_zero() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("index");

        // Create index
        let start_ts = 10_000_000i64;
        let timestamps = vec![10_000_000i64, 20_000_000, 30_000_000];
        create_index_file(&index_path, start_ts, &timestamps);

        // Apply zero offset (no-op)
        apply_time_shift(&index_path, 0).unwrap();

        // Verify unchanged
        let (new_start_ts, new_timestamps) = read_index_file(&index_path);
        assert_eq!(new_start_ts, start_ts);
        assert_eq!(new_timestamps, timestamps);
    }

    #[test]
    fn test_apply_time_shift_uses_saturating_add() {
        let temp = TempDir::new().unwrap();
        let index_path = temp.path().join("index");

        // Create index with max timestamp
        let start_ts = i64::MAX - 100;
        let timestamps = vec![i64::MAX - 100, i64::MAX - 50];
        create_index_file(&index_path, start_ts, &timestamps);

        // Apply large positive offset (should saturate, not overflow)
        let offset = 1000i64;
        apply_time_shift(&index_path, offset).unwrap();

        // Verify saturated to MAX
        let (new_start_ts, new_timestamps) = read_index_file(&index_path);
        assert_eq!(new_start_ts, i64::MAX);
        assert_eq!(new_timestamps, vec![i64::MAX, i64::MAX]);
    }

    /// Helper to create a minimal test database with components
    fn create_test_db(
        dir: &Path,
        components: &[(&str, i64, &[i64])], // (name, start_ts, timestamps)
    ) -> Result<(), Error> {
        use impeller2::types::ComponentId;
        use impeller2_wkt::DbConfig;

        fs::create_dir_all(dir)?;

        // Create db_state
        let config = DbConfig {
            recording: false,
            default_stream_time_step: std::time::Duration::from_millis(10),
            metadata: HashMap::new(),
        };
        config.write(dir.join("db_state"))?;

        // Create components
        for (name, start_ts, timestamps) in components {
            let component_id = ComponentId::new(name);
            let component_dir = dir.join(component_id.to_string());
            fs::create_dir_all(&component_dir)?;

            // Write metadata
            let metadata = ComponentMetadata {
                component_id,
                name: name.to_string(),
                metadata: HashMap::new(),
            };
            metadata.write(component_dir.join("metadata"))?;

            // Write minimal schema
            let schema_data = [10u8, 0, 0, 0, 0, 0, 0, 0, 0];
            fs::write(component_dir.join("schema"), &schema_data)?;

            // Write index with timestamps
            create_index_file(&component_dir.join("index"), *start_ts, timestamps);

            // Write data file with matching size
            let element_size = 8u64;
            let data_len = timestamps.len() * 8;
            let committed_len = (HEADER_SIZE + data_len) as u64;
            let mut data_file = vec![0u8; HEADER_SIZE + data_len];
            data_file[0..8].copy_from_slice(&committed_len.to_le_bytes());
            data_file[16..24].copy_from_slice(&element_size.to_le_bytes());
            fs::write(component_dir.join("data"), &data_file)?;
        }

        Ok(())
    }

    #[test]
    fn test_run_align_all() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database with two components at different start times
        create_test_db(
            &db_path,
            &[
                (
                    "comp1",
                    10_000_000,
                    &[10_000_000i64, 20_000_000, 30_000_000],
                ),
                (
                    "comp2",
                    15_000_000,
                    &[15_000_000i64, 25_000_000, 35_000_000],
                ),
            ],
        )
        .unwrap();

        // Align all to t=0
        run(db_path.clone(), 0.0, true, None, false, true).unwrap();

        // Verify comp1 shifted by -10s
        let comp1_id = impeller2::types::ComponentId::new("comp1");
        let (start_ts1, timestamps1) =
            read_index_file(&db_path.join(comp1_id.to_string()).join("index"));
        assert_eq!(start_ts1, 0);
        assert_eq!(timestamps1, vec![0, 10_000_000, 20_000_000]);

        // Verify comp2 shifted by -15s
        let comp2_id = impeller2::types::ComponentId::new("comp2");
        let (start_ts2, timestamps2) =
            read_index_file(&db_path.join(comp2_id.to_string()).join("index"));
        assert_eq!(start_ts2, 0);
        assert_eq!(timestamps2, vec![0, 10_000_000, 20_000_000]);
    }

    #[test]
    fn test_run_align_single_component() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database with two components
        create_test_db(
            &db_path,
            &[
                (
                    "comp1",
                    10_000_000,
                    &[10_000_000i64, 20_000_000, 30_000_000],
                ),
                (
                    "comp2",
                    15_000_000,
                    &[15_000_000i64, 25_000_000, 35_000_000],
                ),
            ],
        )
        .unwrap();

        // Align only comp1 to t=0
        run(
            db_path.clone(),
            0.0,
            false,
            Some("comp1".to_string()),
            false,
            true,
        )
        .unwrap();

        // Verify comp1 was shifted
        let comp1_id = impeller2::types::ComponentId::new("comp1");
        let (start_ts1, timestamps1) =
            read_index_file(&db_path.join(comp1_id.to_string()).join("index"));
        assert_eq!(start_ts1, 0);
        assert_eq!(timestamps1, vec![0, 10_000_000, 20_000_000]);

        // Verify comp2 was NOT shifted
        let comp2_id = impeller2::types::ComponentId::new("comp2");
        let (start_ts2, timestamps2) =
            read_index_file(&db_path.join(comp2_id.to_string()).join("index"));
        assert_eq!(start_ts2, 15_000_000);
        assert_eq!(timestamps2, vec![15_000_000, 25_000_000, 35_000_000]);
    }

    #[test]
    fn test_run_validation_requires_all_or_component() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        create_test_db(&db_path, &[]).unwrap();

        // Should fail without --all or --component
        let result = run(db_path, 0.0, false, None, false, true);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("--all") || err_msg.contains("--component"));
    }

    #[test]
    fn test_run_validation_rejects_both_all_and_component() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        create_test_db(&db_path, &[]).unwrap();

        // Should fail with both --all and --component
        let result = run(db_path, 0.0, true, Some("comp1".to_string()), false, true);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Cannot specify both"));
    }

    #[test]
    fn test_run_validation_rejects_nan() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        create_test_db(&db_path, &[]).unwrap();

        let result = run(db_path, f64::NAN, true, None, false, true);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("finite"));
    }

    #[test]
    fn test_run_validation_rejects_infinity() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");
        create_test_db(&db_path, &[]).unwrap();

        let result = run(db_path, f64::INFINITY, true, None, false, true);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("finite"));
    }

    #[test]
    fn test_dry_run_does_not_modify() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database
        create_test_db(
            &db_path,
            &[(
                "comp1",
                10_000_000,
                &[10_000_000i64, 20_000_000, 30_000_000],
            )],
        )
        .unwrap();

        // Dry run
        run(db_path.clone(), 0.0, true, None, true, true).unwrap();

        // Verify not modified
        let comp1_id = impeller2::types::ComponentId::new("comp1");
        let (start_ts, timestamps) =
            read_index_file(&db_path.join(comp1_id.to_string()).join("index"));
        assert_eq!(start_ts, 10_000_000);
        assert_eq!(timestamps, vec![10_000_000, 20_000_000, 30_000_000]);
    }
}
