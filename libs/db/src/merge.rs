//! Database merge tool
//!
//! Merges two Elodin databases into one, applying optional prefixes to component names
//! to avoid namespace collisions. This enables viewing simulation and real-world
//! telemetry data simultaneously in the Elodin Editor.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};

use crate::{Error, MetadataExt, copy_dir_native, copy_file_native, sync_dir};
use impeller2::types::ComponentId;
use impeller2_wkt::{ComponentMetadata, DbConfig};

/// Statistics about a merge operation
#[derive(Default, Debug)]
pub struct MergeStats {
    pub components_copied: usize,
    pub msg_logs_copied: usize,
    pub msg_log_conflicts: Vec<String>,
}

/// Information about a source database
struct DatabaseInfo {
    path: PathBuf,
    prefix: Option<String>,
    component_count: usize,
    msg_log_count: usize,
    time_range: Option<(i64, i64)>,
    has_schematic: bool,
}

/// Merge two databases into one with optional prefixes.
///
/// # Arguments
/// * `db1` - Path to the first source database
/// * `db2` - Path to the second source database
/// * `output` - Path for the merged output database
/// * `prefix1` - Optional prefix for first database components
/// * `prefix2` - Optional prefix for second database components
/// * `dry_run` - If true, only show what would be done without creating output
/// * `auto_confirm` - If true, skip the confirmation prompt
/// * `align1` - Optional alignment timestamp (seconds) for an event in DB1
/// * `align2` - Optional alignment timestamp (seconds) for the same event in DB2.
///   The database with the earlier anchor is shifted forward to align with the later one.
///   This ensures timestamps never go negative (important for monotonic datasets).
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(Error)` if the operation fails
#[allow(clippy::too_many_arguments)]
pub fn run(
    db1: PathBuf,
    db2: PathBuf,
    output: PathBuf,
    prefix1: Option<String>,
    prefix2: Option<String>,
    dry_run: bool,
    auto_confirm: bool,
    align1: Option<f64>,
    align2: Option<f64>,
) -> Result<(), Error> {
    // Validate alignment arguments: both must be provided or neither
    // Returns (db1_offset, db2_offset) - one will be 0, the other will be positive
    // We always shift forward (never backward past 0) to avoid negative timestamps
    let (db1_offset, db2_offset) = match (align1, align2) {
        (Some(a1), Some(a2)) => {
            // Validate alignment values are finite numbers
            if !a1.is_finite() {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("--align1 must be a finite number, got: {}", a1),
                )));
            }
            if !a2.is_finite() {
                return Err(Error::Io(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("--align2 must be a finite number, got: {}", a2),
                )));
            }

            // Convert from seconds to microseconds
            let a1_micros = (a1 * 1_000_000.0) as i64;
            let a2_micros = (a2 * 1_000_000.0) as i64;

            // Always shift forward: the dataset with the earlier anchor gets shifted
            // to align with the later anchor
            if a1_micros >= a2_micros {
                // DB1's anchor is later or equal - shift DB2 forward to match
                (0i64, a1_micros - a2_micros)
            } else {
                // DB2's anchor is later - shift DB1 forward to match
                (a2_micros - a1_micros, 0i64)
            }
        }
        (None, None) => (0, 0),
        _ => {
            return Err(Error::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Both --align1 and --align2 must be provided together, or neither",
            )));
        }
    };
    // Validate source databases exist
    if !db1.exists() {
        return Err(Error::MissingDbState(db1));
    }
    if !db2.exists() {
        return Err(Error::MissingDbState(db2));
    }

    // Validate db_state files exist
    let db1_state_path = db1.join("db_state");
    let db2_state_path = db2.join("db_state");
    if !db1_state_path.exists() {
        return Err(Error::MissingDbState(db1_state_path));
    }
    if !db2_state_path.exists() {
        return Err(Error::MissingDbState(db2_state_path));
    }

    // Check output doesn't exist
    if output.exists() {
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("Output directory already exists: {}", output.display()),
        )));
    }

    // Analyze both databases
    let db1_info = analyze_database(&db1, prefix1.clone())?;
    let db2_info = analyze_database(&db2, prefix2.clone())?;

    // Check for potential collisions (including cross-prefix collisions)
    let collision_check = check_for_collisions(&db1, &db2, prefix1.as_deref(), prefix2.as_deref())?;
    if !collision_check.is_empty() {
        eprintln!("Warning: Potential component name collisions detected:");
        for (name1, name2, resulting_name) in &collision_check {
            if name1 == name2 {
                // Same name in both databases
                eprintln!("  - \"{}\" exists in both databases", name1);
            } else {
                // Cross-prefix collision
                eprintln!(
                    "  - \"{}\" (DB1) and \"{}\" (DB2) both become \"{}\"",
                    name1, name2, resulting_name
                );
            }
        }
        if prefix1.is_none() && prefix2.is_none() {
            eprintln!("\nConsider using --prefix1 and --prefix2 to avoid collisions.");
        }
    }

    // Print merge plan
    println!("Merging databases:");
    print_database_info(&db1_info, "Source 1");
    print_database_info(&db2_info, "Source 2");

    let total_components = db1_info.component_count + db2_info.component_count;
    println!("\nOutput: {}", output.display());
    println!("  - {} components total", total_components);

    // Show naming example
    if let Some(example_name) = get_example_component_name(&db1)? {
        println!("\nComponent naming example:");
        let new_name1 = apply_prefix(&example_name, prefix1.as_deref());
        let new_name2 = apply_prefix(&example_name, prefix2.as_deref());
        println!(
            "  \"{}\" -> \"{}\" (from {})",
            example_name,
            new_name1,
            db1.display()
        );
        println!(
            "  \"{}\" -> \"{}\" (from {})",
            example_name,
            new_name2,
            db2.display()
        );
    }

    // Warn about schematics being pruned
    if db1_info.has_schematic || db2_info.has_schematic {
        println!("\nNote: Schematics will be pruned from the merged database.");
        println!("      You can create a new schematic for the combined data.");
    }

    // Show time alignment information if specified
    if let (Some(a1), Some(a2)) = (align1, align2) {
        let db1_offset_secs = db1_offset as f64 / 1_000_000.0;
        let db2_offset_secs = db2_offset as f64 / 1_000_000.0;

        println!("\nTime Alignment:");
        println!("  DB1 anchor: {:.3}s", a1);
        println!("  DB2 anchor: {:.3}s", a2);

        // Show which database is being shifted (always forward, never backward)
        if db1_offset > 0 {
            println!(
                "  Shifting DB1 forward by {:.3}s to align with DB2",
                db1_offset_secs
            );
        } else if db2_offset > 0 {
            println!(
                "  Shifting DB2 forward by {:.3}s to align with DB1",
                db2_offset_secs
            );
        } else {
            println!("  No shift needed (anchors are equal)");
        }

        // Show time range transformation for DB1
        if let Some((start, end)) = db1_info.time_range {
            let start_secs = start as f64 / 1_000_000.0;
            let end_secs = end as f64 / 1_000_000.0;
            if db1_offset > 0 {
                let new_start_secs = start_secs + db1_offset_secs;
                let new_end_secs = end_secs + db1_offset_secs;
                println!(
                    "  DB1 time range: {:.3}s - {:.3}s -> {:.3}s - {:.3}s (shifted forward)",
                    start_secs, end_secs, new_start_secs, new_end_secs
                );
            } else {
                println!(
                    "  DB1 time range: {:.3}s - {:.3}s (unchanged)",
                    start_secs, end_secs
                );
            }
        }

        // Show time range transformation for DB2
        if let Some((start, end)) = db2_info.time_range {
            let start_secs = start as f64 / 1_000_000.0;
            let end_secs = end as f64 / 1_000_000.0;
            if db2_offset > 0 {
                let new_start_secs = start_secs + db2_offset_secs;
                let new_end_secs = end_secs + db2_offset_secs;
                println!(
                    "  DB2 time range: {:.3}s - {:.3}s -> {:.3}s - {:.3}s (shifted forward)",
                    start_secs, end_secs, new_start_secs, new_end_secs
                );
            } else {
                println!(
                    "  DB2 time range: {:.3}s - {:.3}s (unchanged)",
                    start_secs, end_secs
                );
            }
        }

        // Warn if anchor is outside the dataset's time range
        if let Some((start, end)) = db1_info.time_range {
            let a1_micros = (a1 * 1_000_000.0) as i64;
            if a1_micros < start || a1_micros > end {
                eprintln!(
                    "\nWarning: DB1 anchor ({:.3}s) is outside its time range ({:.3}s - {:.3}s)",
                    a1,
                    start as f64 / 1_000_000.0,
                    end as f64 / 1_000_000.0
                );
            }
        }
        if let Some((start, end)) = db2_info.time_range {
            let a2_micros = (a2 * 1_000_000.0) as i64;
            if a2_micros < start || a2_micros > end {
                eprintln!(
                    "\nWarning: DB2 anchor ({:.3}s) is outside its time range ({:.3}s - {:.3}s)",
                    a2,
                    start as f64 / 1_000_000.0,
                    end as f64 / 1_000_000.0
                );
            }
        }
    }

    if dry_run {
        println!("\nDRY RUN - no changes will be made");
        return Ok(());
    }

    // Confirm
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

    // Create output directory structure
    // Note: Path::new("merged").parent() returns Some("") not None, so we check for empty
    let parent_dir = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&parent_dir)?;

    // Use a temporary directory for atomic operation
    let tmp_output = parent_dir.join(format!(
        "{}.tmp",
        output.file_name().unwrap_or_default().to_string_lossy()
    ));
    if tmp_output.exists() {
        fs::remove_dir_all(&tmp_output)?;
    }
    fs::create_dir_all(&tmp_output)?;

    // Perform the merge, cleaning up on failure
    let result = do_merge(
        &db1,
        &db2,
        &tmp_output,
        prefix1.as_deref(),
        prefix2.as_deref(),
        db1_offset,
        db2_offset,
    );

    // If merge failed, clean up the temporary directory
    if result.is_err() {
        eprintln!("Merge failed, cleaning up temporary directory...");
        if let Err(cleanup_err) = fs::remove_dir_all(&tmp_output) {
            eprintln!(
                "Warning: Failed to clean up temporary directory {}: {}",
                tmp_output.display(),
                cleanup_err
            );
        }
        return result.map(|_| ());
    }

    let (stats1, stats2) = result?;

    // Report any message log conflicts
    let all_conflicts: Vec<_> = stats1
        .msg_log_conflicts
        .iter()
        .chain(stats2.msg_log_conflicts.iter())
        .collect();
    if !all_conflicts.is_empty() {
        println!("\nWarning: Message log conflicts (skipped):");
        for conflict in all_conflicts {
            println!("  - {}", conflict);
        }
    }

    // Sync and rename to final location
    sync_dir(&tmp_output)?;
    fs::rename(&tmp_output, &output)?;
    sync_dir(&parent_dir)?;

    println!("\nSuccessfully merged databases to {}", output.display());
    Ok(())
}

/// Helper function that performs the actual merge operations.
/// Returns the merge stats on success, or an error on failure.
fn do_merge(
    db1: &Path,
    db2: &Path,
    tmp_output: &Path,
    prefix1: Option<&str>,
    prefix2: Option<&str>,
    db1_offset: i64,
    db2_offset: i64,
) -> Result<(MergeStats, MergeStats), Error> {
    // Merge databases with their respective offsets
    // One of the offsets will be 0, the other will be positive (shift forward)
    print!("Merging {}... ", db1.display());
    io::stdout().flush()?;
    let db1_offset_opt = if db1_offset != 0 {
        Some(db1_offset)
    } else {
        None
    };
    let stats1 = merge_database(db1, tmp_output, prefix1, db1_offset_opt)?;
    println!("done ({} components)", stats1.components_copied);

    print!("Merging {}... ", db2.display());
    io::stdout().flush()?;
    let db2_offset_opt = if db2_offset != 0 {
        Some(db2_offset)
    } else {
        None
    };
    let stats2 = merge_database(db2, tmp_output, prefix2, db2_offset_opt)?;
    println!("done ({} components)", stats2.components_copied);

    // Merge db_state
    print!("Writing db_state... ");
    io::stdout().flush()?;
    merge_db_state(db1, db2, tmp_output, db1_offset_opt, db2_offset_opt)?;
    println!("done");

    Ok((stats1, stats2))
}

/// Analyze a database and return information about it
fn analyze_database(db_path: &Path, prefix: Option<String>) -> Result<DatabaseInfo, Error> {
    let mut component_count = 0;
    let mut msg_log_count = 0;
    let mut min_timestamp = i64::MAX;
    let mut max_timestamp = i64::MIN;

    // Count components
    for entry in fs::read_dir(db_path)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
        if dir_name == "msgs" {
            // Count message logs
            if let Ok(msgs_dir) = fs::read_dir(&path) {
                for msg_entry in msgs_dir.flatten() {
                    if msg_entry.path().is_dir() {
                        msg_log_count += 1;
                    }
                }
            }
            continue;
        }
        // Skip non-component directories
        if dir_name.parse::<u64>().is_err() {
            continue;
        }
        // Check if it has a schema file (valid component)
        if path.join("schema").exists() {
            component_count += 1;

            // Try to get timestamp range from index file
            match read_timestamp_range(&path.join("index")) {
                Ok(Some((start, end))) => {
                    min_timestamp = min_timestamp.min(start);
                    max_timestamp = max_timestamp.max(end);
                }
                Ok(None) => {
                    // No timestamps in this component
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to read timestamps from {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }
    }

    // Check for schematic in db_state
    let db_state_path = db_path.join("db_state");
    let has_schematic = if db_state_path.exists() {
        let config = DbConfig::read(&db_state_path)?;
        config.schematic_path().is_some() || config.schematic_content().is_some()
    } else {
        false
    };

    let time_range = if min_timestamp <= max_timestamp && min_timestamp != i64::MAX {
        Some((min_timestamp, max_timestamp))
    } else {
        None
    };

    Ok(DatabaseInfo {
        path: db_path.to_path_buf(),
        prefix,
        component_count,
        msg_log_count,
        time_range,
        has_schematic,
    })
}

/// Read the first and last timestamp from an index file
fn read_timestamp_range(index_path: &Path) -> Result<Option<(i64, i64)>, Error> {
    if !index_path.exists() {
        return Ok(None);
    }

    let file = File::open(index_path)?;

    // Header is 24 bytes: committed_len (8) + head_len (8) + extra (8)
    // committed_len includes the header size, so actual data length = committed_len - 24
    const HEADER_SIZE: usize = 24;

    // Read header to get committed length
    use std::io::Read;
    let mut header = [0u8; HEADER_SIZE];
    let mut file = file;
    file.read_exact(&mut header)?;

    // committed_len includes the header size
    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;

    // Calculate actual data length (excluding header)
    let data_len = committed_len.saturating_sub(HEADER_SIZE);
    if data_len < 8 {
        // No timestamps stored
        return Ok(None);
    }

    // Number of timestamps
    let num_timestamps = data_len / 8;
    if num_timestamps == 0 {
        return Ok(None);
    }

    // Read first timestamp (starts right after header)
    let mut first_ts_bytes = [0u8; 8];
    file.read_exact(&mut first_ts_bytes)?;
    let first_ts = i64::from_le_bytes(first_ts_bytes);

    // If only one timestamp, first == last
    if num_timestamps == 1 {
        return Ok(Some((first_ts, first_ts)));
    }

    // Read last timestamp
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(
        (HEADER_SIZE + (num_timestamps - 1) * 8) as u64,
    ))?;
    let mut last_ts_bytes = [0u8; 8];
    file.read_exact(&mut last_ts_bytes)?;
    let last_ts = i64::from_le_bytes(last_ts_bytes);

    Ok(Some((first_ts, last_ts)))
}

/// Print information about a database
fn print_database_info(info: &DatabaseInfo, label: &str) {
    let prefix_str = info
        .prefix
        .as_ref()
        .map(|p| format!("\"{}\"", p))
        .unwrap_or_else(|| "none".to_string());
    println!(
        "  {}: {} (prefix: {})",
        label,
        info.path.display(),
        prefix_str
    );
    println!("    - {} components", info.component_count);
    if info.msg_log_count > 0 {
        println!("    - {} message logs", info.msg_log_count);
    }
    if let Some((start, end)) = info.time_range {
        println!(
            "    - Time range: {:.3}s - {:.3}s",
            start as f64 / 1_000_000.0,
            end as f64 / 1_000_000.0
        );
    }
}

/// Check for potential collisions between two databases
/// Check for component name collisions after applying prefixes.
/// Returns a list of (original_name1, original_name2, resulting_name) tuples for collisions.
/// When both names are the same, it indicates the same component exists in both databases.
fn check_for_collisions(
    db1: &Path,
    db2: &Path,
    prefix1: Option<&str>,
    prefix2: Option<&str>,
) -> Result<Vec<(String, String, String)>, Error> {
    let names1 = collect_component_names(db1)?;
    let names2 = collect_component_names(db2)?;

    // Build a map of prefixed_name -> original_name for DB1
    let mut prefixed_names1: HashMap<String, String> = HashMap::new();
    for name in &names1 {
        let prefixed = apply_prefix(name, prefix1);
        prefixed_names1.insert(prefixed, name.clone());
    }

    // Check each DB2 component against DB1's prefixed names
    let mut collisions = Vec::new();
    for name2 in &names2 {
        let prefixed2 = apply_prefix(name2, prefix2);
        if let Some(original_name1) = prefixed_names1.get(&prefixed2) {
            collisions.push((original_name1.clone(), name2.clone(), prefixed2));
        }
    }
    Ok(collisions)
}

/// Collect all component names from a database
fn collect_component_names(db_path: &Path) -> Result<Vec<String>, Error> {
    let mut names = Vec::new();
    for entry in fs::read_dir(db_path)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
        if dir_name == "msgs" || dir_name.parse::<u64>().is_err() {
            continue;
        }
        let metadata_path = path.join("metadata");
        if metadata_path.exists() {
            let metadata = ComponentMetadata::read(&metadata_path)?;
            names.push(metadata.name);
        }
    }
    Ok(names)
}

/// Get an example component name from a database
fn get_example_component_name(db_path: &Path) -> Result<Option<String>, Error> {
    for entry in fs::read_dir(db_path)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();
        if dir_name == "msgs" || dir_name.parse::<u64>().is_err() {
            continue;
        }
        let metadata_path = path.join("metadata");
        if metadata_path.exists() {
            let metadata = ComponentMetadata::read(&metadata_path)?;
            return Ok(Some(metadata.name));
        }
    }
    Ok(None)
}

/// Apply a prefix to a component name using underscore separator
/// This ensures consistency with snake_case table naming conventions
fn apply_prefix(name: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) if !p.is_empty() => format!("{}_{}", p, name),
        _ => name.to_string(),
    }
}

/// Merge a single source database into the target directory
///
/// # Arguments
/// * `source` - Path to the source database
/// * `target` - Path to the target (output) database directory
/// * `prefix` - Optional prefix to apply to component names
/// * `timestamp_offset` - Optional offset (in microseconds) to apply to all timestamps
fn merge_database(
    source: &Path,
    target: &Path,
    prefix: Option<&str>,
    timestamp_offset: Option<i64>,
) -> Result<MergeStats, Error> {
    let mut stats = MergeStats::default();

    // Process each component directory
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let dir_name = path.file_name().unwrap_or_default().to_string_lossy();

        // Handle message logs separately
        if dir_name == "msgs" {
            merge_message_logs(&path, &target.join("msgs"), &mut stats, timestamp_offset)?;
            continue;
        }

        // Skip non-component directories
        if dir_name.parse::<u64>().is_err() {
            continue;
        }

        // Check if it's a valid component (has schema)
        if !path.join("schema").exists() {
            continue;
        }

        // Copy component with prefix
        copy_component_with_prefix(&path, target, prefix, timestamp_offset)?;
        stats.components_copied += 1;
    }

    Ok(stats)
}

/// Copy a component directory with a new prefixed name
///
/// # Arguments
/// * `src_component_dir` - Path to the source component directory
/// * `target_db_dir` - Path to the target database directory
/// * `prefix` - Optional prefix to apply to the component name
/// * `timestamp_offset` - Optional offset (in microseconds) to apply to timestamps
fn copy_component_with_prefix(
    src_component_dir: &Path,
    target_db_dir: &Path,
    prefix: Option<&str>,
    timestamp_offset: Option<i64>,
) -> Result<(), Error> {
    // Read existing metadata
    let metadata_path = src_component_dir.join("metadata");
    let old_name = if metadata_path.exists() {
        let metadata = ComponentMetadata::read(&metadata_path)?;
        metadata.name
    } else {
        // Use component ID as name if no metadata
        src_component_dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    };

    // Compute new name and ID
    let new_name = apply_prefix(&old_name, prefix);
    let new_component_id = ComponentId::new(&new_name);
    let new_component_dir = target_db_dir.join(new_component_id.to_string());

    // Check for collision in target
    if new_component_dir.exists() {
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "Component collision: {} (from {}) already exists in target",
                new_name, old_name
            ),
        )));
    }

    // Create new component directory
    fs::create_dir_all(&new_component_dir)?;

    // Copy schema file (unchanged)
    let schema_src = src_component_dir.join("schema");
    if schema_src.exists() {
        copy_file_native(&schema_src, &new_component_dir.join("schema"))?;
    }

    // Copy index file - apply timestamp offset if provided
    // Use sparse-aware copy to avoid expanding 8GB sparse files
    let index_src = src_component_dir.join("index");
    if index_src.exists() {
        let index_dst = new_component_dir.join("index");
        if let Some(offset) = timestamp_offset {
            copy_index_with_offset(&index_src, &index_dst, offset)?;
        } else {
            copy_append_log_file(&index_src, &index_dst)?;
        }
    }

    // Copy data file (unchanged - data values are not timestamps)
    // Use sparse-aware copy to avoid expanding 8GB sparse files
    let data_src = src_component_dir.join("data");
    if data_src.exists() {
        copy_append_log_file(&data_src, &new_component_dir.join("data"))?;
    }

    // Write updated metadata with new name and component_id
    let old_metadata = if metadata_path.exists() {
        ComponentMetadata::read(&metadata_path)?
    } else {
        ComponentMetadata {
            component_id: ComponentId::new(&old_name),
            name: old_name.clone(),
            metadata: HashMap::new(),
        }
    };

    let new_metadata = ComponentMetadata {
        component_id: new_component_id,
        name: new_name,
        metadata: old_metadata.metadata,
    };
    new_metadata.write(new_component_dir.join("metadata"))?;

    // Sync the directory
    sync_dir(&new_component_dir)?;

    Ok(())
}

/// Merge message logs from source to target
///
/// # Arguments
/// * `source_msgs` - Path to the source msgs directory
/// * `target_msgs` - Path to the target msgs directory
/// * `stats` - Mutable reference to merge statistics
/// * `timestamp_offset` - Optional offset (in microseconds) to apply to timestamps
fn merge_message_logs(
    source_msgs: &Path,
    target_msgs: &Path,
    stats: &mut MergeStats,
    timestamp_offset: Option<i64>,
) -> Result<(), Error> {
    if !source_msgs.exists() {
        return Ok(());
    }

    fs::create_dir_all(target_msgs)?;

    for entry in fs::read_dir(source_msgs)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let msg_id = path.file_name().unwrap_or_default().to_string_lossy();
        let target_msg_dir = target_msgs.join(msg_id.as_ref());

        // Check for collision
        if target_msg_dir.exists() {
            stats
                .msg_log_conflicts
                .push(format!("Message log ID: {}", msg_id));
            continue;
        }

        // Copy the message log directory, applying timestamp offset to index if needed
        copy_msg_log_with_offset(&path, &target_msg_dir, timestamp_offset)?;
        stats.msg_logs_copied += 1;
    }

    sync_dir(target_msgs)?;
    Ok(())
}

/// Copy a message log directory, optionally applying a timestamp offset to the index
fn copy_msg_log_with_offset(
    src: &Path,
    dst: &Path,
    timestamp_offset: Option<i64>,
) -> Result<(), Error> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if file_type.is_dir() {
            copy_dir_native(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            // Apply offset to index file if timestamp_offset is provided
            // Use sparse-aware copy for AppendLog files (index, data)
            if file_name_str == "index" {
                if let Some(offset) = timestamp_offset {
                    copy_index_with_offset(&src_path, &dst_path, offset)?;
                } else {
                    copy_append_log_file(&src_path, &dst_path)?;
                }
            } else if file_name_str == "data" {
                // Data files are also AppendLog sparse files
                copy_append_log_file(&src_path, &dst_path)?;
            } else {
                copy_file_native(&src_path, &dst_path)?;
            }
        }
    }

    let metadata = fs::metadata(src)?;
    fs::set_permissions(dst, metadata.permissions())?;
    Ok(())
}

/// Merge db_state from both databases, pruning schematics
///
/// # Arguments
/// * `db1` - Path to the first database
/// * `db2` - Path to the second database
/// * `target` - Path to the target (output) database
/// * `db1_offset` - Optional offset (in microseconds) applied to DB1's timestamps
/// * `db2_offset` - Optional offset (in microseconds) applied to DB2's timestamps
fn merge_db_state(
    db1: &Path,
    db2: &Path,
    target: &Path,
    db1_offset: Option<i64>,
    db2_offset: Option<i64>,
) -> Result<(), Error> {
    let config1 = DbConfig::read(db1.join("db_state"))?;
    let config2 = DbConfig::read(db2.join("db_state"))?;

    // Start with config1 as base
    let mut merged_config = config1.clone();

    // Merge metadata, excluding schematic keys
    for (key, value) in &config2.metadata {
        if key != "schematic.path" && key != "schematic.content" {
            merged_config.metadata.insert(key.clone(), value.clone());
        }
    }

    // Remove schematic keys from merged config
    merged_config.metadata.remove("schematic.path");
    merged_config.metadata.remove("schematic.content");

    // Use earliest time.start_timestamp from both (after applying offsets)
    let ts1 = config1.time_start_timestamp_micros().map(|t1| {
        if let Some(offset) = db1_offset {
            t1.saturating_add(offset)
        } else {
            t1
        }
    });
    let ts2 = config2.time_start_timestamp_micros().map(|t2| {
        if let Some(offset) = db2_offset {
            t2.saturating_add(offset)
        } else {
            t2
        }
    });

    match (ts1, ts2) {
        (Some(t1), Some(t2)) => {
            merged_config.set_time_start_timestamp_micros(t1.min(t2));
        }
        (Some(t1), None) => {
            merged_config.set_time_start_timestamp_micros(t1);
        }
        (None, Some(t2)) => {
            merged_config.set_time_start_timestamp_micros(t2);
        }
        (None, None) => {}
    }

    // Write merged config
    merged_config.write(target.join("db_state"))?;
    Ok(())
}
/// Copy an AppendLog file (index or data), reading only the committed portion.
///
/// AppendLog files are sparse files with 8GB logical size but only `committed_len`
/// bytes of actual data. This function reads only the committed portion to avoid
/// expanding the sparse file to its full logical size during copy.
///
/// Uses chunked I/O to avoid allocating large buffers for big databases.
fn copy_append_log_file(src: &Path, dst: &Path) -> Result<(), Error> {
    use std::io::{Read, Seek, SeekFrom, Write};

    const HEADER_SIZE: usize = 24;
    const COMMITTED_LEN_SIZE: usize = 8;
    const CHUNK_SIZE: usize = 64 * 1024; // 64KB chunks

    let mut src_file = File::open(src)?;

    // Read committed_len from header (first 8 bytes)
    let mut committed_len_bytes = [0u8; COMMITTED_LEN_SIZE];
    let bytes_read = src_file.read(&mut committed_len_bytes)?;

    if bytes_read < COMMITTED_LEN_SIZE {
        // File is too small to have a valid header, copy as-is using small buffer
        src_file.seek(SeekFrom::Start(0))?;
        let mut dst_file = File::create(dst)?;
        let mut buf = [0u8; CHUNK_SIZE];
        loop {
            let n = src_file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            dst_file.write_all(&buf[..n])?;
        }
        dst_file.sync_all()?;
        if let Some(parent) = dst.parent() {
            sync_dir(parent)?;
        }
        return Ok(());
    }

    let committed_len = u64::from_le_bytes(committed_len_bytes) as usize;

    // Sanity check: committed_len should be at least HEADER_SIZE
    let committed_len = committed_len.max(HEADER_SIZE);

    // Copy only the committed portion using chunked I/O
    src_file.seek(SeekFrom::Start(0))?;
    let mut dst_file = File::create(dst)?;
    let mut remaining = committed_len;
    let mut buf = vec![0u8; CHUNK_SIZE];

    while remaining > 0 {
        let to_read = remaining.min(CHUNK_SIZE);
        src_file.read_exact(&mut buf[..to_read])?;
        dst_file.write_all(&buf[..to_read])?;
        remaining -= to_read;
    }

    dst_file.sync_all()?;

    if let Some(parent) = dst.parent() {
        sync_dir(parent)?;
    }

    Ok(())
}

/// Copy an index file while applying a timestamp offset.
///
/// This reads only the committed portion of the source index file (not the entire
/// sparse file), applies the offset to:
/// 1. The start_timestamp in the header (bytes 16-24)
/// 2. All timestamp entries in the data section
///
/// Uses saturating arithmetic to prevent overflow.
/// Uses chunked I/O to avoid OOM on large databases.
///
/// IMPORTANT: AppendLog files are sparse files with 8GB logical size but only
/// `committed_len` bytes of actual data. We must read only the committed portion
/// to avoid expanding the sparse file to its full logical size.
fn copy_index_with_offset(src_index: &Path, dst_index: &Path, offset: i64) -> Result<(), Error> {
    use std::io::{Read, Seek, SeekFrom, Write};

    const HEADER_SIZE: usize = 24;
    // Process timestamps in chunks of 8KB (1024 timestamps per chunk)
    const CHUNK_TIMESTAMPS: usize = 1024;
    const CHUNK_SIZE: usize = CHUNK_TIMESTAMPS * 8;

    let mut src_file = File::open(src_index)?;

    // First, read just the header to get committed_len
    let mut header = [0u8; HEADER_SIZE];
    let bytes_read = src_file.read(&mut header)?;

    if bytes_read < HEADER_SIZE {
        // File is too small, just copy what we have
        src_file.seek(SeekFrom::Start(0))?;
        let mut dst_file = File::create(dst_index)?;
        let mut buf = [0u8; 64];
        loop {
            let n = src_file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            dst_file.write_all(&buf[..n])?;
        }
        dst_file.sync_all()?;
        if let Some(parent) = dst_index.parent() {
            sync_dir(parent)?;
        }
        return Ok(());
    }

    // Get committed_len - this tells us how much actual data exists
    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;

    // Sanity check: committed_len should be at least HEADER_SIZE
    let committed_len = committed_len.max(HEADER_SIZE);

    // Apply offset to start_timestamp in header (bytes 16-24)
    let start_ts = i64::from_le_bytes(header[16..24].try_into().unwrap());
    let new_start_ts = start_ts.saturating_add(offset);
    header[16..24].copy_from_slice(&new_start_ts.to_le_bytes());

    // Create destination and write modified header
    let mut dst_file = File::create(dst_index)?;
    dst_file.write_all(&header)?;

    // Calculate how much timestamp data to process
    let data_len = committed_len.saturating_sub(HEADER_SIZE);

    // Validate that data_len is a multiple of 8 (each timestamp is 8 bytes)
    // If not, the index file is malformed and we should report an error rather than
    // silently corrupting the output by leaving trailing bytes without offset applied
    if !data_len.is_multiple_of(8) {
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Malformed index file: data length {} is not a multiple of 8 bytes (timestamp size)",
                data_len
            ),
        )));
    }

    // Process timestamps in chunks to avoid OOM
    let mut remaining = data_len;
    let mut buf = vec![0u8; CHUNK_SIZE];

    while remaining > 0 {
        let to_read = remaining.min(CHUNK_SIZE);
        src_file.read_exact(&mut buf[..to_read])?;

        // Apply offset to each timestamp in the chunk
        let num_timestamps = to_read / 8;
        for i in 0..num_timestamps {
            let start = i * 8;
            let ts = i64::from_le_bytes(buf[start..start + 8].try_into().unwrap());
            let new_ts = ts.saturating_add(offset);
            buf[start..start + 8].copy_from_slice(&new_ts.to_le_bytes());
        }

        dst_file.write_all(&buf[..to_read])?;
        remaining -= to_read;
    }

    dst_file.sync_all()?;

    // Sync parent directory
    if let Some(parent) = dst_index.parent() {
        sync_dir(parent)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_apply_prefix() {
        assert_eq!(
            apply_prefix("rocket.velocity", Some("sim")),
            "sim_rocket.velocity"
        );
        assert_eq!(
            apply_prefix("rocket.velocity", Some("truth")),
            "truth_rocket.velocity"
        );
        assert_eq!(apply_prefix("rocket.velocity", None), "rocket.velocity");
        assert_eq!(apply_prefix("rocket.velocity", Some("")), "rocket.velocity");
    }

    #[test]
    fn test_component_id_with_prefix() {
        let name1 = "rocket.velocity";
        let name2 = apply_prefix(name1, Some("sim"));

        let id1 = ComponentId::new(name1);
        let id2 = ComponentId::new(&name2);

        // IDs should be different after prefix
        assert_ne!(id1, id2);

        // ID should be deterministic
        assert_eq!(id2, ComponentId::new("sim_rocket.velocity"));
    }

    #[test]
    fn test_underscore_preserves_hierarchy() {
        // Original: "drone.rate_pid_state" has hierarchy drone -> rate_pid_state
        // With prefix: "sim_drone.rate_pid_state" has hierarchy sim_drone -> rate_pid_state
        // The underscore separates the prefix from the original entity name

        let original = "drone.rate_pid_state";
        let prefixed = apply_prefix(original, Some("sim"));

        assert_eq!(prefixed, "sim_drone.rate_pid_state");

        // The original hierarchy (entity.component) is preserved
        // Only the entity name gets the prefix
        let parts: Vec<&str> = prefixed.split('.').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "sim_drone"); // entity with prefix
        assert_eq!(parts[1], "rate_pid_state"); // component unchanged
    }

    /// Helper to create a minimal test database
    fn create_test_db(dir: &Path, components: &[(&str, &[u8])]) -> Result<(), Error> {
        fs::create_dir_all(dir)?;

        // Create db_state
        let config = DbConfig {
            recording: false,
            default_stream_time_step: std::time::Duration::from_millis(10),
            metadata: HashMap::new(),
        };
        config.write(dir.join("db_state"))?;

        // Create components
        for (name, _data) in components {
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

            // Write minimal schema (F64 scalar)
            let schema_data = [10u8, 0, 0, 0, 0, 0, 0, 0, 0]; // PrimType::F64, dim=[]
            fs::write(component_dir.join("schema"), &schema_data)?;

            // Write minimal index (header only)
            let index_data = vec![0u8; 24]; // Header: committed_len=0, head_len=0, extra=0
            fs::write(component_dir.join("index"), &index_data)?;

            // Write data file (empty)
            let mut data_file = vec![0u8; 24]; // Header only
            data_file[16..24].copy_from_slice(&8u64.to_le_bytes()); // element_size = 8
            fs::write(component_dir.join("data"), &data_file)?;
        }

        Ok(())
    }

    #[test]
    fn test_merge_basic() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        // Create test databases
        create_test_db(&db1_path, &[("rocket.velocity", b"")]).unwrap();
        create_test_db(&db2_path, &[("rocket.position", b"")]).unwrap();

        // Run merge
        run(
            db1_path,
            db2_path,
            output_path.clone(),
            Some("sim".to_string()),
            Some("truth".to_string()),
            false,
            true, // auto-confirm
            None, // align1
            None, // align2
        )
        .unwrap();

        // Verify output exists
        assert!(output_path.exists());
        assert!(output_path.join("db_state").exists());

        // Verify components were renamed correctly
        let sim_velocity_id = ComponentId::new("sim_rocket.velocity");
        let truth_position_id = ComponentId::new("truth_rocket.position");

        assert!(output_path.join(sim_velocity_id.to_string()).exists());
        assert!(output_path.join(truth_position_id.to_string()).exists());

        // Verify metadata has correct names
        let sim_metadata = ComponentMetadata::read(
            output_path
                .join(sim_velocity_id.to_string())
                .join("metadata"),
        )
        .unwrap();
        assert_eq!(sim_metadata.name, "sim_rocket.velocity");
        assert_eq!(sim_metadata.component_id, sim_velocity_id);

        let truth_metadata = ComponentMetadata::read(
            output_path
                .join(truth_position_id.to_string())
                .join("metadata"),
        )
        .unwrap();
        assert_eq!(truth_metadata.name, "truth_rocket.position");
        assert_eq!(truth_metadata.component_id, truth_position_id);
    }

    #[test]
    fn test_merge_no_prefix() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        // Create test databases with different component names
        create_test_db(&db1_path, &[("rocket.velocity", b"")]).unwrap();
        create_test_db(&db2_path, &[("drone.velocity", b"")]).unwrap();

        // Run merge without prefixes
        run(
            db1_path,
            db2_path,
            output_path.clone(),
            None,
            None,
            false,
            true,
            None, // align1
            None, // align2
        )
        .unwrap();

        // Verify components keep original names
        let rocket_id = ComponentId::new("rocket.velocity");
        let drone_id = ComponentId::new("drone.velocity");

        assert!(output_path.join(rocket_id.to_string()).exists());
        assert!(output_path.join(drone_id.to_string()).exists());
    }

    #[test]
    fn test_merge_schematic_pruned() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        // Create db1 with schematic
        fs::create_dir_all(&db1_path).unwrap();
        let mut config1 = DbConfig {
            recording: false,
            default_stream_time_step: std::time::Duration::from_millis(10),
            metadata: HashMap::new(),
        };
        config1.set_schematic_content("graph \"test\"".to_string());
        config1.write(db1_path.join("db_state")).unwrap();

        // Create db2 without schematic
        create_test_db(&db2_path, &[]).unwrap();

        // Run merge
        run(
            db1_path,
            db2_path,
            output_path.clone(),
            None,
            None,
            false,
            true,
            None, // align1
            None, // align2
        )
        .unwrap();

        // Verify schematic was pruned
        let merged_config = DbConfig::read(output_path.join("db_state")).unwrap();
        assert!(merged_config.schematic_content().is_none());
        assert!(merged_config.schematic_path().is_none());
    }

    #[test]
    fn test_dry_run() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        create_test_db(&db1_path, &[("test", b"")]).unwrap();
        create_test_db(&db2_path, &[("test2", b"")]).unwrap();

        // Run with dry_run=true
        run(
            db1_path,
            db2_path,
            output_path.clone(),
            Some("a".to_string()),
            Some("b".to_string()),
            true, // dry_run
            true,
            None, // align1
            None, // align2
        )
        .unwrap();

        // Verify output was NOT created
        assert!(!output_path.exists());
    }

    #[test]
    fn test_output_exists_error() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        create_test_db(&db1_path, &[]).unwrap();
        create_test_db(&db2_path, &[]).unwrap();
        fs::create_dir_all(&output_path).unwrap();

        // Should fail because output exists
        let result = run(
            db1_path,
            db2_path,
            output_path,
            None,
            None,
            false,
            true,
            None,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_collect_component_names() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        create_test_db(
            &db_path,
            &[
                ("rocket.velocity", b""),
                ("rocket.position", b""),
                ("drone.motor", b""),
            ],
        )
        .unwrap();

        let names = collect_component_names(&db_path).unwrap();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"rocket.velocity".to_string()));
        assert!(names.contains(&"rocket.position".to_string()));
        assert!(names.contains(&"drone.motor".to_string()));
    }

    #[test]
    fn test_check_for_collisions_same_prefix() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");

        // Same component name in both databases, same prefix (both None)
        create_test_db(&db1_path, &[("rocket.velocity", b"")]).unwrap();
        create_test_db(&db2_path, &[("rocket.velocity", b"")]).unwrap();

        let collisions = check_for_collisions(&db1_path, &db2_path, None, None).unwrap();
        assert_eq!(collisions.len(), 1);
        assert_eq!(
            collisions[0],
            (
                "rocket.velocity".to_string(),
                "rocket.velocity".to_string(),
                "rocket.velocity".to_string()
            )
        );
    }

    #[test]
    fn test_check_for_collisions_cross_prefix() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");

        // DB1: "bar" with prefix "foo" becomes "foo_bar"
        // DB2: "foo_bar" with no prefix stays "foo_bar"
        // This should be detected as a collision
        create_test_db(&db1_path, &[("bar", b"")]).unwrap();
        create_test_db(&db2_path, &[("foo_bar", b"")]).unwrap();

        let collisions = check_for_collisions(&db1_path, &db2_path, Some("foo"), None).unwrap();
        assert_eq!(collisions.len(), 1);
        assert_eq!(
            collisions[0],
            (
                "bar".to_string(),
                "foo_bar".to_string(),
                "foo_bar".to_string()
            )
        );
    }

    #[test]
    fn test_check_for_collisions_no_collision_with_different_prefixes() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");

        // Same name but different prefixes - no collision
        create_test_db(&db1_path, &[("rocket.velocity", b"")]).unwrap();
        create_test_db(&db2_path, &[("rocket.velocity", b"")]).unwrap();

        let collisions =
            check_for_collisions(&db1_path, &db2_path, Some("sim"), Some("truth")).unwrap();
        assert!(collisions.is_empty());
    }

    /// Helper to create an index file with timestamps
    fn create_index_file(path: &Path, start_ts: i64, timestamps: &[i64]) {
        use std::io::Write;

        let header_size = 24usize;
        let data_len = timestamps.len() * 8;
        let committed_len = (header_size + data_len) as u64;

        let mut data = vec![0u8; header_size + data_len];

        // Header: committed_len (8) + head_len (8) + start_timestamp (8)
        data[0..8].copy_from_slice(&committed_len.to_le_bytes());
        data[8..16].copy_from_slice(&0u64.to_le_bytes()); // head_len = 0
        data[16..24].copy_from_slice(&start_ts.to_le_bytes());

        // Timestamps
        for (i, ts) in timestamps.iter().enumerate() {
            let offset = header_size + i * 8;
            data[offset..offset + 8].copy_from_slice(&ts.to_le_bytes());
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    /// Helper to read timestamps from an index file
    fn read_index_file(path: &Path) -> (i64, Vec<i64>) {
        use std::io::Read;

        let mut file = File::open(path).unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();

        let committed_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let start_ts = i64::from_le_bytes(data[16..24].try_into().unwrap());

        let header_size = 24;
        let data_len = committed_len.saturating_sub(header_size);
        let num_timestamps = data_len / 8;

        let mut timestamps = Vec::new();
        for i in 0..num_timestamps {
            let offset = header_size + i * 8;
            if offset + 8 <= data.len() {
                let ts = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                timestamps.push(ts);
            }
        }

        (start_ts, timestamps)
    }

    #[test]
    fn test_copy_index_with_positive_offset() {
        let temp = TempDir::new().unwrap();
        let src_index = temp.path().join("src_index");
        let dst_index = temp.path().join("dst_index");

        // Create source index with monotonic timestamps (small values)
        let start_ts = 0i64;
        let timestamps = vec![100_000i64, 200_000, 300_000]; // 0.1s, 0.2s, 0.3s
        create_index_file(&src_index, start_ts, &timestamps);

        // Apply positive offset (shift forward to wall-clock time)
        let offset = 1_000_000_000_000i64; // 1 million seconds in microseconds
        copy_index_with_offset(&src_index, &dst_index, offset).unwrap();

        // Read and verify
        let (new_start_ts, new_timestamps) = read_index_file(&dst_index);
        assert_eq!(new_start_ts, start_ts + offset);
        assert_eq!(new_timestamps.len(), 3);
        assert_eq!(new_timestamps[0], 100_000 + offset);
        assert_eq!(new_timestamps[1], 200_000 + offset);
        assert_eq!(new_timestamps[2], 300_000 + offset);
    }

    #[test]
    fn test_copy_index_with_negative_offset() {
        let temp = TempDir::new().unwrap();
        let src_index = temp.path().join("src_index");
        let dst_index = temp.path().join("dst_index");

        // Create source index with wall-clock timestamps (large values)
        let start_ts = 1_705_000_000_000_000i64; // ~Jan 2024 in microseconds
        let timestamps = vec![
            1_705_000_000_100_000i64,
            1_705_000_000_200_000,
            1_705_000_000_300_000,
        ];
        create_index_file(&src_index, start_ts, &timestamps);

        // Apply negative offset (shift backward to monotonic time)
        let offset = -1_705_000_000_000_000i64;
        copy_index_with_offset(&src_index, &dst_index, offset).unwrap();

        // Read and verify
        let (new_start_ts, new_timestamps) = read_index_file(&dst_index);
        assert_eq!(new_start_ts, 0);
        assert_eq!(new_timestamps.len(), 3);
        assert_eq!(new_timestamps[0], 100_000);
        assert_eq!(new_timestamps[1], 200_000);
        assert_eq!(new_timestamps[2], 300_000);
    }

    #[test]
    fn test_copy_index_with_zero_offset() {
        let temp = TempDir::new().unwrap();
        let src_index = temp.path().join("src_index");
        let dst_index = temp.path().join("dst_index");

        let start_ts = 1000i64;
        let timestamps = vec![1000i64, 2000, 3000];
        create_index_file(&src_index, start_ts, &timestamps);

        // Apply zero offset (no change)
        copy_index_with_offset(&src_index, &dst_index, 0).unwrap();

        // Read and verify - should be unchanged
        let (new_start_ts, new_timestamps) = read_index_file(&dst_index);
        assert_eq!(new_start_ts, start_ts);
        assert_eq!(new_timestamps, timestamps);
    }

    #[test]
    fn test_copy_index_rejects_malformed_data_length() {
        use std::io::Write;
        let temp = TempDir::new().unwrap();
        let src_index = temp.path().join("src_index");
        let dst_index = temp.path().join("dst_index");

        // Create a malformed index file where data_len is not a multiple of 8
        let header_size = 24usize;
        let malformed_data_len = 11; // Not a multiple of 8
        let committed_len = (header_size + malformed_data_len) as u64;

        let mut data = vec![0u8; header_size + malformed_data_len];
        data[0..8].copy_from_slice(&committed_len.to_le_bytes());
        data[8..16].copy_from_slice(&0u64.to_le_bytes()); // head_len = 0
        data[16..24].copy_from_slice(&1000i64.to_le_bytes()); // start_ts

        let mut file = File::create(&src_index).unwrap();
        file.write_all(&data).unwrap();

        // Attempt to copy with an offset should fail with an error about malformed data
        let result = copy_index_with_offset(&src_index, &dst_index, 1000);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not a multiple of 8"),
            "Expected error about data length, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_alignment_validation_both_required() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        create_test_db(&db1_path, &[]).unwrap();
        create_test_db(&db2_path, &[]).unwrap();

        // Should fail if only align1 is provided
        let result = run(
            db1_path.clone(),
            db2_path.clone(),
            output_path.clone(),
            None,
            None,
            false,
            true,
            Some(100.0), // align1
            None,        // align2 missing
        );
        assert!(result.is_err());

        // Should fail if only align2 is provided
        let output_path2 = temp.path().join("merged2");
        let result = run(
            db1_path,
            db2_path,
            output_path2,
            None,
            None,
            false,
            true,
            None,        // align1 missing
            Some(100.0), // align2
        );
        assert!(result.is_err());
    }

    /// Helper to create a test database with timestamps in the index
    fn create_test_db_with_timestamps(
        dir: &Path,
        components: &[(&str, i64, &[i64])], // (name, start_ts, timestamps)
    ) -> Result<(), Error> {
        fs::create_dir_all(dir)?;

        // Create db_state with start timestamp
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

            // Write minimal schema (F64 scalar)
            let schema_data = [10u8, 0, 0, 0, 0, 0, 0, 0, 0];
            fs::write(component_dir.join("schema"), &schema_data)?;

            // Write index with timestamps
            create_index_file(&component_dir.join("index"), *start_ts, timestamps);

            // Write data file with matching size
            let element_size = 8u64;
            let header_size = 24;
            let data_len = timestamps.len() * 8;
            let committed_len = (header_size + data_len) as u64;
            let mut data_file = vec![0u8; header_size + data_len];
            data_file[0..8].copy_from_slice(&committed_len.to_le_bytes());
            data_file[16..24].copy_from_slice(&element_size.to_le_bytes());
            fs::write(component_dir.join("data"), &data_file)?;
        }

        Ok(())
    }

    #[test]
    fn test_merge_with_alignment_db1_shifted() {
        // Test case: Both datasets are monotonic (0-based)
        // DB1 has "boost" event at 15s, DB2 has "boost" event at 45s
        // Since align1 (15s) < align2 (45s), DB1 should be shifted forward by 30s
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        // Create DB1 with monotonic timestamps - "boost" at 15s
        create_test_db_with_timestamps(
            &db1_path,
            &[(
                "sim.velocity",
                0,
                &[5_000_000, 15_000_000, 25_000_000], // 5s, 15s (boost), 25s
            )],
        )
        .unwrap();

        // Create DB2 with monotonic timestamps - "boost" at 45s
        create_test_db_with_timestamps(
            &db2_path,
            &[(
                "truth.velocity",
                0,
                &[35_000_000, 45_000_000, 55_000_000], // 35s, 45s (boost), 55s
            )],
        )
        .unwrap();

        // Merge with alignment: align DB1's 15s with DB2's 45s
        // Since align1 < align2, DB1 gets shifted forward by 30s
        run(
            db1_path,
            db2_path,
            output_path.clone(),
            Some("sim".to_string()),
            Some("truth".to_string()),
            false,
            true,
            Some(15.0), // align1: boost at 15s in DB1
            Some(45.0), // align2: boost at 45s in DB2
        )
        .unwrap();

        // Verify DB1 (sim) was shifted forward by 30s
        let sim_id = ComponentId::new("sim_sim.velocity");
        let sim_index = output_path.join(sim_id.to_string()).join("index");
        let (sim_start_ts, sim_timestamps) = read_index_file(&sim_index);

        // Original: 0, [5s, 15s, 25s] -> After +30s shift: 30s, [35s, 45s, 55s]
        assert_eq!(sim_start_ts, 30_000_000); // 0 + 30s
        assert_eq!(sim_timestamps[0], 35_000_000); // 5s + 30s
        assert_eq!(sim_timestamps[1], 45_000_000); // 15s + 30s (boost, now aligned)
        assert_eq!(sim_timestamps[2], 55_000_000); // 25s + 30s

        // Verify DB2 (truth) was NOT shifted
        let truth_id = ComponentId::new("truth_truth.velocity");
        let truth_index = output_path.join(truth_id.to_string()).join("index");
        let (truth_start_ts, truth_timestamps) = read_index_file(&truth_index);

        assert_eq!(truth_start_ts, 0);
        assert_eq!(truth_timestamps[0], 35_000_000);
        assert_eq!(truth_timestamps[1], 45_000_000); // boost event (reference)
        assert_eq!(truth_timestamps[2], 55_000_000);
    }

    #[test]
    fn test_merge_with_alignment_db2_shifted() {
        // Test case: Both datasets are monotonic (0-based)
        // DB1 has "boost" event at 45s, DB2 has "boost" event at 15s
        // Since align1 (45s) > align2 (15s), DB2 should be shifted forward by 30s
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");
        let output_path = temp.path().join("merged");

        // Create DB1 with monotonic timestamps - "boost" at 45s
        create_test_db_with_timestamps(
            &db1_path,
            &[(
                "sim.velocity",
                0,
                &[35_000_000, 45_000_000, 55_000_000], // 35s, 45s (boost), 55s
            )],
        )
        .unwrap();

        // Create DB2 with monotonic timestamps - "boost" at 15s
        create_test_db_with_timestamps(
            &db2_path,
            &[(
                "truth.velocity",
                0,
                &[5_000_000, 15_000_000, 25_000_000], // 5s, 15s (boost), 25s
            )],
        )
        .unwrap();

        // Merge with alignment: align DB1's 45s with DB2's 15s
        // Since align1 > align2, DB2 gets shifted forward by 30s
        run(
            db1_path,
            db2_path,
            output_path.clone(),
            Some("sim".to_string()),
            Some("truth".to_string()),
            false,
            true,
            Some(45.0), // align1: boost at 45s in DB1
            Some(15.0), // align2: boost at 15s in DB2
        )
        .unwrap();

        // Verify DB1 (sim) was NOT shifted
        let sim_id = ComponentId::new("sim_sim.velocity");
        let sim_index = output_path.join(sim_id.to_string()).join("index");
        let (sim_start_ts, sim_timestamps) = read_index_file(&sim_index);

        assert_eq!(sim_start_ts, 0);
        assert_eq!(sim_timestamps[0], 35_000_000);
        assert_eq!(sim_timestamps[1], 45_000_000); // boost event (reference)
        assert_eq!(sim_timestamps[2], 55_000_000);

        // Verify DB2 (truth) was shifted forward by 30s
        let truth_id = ComponentId::new("truth_truth.velocity");
        let truth_index = output_path.join(truth_id.to_string()).join("index");
        let (truth_start_ts, truth_timestamps) = read_index_file(&truth_index);

        // Original: 0, [5s, 15s, 25s] -> After +30s shift: 30s, [35s, 45s, 55s]
        assert_eq!(truth_start_ts, 30_000_000); // 0 + 30s
        assert_eq!(truth_timestamps[0], 35_000_000); // 5s + 30s
        assert_eq!(truth_timestamps[1], 45_000_000); // 15s + 30s (boost, now aligned)
        assert_eq!(truth_timestamps[2], 55_000_000); // 25s + 30s
    }

    #[test]
    fn test_alignment_rejects_nan_and_infinity() {
        let temp = TempDir::new().unwrap();
        let db1_path = temp.path().join("db1");
        let db2_path = temp.path().join("db2");

        create_test_db(&db1_path, &[]).unwrap();
        create_test_db(&db2_path, &[]).unwrap();

        // Test NaN in align1
        let output_path = temp.path().join("merged_nan1");
        let result = run(
            db1_path.clone(),
            db2_path.clone(),
            output_path,
            None,
            None,
            false,
            true,
            Some(f64::NAN),
            Some(100.0),
        );
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("finite"),
            "Expected 'finite' in error: {}",
            err_msg
        );

        // Test NaN in align2
        let output_path = temp.path().join("merged_nan2");
        let result = run(
            db1_path.clone(),
            db2_path.clone(),
            output_path,
            None,
            None,
            false,
            true,
            Some(100.0),
            Some(f64::NAN),
        );
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("finite"),
            "Expected 'finite' in error: {}",
            err_msg
        );

        // Test positive infinity in align1
        let output_path = temp.path().join("merged_inf1");
        let result = run(
            db1_path.clone(),
            db2_path.clone(),
            output_path,
            None,
            None,
            false,
            true,
            Some(f64::INFINITY),
            Some(100.0),
        );
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("finite"),
            "Expected 'finite' in error: {}",
            err_msg
        );

        // Test negative infinity in align2
        let output_path = temp.path().join("merged_neginf2");
        let result = run(
            db1_path.clone(),
            db2_path.clone(),
            output_path,
            None,
            None,
            false,
            true,
            Some(100.0),
            Some(f64::NEG_INFINITY),
        );
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("finite"),
            "Expected 'finite' in error: {}",
            err_msg
        );
    }
}
