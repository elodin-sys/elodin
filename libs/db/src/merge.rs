//! Database merge tool
//!
//! Merges two Elodin databases into one, applying optional prefixes to component names
//! to avoid namespace collisions. This enables viewing simulation and real-world
//! telemetry data simultaneously in the Elodin Editor.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};

use crate::{Error, MetadataExt};
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
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(Error)` if the operation fails
pub fn run(
    db1: PathBuf,
    db2: PathBuf,
    output: PathBuf,
    prefix1: Option<String>,
    prefix2: Option<String>,
    dry_run: bool,
    auto_confirm: bool,
) -> Result<(), Error> {
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

    // Check for potential collisions if both prefixes are the same (including both None)
    if prefix1 == prefix2 {
        let collision_check = check_for_collisions(&db1, &db2)?;
        if !collision_check.is_empty() {
            eprintln!("Warning: Potential component name collisions detected:");
            for name in &collision_check {
                eprintln!("  - {}", name);
            }
            if prefix1.is_none() {
                eprintln!("\nConsider using --prefix1 and --prefix2 to avoid collisions.");
            }
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
    let parent_dir = output
        .parent()
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

    // Merge databases
    print!("Merging {}... ", db1.display());
    io::stdout().flush()?;
    let stats1 = merge_database(&db1, &tmp_output, prefix1.as_deref())?;
    println!("done ({} components)", stats1.components_copied);

    print!("Merging {}... ", db2.display());
    io::stdout().flush()?;
    let stats2 = merge_database(&db2, &tmp_output, prefix2.as_deref())?;
    println!("done ({} components)", stats2.components_copied);

    // Merge db_state
    print!("Writing db_state... ");
    io::stdout().flush()?;
    merge_db_state(&db1, &db2, &tmp_output)?;
    println!("done");

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
fn check_for_collisions(db1: &Path, db2: &Path) -> Result<Vec<String>, Error> {
    let names1 = collect_component_names(db1)?;
    let names2 = collect_component_names(db2)?;

    let mut collisions = Vec::new();
    for name in &names1 {
        if names2.contains(name) {
            collisions.push(name.clone());
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
fn merge_database(source: &Path, target: &Path, prefix: Option<&str>) -> Result<MergeStats, Error> {
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
            merge_message_logs(&path, &target.join("msgs"), &mut stats)?;
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
        copy_component_with_prefix(&path, target, prefix)?;
        stats.components_copied += 1;
    }

    Ok(stats)
}

/// Copy a component directory with a new prefixed name
fn copy_component_with_prefix(
    src_component_dir: &Path,
    target_db_dir: &Path,
    prefix: Option<&str>,
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

    // Copy index file (unchanged)
    let index_src = src_component_dir.join("index");
    if index_src.exists() {
        copy_file_native(&index_src, &new_component_dir.join("index"))?;
    }

    // Copy data file (unchanged)
    let data_src = src_component_dir.join("data");
    if data_src.exists() {
        copy_file_native(&data_src, &new_component_dir.join("data"))?;
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
fn merge_message_logs(
    source_msgs: &Path,
    target_msgs: &Path,
    stats: &mut MergeStats,
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

        // Copy the message log directory
        copy_dir_native(&path, &target_msg_dir)?;
        stats.msg_logs_copied += 1;
    }

    sync_dir(target_msgs)?;
    Ok(())
}

/// Merge db_state from both databases, pruning schematics
fn merge_db_state(db1: &Path, db2: &Path, target: &Path) -> Result<(), Error> {
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

    // Use earliest time.start_timestamp from both
    let ts1 = config1.time_start_timestamp_micros();
    let ts2 = config2.time_start_timestamp_micros();
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

/// Copy a file using reflink if available
fn copy_file_native(src: &Path, dst: &Path) -> Result<(), Error> {
    let metadata = fs::metadata(src)?;
    reflink_copy::reflink_or_copy(src, dst)?;
    fs::set_permissions(dst, metadata.permissions())?;
    let file = File::open(dst)?;
    file.sync_all()?;
    if let Some(parent) = dst.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

/// Recursively copy a directory
fn copy_dir_native(src: &Path, dst: &Path) -> Result<(), Error> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_native(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            copy_file_native(&src_path, &dst_path)?;
        }
    }
    let metadata = fs::metadata(src)?;
    fs::set_permissions(dst, metadata.permissions())?;
    Ok(())
}

/// Sync a directory to ensure durability
fn sync_dir(path: &Path) -> io::Result<()> {
    #[cfg(target_family = "unix")]
    {
        let dir = File::open(path)?;
        dir.sync_all()
    }
    #[cfg(not(target_family = "unix"))]
    {
        let _ = path;
        Ok(())
    }
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
        let result = run(db1_path, db2_path, output_path, None, None, false, true);

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
}
