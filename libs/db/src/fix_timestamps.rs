//! Database timestamp migration tool
//!
//! Fixes databases where some components have monotonic timestamps (from device boot time)
//! instead of wall-clock timestamps.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::utils::component_label;
use crate::{Error, MetadataExt};
use impeller2_wkt::DbConfig;

const HEADER_SIZE: usize = 24; // committed_len (8) + head_len (8) + extra (8)

/// Fix timestamps in an elodin-db database by aligning non-reference components to the reference.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `dry_run` - If true, only show what would be changed without modifying
/// * `auto_confirm` - If true, skip the confirmation prompt (for non-interactive use)
/// * `reference` - Which clock to treat as the reference set (monotonic or wall-clock)
///
/// # Returns
/// * `Ok(())` if successful or no changes needed
/// * `Err(Error)` if the operation fails
pub fn run(
    db_path: PathBuf,
    dry_run: bool,
    auto_confirm: bool,
    reference: ReferenceClock,
) -> Result<(), Error> {
    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }

    println!("Analyzing database: {}", db_path.display());
    println!("Reference clock: {}", reference);
    let db_state_path = db_path.join("db_state");
    if db_state_path.exists() {
        match DbConfig::read(&db_state_path) {
            Ok(config) => {
                if let Some(start_ts) = config.time_start_timestamp_micros() {
                    println!(
                        "time.start_timestamp: {} ({:.3}s)",
                        start_ts,
                        start_ts as f64 / 1e6
                    );
                } else {
                    println!("time.start_timestamp: <not set>");
                }
            }
            Err(err) => {
                eprintln!(
                    "Warning: failed to read db_state at {}: {}",
                    db_state_path.display(),
                    err
                );
            }
        }
    }
    if dry_run {
        println!("DRY RUN - no changes will be made");
    }
    println!();

    // Collect all components and their timestamp ranges
    let mut components: HashMap<PathBuf, ComponentInfo> = HashMap::new();

    for entry in fs::read_dir(&db_path)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        // Skip non-component directories
        let dir_name = path.file_name().unwrap().to_string_lossy();
        if dir_name == "msgs" || dir_name == "db_state" {
            continue;
        }

        let index_path = path.join("index");
        if !index_path.exists() {
            continue;
        }

        match analyze_component(&index_path) {
            Ok(info) => {
                components.insert(path, info);
            }
            Err(e) => {
                eprintln!("Warning: Failed to analyze {}: {}", path.display(), e);
            }
        }
    }

    if components.is_empty() {
        println!("No components found in database.");
        return Ok(());
    }

    // Separate into reference (good) and to-fix (other) timestamps.
    // Wall-clock timestamps from 2020+ are considered wall-clock.
    // Timestamps before 2000 are considered monotonic.
    // NOTE: Elodin timestamps are in MICROSECONDS, not nanoseconds
    let cutoff_2000: i64 = 946_684_800_000_000; // 2000-01-01 in microseconds
    let cutoff_2020: i64 = 1_577_836_800_000_000; // 2020-01-01 in microseconds

    let mut good_components: Vec<(&PathBuf, &ComponentInfo)> = Vec::new();
    let mut bad_components: Vec<(&PathBuf, &ComponentInfo)> = Vec::new();

    for (path, info) in &components {
        // Skip empty components
        if info.count == 0 || info.min_timestamp == i64::MAX {
            continue;
        }

        let is_wall_clock = info.min_timestamp > cutoff_2020;
        let is_monotonic = info.min_timestamp < cutoff_2000;

        let (is_good, is_bad) = match reference {
            ReferenceClock::WallClock => (is_wall_clock, is_monotonic),
            ReferenceClock::Monotonic => (is_monotonic, is_wall_clock),
        };

        if is_good {
            good_components.push((path, info));
        } else if is_bad {
            bad_components.push((path, info));
        } else {
            println!(
                "Ambiguous component (between 2000-2020): {}",
                path.display()
            );
        }
    }

    println!(
        "Components with {} timestamps (reference): {}",
        reference.label(),
        good_components.len()
    );
    println!(
        "Components with {} timestamps (to fix): {}",
        reference.other_label(),
        bad_components.len()
    );
    println!();

    if bad_components.is_empty() {
        println!("No components need fixing!");
        return Ok(());
    }

    if good_components.is_empty() {
        eprintln!(
            "Error: No reference components with {} timestamps found.",
            reference.label()
        );
        eprintln!("Cannot determine the correct time offset.");
        return Err(Error::FixTimestamps(format!(
            "No reference components with {} timestamps found",
            reference.label()
        )));
    }

    // Calculate offset: reference_min - to_fix_min
    let reference_min = good_components
        .iter()
        .map(|(_, info)| info.min_timestamp)
        .min()
        .unwrap();

    let to_fix_min = bad_components
        .iter()
        .map(|(_, info)| info.min_timestamp)
        .min()
        .unwrap();

    let offset = reference_min - to_fix_min;

    println!("Calculated offset:");
    println!(
        "  Reference min: {} ({:.3}s since epoch)",
        reference_min,
        reference_min as f64 / 1e6
    );
    println!(
        "  To-fix min:    {} ({:.3}s since epoch)",
        to_fix_min,
        to_fix_min as f64 / 1e6
    );
    println!(
        "  Offset:         {} ({:.3} days)",
        offset,
        offset as f64 / 1e6 / 86400.0
    );
    println!();

    // Apply the offset to bad components
    println!("Components to fix:");
    for (path, info) in &bad_components {
        let label = component_label(path);
        println!(
            "  {} ({} entries, min: {:.3}s -> {:.3}s)",
            label,
            info.count,
            info.min_timestamp as f64 / 1e6,
            (info.min_timestamp + offset) as f64 / 1e6
        );
    }
    println!();

    if dry_run {
        println!("Dry run complete. Run without --dry-run to apply changes.");
        return Ok(());
    }

    if !auto_confirm {
        print!("Apply timestamp fixes? [y/N] ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    println!();
    println!("Applying fixes...");

    for (path, info) in &bad_components {
        let index_path = path.join("index");
        match apply_offset(&index_path, offset, info.count) {
            Ok(()) => {
                println!("  Fixed: {}", path.display());
            }
            Err(e) => {
                eprintln!("  Error fixing {}: {}", path.display(), e);
            }
        }
    }

    // Update time.start_timestamp in db_state to match the corrected data
    let db_state_path = db_path.join("db_state");
    if db_state_path.exists()
        && let Ok(mut config) = DbConfig::read(&db_state_path)
        && let Some(old_start) = config.time_start_timestamp_micros()
    {
        let is_bad = match reference {
            ReferenceClock::WallClock => old_start < cutoff_2000,
            ReferenceClock::Monotonic => old_start > cutoff_2020,
        };
        if is_bad {
            let new_start = old_start + offset;
            println!(
                "Updating time.start_timestamp in db_state: {} -> {}",
                old_start, new_start
            );
            config.set_time_start_timestamp_micros(new_start);
            if let Err(e) = config.write(&db_state_path) {
                eprintln!("Warning: failed to update db_state: {}", e);
            }
        }
    }

    println!();
    println!("Done! Database timestamps have been normalized.");
    println!("You may need to restart elodin-db to see the changes.");

    Ok(())
}

struct ComponentInfo {
    min_timestamp: i64,
    count: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceClock {
    WallClock,
    Monotonic,
}

impl ReferenceClock {
    fn label(self) -> &'static str {
        match self {
            ReferenceClock::WallClock => "wall-clock",
            ReferenceClock::Monotonic => "monotonic",
        }
    }

    fn other_label(self) -> &'static str {
        match self {
            ReferenceClock::WallClock => "monotonic",
            ReferenceClock::Monotonic => "wall-clock",
        }
    }
}

impl std::fmt::Display for ReferenceClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

fn analyze_component(index_path: &Path) -> Result<ComponentInfo, std::io::Error> {
    let mut file = File::open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap());
    let _head_len = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let _start_timestamp = i64::from_le_bytes(header[16..24].try_into().unwrap());

    let data_len = committed_len as usize - HEADER_SIZE;
    let count = data_len / 8; // Each timestamp is 8 bytes

    if count == 0 {
        return Ok(ComponentInfo {
            min_timestamp: i64::MAX,
            count: 0,
        });
    }

    // Read all timestamps
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    let mut min_ts = i64::MAX;

    for chunk in data.chunks_exact(8) {
        let ts = i64::from_le_bytes(chunk.try_into().unwrap());
        min_ts = min_ts.min(ts);
    }

    Ok(ComponentInfo {
        min_timestamp: min_ts,
        count,
    })
}

fn apply_offset(index_path: &Path, offset: i64, count: usize) -> Result<(), std::io::Error> {
    let mut file = OpenOptions::new().read(true).write(true).open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    // Update start_timestamp in header
    let start_ts = i64::from_le_bytes(header[16..24].try_into().unwrap());
    let new_start_ts = start_ts + offset;
    header[16..24].copy_from_slice(&new_start_ts.to_le_bytes());

    // Read all timestamps
    let data_len = count * 8;
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    // Apply offset to all timestamps
    for chunk in data.chunks_exact_mut(8) {
        let ts = i64::from_le_bytes(chunk.try_into().unwrap());
        let new_ts = ts + offset;
        chunk.copy_from_slice(&new_ts.to_le_bytes());
    }

    // Write back
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    file.write_all(&data)?;
    file.sync_all()?;

    Ok(())
}
