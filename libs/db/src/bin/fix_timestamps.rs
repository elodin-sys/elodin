#!/usr/bin/env -S cargo +nightly -Zscript
//! Database timestamp migration tool
//!
//! Fixes databases where some components have monotonic timestamps (from device boot time)
//! instead of wall-clock timestamps.
//!
//! Usage:
//!   cargo run --bin fix_timestamps -- <db_path> [--dry-run]

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const HEADER_SIZE: usize = 24; // committed_len (8) + head_len (8) + extra (8)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <db_path> [--dry-run]", args[0]);
        eprintln!();
        eprintln!("Fixes timestamps in an elodin-db database by aligning monotonic");
        eprintln!("timestamps (1970 dates) with wall-clock timestamps (2025 dates).");
        std::process::exit(1);
    }

    let db_path = PathBuf::from(&args[1]);
    let dry_run = args.get(2).map(|s| s == "--dry-run").unwrap_or(false);

    if !db_path.exists() {
        eprintln!("Error: Database path does not exist: {}", db_path.display());
        std::process::exit(1);
    }

    println!("Analyzing database: {}", db_path.display());
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

    // Separate into good (wall-clock) and bad (monotonic) timestamps
    // Wall-clock timestamps from 2020+ are considered "good"
    // Timestamps before 2000 are considered "bad" (monotonic)
    // NOTE: Elodin timestamps are in MICROSECONDS, not nanoseconds
    let cutoff_2000: i64 = 946684800_000_000; // 2000-01-01 in microseconds
    let cutoff_2020: i64 = 1577836800_000_000; // 2020-01-01 in microseconds

    let mut good_components: Vec<(&PathBuf, &ComponentInfo)> = Vec::new();
    let mut bad_components: Vec<(&PathBuf, &ComponentInfo)> = Vec::new();

    for (path, info) in &components {
        // Skip empty components
        if info.count == 0 || info.min_timestamp == i64::MAX {
            continue;
        }

        if info.min_timestamp > cutoff_2020 {
            good_components.push((path, info));
        } else if info.min_timestamp < cutoff_2000 {
            bad_components.push((path, info));
        } else {
            println!(
                "Ambiguous component (between 2000-2020): {}",
                path.display()
            );
        }
    }

    println!(
        "Components with wall-clock timestamps (2020+): {}",
        good_components.len()
    );
    println!(
        "Components with monotonic timestamps (pre-2000): {}",
        bad_components.len()
    );
    println!();

    if bad_components.is_empty() {
        println!("No components need fixing!");
        return Ok(());
    }

    if good_components.is_empty() {
        eprintln!("Error: No reference components with wall-clock timestamps found.");
        eprintln!("Cannot determine the correct time offset.");
        std::process::exit(1);
    }

    // Calculate offset: wall_clock_min - monotonic_min
    let wall_clock_min = good_components
        .iter()
        .map(|(_, info)| info.min_timestamp)
        .min()
        .unwrap();

    let monotonic_min = bad_components
        .iter()
        .map(|(_, info)| info.min_timestamp)
        .min()
        .unwrap();

    let offset = wall_clock_min - monotonic_min;

    println!("Calculated offset:");
    println!(
        "  Wall-clock min: {} ({:.3}s since epoch)",
        wall_clock_min,
        wall_clock_min as f64 / 1e6
    );
    println!(
        "  Monotonic min:  {} ({:.3}s since epoch)",
        monotonic_min,
        monotonic_min as f64 / 1e6
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
        let metadata_path = path.join("metadata");
        let name = if metadata_path.exists() {
            read_component_name(&metadata_path)
                .unwrap_or_else(|_| path.file_name().unwrap().to_string_lossy().to_string())
        } else {
            path.file_name().unwrap().to_string_lossy().to_string()
        };

        println!(
            "  {} ({} entries, min: {:.3}s -> {:.3}s)",
            name,
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

    print!("Apply timestamp fixes? [y/N] ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Aborted.");
        return Ok(());
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

    println!();
    println!("Done! Database timestamps have been normalized.");
    println!("You may need to restart elodin-db to see the changes.");

    Ok(())
}

struct ComponentInfo {
    min_timestamp: i64,
    count: usize,
}

fn analyze_component(index_path: &Path) -> Result<ComponentInfo, Box<dyn std::error::Error>> {
    let mut file = File::open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into()?);
    let _head_len = u64::from_le_bytes(header[8..16].try_into()?);
    let _start_timestamp = i64::from_le_bytes(header[16..24].try_into()?);

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
        let ts = i64::from_le_bytes(chunk.try_into()?);
        min_ts = min_ts.min(ts);
    }

    Ok(ComponentInfo {
        min_timestamp: min_ts,
        count,
    })
}

fn apply_offset(
    index_path: &Path,
    offset: i64,
    count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().read(true).write(true).open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    // Update start_timestamp in header
    let start_ts = i64::from_le_bytes(header[16..24].try_into()?);
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

fn read_component_name(metadata_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    // The metadata is stored as postcard-serialized ComponentMetadata
    // For simplicity, we'll just try to extract the name string
    let data = fs::read(metadata_path)?;

    // ComponentMetadata has component_id (u64), then name (String), then metadata
    // postcard encodes String as varint length + utf8 bytes
    if data.len() < 9 {
        return Err("metadata too short".into());
    }

    // Skip component_id (8 bytes)
    let name_data = &data[8..];

    // Read varint length
    let (len, bytes_read) = decode_varint(name_data)?;
    let name_start = bytes_read;
    let name_end = name_start + len as usize;

    if name_end > name_data.len() {
        return Err("name length exceeds data".into());
    }

    let name = String::from_utf8(name_data[name_start..name_end].to_vec())?;
    Ok(name)
}

fn decode_varint(data: &[u8]) -> Result<(u64, usize), Box<dyn std::error::Error>> {
    let mut result: u64 = 0;
    let mut shift = 0;

    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift > 63 {
            return Err("varint too long".into());
        }
    }

    Err("incomplete varint".into())
}
