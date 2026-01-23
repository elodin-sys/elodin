//! Database pruning tool
//!
//! Removes empty components from an Elodin database.

use std::fs::{self, File};
use std::io::{Read, Write as IoWrite};
use std::path::{Path, PathBuf};

use crate::Error;
use crate::utils::component_label;

const HEADER_SIZE: usize = 24; // committed_len (8) + head_len (8) + extra (8)

/// Remove empty components from an elodin-db database.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `dry_run` - If true, only show what would be pruned without modifying
/// * `auto_confirm` - If true, skip the confirmation prompt (for non-interactive use)
///
/// # Returns
/// * `Ok(())` if successful or no changes needed
/// * `Err(Error)` if the operation fails
pub fn run(db_path: PathBuf, dry_run: bool, auto_confirm: bool) -> Result<(), Error> {
    if !db_path.exists() {
        return Err(Error::MissingDbState(db_path));
    }

    println!("Analyzing database: {}", db_path.display());
    if dry_run {
        println!("DRY RUN - no changes will be made");
    }
    println!();

    // Find all empty components
    let mut empty_components: Vec<PathBuf> = Vec::new();
    let mut total_components = 0;

    for entry in fs::read_dir(&db_path)? {
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

        let index_path = path.join("index");
        if !index_path.exists() {
            continue;
        }

        total_components += 1;

        match is_component_empty(&index_path) {
            Ok(true) => {
                empty_components.push(path);
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("Warning: Failed to analyze {}: {}", path.display(), e);
            }
        }
    }

    println!("Total components: {}", total_components);
    println!("Empty components: {}", empty_components.len());
    println!();

    if empty_components.is_empty() {
        println!("No empty components to prune.");
        return Ok(());
    }

    // Display empty components
    println!("Empty components to prune:");
    for path in &empty_components {
        println!("  {}", component_label(path));
    }
    println!();

    if dry_run {
        println!("Dry run complete. Run without --dry-run to apply changes.");
        return Ok(());
    }

    if !auto_confirm {
        print!("Prune {} empty components? [y/N] ", empty_components.len());
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    println!("Pruning empty components...");
    let mut pruned = 0;
    for path in &empty_components {
        match fs::remove_dir_all(path) {
            Ok(()) => {
                println!("  Pruned: {}", component_label(path));
                pruned += 1;
            }
            Err(e) => {
                eprintln!("  Error pruning {}: {}", path.display(), e);
            }
        }
    }

    println!();
    println!("Done! Pruned {} empty components.", pruned);

    Ok(())
}

/// Check if a component is empty (has no data entries).
fn is_component_empty(index_path: &Path) -> Result<bool, std::io::Error> {
    let mut file = File::open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap());
    let data_len = committed_len as usize - HEADER_SIZE;
    let count = data_len / 8; // Each timestamp is 8 bytes

    Ok(count == 0)
}
