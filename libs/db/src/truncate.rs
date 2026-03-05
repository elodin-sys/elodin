//! Database truncation tool
//!
//! Clears all data from an Elodin database while preserving schemas and metadata.

use std::io::Write as IoWrite;
use std::path::PathBuf;

use crate::{DB, Error};

/// Truncate all data in an elodin-db database, preserving schemas and metadata.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `dry_run` - If true, only show what would be truncated without modifying
/// * `auto_confirm` - If true, skip the confirmation prompt (for non-interactive use)
///
/// # Returns
/// * `Ok(())` if successful
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

    // Open the database
    let db = DB::open(db_path)?;

    // Get component and message log counts
    let (component_count, msg_log_count, total_data_entries) = db.with_state(|state| {
        let component_count = state.components.len();
        let msg_log_count = state.msg_logs.len();

        // Count total data entries across all components
        // Each timestamp entry is 8 bytes (i64)
        let total_data_entries: u64 = state
            .components
            .values()
            .map(|c| c.time_series.index().len() / 8)
            .sum();

        (component_count, msg_log_count, total_data_entries)
    });

    println!("Database contents:");
    println!("  Components: {}", component_count);
    println!("  Message logs: {}", msg_log_count);
    println!("  Total data entries: {}", total_data_entries);
    println!();

    if component_count == 0 && msg_log_count == 0 {
        println!("Database is already empty. Nothing to truncate.");
        return Ok(());
    }

    if total_data_entries == 0 {
        println!("No data entries to truncate. Components and message logs are already empty.");
        return Ok(());
    }

    println!(
        "Truncating will clear all {} data entries while preserving:",
        total_data_entries
    );
    println!("  - Component schemas");
    println!("  - Component metadata");
    println!("  - Message log metadata");
    println!();

    if dry_run {
        println!("Dry run complete. Run without --dry-run to apply changes.");
        return Ok(());
    }

    if !auto_confirm {
        print!("Truncate all data? This cannot be undone. [y/N] ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return Ok(());
        }
    }

    println!("Truncating database...");
    db.truncate();

    println!();
    println!("Done! Database has been truncated.");
    println!("All data cleared. Schemas and metadata preserved.");

    Ok(())
}
