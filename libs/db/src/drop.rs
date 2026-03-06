//! Database component drop tool
//!
//! Removes components from an Elodin database with support for fuzzy matching,
//! glob patterns, and bulk removal.

use std::fs::{self, File};
use std::io::{Read, Write as IoWrite};
use std::path::{Path, PathBuf};

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;

use crate::utils::component_label;
use crate::{Error, MetadataExt};
use impeller2_wkt::ComponentMetadata;

const HEADER_SIZE: usize = 24; // committed_len (8) + head_len (8) + start_timestamp (8)

/// Information about a component for dropping
#[derive(Debug)]
struct ComponentInfo {
    path: PathBuf,
    name: String,
    entry_count: usize,
    fuzzy_score: Option<i64>,
}

/// Matching mode for selecting components to drop
#[derive(Debug)]
pub enum MatchMode {
    /// Fuzzy match against component names
    Fuzzy(String),
    /// Glob pattern match (supports * and ?)
    Pattern(String),
    /// Drop all components
    All,
}

/// Drop components from an elodin-db database.
///
/// # Arguments
/// * `db_path` - Path to the database directory
/// * `match_mode` - How to select components to drop
/// * `dry_run` - If true, only show what would be dropped without modifying
/// * `auto_confirm` - If true, skip the confirmation prompt
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(Error)` if the operation fails
pub fn run(
    db_path: PathBuf,
    match_mode: MatchMode,
    dry_run: bool,
    auto_confirm: bool,
) -> Result<(), Error> {
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

    // Collect all components
    let all_components = collect_all_components(&db_path)?;

    if all_components.is_empty() {
        println!("No components found in database.");
        return Ok(());
    }

    // Find matching components based on mode
    let components_to_drop = match &match_mode {
        MatchMode::Fuzzy(pattern) => fuzzy_match_components(&all_components, pattern),
        MatchMode::Pattern(pattern) => glob_match_components(&all_components, pattern),
        MatchMode::All => all_components,
    };

    if components_to_drop.is_empty() {
        match &match_mode {
            MatchMode::Fuzzy(pattern) => {
                println!("No components matching \"{}\" found.", pattern);
            }
            MatchMode::Pattern(pattern) => {
                println!("No components matching pattern \"{}\" found.", pattern);
            }
            MatchMode::All => {
                println!("No components found in database.");
            }
        }
        return Ok(());
    }

    // Display components to drop
    match &match_mode {
        MatchMode::Fuzzy(pattern) => {
            println!("Components matching \"{}\":", pattern);
        }
        MatchMode::Pattern(pattern) => {
            println!("Components matching pattern \"{}\":", pattern);
        }
        MatchMode::All => {
            println!("All components:");
        }
    }

    for (i, component) in components_to_drop.iter().enumerate() {
        let score_str = component
            .fuzzy_score
            .map(|s| format!(" (score: {})", s))
            .unwrap_or_default();
        println!(
            "  {}. {}{} - {} entries",
            i + 1,
            component.name,
            score_str,
            component.entry_count
        );
    }
    println!();

    // Calculate total entries being dropped
    let total_entries: usize = components_to_drop.iter().map(|c| c.entry_count).sum();
    println!(
        "WARNING: This will permanently delete {} component{} ({} total entries).",
        components_to_drop.len(),
        if components_to_drop.len() == 1 {
            ""
        } else {
            "s"
        },
        total_entries
    );
    println!("This action cannot be undone.");
    println!();

    if dry_run {
        println!("Dry run complete. Run without --dry-run to apply changes.");
        return Ok(());
    }

    // Confirm
    if !auto_confirm {
        print!(
            "Drop {} component{}? [y/N] ",
            components_to_drop.len(),
            if components_to_drop.len() == 1 {
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

    println!("Dropping components...");

    let mut dropped_count = 0;
    for component in &components_to_drop {
        match fs::remove_dir_all(&component.path) {
            Ok(()) => {
                println!("  Dropped: {}", component.name);
                dropped_count += 1;
            }
            Err(e) => {
                eprintln!("  Error dropping {}: {}", component.name, e);
            }
        }
    }

    println!();
    println!(
        "Done! Dropped {} component{}.",
        dropped_count,
        if dropped_count == 1 { "" } else { "s" }
    );

    Ok(())
}

/// Collect all components from the database.
fn collect_all_components(db_path: &Path) -> Result<Vec<ComponentInfo>, Error> {
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

        // Check if it has an index file (valid component)
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

        // Get entry count
        let entry_count = match get_entry_count(&index_path) {
            Ok(count) => count,
            Err(e) => {
                eprintln!("Warning: Failed to analyze {}: {}", path.display(), e);
                0
            }
        };

        components.push(ComponentInfo {
            path,
            name,
            entry_count,
            fuzzy_score: None,
        });
    }

    // Sort by name for consistent output
    components.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(components)
}

/// Fuzzy match components against a pattern.
fn fuzzy_match_components(components: &[ComponentInfo], pattern: &str) -> Vec<ComponentInfo> {
    let matcher = SkimMatcherV2::default().smart_case();
    let mut matches: Vec<ComponentInfo> = Vec::new();

    for component in components {
        if let Some(score) = matcher.fuzzy_match(&component.name, pattern) {
            matches.push(ComponentInfo {
                path: component.path.clone(),
                name: component.name.clone(),
                entry_count: component.entry_count,
                fuzzy_score: Some(score),
            });
        }
    }

    // Sort by score descending (best matches first)
    matches.sort_by(|a, b| b.fuzzy_score.unwrap_or(0).cmp(&a.fuzzy_score.unwrap_or(0)));

    matches
}

/// Glob pattern match components.
/// Supports * (any characters) and ? (single character).
fn glob_match_components(components: &[ComponentInfo], pattern: &str) -> Vec<ComponentInfo> {
    let mut matches: Vec<ComponentInfo> = Vec::new();

    for component in components {
        if glob_matches(&component.name, pattern) {
            matches.push(ComponentInfo {
                path: component.path.clone(),
                name: component.name.clone(),
                entry_count: component.entry_count,
                fuzzy_score: None,
            });
        }
    }

    // Sort by name for consistent output
    matches.sort_by(|a, b| a.name.cmp(&b.name));

    matches
}

/// Simple glob pattern matching.
/// Supports:
/// - `*` matches any sequence of characters
/// - `?` matches any single character
fn glob_matches(text: &str, pattern: &str) -> bool {
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    glob_matches_impl(&text_chars, &pattern_chars)
}

fn glob_matches_impl(text: &[char], pattern: &[char]) -> bool {
    if pattern.is_empty() {
        return text.is_empty();
    }

    let (p_first, p_rest) = pattern.split_first().unwrap();

    match p_first {
        '*' => {
            // * matches zero or more characters
            // Try matching zero characters, or one character and continue
            glob_matches_impl(text, p_rest)
                || (!text.is_empty() && glob_matches_impl(&text[1..], pattern))
        }
        '?' => {
            // ? matches exactly one character
            !text.is_empty() && glob_matches_impl(&text[1..], p_rest)
        }
        c => {
            // Literal character match
            !text.is_empty() && text[0] == *c && glob_matches_impl(&text[1..], p_rest)
        }
    }
}

/// Get the number of entries in a component.
fn get_entry_count(index_path: &Path) -> Result<usize, std::io::Error> {
    let mut file = File::open(index_path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    let committed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let data_len = committed_len.saturating_sub(HEADER_SIZE);
    let count = data_len / 8; // Each timestamp is 8 bytes

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_glob_matches_exact() {
        assert!(glob_matches("rocket.velocity", "rocket.velocity"));
        assert!(!glob_matches("rocket.velocity", "rocket.position"));
    }

    #[test]
    fn test_glob_matches_star() {
        assert!(glob_matches("rocket.velocity", "rocket.*"));
        assert!(glob_matches("rocket.position", "rocket.*"));
        assert!(glob_matches("rocket.velocity", "*.velocity"));
        assert!(glob_matches("drone.velocity", "*.velocity"));
        assert!(glob_matches("anything", "*"));
        assert!(glob_matches("", "*"));
        assert!(!glob_matches("rocket.velocity", "drone.*"));
    }

    #[test]
    fn test_glob_matches_question() {
        assert!(glob_matches("cat", "c?t"));
        assert!(glob_matches("cot", "c?t"));
        assert!(!glob_matches("ct", "c?t"));
        assert!(!glob_matches("cart", "c?t"));
    }

    #[test]
    fn test_glob_matches_combined() {
        assert!(glob_matches("rocket.velocity", "r*.v*"));
        assert!(glob_matches("rocket.velocity", "r?cket.*"));
        assert!(glob_matches("test123.data", "test???.data"));
        assert!(!glob_matches("test12.data", "test???.data"));
    }

    #[test]
    fn test_fuzzy_match_components() {
        let components = vec![
            ComponentInfo {
                path: PathBuf::from("/test/1"),
                name: "rocket.velocity".to_string(),
                entry_count: 100,
                fuzzy_score: None,
            },
            ComponentInfo {
                path: PathBuf::from("/test/2"),
                name: "rocket.position".to_string(),
                entry_count: 200,
                fuzzy_score: None,
            },
            ComponentInfo {
                path: PathBuf::from("/test/3"),
                name: "drone.velocity".to_string(),
                entry_count: 50,
                fuzzy_score: None,
            },
        ];

        // Should match "rocket" related components
        let matches = fuzzy_match_components(&components, "rocket");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.name.contains("rocket")));

        // Should match "velocity" related components
        let matches = fuzzy_match_components(&components, "vel");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.name.contains("velocity")));
    }

    #[test]
    fn test_glob_match_components() {
        let components = vec![
            ComponentInfo {
                path: PathBuf::from("/test/1"),
                name: "rocket.velocity".to_string(),
                entry_count: 100,
                fuzzy_score: None,
            },
            ComponentInfo {
                path: PathBuf::from("/test/2"),
                name: "rocket.position".to_string(),
                entry_count: 200,
                fuzzy_score: None,
            },
            ComponentInfo {
                path: PathBuf::from("/test/3"),
                name: "drone.velocity".to_string(),
                entry_count: 50,
                fuzzy_score: None,
            },
        ];

        // Pattern matching "rocket.*"
        let matches = glob_match_components(&components, "rocket.*");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.name.starts_with("rocket.")));

        // Pattern matching "*.velocity"
        let matches = glob_match_components(&components, "*.velocity");
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.name.ends_with(".velocity")));
    }

    /// Helper to create an index file with a specific entry count
    fn create_index_file(path: &Path, entry_count: usize) {
        use std::io::Write;

        let data_len = entry_count * 8;
        let committed_len = (HEADER_SIZE + data_len) as u64;

        let mut data = vec![0u8; HEADER_SIZE + data_len];

        // Header: committed_len (8) + head_len (8) + start_timestamp (8)
        data[0..8].copy_from_slice(&committed_len.to_le_bytes());
        data[8..16].copy_from_slice(&0u64.to_le_bytes()); // head_len = 0
        data[16..24].copy_from_slice(&0i64.to_le_bytes()); // start_timestamp

        // Fill with dummy timestamps
        for i in 0..entry_count {
            let offset = HEADER_SIZE + i * 8;
            let ts = (i as i64) * 1_000_000;
            data[offset..offset + 8].copy_from_slice(&ts.to_le_bytes());
        }

        let mut file = File::create(path).unwrap();
        file.write_all(&data).unwrap();
    }

    /// Helper to create a test database with components
    fn create_test_db(
        dir: &Path,
        components: &[(&str, usize)], // (name, entry_count)
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
        for (name, entry_count) in components {
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

            // Write index with entry count
            create_index_file(&component_dir.join("index"), *entry_count);

            // Write data file
            let element_size = 8u64;
            let data_len = entry_count * 8;
            let committed_len = (HEADER_SIZE + data_len) as u64;
            let mut data_file = vec![0u8; HEADER_SIZE + data_len];
            data_file[0..8].copy_from_slice(&committed_len.to_le_bytes());
            data_file[16..24].copy_from_slice(&element_size.to_le_bytes());
            fs::write(component_dir.join("data"), &data_file)?;
        }

        Ok(())
    }

    #[test]
    fn test_run_drop_all() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database with components
        create_test_db(
            &db_path,
            &[("rocket.velocity", 100), ("rocket.position", 200)],
        )
        .unwrap();

        // Verify components exist
        let components = collect_all_components(&db_path).unwrap();
        assert_eq!(components.len(), 2);

        // Drop all components
        run(db_path.clone(), MatchMode::All, false, true).unwrap();

        // Verify components are gone
        let components = collect_all_components(&db_path).unwrap();
        assert_eq!(components.len(), 0);
    }

    #[test]
    fn test_run_drop_fuzzy() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database with components
        create_test_db(
            &db_path,
            &[
                ("rocket.velocity", 100),
                ("rocket.position", 200),
                ("drone.velocity", 50),
            ],
        )
        .unwrap();

        // Drop only "rocket" components using fuzzy match
        run(
            db_path.clone(),
            MatchMode::Fuzzy("rocket".to_string()),
            false,
            true,
        )
        .unwrap();

        // Verify only drone component remains
        let components = collect_all_components(&db_path).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].name, "drone.velocity");
    }

    #[test]
    fn test_run_drop_pattern() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database with components
        create_test_db(
            &db_path,
            &[
                ("rocket.velocity", 100),
                ("rocket.position", 200),
                ("drone.velocity", 50),
            ],
        )
        .unwrap();

        // Drop components matching "*.velocity"
        run(
            db_path.clone(),
            MatchMode::Pattern("*.velocity".to_string()),
            false,
            true,
        )
        .unwrap();

        // Verify only rocket.position remains
        let components = collect_all_components(&db_path).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].name, "rocket.position");
    }

    #[test]
    fn test_dry_run_does_not_modify() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database
        create_test_db(&db_path, &[("comp1", 100), ("comp2", 200)]).unwrap();

        // Dry run
        run(db_path.clone(), MatchMode::All, true, true).unwrap();

        // Verify components still exist
        let components = collect_all_components(&db_path).unwrap();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_no_matches_found() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("db");

        // Create test database
        create_test_db(&db_path, &[("comp1", 100)]).unwrap();

        // Try to drop non-existent component
        run(
            db_path.clone(),
            MatchMode::Fuzzy("nonexistent".to_string()),
            false,
            true,
        )
        .unwrap();

        // Verify component still exists
        let components = collect_all_components(&db_path).unwrap();
        assert_eq!(components.len(), 1);
    }
}
