//! Database compact tool
//!
//! AppendLog-backed files (component `data`/`index`, message-log
//! `timestamps`/`offsets`/`data_log`) are preallocated as 8 GB sparse files.
//! Their apparent size makes archived databases hostile to anything that walks
//! them naively (rsync, tar, S3 upload). Compacting truncates each file to its
//! committed length — the write head recorded in the file header — so apparent
//! size matches real size.
//!
//! Compacted databases stay fully readable (open, export, query, replay);
//! further *writes* require the map headroom that compaction removed, so only
//! compact databases that are done recording. Never compact a database that a
//! live server currently has open.

use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::Error;

/// Minimum AppendLog header size: committed_len (8) + head_len (8). Component
/// logs carry an extra field on top of this; `committed_len` always includes
/// the full header, so the minimum is only used as a corruption guard.
const MIN_HEADER_SIZE: u64 = 16;

const APPEND_LOG_FILE_NAMES: &[&str] = &["data", "index", "timestamps", "offsets", "data_log"];

#[derive(Debug, Default)]
pub struct CompactStats {
    pub files: usize,
    pub apparent_bytes_before: u64,
    pub apparent_bytes_after: u64,
}

/// Truncate every AppendLog file under `db_path` to its committed length.
pub fn compact(db_path: &Path, dry_run: bool) -> Result<CompactStats, Error> {
    if !db_path.join("db_state").exists() {
        return Err(Error::MissingDbState(db_path.to_path_buf()));
    }
    let mut stats = CompactStats::default();
    let mut stack = vec![db_path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if !APPEND_LOG_FILE_NAMES.contains(&name) {
                continue;
            }
            compact_file(&path, dry_run, &mut stats)?;
        }
    }
    Ok(stats)
}

fn compact_file(path: &PathBuf, dry_run: bool, stats: &mut CompactStats) -> Result<(), Error> {
    let file_len = fs::metadata(path)?.len();
    let mut header = [0u8; 8];
    {
        let mut file = fs::File::open(path)?;
        if file.read(&mut header)? < 8 {
            return Ok(());
        }
    }
    let committed_len = u64::from_le_bytes(header);
    if committed_len < MIN_HEADER_SIZE || committed_len > file_len {
        // Corrupt or already shorter than its own header claims; leave it.
        return Ok(());
    }
    if committed_len >= file_len {
        return Ok(());
    }
    stats.files += 1;
    stats.apparent_bytes_before += file_len;
    stats.apparent_bytes_after += committed_len;
    if !dry_run {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(committed_len)?;
        file.sync_all()?;
    }
    Ok(())
}

/// CLI entry point: compact with progress output.
pub fn run(db_path: PathBuf, dry_run: bool) -> Result<(), Error> {
    let stats = compact(&db_path, dry_run)?;
    let saved = stats
        .apparent_bytes_before
        .saturating_sub(stats.apparent_bytes_after);
    if dry_run {
        println!(
            "DRY RUN: would truncate {} file(s), reclaiming {:.1} GiB of apparent size",
            stats.files,
            saved as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    } else {
        println!(
            "Compacted {} file(s): apparent size reduced by {:.1} GiB",
            stats.files,
            saved as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::append_log::AppendLog;

    #[test]
    fn compact_truncates_to_committed_len() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("db");
        let component = db_path.join("42");
        fs::create_dir_all(&component).unwrap();
        fs::write(db_path.join("db_state"), b"{}").unwrap();

        let log = AppendLog::create(component.join("index"), 0u64).unwrap();
        log.write(&7i64.to_le_bytes()).unwrap();
        log.write(&9i64.to_le_bytes()).unwrap();
        log.sync_all().unwrap();
        let committed = log.len();
        drop(log);

        let before = fs::metadata(component.join("index")).unwrap().len();
        assert_eq!(before, 8 * 1024 * 1024 * 1024 + 1);

        let stats = compact(&db_path, false).unwrap();
        assert_eq!(stats.files, 1);
        let after = fs::metadata(component.join("index")).unwrap().len();
        // committed_len includes the 24-byte header (u64 extra).
        assert_eq!(after, committed + 24);

        // Reopen and read back the committed data.
        let log = AppendLog::<u64>::open(component.join("index")).unwrap();
        assert_eq!(log.len(), committed);
        assert_eq!(log.get(0..8).unwrap(), &7i64.to_le_bytes());

        // Idempotent: nothing left to reclaim.
        let stats = compact(&db_path, false).unwrap();
        assert_eq!(stats.files, 0);
    }

    #[test]
    fn dry_run_leaves_files_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("db");
        let component = db_path.join("7");
        fs::create_dir_all(&component).unwrap();
        fs::write(db_path.join("db_state"), b"{}").unwrap();
        let log = AppendLog::create(component.join("data"), 8u64).unwrap();
        log.write(&[1u8; 8]).unwrap();
        log.sync_all().unwrap();
        drop(log);

        let before = fs::metadata(component.join("data")).unwrap().len();
        let stats = compact(&db_path, true).unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(fs::metadata(component.join("data")).unwrap().len(), before);
    }
}
