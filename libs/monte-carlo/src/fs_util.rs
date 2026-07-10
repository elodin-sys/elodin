//! Sparse-aware directory move for scratch-dir finalize: preserves file holes
//! (elodin-db AppendLogs are preallocated 8 GB sparse files) so moving a run
//! from tmpfs scratch to the artifact volume copies real bytes, not apparent
//! size.

use std::fs;
use std::io;
use std::path::Path;

/// Recursively move `src` into `dst` (created fresh), preserving sparseness,
/// then remove `src`. Falls back to copy when `src` and `dst` are on
/// different filesystems (the expected case for tmpfs scratch).
pub fn move_dir_sparse(src: &Path, dst: &Path) -> io::Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    // Same-filesystem fast path.
    if fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    copy_dir_sparse(src, dst)?;
    fs::remove_dir_all(src)
}

fn copy_dir_sparse(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_sparse(&entry.path(), &target)?;
        } else if file_type.is_file() {
            copy_file_sparse(&entry.path(), &target)?;
        } else if file_type.is_symlink() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(fs::read_link(entry.path())?, &target)?;
        }
    }
    Ok(())
}

/// Copy only the data extents of `src` (SEEK_DATA/SEEK_HOLE), leaving holes
/// as holes in `dst`. Equivalent fidelity to `cp --sparse=always` for files
/// that are already sparse.
#[cfg(target_os = "linux")]
pub fn copy_file_sparse(src: &Path, dst: &Path) -> io::Result<()> {
    use nix::unistd::{Whence, lseek};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;

    let mut input = fs::File::open(src)?;
    let len = input.metadata()?.len() as i64;
    let output = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(dst)?;
    output.set_len(len as u64)?;
    let mut writer = io::BufWriter::new(output);

    let mut offset: i64 = 0;
    let mut buf = vec![0u8; 1024 * 1024];
    while offset < len {
        let data_start = match lseek(input.as_raw_fd(), offset, Whence::SeekData) {
            Ok(start) => start,
            // ENXIO: no more data extents — the rest of the file is a hole.
            Err(nix::errno::Errno::ENXIO) => break,
            // Filesystem without SEEK_DATA support: fall back to a full copy.
            Err(_) => {
                drop(writer);
                fs::copy(src, dst)?;
                return Ok(());
            }
        };
        let data_end = lseek(input.as_raw_fd(), data_start, Whence::SeekHole).unwrap_or(len);
        input.seek(SeekFrom::Start(data_start as u64))?;
        writer.seek(SeekFrom::Start(data_start as u64))?;
        let mut remaining = (data_end - data_start) as u64;
        while remaining > 0 {
            let chunk = remaining.min(buf.len() as u64) as usize;
            input.read_exact(&mut buf[..chunk])?;
            writer.write_all(&buf[..chunk])?;
            remaining -= chunk as u64;
        }
        offset = data_end;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn copy_file_sparse(src: &Path, dst: &Path) -> io::Result<()> {
    fs::copy(src, dst).map(|_| ())
}

/// Elodin-DB AppendLog file names eligible for compaction. Mirrors
/// `elodin-db compact` (libs/db/src/compact.rs) without pulling the full DB
/// crate into the runner: each file's first 8 bytes are its committed length
/// (header included), everything past that is preallocated slack.
const APPEND_LOG_FILE_NAMES: &[&str] = &["data", "index", "timestamps", "offsets", "data_log"];
const APPEND_LOG_MIN_HEADER: u64 = 16;

/// Truncate a finalized run DB's preallocated files to their committed
/// length, so retained DBs have apparent size ≈ real size. Safe only once the
/// writing process has exited. Returns the apparent bytes reclaimed.
pub fn compact_run_db(db_path: &Path) -> io::Result<u64> {
    let mut reclaimed = 0_u64;
    if !db_path.is_dir() {
        return Ok(0);
    }
    let mut stack = vec![db_path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(entry.path());
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
            let path = entry.path();
            let file_len = entry.metadata()?.len();
            let mut header = [0u8; 8];
            {
                use std::io::Read;
                let mut file = fs::File::open(&path)?;
                if file.read(&mut header)? < 8 {
                    continue;
                }
            }
            let committed_len = u64::from_le_bytes(header);
            if committed_len < APPEND_LOG_MIN_HEADER || committed_len >= file_len {
                continue;
            }
            let file = fs::OpenOptions::new().write(true).open(&path)?;
            file.set_len(committed_len)?;
            reclaimed += file_len - committed_len;
        }
    }
    Ok(reclaimed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn sparse_copy_preserves_holes_and_data() {
        use std::io::{Seek, SeekFrom, Write};
        use std::os::unix::fs::MetadataExt;

        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("sparse");
        let mut file = fs::File::create(&src).unwrap();
        file.write_all(b"head").unwrap();
        // 64 MiB hole, then a tail.
        file.seek(SeekFrom::Start(64 * 1024 * 1024)).unwrap();
        file.write_all(b"tail").unwrap();
        file.sync_all().unwrap();
        drop(file);

        let dst = dir.path().join("copy");
        copy_file_sparse(&src, &dst).unwrap();

        let src_meta = fs::metadata(&src).unwrap();
        let dst_meta = fs::metadata(&dst).unwrap();
        assert_eq!(src_meta.len(), dst_meta.len());
        // tmpdir may be tmpfs (always dense) — only assert sparseness when the
        // source itself has holes.
        if src_meta.blocks() * 512 < src_meta.len() {
            assert!(
                dst_meta.blocks() * 512 < dst_meta.len(),
                "copy lost sparseness: {} blocks for {} bytes",
                dst_meta.blocks(),
                dst_meta.len()
            );
        }
        let contents = fs::read(&dst).unwrap();
        assert_eq!(&contents[..4], b"head");
        assert_eq!(&contents[64 * 1024 * 1024..], b"tail");
    }

    #[test]
    fn move_dir_sparse_moves_tree() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(src.join("nested")).unwrap();
        fs::write(src.join("nested/file.txt"), b"hello").unwrap();
        let dst = dir.path().join("dst");
        move_dir_sparse(&src, &dst).unwrap();
        assert!(!src.exists());
        assert_eq!(fs::read(dst.join("nested/file.txt")).unwrap(), b"hello");
    }
}
