//! Sparse-aware `.tar.zst` archives of a database directory.
//!
//! AppendLog files are 8 GiB sparse mmaps whose header says how much is committed, so each
//! one is written as a GNU sparse entry holding only its header, committed data, and head.
//! Filesystem hole detection can't do this on macOS, nor skip allocated zeros anywhere.
//! Output is deterministic (sorted paths, zeroed mtimes, single-threaded zstd), so an
//! interrupted upload can regenerate the same stream and resume.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use tar::{EntryType, Header};

use crate::Error;

const HEADER_LEN: u64 = 16;
const ZSTD_LEVEL: i32 = 3;

/// Write `db_path` as a zstd tar whose entries all sit under the directory's own name.
pub fn write(db_path: &Path, out: impl Write) -> Result<(), Error> {
    let root = db_path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "database path has no name"))?;
    if !db_path.join("db_state").exists() {
        return Err(Error::MissingDbState(db_path.join("db_state")));
    }
    let zstd = zstd::stream::write::Encoder::new(out, ZSTD_LEVEL)?;
    let mut tar = tar::Builder::new(zstd);
    append_dir(&mut tar, db_path, &PathBuf::from(root))?;
    tar.into_inner()?.finish()?.flush()?;
    Ok(())
}

fn append_dir<W: Write>(tar: &mut tar::Builder<W>, dir: &Path, name: &Path) -> io::Result<()> {
    let mut header = new_header(EntryType::Directory, 0o755, 0);
    tar.append_data(&mut header, name, io::empty())?;
    let mut entries = fs::read_dir(dir)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let name = name.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            append_dir(tar, &path, &name)?;
        } else if kind.is_file() {
            append_file(tar, &path, &name)?;
        }
    }
    Ok(())
}

fn append_file<W: Write>(tar: &mut tar::Builder<W>, path: &Path, name: &Path) -> io::Result<()> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    match logical_len(path, &mut file, len)? {
        Some(logical) => {
            let stored = logical.div_ceil(512).saturating_mul(512).min(len);
            let mut header = new_header(EntryType::GNUSparse, 0o644, stored);
            let gnu = header.as_gnu_mut().expect("gnu header");
            gnu.set_real_size(len);
            gnu.sparse[0].set_offset(0);
            gnu.sparse[0].set_length(stored);
            // The terminal chunk marks the file's full length, as GNU tar writes it.
            gnu.sparse[1].set_offset(len);
            gnu.sparse[1].set_length(0);
            tar.append_data(&mut header, name, file.take(stored))
        }
        None => {
            let mut header = new_header(EntryType::Regular, 0o644, len);
            tar.append_data(&mut header, name, file.take(len))
        }
    }
}

/// `committed_len + head_len` for a plausible AppendLog that is worth storing sparsely.
fn logical_len(path: &Path, file: &mut File, len: u64) -> io::Result<Option<u64>> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let append_log = matches!(
        name,
        "index" | "data" | "timestamps" | "offsets" | "data_log"
    ) || name.starts_with("data_log.");
    if !append_log || len < HEADER_LEN {
        return Ok(None);
    }
    let mut buf = [0u8; HEADER_LEN as usize];
    file.read_exact(&mut buf)?;
    let committed = u64::from_le_bytes(buf[..8].try_into().expect("8 bytes"));
    let head = u64::from_le_bytes(buf[8..].try_into().expect("8 bytes"));
    file.seek(SeekFrom::Start(0))?;
    let logical = committed.checked_add(head);
    Ok(match logical {
        Some(l) if committed >= HEADER_LEN && l < len => Some(l),
        _ => None,
    })
}

fn new_header(kind: EntryType, mode: u32, size: u64) -> Header {
    let mut header = Header::new_gnu();
    header.set_entry_type(kind);
    header.set_mode(mode);
    header.set_size(size);
    header.set_mtime(0);
    header.set_uid(0);
    header.set_gid(0);
    header
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_log(path: &Path, committed: &[u8], head: u64, full: u64) {
        let mut f = File::create(path).unwrap();
        f.write_all(&(HEADER_LEN + committed.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&head.to_le_bytes()).unwrap();
        f.write_all(committed).unwrap();
        f.seek(SeekFrom::Start(full - 1)).unwrap();
        f.write_all(&[0]).unwrap();
    }

    #[test]
    fn round_trips_through_the_tar_crate() {
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("rec");
        fs::create_dir_all(db.join("42")).unwrap();
        fs::write(db.join("db_state"), b"state").unwrap();
        fs::write(db.join("42/schema"), b"schema").unwrap();
        let full = 64 << 20;
        sparse_log(&db.join("42/index"), &[7u8; 1000], 8, full);
        // Allocated zeros: a hole detector would store all of this.
        let mut zeros = File::create(db.join("42/data")).unwrap();
        zeros.write_all(&(HEADER_LEN + 3).to_le_bytes()).unwrap();
        zeros.write_all(&0u64.to_le_bytes()).unwrap();
        zeros.write_all(&[1, 2, 3]).unwrap();
        zeros.write_all(&vec![0u8; 4 << 20]).unwrap();
        drop(zeros);

        let mut a = Vec::new();
        write(&db, &mut a).unwrap();
        let mut b = Vec::new();
        write(&db, &mut b).unwrap();
        assert_eq!(a, b, "archives are deterministic");
        assert!(
            a.len() < 64 << 10,
            "only committed bytes are stored: {}",
            a.len()
        );

        let out = tmp.path().join("out");
        tar::Archive::new(zstd::stream::read::Decoder::new(&a[..]).unwrap())
            .unpack(&out)
            .unwrap();
        let index = fs::read(out.join("rec/42/index")).unwrap();
        assert_eq!(index.len() as u64, full);
        assert_eq!(&index[16..1016], &[7u8; 1000][..]);
        assert!(index[1024..].iter().all(|&b| b == 0));
        let data = fs::read(out.join("rec/42/data")).unwrap();
        assert_eq!(data.len(), 16 + 3 + (4 << 20));
        assert_eq!(&data[16..19], &[1, 2, 3]);
        assert_eq!(fs::read(out.join("rec/42/schema")).unwrap(), b"schema");
    }
}
