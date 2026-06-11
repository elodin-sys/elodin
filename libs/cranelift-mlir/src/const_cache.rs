use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use memmap2::MmapRaw;

use crate::ir::ElementType;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct CachedConst {
    pub hash: String,
    pub symbol: String,
    pub byte_len: usize,
    pub path: PathBuf,
    map: MmapRaw,
}

impl CachedConst {
    pub fn ptr(&self) -> *const u8 {
        self.map.as_mut_ptr().cast_const()
    }
}

pub fn intern_bytes(elem_type: ElementType, bytes: &[u8]) -> Result<Arc<CachedConst>, String> {
    if !cfg!(target_endian = "little") {
        return Err("large constant cache requires a little-endian target".to_string());
    }

    let hash = hash_bytes(elem_type, bytes);
    let symbol = format!("__elodin_const_{hash}");
    let dir = cache_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("create constant cache dir {dir:?}: {e}"))?;
    let path = dir.join(format!("{hash}.bin"));

    let cache_entry_valid = path.exists()
        && fs::metadata(&path)
            .map(|metadata| metadata.len() as usize == bytes.len())
            .unwrap_or(false);
    if !cache_entry_valid {
        // Repair missing or truncated cache entries before mapping; byte_len must match file size.
        let _ = fs::remove_file(&path);
        write_atomic(&path, bytes)?;
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| format!("open cached constant {path:?}: {e}"))?;
    let map = MmapRaw::map_raw(file.as_raw_fd()).map_err(|e| format!("mmap {path:?}: {e}"))?;
    Ok(Arc::new(CachedConst {
        hash,
        symbol,
        byte_len: bytes.len(),
        path,
        map,
    }))
}

fn hash_bytes(elem_type: ElementType, bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"elodin-const-cache-v1");
    hasher.update(&[element_tag(elem_type)]);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}

fn element_tag(elem_type: ElementType) -> u8 {
    match elem_type {
        ElementType::F64 => 1,
        ElementType::F32 => 2,
        ElementType::I64 => 3,
        ElementType::I32 => 4,
        ElementType::I1 => 5,
        ElementType::UI32 => 6,
        ElementType::UI64 => 7,
    }
}

fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ELODIN_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/elodin/const-cache")
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("cached constant path has no parent: {path:?}"))?;
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    // Same-process concurrent interners of one hash must not share a temp path.
    let tmp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("constant"),
        std::process::id(),
        seq
    ));

    {
        let mut file =
            File::create(&tmp).map_err(|e| format!("create temp constant {tmp:?}: {e}"))?;
        file.write_all(bytes)
            .map_err(|e| format!("write temp constant {tmp:?}: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("sync temp constant {tmp:?}: {e}"))?;
    }

    match fs::hard_link(&tmp, path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(err) => {
            let _ = fs::remove_file(&tmp);
            return Err(format!(
                "publish cached constant {tmp:?} -> {path:?}: {err}"
            ));
        }
    }
    let _ = fs::remove_file(&tmp);
    Ok(())
}
