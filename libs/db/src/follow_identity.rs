//! Durable message-log identity. Replaced or truncated logs must not reuse a cursor.
use crate::Error;
use std::{fs, io::Write, path::Path};

pub(crate) fn read<const N: usize>(path: &Path) -> Result<Option<[u8; N]>, Error> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes.try_into().map_err(|_| Error::BadMessage)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub(crate) fn write<const N: usize>(path: &Path, id: [u8; N]) -> Result<(), Error> {
    let temporary = path.with_extension("pending");
    let mut file = fs::File::create(&temporary)?;
    file.write_all(&id)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    if let Some(parent) = path.parent() {
        crate::sync_dir(parent)?;
    }
    Ok(())
}

pub(crate) fn load_or_create(path: &Path) -> Result<[u8; 16], Error> {
    if let Some(id) = read(path)? {
        return Ok(id);
    }
    let id = fastrand::u128(..).to_le_bytes();
    write(path, id)?;
    Ok(id)
}
