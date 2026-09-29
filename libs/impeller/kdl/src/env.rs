use std::{
    io,
    path::{Path, PathBuf},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Cannot get current directory: {0:?}")]
    Io(#[from] io::Error),
}

/// Returns the current working directory.
pub fn schematic_dir_or_cwd() -> Result<PathBuf, Error> {
    Ok(std::env::current_dir()?)
}

/// Given an absolute file path, return it unchanged. Otherwise resolve relative
/// to the current working directory (e.g. `elodin editor --kdl drone.kdl`).
pub fn schematic_file(path: &Path) -> PathBuf {
    if path.is_absolute() {
        PathBuf::from(path)
    } else {
        schematic_dir_or_cwd().unwrap_or_default().join(path)
    }
}
