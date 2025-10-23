use std::{
    io,
    path::{Path, PathBuf},
};
use thiserror::Error;
use tracing::{error, warn};

#[derive(Error, Debug)]
pub enum Error {
    #[error("Cannot get current directory: {0:?}")]
    Io(#[from] io::Error),
    #[error("No such directory ELODIN_KDL_DIR {0:?}")]
    NoSuchDir(PathBuf),
    #[error("Not a directory ELODIN_KDL_DIR {0:?}")]
    NotDir(PathBuf),
}

/// Returns a path if the environment variable `ELODIN_KDL_DIR` is set and it
/// exists and is a directory.
pub fn schematic_dir() -> Result<Option<PathBuf>, Error> {
    if let Some(d) = std::env::var_os("ELODIN_KDL_DIR") {
        let p = PathBuf::from(d);
        if !p.exists() {
            let err = Error::NoSuchDir(p.clone());
            warn!("{err}, falling back to current working directory.");
            Ok(None)
        } else if !p.is_dir() {
            let err = Error::NotDir(p.clone());
            warn!("{err}, falling back to current working directory.");
            Ok(None)
        } else {
            Ok(Some(p))
        }
    } else {
        Ok(None)
    }
}

/// Returns the `ELODIN_KDL_DIR` or the current working directory or an error.
pub fn schematic_dir_or_cwd() -> Result<PathBuf, Error> {
    if let Some(dir) = schematic_dir()? {
        Ok(dir)
    } else {
        Ok(std::env::current_dir()?)
    }
}

/// Given an absolute file path, return it unchanged. Otherwise append to
/// `ELODIN_KDL_DIR` if set.
pub fn schematic_file(path: &Path) -> PathBuf {
    if path.is_absolute() {
        PathBuf::from(path)
    } else {
        let mut file = schematic_dir()
            // .inspect(|p| if let Some(p) = p {
            //     info!("Using ELODIN_KDL_DIR {:?}", p.display());
            // })
            .inspect_err(|e| error!("{}, ignoring setting.", e))
            .ok()
            .and_then(|x| x)
            .unwrap_or_default();
        file.push(path);
        file
    }
}
