use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

use dashmap::{DashMap, mapref::one::Ref};
use impeller2_wkt::AssetId;
use memmap2::Mmap;

use crate::Error;

pub struct Assets {
    path: PathBuf,
    pub items: DashMap<AssetId, Mmap>,
}

impl Assets {
    pub fn open(path: PathBuf) -> Result<Assets, Error> {
        fs::create_dir_all(&path)?;
        let items = DashMap::default();
        for elem in fs::read_dir(&path)? {
            let Ok(elem) = elem else { continue };
            let path = elem.path();
            let asset_id: AssetId = path
                .file_name()
                .and_then(|p| p.to_str())
                .and_then(|p| p.parse().ok())
                .ok_or(Error::InvalidAssetId)?;
            let file = OpenOptions::new().read(true).open(path)?;
            let map = unsafe { Mmap::map(&file) }?;
            items.insert(asset_id, map);
        }
        Ok(Self { path, items })
    }

    pub fn insert(&self, id: AssetId, buf: &[u8]) -> Result<(), Error> {
        let path = self.path.join(id.to_string());
        if path.exists() {
            fs::remove_file(&path)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(path)?;
        file.write_all(buf)?;
        let map = unsafe { Mmap::map(&file) }?;
        self.items.insert(id, map);
        Ok(())
    }

    pub fn get(&self, id: AssetId) -> Option<Ref<AssetId, Mmap>> {
        self.items.get(&id)
    }
}
