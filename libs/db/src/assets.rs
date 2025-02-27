use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

use impeller2_wkt::AssetId;

use crate::Error;

pub fn open_assets(db_path: &Path) -> Result<HashMap<AssetId, memmap2::Mmap>, Error> {
    let path = db_path.join("assets");
    fs::create_dir_all(&path)?;
    let mut items = HashMap::default();
    for elem in fs::read_dir(&path)? {
        let Ok(elem) = elem else { continue };
        let path = elem.path();
        let asset_id: AssetId = path
            .file_name()
            .and_then(|p| p.to_str())
            .and_then(|p| p.parse().ok())
            .ok_or(Error::InvalidAssetId)?;
        let file = OpenOptions::new().read(true).open(path)?;
        let map = unsafe { memmap2::Mmap::map(&file) }?;
        items.insert(asset_id, map);
    }
    Ok(items)
}

pub fn insert_asset(
    db_path: &Path,
    assets: &mut HashMap<AssetId, memmap2::Mmap>,
    id: AssetId,
    buf: &[u8],
) -> Result<(), Error> {
    let path = db_path.join("assets").join(id.to_string());
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
    let map = unsafe { memmap2::Mmap::map(&file) }?;
    assets.insert(id, map);
    Ok(())
}
