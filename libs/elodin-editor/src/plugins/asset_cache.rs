use std::path::PathBuf;

pub trait AssetCache: Send + Sync {
    fn get(&self, url: &str) -> Option<CachedAsset>;
    fn put(&self, url: &str, asset: CachedAsset);
}

#[cfg(target_family = "wasm")]
pub fn cache() -> NoCache {
    NoCache
}

#[cfg(not(target_family = "wasm"))]
pub fn cache() -> FsCache {
    FsCache::new()
}

pub struct CachedAsset {
    pub data: Vec<u8>,
    pub etag: String,
}

#[derive(Clone)]
pub struct NoCache;

impl AssetCache for NoCache {
    fn get(&self, _url: &str) -> Option<CachedAsset> {
        None
    }

    fn put(&self, _url: &str, _asset: CachedAsset) {}
}

#[derive(Clone)]
pub struct FsCache {
    cache_dir: PathBuf,
}

impl FsCache {
    #[cfg(not(target_family = "wasm"))]
    pub fn new() -> Self {
        let dirs = directories::ProjectDirs::from("systems", "elodin", "cli").unwrap();
        let cache_dir = dirs.cache_dir().to_path_buf();
        std::fs::create_dir_all(&cache_dir).unwrap();
        Self { cache_dir }
    }
}

impl AssetCache for FsCache {
    fn get(&self, url: &str) -> Option<CachedAsset> {
        let url_hex = hex::encode(url);
        let asset_path = self.cache_dir.join(url_hex);
        let etag_path = asset_path.with_extension("etag");
        let data = std::fs::read(&asset_path).ok()?;
        let etag = std::fs::read_to_string(&etag_path).ok()?;
        Some(CachedAsset { data, etag })
    }

    fn put(&self, url: &str, asset: CachedAsset) {
        let url_hex = hex::encode(url);
        let asset_path = self.cache_dir.join(url_hex);
        let etag_path = asset_path.with_extension("etag");
        if let Err(err) = std::fs::write(&asset_path, asset.data) {
            eprintln!("Failed to write asset to cache: {}", err);
            return;
        }
        if let Err(err) = std::fs::write(&etag_path, asset.etag) {
            eprintln!("Failed to write etag to cache: {}", err);
        }
    }
}
