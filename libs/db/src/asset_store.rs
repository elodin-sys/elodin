use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use impeller2_wkt::{AssetEntry, AssetManifest, ASSET_MANIFEST_VERSION, ASSETS_MANIFEST_KEY};
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::DB;

const INDEX_FILE: &str = "index";
const BLOBS_DIR: &str = "blobs";

pub struct AssetStore {
    root: PathBuf,
    manifest: AssetManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingAssetUpload {
    pub logical_path: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
    pub original_path: Option<String>,
}

impl AssetStore {
    pub fn open(db_path: &Path) -> Result<Self, Error> {
        let root = db_path.join("assets");
        fs::create_dir_all(root.join(BLOBS_DIR))?;

        let index_path = root.join(INDEX_FILE);
        let manifest = if index_path.exists() {
            let data = fs::read(&index_path)?;
            serde_json::from_slice(&data)?
        } else {
            AssetManifest::new()
        };

        Ok(Self { root, manifest })
    }

    pub fn asset_manifest(&self) -> AssetManifest {
        self.manifest.clone()
    }

    pub fn contains(&self, logical_path: &str) -> bool {
        self.manifest.find(logical_path).is_some()
    }

    pub fn put_asset(
        &mut self,
        logical_path: &str,
        media_type: &str,
        bytes: &[u8],
        original_path: Option<&str>,
    ) -> Result<AssetEntry, Error> {
        validate_logical_path(logical_path)?;

        let hash = blake3::hash(bytes).to_hex().to_string();
        let blob_path = self.blob_path(&hash);
        if !blob_path.exists() {
            write_blob_atomic(&blob_path, bytes)?;
        }

        let entry = AssetEntry {
            logical_path: logical_path.to_string(),
            hash,
            media_type: if media_type.is_empty() {
                media_type_from_path(logical_path)
            } else {
                media_type.to_string()
            },
            byte_len: bytes.len() as u64,
            original_path: original_path.map(str::to_string),
        };

        if let Some(existing) = self
            .manifest
            .assets
            .iter()
            .position(|item| item.logical_path == logical_path)
        {
            self.manifest.assets[existing] = entry.clone();
        } else {
            self.manifest.assets.push(entry.clone());
        }

        self.persist()?;
        Ok(entry)
    }

    pub fn get_asset(&self, logical_path: &str) -> Result<(AssetEntry, Vec<u8>), Error> {
        let entry = self
            .manifest
            .find(logical_path)
            .cloned()
            .ok_or_else(|| Error::AssetNotFound(logical_path.to_string()))?;
        let blob_path = self.blob_path(&entry.hash);
        if !blob_path.exists() {
            return Err(Error::AssetBlobMissing {
                logical_path: logical_path.to_string(),
                hash: entry.hash,
            });
        }
        let bytes = fs::read(blob_path)?;
        Ok((entry, bytes))
    }

    pub fn get_asset_by_hash(&self, hash: &str) -> Result<Vec<u8>, Error> {
        let blob_path = self.blob_path(hash);
        if !blob_path.exists() {
            return Err(Error::AssetBlobMissing {
                logical_path: String::new(),
                hash: hash.to_string(),
            });
        }
        Ok(fs::read(blob_path)?)
    }

    pub fn sync_metadata(&self, db_config: &mut impeller2_wkt::DbConfig) {
        let json = serde_json::to_string(&self.manifest).expect("serialize asset manifest");
        db_config
            .metadata
            .insert(ASSETS_MANIFEST_KEY.to_string(), json);
    }

    pub fn load_metadata(db_config: &impeller2_wkt::DbConfig) -> Option<AssetManifest> {
        db_config
            .metadata
            .get(ASSETS_MANIFEST_KEY)
            .and_then(|json| serde_json::from_str(json).ok())
    }

    fn blob_path(&self, hash: &str) -> PathBuf {
        self.root.join(BLOBS_DIR).join(hash)
    }

    fn persist(&mut self) -> Result<(), Error> {
        self.manifest.version = ASSET_MANIFEST_VERSION;
        let data = serde_json::to_vec(&self.manifest)?;
        fs::write(self.root.join(INDEX_FILE), data)?;
        Ok(())
    }
}

pub fn validate_logical_path(path: &str) -> Result<(), Error> {
    if path.is_empty() {
        return Err(Error::InvalidAssetPath(
            "logical path must not be empty".to_string(),
        ));
    }
    if path.starts_with('/') || path.contains('\\') {
        return Err(Error::InvalidAssetPath(format!(
            "logical path must be relative with forward slashes: {path}"
        )));
    }
    if path.contains("://") {
        return Err(Error::InvalidAssetPath(format!(
            "logical path must not be a URL: {path}"
        )));
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        return Err(Error::InvalidAssetPath(format!(
            "logical path must be relative: {path}"
        )));
    }
    for component in parsed.components() {
        match component {
            Component::ParentDir => {
                return Err(Error::InvalidAssetPath(format!(
                    "logical path must not contain '..': {path}"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(Error::InvalidAssetPath(format!(
                    "logical path must be relative: {path}"
                )));
            }
            Component::Normal(_) | Component::CurDir => {}
        }
    }
    Ok(())
}

pub fn media_type_from_path(path: &str) -> String {
    match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("glb") => "model/gltf-binary".to_string(),
        Some("gltf") => "model/gltf+json".to_string(),
        Some("png") => "image/png".to_string(),
        Some("jpg") | Some("jpeg") => "image/jpeg".to_string(),
        Some("ktx2") => "image/ktx2".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

fn write_blob_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("tmp");
    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn put_read_roundtrip() {
        let dir = tempdir().unwrap();
        let mut store = AssetStore::open(dir.path()).unwrap();
        let bytes = b"hello glb";
        let entry = store
            .put_asset("f22.glb", "model/gltf-binary", bytes, None)
            .unwrap();
        assert_eq!(entry.logical_path, "f22.glb");
        assert_eq!(entry.byte_len, bytes.len() as u64);

        let (read_entry, read_bytes) = store.get_asset("f22.glb").unwrap();
        assert_eq!(read_entry.hash, entry.hash);
        assert_eq!(read_bytes, bytes);
    }

    #[test]
    fn deduplicates_identical_bytes() {
        let dir = tempdir().unwrap();
        let mut store = AssetStore::open(dir.path()).unwrap();
        let bytes = b"shared blob";
        let first = store
            .put_asset("a.glb", "", bytes, None)
            .unwrap();
        let second = store
            .put_asset("b.glb", "", bytes, None)
            .unwrap();
        assert_eq!(first.hash, second.hash);
        assert_eq!(
            fs::read_dir(dir.path().join("assets/blobs"))
                .unwrap()
                .count(),
            1
        );
    }

    #[test]
    fn rejects_invalid_paths() {
        assert!(validate_logical_path("../secret.glb").is_err());
        assert!(validate_logical_path("/etc/passwd").is_err());
        assert!(validate_logical_path("https://example.com/a.glb").is_err());
        assert!(validate_logical_path("").is_err());
        assert!(validate_logical_path("models\\jet.glb").is_err());
        assert!(validate_logical_path("f22.glb").is_ok());
        assert!(validate_logical_path("models/f22.glb").is_ok());
    }

    #[test]
    fn persists_after_reopen() {
        let dir = tempdir().unwrap();
        let hash = {
            let mut store = AssetStore::open(dir.path()).unwrap();
            store
                .put_asset("f22.glb", "", b"persistent", None)
                .unwrap()
                .hash
        };

        let index_bytes = fs::read(dir.path().join("assets/index")).unwrap();
        assert!(
            !index_bytes.is_empty(),
            "index should not be empty, len={}",
            index_bytes.len()
        );

        let store = AssetStore::open(dir.path()).unwrap();
        let (entry, bytes) = store.get_asset("f22.glb").unwrap();
        assert_eq!(entry.hash, hash);
        assert_eq!(bytes, b"persistent");
    }

    #[test]
    fn syncs_metadata_json() {
        let dir = tempdir().unwrap();
        let mut store = AssetStore::open(dir.path()).unwrap();
        store.put_asset("f22.glb", "", b"meta", None).unwrap();

        let mut config = impeller2_wkt::DbConfig::default();
        store.sync_metadata(&mut config);
        let manifest = AssetStore::load_metadata(&config).unwrap();
        assert_eq!(manifest.assets.len(), 1);
        assert_eq!(manifest.assets[0].logical_path, "f22.glb");
    }

    #[test]
    fn replaces_existing_logical_path() {
        let dir = tempdir().unwrap();
        let mut store = AssetStore::open(dir.path()).unwrap();
        store.put_asset("f22.glb", "", b"v1", None).unwrap();
        let updated = store.put_asset("f22.glb", "", b"v2", None).unwrap();
        assert_eq!(store.manifest.assets.len(), 1);
        assert_eq!(updated.byte_len, 2);
        assert_eq!(store.get_asset("f22.glb").unwrap().1, b"v2");
    }

    #[test]
    fn db_put_asset_persists_after_reopen() {
        let dir = tempdir().unwrap();
        let db = DB::create(dir.path().to_path_buf()).unwrap();
        db.put_asset("f22.glb", "", b"db blob", None).unwrap();
        let manifest = db.asset_manifest().unwrap();
        assert_eq!(manifest.assets.len(), 1);

        let db = DB::open(dir.path().to_path_buf()).unwrap();
        let (_, bytes) = db.get_asset("f22.glb").unwrap();
        assert_eq!(bytes, b"db blob");
        let config = db.db_config();
        assert!(config.metadata.contains_key(ASSETS_MANIFEST_KEY));
    }

    #[test]
    fn stores_real_f22_glb_when_present() {
        let glb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/f22.glb");
        if !glb.exists() {
            return;
        }
        let bytes = fs::read(&glb).unwrap();
        let dir = tempdir().unwrap();
        let db = DB::create(dir.path().to_path_buf()).unwrap();
        db.put_asset("f22.glb", "", &bytes, None).unwrap();
        let (_, read) = db.get_asset("f22.glb").unwrap();
        assert_eq!(read, bytes);
    }
}
