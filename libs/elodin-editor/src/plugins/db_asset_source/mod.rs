use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use bevy::asset::io::*;
use bevy::prelude::*;
use impeller2::types::IntoLenPacket;
use impeller2_stellar::Client;
use impeller2_wkt::{AssetChunk, AssetManifest, GetAsset, ASSETS_MANIFEST_KEY};

use super::env_asset_source::resolve_assets_dir;

#[derive(Resource, Clone, Default)]
pub struct DbAssetManifest(pub Option<AssetManifest>);

impl DbAssetManifest {
    pub fn sync_from_config(config: &impeller2_wkt::DbConfig) -> Self {
        Self(
            config
                .metadata
                .get(ASSETS_MANIFEST_KEY)
                .and_then(|json| serde_json::from_str(json).ok()),
        )
    }
}

#[derive(Resource, Clone)]
struct DbAssetShared {
    addr: Arc<Mutex<Option<std::net::SocketAddr>>>,
    manifest: Arc<RwLock<Option<AssetManifest>>>,
}

impl Default for DbAssetShared {
    fn default() -> Self {
        Self {
            addr: Arc::new(Mutex::new(None)),
            manifest: Arc::new(RwLock::new(None)),
        }
    }
}

pub(crate) fn sync_db_asset_manifest(
    config: Res<impeller2_wkt::DbConfig>,
    mut manifest: ResMut<DbAssetManifest>,
    shared: Res<DbAssetShared>,
) {
    if config.is_changed() {
        *manifest = DbAssetManifest::sync_from_config(&config);
        *shared.manifest.write().unwrap() = manifest.0.clone();
    }
}

pub(crate) fn sync_db_asset_connection(
    connection: Option<Res<impeller2_bevy::ConnectionAddr>>,
    shared: Res<DbAssetShared>,
) {
    let Some(connection) = connection else {
        return;
    };
    *shared.addr.lock().unwrap() = Some(connection.0);
}

#[derive(Default)]
pub struct DbAssetPlugin;

#[derive(Clone)]
struct DbAssetReader {
    rt: Runtime,
    shared: DbAssetShared,
    cache_dir: PathBuf,
}

#[derive(Clone)]
struct Runtime {
    rt: Arc<tokio::runtime::Runtime>,
}

impl Runtime {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime for db assets");
        Self {
            rt: Arc::new(rt),
        }
    }

    async fn spawn<F>(&self, future: F) -> F::Output
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(future).await.unwrap()
    }
}

impl Plugin for DbAssetPlugin {
    fn build(&self, app: &mut App) {
        let rt = Runtime::new();
        let dirs = directories::ProjectDirs::from("systems", "elodin", "cli")
            .expect("valid project dirs");
        let cache_dir = dirs.cache_dir().join("db_assets");
        let _ = std::fs::create_dir_all(&cache_dir);

        let shared = DbAssetShared::default();
        let reader = DbAssetReader {
            rt,
            shared: shared.clone(),
            cache_dir,
        };

        app.init_resource::<DbAssetManifest>()
            .insert_resource(shared)
            .add_systems(Update, (sync_db_asset_connection, sync_db_asset_manifest))
            .register_asset_source("db", reader.asset_source());
    }
}

impl DbAssetReader {
    fn asset_source(self) -> AssetSourceBuilder {
        AssetSourceBuilder::new(move || Box::new(self.clone()))
    }

    fn cache_path(&self, hash: &str) -> PathBuf {
        self.cache_dir.join(hash)
    }

    async fn fetch_asset(&self, logical_path: &str) -> Result<Vec<u8>, AssetReaderError> {
        let hash = {
            let manifest = self.shared.manifest.read().unwrap();
            let manifest = manifest
                .as_ref()
                .ok_or_else(|| io_error("missing asset manifest"))?;
            manifest
                .find(logical_path)
                .ok_or_else(|| io_error(format!("asset not in manifest: {logical_path}")))?
                .hash
                .clone()
        };

        let cache_path = self.cache_path(&hash);
        if cache_path.is_file() {
            return std::fs::read(&cache_path)
                .map_err(|err| AssetReaderError::Io(std::sync::Arc::new(err)));
        }

        let addr = self
            .shared
            .addr
            .lock()
            .unwrap()
            .ok_or_else(|| io_error("db connection address unavailable"))?;

        let mut client = Client::connect(addr)
            .await
            .map_err(|err| io_error(err.to_string()))?;

        let req_id = 1;
        client
            .send(
                GetAsset {
                    logical_path: logical_path.to_string(),
                }
                .with_request_id(req_id),
            )
            .await
            .0
            .map_err(|_| io_error("failed to send GetAsset"))?;

        let first: AssetChunk = client
            .recv(req_id)
            .await
            .map_err(|err| io_error(err.to_string()))?;
        let total_len = first.total_len as usize;
        let mut bytes = vec![0u8; total_len];
        Self::copy_chunk(&mut bytes, &first);

        let mut received = first.data.len();
        while received < total_len {
            let chunk: AssetChunk = client
                .recv(req_id)
                .await
                .map_err(|err| io_error(err.to_string()))?;
            Self::copy_chunk(&mut bytes, &chunk);
            received = received.saturating_add(chunk.data.len());
            if chunk.offset + chunk.data.len() as u64 >= chunk.total_len {
                break;
            }
        }

        if let Some(parent) = cache_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&cache_path, &bytes)
            .map_err(|err| AssetReaderError::Io(std::sync::Arc::new(err)))?;
        Ok(bytes)
    }

    fn copy_chunk(bytes: &mut [u8], chunk: &AssetChunk) {
        let start = chunk.offset as usize;
        let end = start + chunk.data.len();
        if end <= bytes.len() {
            bytes[start..end].copy_from_slice(&chunk.data);
        }
    }
}

impl AssetReader for DbAssetReader {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<Box<dyn Reader>, AssetReaderError> {
        let logical_path = path
            .to_str()
            .ok_or_else(|| io_error(format!("invalid db asset path: {}", path.display())))?;
        let bytes = self
            .rt
            .spawn({
                let reader = self.clone();
                let logical_path = logical_path.to_string();
                async move { reader.fetch_asset(&logical_path).await }
            })
            .await;
        Ok(Box::new(VecReader::new(bytes?)))
    }

    async fn read_meta<'a>(&'a self, path: &'a Path) -> Result<Box<dyn Reader>, AssetReaderError> {
        Err(AssetReaderError::NotFound(path.to_owned()))
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<Box<PathStream>, AssetReaderError> {
        Err(AssetReaderError::NotFound(path.to_owned()))
    }

    async fn is_directory<'a>(&'a self, _path: &'a Path) -> Result<bool, AssetReaderError> {
        Ok(false)
    }
}

fn io_error(message: impl Into<String>) -> AssetReaderError {
    AssetReaderError::Io(io::Error::new(io::ErrorKind::NotFound, message.into()).into())
}

pub fn resolve_glb_asset_url(
    path: &str,
    db_content: bool,
    manifest: Option<&AssetManifest>,
) -> String {
    if path.contains("://") {
        return format!("{path}#Scene0");
    }
    if Path::new(path).is_absolute() {
        return format!("{path}#Scene0");
    }
    if resolve_assets_dir().is_some_and(|dir| dir.join(path).is_file()) {
        return format!("{path}#Scene0");
    }
    if db_content && manifest.is_some_and(|manifest| manifest.find(path).is_some()) {
        return format!("db://{path}#Scene0");
    }
    format!("{path}#Scene0")
}

#[cfg(test)]
mod tests {
    use super::*;
    use impeller2_wkt::{AssetEntry, AssetManifest};

    #[test]
    fn resolve_glb_prefers_https() {
        let manifest = AssetManifest {
            version: 1,
            assets: vec![AssetEntry {
                logical_path: "f22.glb".to_string(),
                hash: "abc".to_string(),
                media_type: "model/gltf-binary".to_string(),
                byte_len: 1,
                original_path: None,
            }],
        };
        let url = resolve_glb_asset_url("https://example.com/a.glb", true, Some(&manifest));
        assert_eq!(url, "https://example.com/a.glb#Scene0");
    }

    #[test]
    fn resolve_glb_uses_db_when_missing_local() {
        let manifest = AssetManifest {
            version: 1,
            assets: vec![AssetEntry {
                logical_path: "missing.glb".to_string(),
                hash: "abc".to_string(),
                media_type: "model/gltf-binary".to_string(),
                byte_len: 1,
                original_path: None,
            }],
        };
        let url = resolve_glb_asset_url("missing.glb", true, Some(&manifest));
        assert_eq!(url, "db://missing.glb#Scene0");
    }
}
