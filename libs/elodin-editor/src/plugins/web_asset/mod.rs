use std::collections::hash_map::DefaultHasher;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use bevy::asset::io::*;
use bevy::prelude::*;
use impeller2_bevy::ConnectionAddr;
use reqwest::StatusCode;

use super::asset_cache::{self, CachedAsset};
use crate::object_3d::assets_http_addr;

#[derive(Default)]
pub struct WebAssetPlugin;

/// Live `ip:port` of the DB assets HTTP server, shared with the `http` asset
/// reader so failed fetches of DB assets can fall back to local files.
#[derive(Resource, Clone)]
pub struct DbAssetsEndpoint(Arc<RwLock<SocketAddr>>);

impl Default for DbAssetsEndpoint {
    fn default() -> Self {
        let default_db: SocketAddr = "127.0.0.1:2240".parse().expect("valid addr");
        Self(Arc::new(RwLock::new(assets_http_addr(default_db))))
    }
}

/// Keeps the fallback endpoint in sync with the live DB connection.
fn sync_db_assets_endpoint(
    endpoint: Res<DbAssetsEndpoint>,
    connection: Option<Res<ConnectionAddr>>,
) {
    let Some(connection) = connection else {
        return;
    };
    if !connection.is_changed() {
        return;
    }
    if let Ok(mut current) = endpoint.0.write() {
        *current = assets_http_addr(connection.0);
    }
}

struct Client {
    #[allow(dead_code)]
    rt: Runtime,
    source: Source,
    cache: Box<dyn asset_cache::AssetCache>,
    /// URLs under this endpoint are DB assets eligible for local fallback.
    db_endpoint: Option<Arc<RwLock<SocketAddr>>>,
    /// Local assets root searched when a DB asset fetch fails.
    local_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum Source {
    Http,
    Https,
}

#[derive(Clone)]
#[cfg(not(target_family = "wasm"))]
struct Runtime {
    rt: std::sync::Arc<tokio::runtime::Runtime>,
}

#[cfg(not(target_family = "wasm"))]
impl Runtime {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let rt = std::sync::Arc::new(rt);
        Self { rt }
    }

    async fn spawn<F>(&self, future: F) -> F::Output
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(future).await.unwrap()
    }
}

#[cfg(target_family = "wasm")]
#[derive(Clone)]
struct Runtime;

#[cfg(target_family = "wasm")]
impl Runtime {
    fn new() -> Self {
        Self
    }

    async fn spawn<F>(&self, future: F) -> F::Output
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        future.await
    }
}

impl Plugin for WebAssetPlugin {
    fn build(&self, app: &mut App) {
        let rt = Runtime::new();
        let endpoint = DbAssetsEndpoint::default();
        let local_root = super::env_asset_source::resolve_assets_dir();
        app.insert_resource(endpoint.clone());
        app.add_systems(Update, sync_db_assets_endpoint);
        app.register_asset_source(
            "http",
            Client::asset_source(
                rt.clone(),
                Source::Http,
                Some(endpoint.0.clone()),
                local_root,
            ),
        );
        app.register_asset_source(
            "https",
            Client::asset_source(rt.clone(), Source::Https, None, None),
        );
    }
}

impl Client {
    fn url(&self, path: &Path) -> Result<String, AssetReaderError> {
        let path = path.to_str().ok_or(AssetReaderError::Io(
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid path: {}", path.display()),
            )
            .into(),
        ))?;
        let url = match self.source {
            Source::Http => format!("http://{}", path),
            Source::Https => format!("https://{}", path),
        };
        Ok(url)
    }

    async fn get(
        &self,
        url: String,
        cached_asset: Option<CachedAsset>,
    ) -> Result<(Vec<u8>, Option<String>), AssetReaderError> {
        self.rt
            .spawn(async move {
                let client = reqwest::Client::new();
                let mut request = client.get(&url);
                if let Some(CachedAsset { etag, .. }) = &cached_asset {
                    request = request.header("If-None-Match", etag);
                }
                let response = request.send().await.map_err(|err| http_err(&url, err))?;
                if let Some(CachedAsset { data, .. }) = cached_asset
                    && response.status() == StatusCode::NOT_MODIFIED
                {
                    return Ok((data, None));
                }
                // Error statuses must not flow into asset loaders as bytes
                // (a 404 body is not a GLB) and must not be cached.
                let status = response.status();
                if !status.is_success() {
                    return Err(AssetReaderError::HttpError(status.as_u16()));
                }
                let etag = response
                    .headers()
                    .get("ETag")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_owned());

                let data = response
                    .bytes()
                    .await
                    .map_err(|err| http_err(&url, err))?
                    .to_vec();
                Ok((data, etag))
            })
            .await
    }

    fn asset_source(
        rt: Runtime,
        source: Source,
        db_endpoint: Option<Arc<RwLock<SocketAddr>>>,
        local_root: Option<PathBuf>,
    ) -> AssetSourceBuilder {
        AssetSourceBuilder::new(move || {
            Box::new(Client {
                rt: rt.clone(),
                source,
                cache: Box::new(asset_cache::cache()),
                db_endpoint: db_endpoint.clone(),
                local_root: local_root.clone(),
            })
        })
    }

    /// Serves `<local_root>/<key>` for DB-asset URLs that failed to fetch, so
    /// local iteration (`--kdl`) works without injecting every asset into the
    /// DB. Only URLs under the DB assets endpoint are eligible.
    fn local_fallback(&self, path: &Path, url: &str) -> Option<Box<dyn Reader>> {
        let endpoint = *self.db_endpoint.as_ref()?.read().ok()?;
        let key = db_asset_key(path.to_str()?, endpoint)?;
        let file = self.local_root.as_ref()?.join(key);
        let bytes = std::fs::read(&file).ok()?;
        tracing::info!(
            url = %url,
            file = %file.display(),
            "db asset unavailable; serving local file"
        );
        Some(Box::new(VecReader::new(bytes)) as Box<dyn Reader>)
    }
}

/// `host:port/key` reader path -> `key`, when `host:port` is the DB assets
/// endpoint. Rejects parent-dir components so a malformed key cannot escape
/// the assets root.
fn db_asset_key(path: &str, endpoint: SocketAddr) -> Option<&str> {
    let key = path
        .strip_prefix(&endpoint.to_string())?
        .strip_prefix('/')?;
    if key.is_empty() || key.split('/').any(|part| part == "..") {
        return None;
    }
    Some(key)
}

/// Content fingerprint used when the HTTP server omits `ETag` (historically
/// elodin-db's asset server). Without a stored tag the disk cache never
/// persisted successful fetches, so every thruster reload re-hit the network.
fn weak_etag(bytes: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    format!("W/\"{:016x}-{}\"", hasher.finish(), bytes.len())
}

impl AssetReader for Client {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<Box<dyn Reader>, AssetReaderError> {
        let url = self.url(path)?;
        let cached_asset = self.cache.get(&url);

        match self.get(url.clone(), cached_asset.clone()).await {
            Ok((bytes, etag)) => {
                // Always persist: synthesize a weak tag when the server sends none
                // so the next load can short-circuit / revalidate.
                let etag = etag.unwrap_or_else(|| weak_etag(&bytes));
                self.cache.put(
                    &url,
                    CachedAsset {
                        data: bytes.clone(),
                        etag,
                    },
                );
                let reader = VecReader::new(bytes);
                Ok(Box::new(reader) as Box<dyn Reader>)
            }
            Err(err) => {
                // 404: the DB genuinely lacks the key. Don't resurrect a stale
                // cached copy; do try the local assets tree (--kdl iteration).
                if matches!(err, AssetReaderError::HttpError(404)) {
                    if let Some(reader) = self.local_fallback(path, &url) {
                        return Ok(reader);
                    }
                    return Err(err);
                }
                // Stale-while-error: if the DB asset HTTP port is already down
                // (sim teardown) or briefly unreachable, keep using the last
                // good copy instead of spamming failed fetches.
                if let Some(CachedAsset { data, .. }) = cached_asset {
                    tracing::warn!(
                        url = %url,
                        error = %err,
                        "serving cached web asset after fetch failure"
                    );
                    let reader = VecReader::new(data);
                    return Ok(Box::new(reader) as Box<dyn Reader>);
                }
                if let Some(reader) = self.local_fallback(path, &url) {
                    return Ok(reader);
                }
                Err(err)
            }
        }
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

fn http_err(url: &str, err: reqwest::Error) -> AssetReaderError {
    if let Some(status) = err.status() {
        return AssetReaderError::HttpError(status.as_u16());
    }
    let message = format!("{url}: {err}");
    tracing::warn!(error = %message, "failed to fetch web asset");
    AssetReaderError::Io(io::Error::other(message).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_etag_is_stable_for_same_bytes() {
        let a = weak_etag(b"hello");
        let b = weak_etag(b"hello");
        let c = weak_etag(b"world");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.starts_with("W/\""));
    }

    #[test]
    fn db_asset_key_extracts_key_only_for_matching_endpoint() {
        let endpoint: SocketAddr = "127.0.0.1:2241".parse().unwrap();
        assert_eq!(
            db_asset_key("127.0.0.1:2241/textures/x.png", endpoint),
            Some("textures/x.png")
        );
        // Different host or port: not a DB asset.
        assert_eq!(db_asset_key("10.0.0.5:2241/textures/x.png", endpoint), None);
        assert_eq!(
            db_asset_key("127.0.0.1:8080/textures/x.png", endpoint),
            None
        );
        // Parent-dir escapes rejected.
        assert_eq!(db_asset_key("127.0.0.1:2241/../secrets", endpoint), None);
        // Empty key rejected.
        assert_eq!(db_asset_key("127.0.0.1:2241/", endpoint), None);
    }

    #[test]
    fn db_asset_key_handles_ipv6_endpoints() {
        let endpoint: SocketAddr = "[::1]:2241".parse().unwrap();
        assert_eq!(
            db_asset_key("[::1]:2241/models/a.glb", endpoint),
            Some("models/a.glb")
        );
    }

    struct NoopCache;

    impl asset_cache::AssetCache for NoopCache {
        fn get(&self, _url: &str) -> Option<CachedAsset> {
            None
        }

        fn put(&self, _url: &str, _asset: CachedAsset) {}
    }

    #[test]
    fn read_serves_local_file_when_db_endpoint_is_unreachable() {
        let root =
            std::env::temp_dir().join(format!("elodin-web-asset-fallback-{}", std::process::id()));
        std::fs::create_dir_all(root.join("textures")).unwrap();
        std::fs::write(root.join("textures/x.png"), b"png-bytes").unwrap();

        // Port 1 on loopback: connection refused, deterministic transport error.
        let endpoint: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let client = Client {
            rt: Runtime::new(),
            source: Source::Http,
            cache: Box::new(NoopCache),
            db_endpoint: Some(Arc::new(RwLock::new(endpoint))),
            local_root: Some(root.clone()),
        };

        let mut buf = Vec::new();
        bevy::tasks::block_on(async {
            let mut reader = AssetReader::read(&client, Path::new("127.0.0.1:1/textures/x.png"))
                .await
                .expect("local fallback should serve the file");
            reader.read_to_end(&mut buf).await.unwrap();
        });
        assert_eq!(buf, b"png-bytes");

        // A URL outside the DB endpoint must not fall back.
        bevy::tasks::block_on(async {
            let result = AssetReader::read(&client, Path::new("127.0.0.1:2/textures/x.png")).await;
            assert!(result.is_err(), "non-DB URLs must not serve local files");
        });

        std::fs::remove_dir_all(&root).ok();
    }
}
