use std::future::Future;
use std::io;
use std::path::Path;

use bevy::asset::io::*;
use bevy::prelude::*;
use reqwest::StatusCode;

use super::asset_cache::{self, CachedAsset};

#[derive(Default)]
pub struct WebAssetPlugin;

struct Client {
    #[allow(dead_code)]
    rt: Runtime,
    source: Source,
    cache: Box<dyn asset_cache::AssetCache>,
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
        app.register_asset_source("http", Client::asset_source(rt.clone(), Source::Http));
        app.register_asset_source("https", Client::asset_source(rt.clone(), Source::Https));
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
                let response = request.send().await.map_err(http_err)?;
                if let Some(CachedAsset { data, .. }) = cached_asset {
                    if response.status() == StatusCode::NOT_MODIFIED {
                        return Ok((data, None));
                    }
                }
                let etag = response
                    .headers()
                    .get("ETag")
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_owned());

                let data = response.bytes().await.map_err(http_err)?.to_vec();
                Ok((data, etag))
            })
            .await
    }

    fn asset_source(rt: Runtime, source: Source) -> AssetSourceBuilder {
        AssetSource::build().with_reader(move || {
            let rt = rt.clone();
            let cache = Box::new(asset_cache::cache());
            Box::new(Client { rt, source, cache })
        })
    }
}

impl AssetReader for Client {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<Box<dyn Reader>, AssetReaderError> {
        let url = self.url(path)?;
        let cached_asset = self.cache.get(&url);

        let (bytes, etag) = self.get(url.clone(), cached_asset).await?;
        if let Some(etag) = etag {
            let cached_asset = CachedAsset {
                data: bytes.clone(),
                etag,
            };
            self.cache.put(&url, cached_asset);
        }
        let reader = VecReader::new(bytes);
        Ok(Box::new(reader) as Box<dyn Reader>)
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

fn http_err(err: reqwest::Error) -> AssetReaderError {
    let status_code = err
        .status()
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        .as_u16();
    AssetReaderError::HttpError(status_code)
}
