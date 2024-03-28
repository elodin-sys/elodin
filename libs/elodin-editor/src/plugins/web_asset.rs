use std::future::Future;
use std::io;
use std::path::Path;

use bevy::asset::io::*;
use bevy::prelude::*;
use bevy::utils::BoxedFuture;
use reqwest::StatusCode;

#[derive(Default)]
pub struct WebAssetPlugin;

#[derive(Clone)]
struct Client {
    #[allow(dead_code)]
    rt: Runtime,
    source: Source,
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

    async fn get(&self, url: String) -> Result<Vec<u8>, AssetReaderError> {
        self.rt
            .spawn(async move {
                let bytes = reqwest::get(url)
                    .await
                    .and_then(|r| r.error_for_status())
                    .map_err(http_err)?
                    .bytes()
                    .await
                    .map_err(http_err)?;
                Ok(bytes.to_vec())
            })
            .await
    }

    fn asset_source(rt: Runtime, source: Source) -> AssetSourceBuilder {
        let client = Client { rt, source };
        AssetSource::build().with_reader(move || Box::new(client.clone()))
    }
}

impl AssetReader for Client {
    fn read<'a>(
        &'a self,
        path: &'a Path,
    ) -> BoxedFuture<'a, Result<Box<Reader<'a>>, AssetReaderError>> {
        Box::pin(async move {
            let url = self.url(path)?;
            let bytes = self.get(url).await?;
            let reader = VecReader::new(bytes);
            Ok(Box::new(reader) as Box<Reader<'a>>)
        })
    }

    fn read_meta<'a>(
        &'a self,
        path: &'a Path,
    ) -> BoxedFuture<'a, Result<Box<Reader<'a>>, AssetReaderError>> {
        Box::pin(async move { Err(AssetReaderError::NotFound(path.to_owned())) })
    }

    fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> BoxedFuture<'a, Result<Box<PathStream>, AssetReaderError>> {
        Box::pin(async move { Err(AssetReaderError::NotFound(path.to_owned())) })
    }

    fn is_directory<'a>(
        &'a self,
        _path: &'a Path,
    ) -> BoxedFuture<'a, Result<bool, AssetReaderError>> {
        Box::pin(async { Ok(false) })
    }
}

fn http_err(err: reqwest::Error) -> AssetReaderError {
    let status_code = err
        .status()
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        .as_u16();
    AssetReaderError::HttpError(status_code)
}
