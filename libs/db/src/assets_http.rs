use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use miette::IntoDiagnostic;
use std::io;
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

pub use impeller2::ASSETS_HTTP_PORT_OFFSET;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SanitizeError {
    #[error("invalid asset path")]
    Invalid,
}

#[derive(Clone)]
struct AssetsState {
    assets_dir: PathBuf,
}

pub fn assets_http_addr(tcp: SocketAddr) -> SocketAddr {
    SocketAddr::new(tcp.ip(), tcp.port().saturating_add(ASSETS_HTTP_PORT_OFFSET))
}

pub fn assets_dir(db_path: &Path) -> PathBuf {
    db_path.join("assets")
}

pub fn sanitize_asset_path(path: &str) -> Result<PathBuf, SanitizeError> {
    if path.is_empty() || path.contains('\0') {
        return Err(SanitizeError::Invalid);
    }
    let path = Path::new(path);
    for component in path.components() {
        match component {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(SanitizeError::Invalid);
            }
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    Ok(path.to_path_buf())
}

pub fn write_asset_file(assets_dir: &Path, name: &str, data: &[u8]) -> io::Result<()> {
    let rel = sanitize_asset_path(name)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid asset path"))?;
    std::fs::create_dir_all(assets_dir)?;
    let full = assets_dir.join(&rel);
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = full.with_extension("elodin-upload");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, &full)?;
    Ok(())
}

pub fn read_asset_file(assets_dir: &Path, name: &str) -> io::Result<Vec<u8>> {
    let rel = sanitize_asset_path(name)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid asset path"))?;
    std::fs::read(assets_dir.join(rel))
}

async fn get_asset(
    AxumPath(path): AxumPath<String>,
    State(state): State<Arc<AssetsState>>,
) -> Result<impl IntoResponse, StatusCode> {
    if path.contains("..") {
        return Err(StatusCode::BAD_REQUEST);
    }
    let rel = sanitize_asset_path(&path).map_err(|_| StatusCode::BAD_REQUEST)?;
    let full = state.assets_dir.join(rel);
    match tokio::task::spawn_blocking(move || std::fs::read(full)).await {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(err)) if err.kind() == io::ErrorKind::NotFound => Err(StatusCode::NOT_FOUND),
        Ok(Err(_)) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn put_asset(
    AxumPath(path): AxumPath<String>,
    State(state): State<Arc<AssetsState>>,
    body: Bytes,
) -> Result<StatusCode, StatusCode> {
    if path.contains("..") {
        return Err(StatusCode::BAD_REQUEST);
    }
    write_asset_file(&state.assets_dir, &path, &body).map_err(|err| {
        if err.kind() == io::ErrorKind::InvalidInput {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    })?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn serve_assets(addr: SocketAddr, assets_dir: PathBuf) -> miette::Result<()> {
    std::fs::create_dir_all(&assets_dir).into_diagnostic()?;
    let state = Arc::new(AssetsState { assets_dir });
    let app = Router::new()
        .route("/{*path}", get(get_asset).put(put_asset))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .into_diagnostic()?;
    tracing::info!(?addr, "assets http server listening");
    axum::serve(listener, app).await.into_diagnostic()?;
    Ok(())
}

pub fn spawn_assets_http(db_path: &Path, tcp_addr: SocketAddr) {
    let addr = assets_http_addr(tcp_addr);
    let assets_dir = assets_dir(db_path);
    stellarator::struc_con::tokio(move |_| async move {
        if let Err(err) = serve_assets(addr, assets_dir).await {
            tracing::error!(?err, "assets http server failed");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn sanitize_rejects_traversal_and_empty() {
        assert_eq!(
            sanitize_asset_path("rocket.glb").unwrap(),
            PathBuf::from("rocket.glb")
        );
        assert_eq!(
            sanitize_asset_path("models/rocket.glb").unwrap(),
            PathBuf::from("models/rocket.glb")
        );
        assert!(sanitize_asset_path("").is_err());
        assert!(sanitize_asset_path("../secret").is_err());
        assert!(sanitize_asset_path("foo/../../secret").is_err());
        assert!(sanitize_asset_path("/etc/passwd").is_err());
    }

    #[test]
    fn write_and_read_asset_file_round_trip() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        write_asset_file(&assets, "rocket.glb", b"glb-bytes").unwrap();
        assert_eq!(
            read_asset_file(&assets, "rocket.glb").unwrap(),
            b"glb-bytes".to_vec()
        );
        assert!(assets.join("rocket.glb").is_file());
    }

    #[test]
    fn assets_http_addr_uses_port_offset() {
        let tcp: SocketAddr = "127.0.0.1:2240".parse().unwrap();
        assert_eq!(assets_http_addr(tcp), "127.0.0.1:2241".parse().unwrap());
    }

    #[tokio::test]
    async fn put_and_get_over_http() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();

        let state = Arc::new(AssetsState {
            assets_dir: assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset).put(put_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let put_url = format!("http://{bound}/rocket.glb");
        let response = client
            .put(&put_url)
            .body(b"rocket-payload".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            std::fs::read(assets.join("rocket.glb")).unwrap(),
            b"rocket-payload".to_vec()
        );

        let get_response = client.get(&put_url).send().await.unwrap();
        assert_eq!(get_response.status(), StatusCode::OK);
        assert_eq!(
            get_response.bytes().await.unwrap().as_ref(),
            b"rocket-payload"
        );
    }

    #[tokio::test]
    async fn get_missing_asset_returns_404() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();

        let state = Arc::new(AssetsState { assets_dir: assets });
        let app = Router::new()
            .route("/{*path}", get(get_asset).put(put_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let response = reqwest::Client::new()
            .get(format!("http://{bound}/missing.glb"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn open_skips_assets_directory() {
        use crate::DB;

        let dir = tempdir().unwrap();
        let db = DB::create(dir.path().to_path_buf()).unwrap();
        drop(db);

        let assets = dir.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("rocket.glb"), b"payload").unwrap();

        DB::open(dir.path().to_path_buf()).expect("open should ignore assets/");
    }

    #[tokio::test]
    async fn put_with_encoded_parent_dir_returns_400() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();

        let state = Arc::new(AssetsState { assets_dir: assets.clone() });
        let app = Router::new()
            .route("/{*path}", get(get_asset).put(put_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let response = reqwest::Client::new()
            .put(format!("http://{bound}/..%2frocket.glb"))
            .body(b"x".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(!assets.join("rocket.glb").exists());
    }
}
