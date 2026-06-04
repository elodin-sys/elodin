use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use miette::IntoDiagnostic;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
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

fn client_asset_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        other => other,
    }
}

pub fn assets_http_base_url(tcp: SocketAddr) -> String {
    let addr = assets_http_addr(tcp);
    let ip = client_asset_ip(addr.ip());
    match ip {
        IpAddr::V4(v4) => format!("http://{v4}:{}", addr.port()),
        IpAddr::V6(v6) => format!("http://[{v6}]:{}", addr.port()),
    }
}

#[derive(Error, Debug)]
pub enum SyncAssetsError {
    #[error("failed to parse schematic KDL")]
    Parse(#[source] impeller2_kdl::KdlSchematicError),
    #[error("HTTP request failed")]
    Http(#[from] reqwest::Error),
    #[error("IO error")]
    Io(#[from] io::Error),
}

/// Copies `db:` GLB assets referenced in schematic KDL from a source elodin-db assets server.
pub async fn sync_schematic_assets_from_source(
    source_tcp: SocketAddr,
    db_path: &Path,
    schematic_kdl: &str,
) -> Result<(), SyncAssetsError> {
    let schematic =
        impeller2_kdl::parse_schematic(schematic_kdl).map_err(SyncAssetsError::Parse)?;
    let names = impeller2_kdl::collect_db_glb_asset_names(&schematic);
    if names.is_empty() {
        return Ok(());
    }

    let base = assets_http_base_url(source_tcp);
    let assets_dir = assets_dir(db_path);
    std::fs::create_dir_all(&assets_dir)?;
    let client = reqwest::Client::new();

    for name in names {
        let url = format!("{base}/{name}");
        let response = client.get(&url).send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            tracing::warn!(asset = %name, %url, "schematic asset missing on source");
            continue;
        }
        if !response.status().is_success() {
            tracing::warn!(
                asset = %name,
                status = %response.status(),
                %url,
                "failed to fetch schematic asset from source"
            );
            continue;
        }
        let bytes = response.bytes().await?;
        if let Err(err) = write_asset_file(&assets_dir, &name, &bytes) {
            tracing::warn!(
                asset = %name,
                %url,
                error = %err,
                "failed to write schematic asset from source"
            );
            continue;
        }
        tracing::info!(asset = %name, "synced schematic asset from source");
    }

    Ok(())
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
    write_asset_file(&state.assets_dir, &path, &body).map_err(|err| {
        if err.kind() == io::ErrorKind::InvalidInput {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    })?;
    Ok(StatusCode::NO_CONTENT)
}

async fn serve_assets_with_listener(
    listener: tokio::net::TcpListener,
    addr: SocketAddr,
    assets_dir: PathBuf,
) -> miette::Result<()> {
    let state = Arc::new(AssetsState { assets_dir });
    let app = Router::new()
        .route("/{*path}", get(get_asset).put(put_asset))
        .with_state(state);
    tracing::info!(?addr, "assets http server listening");
    axum::serve(listener, app).await.into_diagnostic()?;
    Ok(())
}

pub async fn serve_assets(addr: SocketAddr, assets_dir: PathBuf) -> miette::Result<()> {
    std::fs::create_dir_all(&assets_dir).into_diagnostic()?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .into_diagnostic()?;
    serve_assets_with_listener(listener, addr, assets_dir).await
}

pub fn spawn_assets_http(db_path: &Path, tcp_addr: SocketAddr) -> io::Result<()> {
    let addr = assets_http_addr(tcp_addr);
    let assets_dir = assets_dir(db_path);
    std::fs::create_dir_all(&assets_dir)?;
    let listener = std::net::TcpListener::bind(addr)?;
    listener.set_nonblocking(true)?;
    stellarator::struc_con::tokio(move |_| async move {
        match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => {
                if let Err(err) = serve_assets_with_listener(listener, addr, assets_dir).await {
                    tracing::error!(?err, "assets http server failed");
                }
            }
            Err(err) => {
                tracing::error!(?err, "assets http server listener setup failed");
            }
        }
    });
    Ok(())
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
        assert_eq!(
            sanitize_asset_path("model..v2.glb").unwrap(),
            PathBuf::from("model..v2.glb")
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

    #[test]
    fn spawn_assets_http_reports_bind_conflict() {
        let dir = tempdir().unwrap();
        let tcp: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = std::net::TcpListener::bind(tcp).unwrap();
        let bound = listener.local_addr().unwrap();
        let assets_addr = assets_http_addr(bound);
        let _conflict = std::net::TcpListener::bind(assets_addr).unwrap();

        let err = spawn_assets_http(dir.path(), bound).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AddrInUse);
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
        let put_url = format!("http://{bound}/model..v2.glb");
        let response = client
            .put(&put_url)
            .body(b"rocket-payload".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert_eq!(
            std::fs::read(assets.join("model..v2.glb")).unwrap(),
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
    async fn sync_schematic_assets_from_source_copies_db_glb_files() {
        let dir = tempdir().unwrap();
        let source_assets = dir.path().join("source_assets");
        std::fs::create_dir_all(source_assets.join("models")).unwrap();
        std::fs::write(source_assets.join("models/rocket.glb"), b"from-source").unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        let state = Arc::new(AssetsState {
            assets_dir: source_assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset).put(put_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let follower_db = dir.path().join("follower_db");
        let kdl = r#"
object_3d "rocket.world_pos" {
    glb path="db:models/rocket.glb"
}
"#;
        sync_schematic_assets_from_source(tcp, &follower_db, kdl)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("models/rocket.glb")).unwrap(),
            b"from-source".to_vec()
        );
    }

    #[tokio::test]
    async fn sync_schematic_assets_from_source_continues_after_write_error() {
        let dir = tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        let app = Router::new().route(
            "/{*path}",
            get(|| async { Bytes::from_static(b"from-source") }),
        );

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let follower_db = dir.path().join("follower_db");
        let kdl = r#"
object_3d "bad.world_pos" {
    glb path="db:bad/../invalid.glb"
}

object_3d "rocket.world_pos" {
    glb path="db:models/rocket.glb"
}
"#;
        sync_schematic_assets_from_source(tcp, &follower_db, kdl)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("models/rocket.glb")).unwrap(),
            b"from-source".to_vec()
        );
    }

    #[tokio::test]
    async fn put_with_encoded_parent_dir_returns_400() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
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
