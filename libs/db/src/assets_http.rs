use axum::Router;
use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use miette::IntoDiagnostic;
use std::collections::HashSet;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tracing::warn;

use impeller2_wkt::{DbConfig, Schematic};

use crate::DB;

pub use impeller2::ASSETS_HTTP_PORT_OFFSET;

/// Active skybox for asset sync: `skybox.active` metadata overrides schematic KDL.
pub fn skybox_name_for_schematic_sync(
    db_config: Option<&DbConfig>,
    schematic: &Schematic,
) -> Option<String> {
    if let Some(config) = db_config
        && let Some(desired) = config.skybox_active_desired()
    {
        return desired;
    }
    schematic.skybox.as_ref().map(|skybox| skybox.name.clone())
}

/// `db:` asset keys to fetch plus the active skybox name for cubemap resolution.
fn schematic_sync_asset_names(
    schematic: &Schematic,
    db_config: Option<&DbConfig>,
) -> (Vec<String>, Option<String>) {
    let skybox_name = skybox_name_for_schematic_sync(db_config, schematic);
    let mut names = impeller2_kdl::collect_db_asset_names(schematic);
    if skybox_name.is_some()
        && !names
            .iter()
            .any(|name| name == impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME)
    {
        names.push(impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME.to_string());
    }
    (names, skybox_name)
}

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
    #[error("IO error")]
    Io(#[from] io::Error),
}

const ASSET_FETCH_ATTEMPTS: usize = 8;
const ASSET_FETCH_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(100);
const ASSET_FETCH_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);
const ASSET_FETCH_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

fn sync_http_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(ASSET_FETCH_CONNECT_TIMEOUT)
        .timeout(ASSET_FETCH_REQUEST_TIMEOUT)
        .build()
}

fn fetch_error_retryable(err: &reqwest::Error) -> bool {
    err.is_connect() || err.is_timeout()
}

async fn fetch_source_asset(client: &reqwest::Client, url: &str, asset: &str) -> Option<Vec<u8>> {
    for attempt in 0..ASSET_FETCH_ATTEMPTS {
        match client.get(url).send().await {
            Ok(response) => {
                if response.status() == reqwest::StatusCode::NOT_FOUND {
                    tracing::warn!(asset = %asset, %url, "schematic asset missing on source");
                    return None;
                }
                if !response.status().is_success() {
                    tracing::warn!(
                        asset = %asset,
                        status = %response.status(),
                        %url,
                        "failed to fetch schematic asset from source"
                    );
                    return None;
                }
                match response.bytes().await {
                    Ok(bytes) => return Some(bytes.to_vec()),
                    Err(err) => {
                        tracing::warn!(
                            asset = %asset,
                            %url,
                            attempt,
                            error = %err,
                            "failed to read schematic asset response from source"
                        );
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    asset = %asset,
                    %url,
                    attempt,
                    error = %err,
                    retryable = fetch_error_retryable(&err),
                    "failed to fetch schematic asset from source"
                );
                if !fetch_error_retryable(&err) {
                    return None;
                }
            }
        }
        if attempt + 1 < ASSET_FETCH_ATTEMPTS {
            tokio::time::sleep(ASSET_FETCH_RETRY_DELAY).await;
        }
    }
    None
}

async fn sync_one_schematic_asset(
    client: &reqwest::Client,
    base: &str,
    assets_dir: &Path,
    name: &str,
) -> Option<Vec<u8>> {
    let url = format!("{base}/{name}");
    let bytes = fetch_source_asset(client, &url, name).await?;
    if let Err(err) = write_asset_file(assets_dir, name, &bytes) {
        tracing::warn!(
            asset = %name,
            %url,
            error = %err,
            "failed to write schematic asset from source"
        );
        return None;
    }
    tracing::info!(asset = %name, "synced schematic asset from source");
    Some(bytes)
}

/// Fetches schematic `db:` assets from a source DB Asset Server into `db`'s local `assets/` dir.
pub async fn sync_schematic_assets_for_db_from_source(source_tcp: SocketAddr, db: &DB) {
    let (content, db_config) = db.with_state(|s| {
        (
            s.db_config.schematic_content().map(str::to_owned),
            s.db_config.clone(),
        )
    });
    let Some(content) = content else {
        return;
    };
    if let Err(err) =
        sync_schematic_assets_from_source(source_tcp, &db.path, &content, Some(&db_config)).await
    {
        warn!(
            ?err,
            "failed to sync schematic assets from source; db: paths may not load"
        );
    }
}

/// Copies `db:` assets referenced in schematic KDL from a source elodin-db assets server.
pub async fn sync_schematic_assets_from_source(
    source_tcp: SocketAddr,
    db_path: &Path,
    schematic_kdl: &str,
    db_config: Option<&DbConfig>,
) -> Result<(), SyncAssetsError> {
    let schematic =
        impeller2_kdl::parse_schematic(schematic_kdl).map_err(SyncAssetsError::Parse)?;
    let (names, skybox_name) = schematic_sync_asset_names(&schematic, db_config);
    if names.is_empty() {
        return Ok(());
    }

    let base = assets_http_base_url(source_tcp);
    let assets_dir = assets_dir(db_path);
    std::fs::create_dir_all(&assets_dir)?;
    let client = sync_http_client().map_err(io::Error::other)?;
    let mut synced = HashSet::new();
    let mut skybox_manifest = None;

    for name in names {
        if let Some(bytes) = sync_one_schematic_asset(&client, &base, &assets_dir, &name).await {
            if name == impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME {
                skybox_manifest = Some(bytes);
            }
            synced.insert(name);
        }
    }

    if let (Some(skybox_name), Some(manifest)) = (skybox_name, skybox_manifest)
        && let Ok(manifest) = std::str::from_utf8(&manifest)
    {
        match impeller2_kdl::skybox_manifest_cubemap_asset_name(manifest, &skybox_name) {
            Ok(Some(cubemap_name)) if !synced.contains(&cubemap_name) => {
                let _ = sync_one_schematic_asset(&client, &base, &assets_dir, &cubemap_name).await;
            }
            Ok(_) => {}
            Err(err) => {
                tracing::warn!(
                    skybox = %skybox_name,
                    error = %err,
                    "failed to resolve skybox cubemap from source manifest"
                );
            }
        }
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
        Ok(Err(err)) if err.kind() == io::ErrorKind::NotFound => {
            // A 404 is a routine client outcome (e.g. a mirror polling for an
            // asset still being uploaded), not a server fault — log at info.
            tracing::info!(asset = %path, "asset http 404 (not found)");
            Err(StatusCode::NOT_FOUND)
        }
        Ok(Err(err)) => {
            tracing::error!(asset = %path, ?err, "asset http 500 (read error)");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(err) => {
            tracing::error!(asset = %path, ?err, "asset http 500 (read task failed)");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn serve_assets_with_listener(
    listener: tokio::net::TcpListener,
    addr: SocketAddr,
    assets_dir: PathBuf,
) -> miette::Result<()> {
    let state = Arc::new(AssetsState { assets_dir });
    let app = Router::new()
        .route("/{*path}", get(get_asset))
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
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    stellarator::struc_con::tokio(move |_| async move {
        match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => {
                if ready_tx.send(Ok(())).is_ok()
                    && let Err(err) = serve_assets_with_listener(listener, addr, assets_dir).await
                {
                    tracing::error!(?err, "assets http server failed");
                }
            }
            Err(err) => {
                tracing::error!(?err, "assets http server listener setup failed");
                let _ = ready_tx.send(Err(err));
            }
        }
    });
    match ready_rx.recv() {
        Ok(result) => result,
        Err(_) => Err(io::Error::other(
            "assets http server thread exited before listener setup completed",
        )),
    }
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
    async fn spawn_assets_http_serves_after_startup() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        write_asset_file(&assets, "rocket.glb", b"spawned-payload").unwrap();
        let tcp: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = std::net::TcpListener::bind(tcp).unwrap();
        let bound = listener.local_addr().unwrap();
        drop(listener);

        spawn_assets_http(dir.path(), bound).unwrap();

        let assets_addr = assets_http_addr(bound);
        let response = reqwest::Client::new()
            .get(format!("http://{assets_addr}/rocket.glb"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"spawned-payload");
    }

    #[tokio::test]
    async fn get_over_http_allows_dot_dot_in_asset_name() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        write_asset_file(&assets, "model..v2.glb", b"rocket-payload").unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();

        let state = Arc::new(AssetsState {
            assets_dir: assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let get_response = client
            .get(format!("http://{bound}/model..v2.glb"))
            .send()
            .await
            .unwrap();
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
            .route("/{*path}", get(get_asset))
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
    async fn sync_schematic_assets_from_source_copies_db_assets() {
        let dir = tempdir().unwrap();
        let source_assets = dir.path().join("source_assets");
        std::fs::create_dir_all(source_assets.join("models")).unwrap();
        std::fs::create_dir_all(source_assets.join("icons")).unwrap();
        std::fs::create_dir_all(source_assets.join("skyboxes")).unwrap();
        std::fs::write(source_assets.join("models/rocket.glb"), b"from-source").unwrap();
        std::fs::write(source_assets.join("icons/marker.png"), b"png-source").unwrap();
        let skybox_manifest = r#"
(
    version: 2,
    entries: [
        (
            name: "desert_night",
            prompt: "mojave",
            style: M3Photoreal,
            blockade: None,
            cubemap_file: "desert_night.cubemap.ktx2",
            face_size: 2048,
            created_at: "2026-05-11T05:34:26Z",
        ),
    ],
    default: Some("desert_night"),
)
"#;
        std::fs::write(source_assets.join("skyboxes/manifest.ron"), skybox_manifest).unwrap();
        std::fs::write(
            source_assets.join("skyboxes/desert_night.cubemap.ktx2"),
            b"ktx2-source",
        )
        .unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        let state = Arc::new(AssetsState {
            assets_dir: source_assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let follower_db = dir.path().join("follower_db");
        let kdl = r#"
skybox name="desert_night"

object_3d "rocket.world_pos" {
    glb path="db:models/rocket.glb"
    icon path="db:icons/marker.png"
}
"#;
        sync_schematic_assets_from_source(tcp, &follower_db, kdl, None)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("models/rocket.glb")).unwrap(),
            b"from-source".to_vec()
        );
        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("icons/marker.png")).unwrap(),
            b"png-source".to_vec()
        );
        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("skyboxes/manifest.ron")).unwrap(),
            skybox_manifest.as_bytes().to_vec()
        );
        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("skyboxes/desert_night.cubemap.ktx2"))
                .unwrap(),
            b"ktx2-source".to_vec()
        );
    }

    #[tokio::test]
    async fn sync_schematic_assets_uses_skybox_active_metadata() {
        let dir = tempdir().unwrap();
        let source_assets = dir.path().join("source_assets");
        std::fs::create_dir_all(source_assets.join("skyboxes")).unwrap();
        let skybox_manifest = r#"
(
    version: 2,
    entries: [
        (
            name: "desert_night",
            prompt: "mojave",
            style: M3Photoreal,
            blockade: None,
            cubemap_file: "desert_night.cubemap.ktx2",
            face_size: 2048,
            created_at: "2026-05-11T05:34:26Z",
        ),
        (
            name: "alpine",
            prompt: "alpine",
            style: M3Photoreal,
            blockade: None,
            cubemap_file: "alpine.cubemap.ktx2",
            face_size: 2048,
            created_at: "2026-05-11T05:34:26Z",
        ),
    ],
    default: Some("desert_night"),
)
"#;
        std::fs::write(source_assets.join("skyboxes/manifest.ron"), skybox_manifest).unwrap();
        std::fs::write(
            source_assets.join("skyboxes/desert_night.cubemap.ktx2"),
            b"mojave",
        )
        .unwrap();
        std::fs::write(
            source_assets.join("skyboxes/alpine.cubemap.ktx2"),
            b"alpine",
        )
        .unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        let state = Arc::new(AssetsState {
            assets_dir: source_assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let follower_db = dir.path().join("follower_db");
        let kdl = r#"
skybox name="desert_night"
"#;
        let mut db_config = DbConfig::default();
        db_config.set_skybox_active(Some("alpine"));

        sync_schematic_assets_from_source(tcp, &follower_db, kdl, Some(&db_config))
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("skyboxes/alpine.cubemap.ktx2")).unwrap(),
            b"alpine".to_vec()
        );
        assert!(
            !assets_dir(&follower_db)
                .join("skyboxes/desert_night.cubemap.ktx2")
                .exists()
        );
    }

    #[tokio::test]
    async fn sync_schematic_assets_from_source_continues_after_write_error() {
        let dir = tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        let app = Router::new().route("/{*path}", get(|| async { "from-source" }));

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
        sync_schematic_assets_from_source(tcp, &follower_db, kdl, None)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("models/rocket.glb")).unwrap(),
            b"from-source".to_vec()
        );
    }

    #[tokio::test]
    async fn sync_returns_ok_when_assets_server_unreachable() {
        let dir = tempdir().unwrap();
        let tcp: SocketAddr = "127.0.0.1:49152".parse().unwrap();

        let kdl = r#"
object_3d "rocket.world_pos" {
    glb path="db:rocket.glb"
}
"#;
        sync_schematic_assets_from_source(tcp, dir.path(), kdl, None)
            .await
            .unwrap();
        assert!(!assets_dir(dir.path()).join("rocket.glb").exists());
    }

    #[tokio::test]
    async fn sync_waits_for_late_assets_server() {
        let dir = tempdir().unwrap();
        let source_assets = dir.path().join("source_assets");
        std::fs::create_dir_all(source_assets.join("models")).unwrap();
        std::fs::write(source_assets.join("models/rocket.glb"), b"from-source").unwrap();

        let state = Arc::new(AssetsState {
            assets_dir: source_assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);

        let reserved = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let assets_addr = reserved.local_addr().unwrap();
        let tcp = SocketAddr::new(assets_addr.ip(), assets_addr.port().saturating_sub(1));
        drop(reserved);

        let server = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let listener = tokio::net::TcpListener::bind(assets_addr).await.unwrap();
            axum::serve(listener, app).await.unwrap();
        });

        let follower_db = dir.path().join("follower_db");
        let kdl = r#"
object_3d "rocket.world_pos" {
    glb path="db:models/rocket.glb"
}
"#;
        sync_schematic_assets_from_source(tcp, &follower_db, kdl, None)
            .await
            .unwrap();
        server.abort();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("models/rocket.glb")).unwrap(),
            b"from-source".to_vec()
        );
    }

    #[test]
    fn skybox_name_for_schematic_sync_prefers_metadata() {
        let schematic = Schematic {
            skybox: Some(impeller2_wkt::SkyboxConfig {
                name: "desert_night".to_string(),
            }),
            ..Default::default()
        };
        let mut config = DbConfig::default();
        config.set_skybox_active(Some("alpine"));
        assert_eq!(
            skybox_name_for_schematic_sync(Some(&config), &schematic).as_deref(),
            Some("alpine")
        );
    }

    #[test]
    fn schematic_sync_asset_names_includes_manifest_for_metadata_only_skybox() {
        let schematic = Schematic::default();
        let mut config = DbConfig::default();
        config.set_skybox_active(Some("alpine"));
        let (names, skybox) = schematic_sync_asset_names(&schematic, Some(&config));
        assert_eq!(skybox.as_deref(), Some("alpine"));
        assert_eq!(
            names,
            vec![impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME.to_string()]
        );
    }

    #[tokio::test]
    async fn sync_schematic_assets_metadata_only_skybox_active() {
        let dir = tempdir().unwrap();
        let source_assets = dir.path().join("source_assets");
        std::fs::create_dir_all(source_assets.join("skyboxes")).unwrap();
        let skybox_manifest = r#"
(
    version: 2,
    entries: [
        (
            name: "alpine",
            prompt: "alpine",
            style: M3Photoreal,
            blockade: None,
            cubemap_file: "alpine.cubemap.ktx2",
            face_size: 2048,
            created_at: "2026-05-11T05:34:26Z",
        ),
    ],
    default: Some("alpine"),
)
"#;
        std::fs::write(source_assets.join("skyboxes/manifest.ron"), skybox_manifest).unwrap();
        std::fs::write(
            source_assets.join("skyboxes/alpine.cubemap.ktx2"),
            b"alpine",
        )
        .unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        let state = Arc::new(AssetsState {
            assets_dir: source_assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let follower_db = dir.path().join("follower_db");
        let kdl = r#"
viewport "main" { }
"#;
        let mut db_config = DbConfig::default();
        db_config.set_skybox_active(Some("alpine"));

        sync_schematic_assets_from_source(tcp, &follower_db, kdl, Some(&db_config))
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(assets_dir(&follower_db).join("skyboxes/alpine.cubemap.ktx2")).unwrap(),
            b"alpine".to_vec()
        );
    }

    #[tokio::test]
    async fn put_over_http_is_not_allowed() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        std::fs::create_dir_all(&assets).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();

        let state = Arc::new(AssetsState {
            assets_dir: assets.clone(),
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let response = reqwest::Client::new()
            .put(format!("http://{bound}/rocket.glb"))
            .body(b"x".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert!(!assets.join("rocket.glb").exists());
    }
}
