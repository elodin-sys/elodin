use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
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
    /// When false (e.g. on a follower replica), `PUT` uploads are rejected so
    /// the read-only mirror cannot diverge from its source.
    writable: bool,
    /// DB handle used to bump `assets.revision` on a successful `PUT`, so
    /// followers re-mirror and editors reload after a byte-only change (RFD
    /// #724). `None` when the server runs without a co-located DB.
    db: Option<Arc<DB>>,
}

/// Upper bound on a single `PUT` asset upload. Generous enough for large GLB
/// meshes while bounding memory use from a hostile client.
const MAX_ASSET_UPLOAD_BYTES: usize = 256 * 1024 * 1024;

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
    #[error("asset index returned HTTP {0}")]
    IndexStatus(reqwest::StatusCode),
    #[error("failed to fetch asset index from source")]
    IndexFetch(#[from] reqwest::Error),
    #[error("follower asset mirror incomplete: {failed} of {total} assets failed")]
    PartialMirror { failed: usize, total: usize },
    #[error("IO error")]
    Io(#[from] io::Error),
}

const ASSET_FETCH_ATTEMPTS: usize = 8;
const ASSET_FETCH_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(100);
const ASSET_FETCH_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);
/// Per-read timeout (reqwest resets it after each successful chunk), **not** a
/// total-request timeout. A stalled or dead connection still fails fast and
/// retries, but a large yet healthy mesh/cubemap download is never capped
/// mid-transfer the way a fixed total timeout did (RFD #724, Bug 2).
const ASSET_FETCH_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn sync_http_client() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .connect_timeout(ASSET_FETCH_CONNECT_TIMEOUT)
        .read_timeout(ASSET_FETCH_READ_TIMEOUT)
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

/// Serializes follower full-tree asset mirrors. Each `SetDbConfig` that requests
/// an asset sync would otherwise spawn an independent mirror; overlapping passes
/// can run `prune_stale_local_assets` from different index snapshots — deleting
/// keys another in-flight pass still needs, or leaving a mixed, inconsistent
/// follower asset tree (RFD #724, Bug 1).
///
/// Single-flight with coalescing: at most one mirror runs at a time. Requests
/// arriving while a pass runs set a `rerun` flag so exactly one more pass runs
/// afterwards, capturing the latest source state without unbounded stacking.
#[derive(Default)]
pub struct AssetMirrorCoordinator {
    inner: std::sync::Mutex<MirrorRunState>,
}

#[derive(Default)]
struct MirrorRunState {
    running: bool,
    rerun: bool,
}

impl AssetMirrorCoordinator {
    /// Record a mirror request and try to claim the runner slot. Returns `true`
    /// if this caller must run the mirror loop, `false` if a pass is already in
    /// flight (it will observe the `rerun` we set and do one more pass).
    fn try_begin(&self) -> bool {
        let mut st = self.inner.lock().unwrap();
        st.rerun = true;
        if st.running {
            return false;
        }
        st.running = true;
        true
    }

    /// Decide, atomically, whether the runner should do another pass. Clears
    /// `rerun` and returns `true` when a request arrived during the last pass;
    /// otherwise releases the runner slot and returns `false`. Doing both under
    /// one lock avoids a lost wakeup (a request setting `rerun` between an
    /// unlocked check and a separate clear).
    fn finish_or_continue(&self) -> bool {
        let mut st = self.inner.lock().unwrap();
        if st.rerun {
            st.rerun = false;
            true
        } else {
            st.running = false;
            false
        }
    }

    /// Run a full-tree mirror, coalescing concurrent requests. Either performs
    /// the pass(es) itself as the sole runner, or hands off to the pass already
    /// in flight (which will do one more pass on this request's behalf).
    pub async fn mirror_all(&self, source_tcp: SocketAddr, db_path: &Path) {
        if !self.try_begin() {
            return;
        }
        // Clear `running` even if a pass panics, so a crashed mirror never wedges
        // every future sync. Fires only on unwind; the normal path releases the
        // slot via `finish_or_continue`.
        struct PanicGuard<'a>(&'a AssetMirrorCoordinator);
        impl Drop for PanicGuard<'_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.inner.lock().unwrap().running = false;
                }
            }
        }
        let _panic_guard = PanicGuard(self);

        loop {
            if !self.finish_or_continue() {
                return;
            }
            if let Err(err) = sync_all_assets_from_source(source_tcp, db_path).await {
                warn!(
                    ?err,
                    "failed to sync assets from source; db: paths may not load"
                );
            }
        }
    }
}

/// Fetches schematic `db:` assets from a source DB Asset Server into `db`'s local
/// `assets/` dir, serialized through the DB's mirror coordinator so concurrent
/// syncs never overlap (RFD #724, Bug 1).
pub async fn sync_schematic_assets_for_db_from_source(source_tcp: SocketAddr, db: &DB) {
    db.asset_mirror.mirror_all(source_tcp, &db.path).await;
}

async fn fetch_source_index(
    client: &reqwest::Client,
    base: &str,
) -> Result<Vec<crate::assets::AssetEntry>, SyncAssetsError> {
    let url = format!("{base}/{INDEX_KEY}");
    let response = client.get(&url).send().await?;
    if !response.status().is_success() {
        return Err(SyncAssetsError::IndexStatus(response.status()));
    }
    let bytes = response.bytes().await?;
    serde_json::from_slice(&bytes)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err).into())
}

fn local_asset_matches(assets_dir: &Path, key: &str, remote: &[u8]) -> bool {
    read_asset_file(assets_dir, key)
        .ok()
        .is_some_and(|local| local == remote)
}

async fn mirror_asset_from_source(
    client: &reqwest::Client,
    base: &str,
    assets_dir: &Path,
    key: &str,
) -> bool {
    let url = format!("{base}/{key}");
    let Some(remote) = fetch_source_asset(client, &url, key).await else {
        return false;
    };
    if local_asset_matches(assets_dir, key, &remote) {
        return true;
    }
    write_asset_file(assets_dir, key, &remote)
        .inspect_err(|err| tracing::warn!(asset = %key, ?err, "failed to write mirrored asset"))
        .is_ok()
}

/// Delete local assets whose keys the source no longer advertises, so a follower
/// stops serving files removed upstream. Best-effort: individual prune failures
/// are logged, not fatal. `index_assets_in` already excludes dotfiles (e.g. the
/// ingest marker); reserved keys are skipped defensively.
fn prune_stale_local_assets(assets_dir: &Path, source_keys: &HashSet<String>) {
    let local = match crate::assets::index_assets_in(assets_dir, None) {
        Ok(local) => local,
        Err(err) => {
            warn!(?err, "failed to list local assets for follower prune");
            return;
        }
    };
    for entry in local {
        if source_keys.contains(&entry.key) || is_reserved_asset_key(&entry.key) {
            continue;
        }
        match remove_asset_file(assets_dir, &entry.key) {
            Ok(()) => {
                tracing::info!(asset = %entry.key, "pruned stale local asset removed upstream")
            }
            Err(err) => warn!(asset = %entry.key, ?err, "failed to prune stale local asset"),
        }
    }
}

/// Mirrors the source DB's full asset tree into `db_path/assets/` via `GET /__index__`.
///
/// Assets absent from the source index are pruned locally so the follower never
/// serves files deleted upstream. Returns [`SyncAssetsError::PartialMirror`] when
/// any advertised asset failed to download, so callers don't treat an incomplete
/// tree as a completed sync.
pub async fn sync_all_assets_from_source(
    source_tcp: SocketAddr,
    db_path: &Path,
) -> Result<(), SyncAssetsError> {
    let base = assets_http_base_url(source_tcp);
    let assets_dir = assets_dir(db_path);
    std::fs::create_dir_all(&assets_dir)?;
    let client = sync_http_client().map_err(io::Error::other)?;

    let entries = fetch_source_index(&client, &base).await?;
    // Remember every key the source advertises so stale local files can be
    // pruned once the mirror pass completes.
    let mut source_keys = HashSet::new();
    let mut synced = 0usize;
    let mut failed = 0usize;
    for entry in entries {
        if is_reserved_asset_key(&entry.key) {
            continue;
        }
        source_keys.insert(entry.key.clone());
        if mirror_asset_from_source(&client, &base, &assets_dir, &entry.key).await {
            synced += 1;
        } else {
            failed += 1;
        }
    }

    // Prune is destructive, so only run it on a fully successful pass. On a
    // partial pass the source (or its index) may be transiently unhealthy, and
    // deleting local assets against that snapshot would compound the incomplete
    // download with lost files. Skip pruning and return `PartialMirror`; the
    // coordinator reruns and the next complete pass prunes on a trustworthy view
    // (RFD #724).
    if failed > 0 {
        warn!(
            synced,
            failed, "follower asset mirror finished with missing assets; skipping prune"
        );
        return Err(SyncAssetsError::PartialMirror {
            failed,
            total: synced + failed,
        });
    }

    prune_stale_local_assets(&assets_dir, &source_keys);

    Ok(())
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
    if is_reserved_asset_key(name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "reserved asset key",
        ));
    }
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

/// Remove a stored asset by key. Sanitized like the read/write paths so a key
/// can never escape the assets root. A missing file is treated as already gone.
pub fn remove_asset_file(assets_dir: &Path, name: &str) -> io::Result<()> {
    let rel = sanitize_asset_path(name)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid asset path"))?;
    match std::fs::remove_file(assets_dir.join(rel)) {
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}

/// Reserved key prefix for the asset index listing (network replacement for a
/// filesystem `read_dir`): `GET /__index__` or `GET /__index__/<prefix>`.
pub(crate) const INDEX_KEY: &str = "__index__";

/// Keys reserved by the DB asset layer that must never be stored as real
/// assets. Storing them would be unservable: the index namespace (`__index__`,
/// `__index__/…`) is shadowed by the listing route, and the ingest marker
/// (`.elodin-ingested`) would forge the copy-once guard. Enforced at every write
/// path (uploads, skybox sync, ingest) and skipped when copying a source tree.
pub(crate) fn is_reserved_asset_key(key: &str) -> bool {
    key == crate::assets::INGEST_MARKER
        || key == INDEX_KEY
        || key
            .strip_prefix(INDEX_KEY)
            .is_some_and(|rest| rest.starts_with('/'))
}

async fn get_asset(
    AxumPath(path): AxumPath<String>,
    State(state): State<Arc<AssetsState>>,
) -> Response {
    if path == INDEX_KEY || path == "__index__/" {
        return index_response(state.assets_dir.clone(), None).await;
    }
    if let Some(prefix) = path.strip_prefix("__index__/") {
        return index_response(state.assets_dir.clone(), Some(prefix.to_string())).await;
    }

    let Ok(rel) = sanitize_asset_path(&path) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let full = state.assets_dir.join(rel);
    match tokio::task::spawn_blocking(move || std::fs::read(full)).await {
        Ok(Ok(bytes)) => bytes.into_response(),
        Ok(Err(err)) if err.kind() == io::ErrorKind::NotFound => {
            // A 404 is a routine client outcome (e.g. a mirror polling for an
            // asset still being uploaded), not a server fault — log at info.
            tracing::info!(asset = %path, "asset http 404 (not found)");
            StatusCode::NOT_FOUND.into_response()
        }
        Ok(Err(err)) => {
            tracing::error!(asset = %path, ?err, "asset http 500 (read error)");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(err) => {
            tracing::error!(asset = %path, ?err, "asset http 500 (read task failed)");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

/// `PUT /<key>` writes an asset, the network replacement for a filesystem
/// write. Path sanitization and reserved-key rejection live in
/// `write_asset_file`; here we add the write gate (`writable`) so a follower
/// replica returns `405` instead of diverging from its source.
async fn put_asset(
    AxumPath(path): AxumPath<String>,
    State(state): State<Arc<AssetsState>>,
    body: Bytes,
) -> Response {
    if !state.writable {
        tracing::warn!(asset = %path, "asset http PUT rejected (read-only server)");
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let assets_dir = state.assets_dir.clone();
    let name = path.clone();
    let db = state.db.clone();
    match tokio::task::spawn_blocking(move || {
        write_asset_file(&assets_dir, &name, &body)?;
        // Bytes changed: bump the revision so followers re-mirror and editors
        // reload even under an unchanged `schematic.active`. A persistence
        // hiccup must not fail the (already written) upload — log and move on.
        if let Some(db) = db
            && let Err(err) = db.bump_assets_revision()
        {
            tracing::warn!(?err, "failed to bump asset revision after PUT");
        }
        Ok::<(), io::Error>(())
    })
    .await
    {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(err)) if err.kind() == io::ErrorKind::InvalidInput => {
            // Reserved key or path escaping the assets root: a client fault.
            tracing::warn!(asset = %path, ?err, "asset http PUT 400 (invalid key)");
            StatusCode::BAD_REQUEST.into_response()
        }
        Ok(Err(err)) => {
            tracing::error!(asset = %path, ?err, "asset http PUT 500 (write error)");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(err) => {
            tracing::error!(asset = %path, ?err, "asset http PUT 500 (write task failed)");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn index_response(assets_dir: PathBuf, prefix: Option<String>) -> Response {
    let walk = tokio::task::spawn_blocking(move || {
        crate::assets::index_assets_in(&assets_dir, prefix.as_deref())
    })
    .await;
    match walk {
        Ok(Ok(entries)) => Json(entries).into_response(),
        Ok(Err(err)) => {
            tracing::error!(?err, "asset index 500 (walk error)");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
        Err(err) => {
            tracing::error!(?err, "asset index 500 (walk task failed)");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn serve_assets_with_listener(
    listener: tokio::net::TcpListener,
    addr: SocketAddr,
    assets_dir: PathBuf,
    writable: bool,
    db: Option<Arc<DB>>,
) -> miette::Result<()> {
    let state = Arc::new(AssetsState {
        assets_dir,
        writable,
        db,
    });
    let app = Router::new()
        .route("/{*path}", get(get_asset).put(put_asset))
        .layer(DefaultBodyLimit::max(MAX_ASSET_UPLOAD_BYTES))
        .with_state(state);
    tracing::info!(?addr, writable, "assets http server listening");
    axum::serve(listener, app).await.into_diagnostic()?;
    Ok(())
}

pub async fn serve_assets(
    addr: SocketAddr,
    assets_dir: PathBuf,
    writable: bool,
) -> miette::Result<()> {
    std::fs::create_dir_all(&assets_dir).into_diagnostic()?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .into_diagnostic()?;
    serve_assets_with_listener(listener, addr, assets_dir, writable, None).await
}

pub fn spawn_assets_http(
    db_path: &Path,
    tcp_addr: SocketAddr,
    writable: bool,
    db: Option<Arc<DB>>,
) -> io::Result<()> {
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
                    && let Err(err) =
                        serve_assets_with_listener(listener, addr, assets_dir, writable, db).await
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
    fn mirror_coordinator_coalesces_overlapping_requests() {
        let coord = AssetMirrorCoordinator::default();

        // First request claims the runner slot.
        assert!(coord.try_begin(), "first request becomes the runner");
        // Any number of requests arriving while a pass runs are absorbed, not
        // started — this is what prevents overlapping prune/download passes.
        assert!(
            !coord.try_begin(),
            "concurrent request must not start a pass"
        );
        assert!(
            !coord.try_begin(),
            "further concurrent requests also absorbed"
        );

        // After the pass, exactly one more pass runs to capture the latest
        // source state, regardless of how many requests were absorbed.
        assert!(
            coord.finish_or_continue(),
            "absorbed requests trigger one rerun"
        );
        // No new request during that rerun: the runner releases the slot.
        assert!(
            !coord.finish_or_continue(),
            "no pending request must release the runner slot"
        );

        // Slot is free again: a later request starts a fresh runner, which runs
        // one pass for that request and then releases the slot when idle.
        assert!(coord.try_begin(), "slot released, new runner can claim it");
        assert!(
            coord.finish_or_continue(),
            "runner performs the pass for its initiating request"
        );
        assert!(
            !coord.finish_or_continue(),
            "and releases cleanly when idle"
        );
    }

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

        let err = spawn_assets_http(dir.path(), bound, true, None).unwrap_err();
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

        spawn_assets_http(dir.path(), bound, true, None).unwrap();

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
            writable: true,
            db: None,
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

        let state = Arc::new(AssetsState {
            assets_dir: assets,
            writable: true,
            db: None,
        });
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

    /// Builds an in-process asset server and returns its bound address.
    #[cfg(test)]
    async fn spawn_test_asset_server(assets_dir: PathBuf, writable: bool) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let state = Arc::new(AssetsState {
            assets_dir,
            writable,
            db: None,
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset).put(put_asset))
            .layer(DefaultBodyLimit::max(MAX_ASSET_UPLOAD_BYTES))
            .with_state(state);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        bound
    }

    #[tokio::test]
    async fn put_then_get_round_trips_when_writable() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        let bound = spawn_test_asset_server(assets, true).await;

        let client = reqwest::Client::new();
        let put = client
            .put(format!("http://{bound}/models/rocket.glb"))
            .body(b"glb-bytes".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(put.status(), StatusCode::NO_CONTENT);

        let get = client
            .get(format!("http://{bound}/models/rocket.glb"))
            .send()
            .await
            .unwrap();
        assert_eq!(get.status(), StatusCode::OK);
        assert_eq!(get.bytes().await.unwrap().as_ref(), b"glb-bytes");
    }

    #[tokio::test]
    async fn put_rejected_on_read_only_server() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        let bound = spawn_test_asset_server(assets.clone(), false).await;

        let put = reqwest::Client::new()
            .put(format!("http://{bound}/models/rocket.glb"))
            .body(b"glb-bytes".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(put.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert!(!assets.join("models/rocket.glb").exists());
    }

    #[tokio::test]
    async fn put_rejects_reserved_key() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        let bound = spawn_test_asset_server(assets, true).await;

        let put = reqwest::Client::new()
            .put(format!("http://{bound}/{INDEX_KEY}"))
            .body(b"forged-index".to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(put.status(), StatusCode::BAD_REQUEST);
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
            writable: true,
            db: None,
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
            writable: true,
            db: None,
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
            writable: true,
            db: None,
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
            writable: true,
            db: None,
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
    async fn index_endpoint_lists_assets_and_filters_prefix() {
        use crate::assets::AssetEntry;

        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");
        write_asset_file(&assets, "meshes/rocket.glb", b"glb").unwrap();
        write_asset_file(&assets, "schematics/main.kdl", b"kdl").unwrap();
        // The marker is a reserved key; create it directly to exercise the
        // index's dotfile exclusion without going through `write_asset_file`.
        std::fs::write(assets.join(".elodin-ingested"), b"{}").unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let state = Arc::new(AssetsState {
            assets_dir: assets.clone(),
            writable: true,
            db: None,
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let all_bytes = client
            .get(format!("http://{bound}/__index__"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let all: Vec<AssetEntry> = serde_json::from_slice(&all_bytes).unwrap();
        let keys: Vec<_> = all.iter().map(|e| e.key.as_str()).collect();
        assert_eq!(keys, vec!["meshes/rocket.glb", "schematics/main.kdl"]);

        let schematics_bytes = client
            .get(format!("http://{bound}/__index__/schematics/"))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let schematics: Vec<AssetEntry> = serde_json::from_slice(&schematics_bytes).unwrap();
        assert_eq!(schematics.len(), 1);
        assert_eq!(schematics[0].key, "schematics/main.kdl");
    }

    #[test]
    fn write_asset_file_rejects_reserved_keys() {
        let dir = tempdir().unwrap();
        let assets = dir.path().join("assets");

        for key in ["__index__", "__index__/foo", ".elodin-ingested"] {
            let err = write_asset_file(&assets, key, b"x").unwrap_err();
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "key {key} allowed");
            assert!(!assets.join(key).exists(), "key {key} was written");
        }

        // Names that merely share the prefix are normal assets.
        write_asset_file(&assets, "__index__notreserved.glb", b"x").unwrap();
        assert!(assets.join("__index__notreserved.glb").is_file());
    }

    #[tokio::test]
    async fn follower_sync_fetches_active_schematic_file() {
        use crate::DB;

        let dir = tempdir().unwrap();
        let active_key = "schematics/main.kdl";
        let active_kdl = "viewport {\n}\n";

        // Source serves the active schematic file over its asset server.
        let source_assets = dir.path().join("source_assets");
        write_asset_file(&source_assets, active_key, active_kdl.as_bytes()).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));
        let state = Arc::new(AssetsState {
            assets_dir: source_assets,
            writable: true,
            db: None,
        });
        let app = Router::new()
            .route("/{*path}", get(get_asset))
            .with_state(state);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Follower state after a pointer-only repoint replicated from the
        // source: the active pointer is set but the asset file is still missing
        // locally.
        let follower_db = DB::create(dir.path().join("follower_db")).unwrap();
        follower_db.with_state_mut(|s| s.db_config.set_schematic_active(active_key));

        sync_schematic_assets_for_db_from_source(tcp, &follower_db).await;

        // The active file is fetched from the source so the follower's asset
        // server can serve the key its config points at.
        assert_eq!(
            std::fs::read(assets_dir(&follower_db.path).join(active_key)).unwrap(),
            active_kdl.as_bytes()
        );
        assert_eq!(
            follower_db.read_active_schematic().as_deref(),
            Some(active_kdl)
        );
    }

    #[tokio::test]
    async fn sync_all_assets_prunes_assets_removed_from_source() {
        let dir = tempdir().unwrap();
        let source_assets = dir.path().join("source_assets");
        write_asset_file(&source_assets, "models/keep.glb", b"keep").unwrap();
        let bound = spawn_test_asset_server(source_assets, false).await;
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));

        // Follower already holds a stale asset the source no longer lists.
        let follower_db = dir.path().join("follower_db");
        let follower_assets = assets_dir(&follower_db);
        write_asset_file(&follower_assets, "models/stale.glb", b"stale").unwrap();

        sync_all_assets_from_source(tcp, &follower_db)
            .await
            .unwrap();

        assert_eq!(
            std::fs::read(follower_assets.join("models/keep.glb")).unwrap(),
            b"keep"
        );
        assert!(
            !follower_assets.join("models/stale.glb").exists(),
            "asset removed upstream must be pruned locally (Bug 1)"
        );
    }

    #[tokio::test]
    async fn sync_all_assets_errors_on_partial_fetch() {
        // Source advertises a key in its index but 404s the actual GET, so the
        // mirror completes with a missing asset and must report failure (Bug 2).
        async fn index() -> Response {
            Json(vec![crate::assets::AssetEntry {
                key: "models/rocket.glb".to_string(),
                size: 3,
            }])
            .into_response()
        }
        async fn missing() -> Response {
            StatusCode::NOT_FOUND.into_response()
        }
        let app = Router::new()
            .route(&format!("/{INDEX_KEY}"), get(index))
            .route("/{*path}", get(missing));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        let dir = tempdir().unwrap();
        let follower_db = dir.path().join("follower_db");
        let err = sync_all_assets_from_source(tcp, &follower_db)
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                SyncAssetsError::PartialMirror {
                    failed: 1,
                    total: 1
                }
            ),
            "expected PartialMirror, got {err:?}"
        );
    }

    #[tokio::test]
    async fn sync_all_assets_skips_prune_on_partial_fetch() {
        // Source advertises a key it then 404s, so the pass is partial. A stale
        // local asset absent from the index must NOT be pruned during that
        // incomplete, possibly-unhealthy pass — pruning waits for a full pass so
        // the follower never loses files against an untrustworthy snapshot.
        async fn index() -> Response {
            Json(vec![crate::assets::AssetEntry {
                key: "models/rocket.glb".to_string(),
                size: 3,
            }])
            .into_response()
        }
        async fn missing() -> Response {
            StatusCode::NOT_FOUND.into_response()
        }
        let app = Router::new()
            .route(&format!("/{INDEX_KEY}"), get(index))
            .route("/{*path}", get(missing));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        let tcp = SocketAddr::new(bound.ip(), bound.port().saturating_sub(1));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        let dir = tempdir().unwrap();
        let follower_db = dir.path().join("follower_db");
        let follower_assets = assets_dir(&follower_db);
        write_asset_file(&follower_assets, "models/stale.glb", b"stale").unwrap();

        let err = sync_all_assets_from_source(tcp, &follower_db)
            .await
            .unwrap_err();
        assert!(
            matches!(err, SyncAssetsError::PartialMirror { failed: 1, .. }),
            "expected PartialMirror, got {err:?}"
        );
        assert!(
            follower_assets.join("models/stale.glb").exists(),
            "stale asset must survive a partial pass; prune only runs on a full pass"
        );
    }
}
