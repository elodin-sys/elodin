use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task, futures_lite::future};
use bevy_ai_skybox::{
    ManifestEntry, SkyboxManifest,
    prelude::{SetActiveSkybox, SkyboxAssetSettings, SkyboxCache},
};
use impeller2_bevy::{ConnectionAddr, PacketTx};
use impeller2_wkt::{DbConfig, StoreAsset};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::object_3d::assets_http_base;

const RETRY_DELAY: Duration = Duration::from_secs(2);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, PartialEq, Eq)]
struct MirrorKey {
    addr: SocketAddr,
    skybox: String,
}

#[derive(Resource, Default)]
pub struct DbSkyboxAssetMirror {
    synced: Option<MirrorKey>,
    last_failed: Option<(MirrorKey, Instant)>,
}

#[derive(Resource, Default)]
pub struct DbSkyboxSyncInFlight {
    key: Option<MirrorKey>,
    task: Option<Task<Result<SkyboxDownloadPayload, String>>>,
}

struct SkyboxDownloadPayload {
    entry: ManifestEntry,
    cubemap_bytes: Vec<u8>,
}

fn skybox_still_desired(
    config: &DbConfig,
    connection_addr: Option<&ConnectionAddr>,
    key: &MirrorKey,
) -> bool {
    let Some(addr) = connection_addr.map(|addr| addr.0) else {
        return false;
    };
    matches!(
        desired_skybox_from_config(config),
        Some(Some(skybox)) if key.addr == addr && key.skybox == skybox
    )
}

fn skybox_in_flight_still_desired(
    config: &DbConfig,
    connection_addr: Option<&ConnectionAddr>,
    in_flight: &DbSkyboxSyncInFlight,
) -> bool {
    let Some(key) = in_flight.key.as_ref() else {
        return false;
    };
    skybox_still_desired(config, connection_addr, key)
}

/// Returns `true` when DB skybox assets were mirrored and activation was already dispatched.
#[cfg(any(test, all(not(target_family = "wasm"), target_family = "unix")))]
pub fn db_skybox_mirror_synced(
    connection_addr: SocketAddr,
    skybox: &str,
    mirror: &DbSkyboxAssetMirror,
) -> bool {
    mirror
        .synced
        .as_ref()
        .is_some_and(|key| key.addr == connection_addr && key.skybox == skybox)
}

/// Returns `true` while headless (or other consumers) should wait before activating a DB skybox.
#[cfg(any(test, all(not(target_family = "wasm"), target_family = "unix")))]
pub fn db_skybox_mirror_pending(
    connection_addr: SocketAddr,
    skybox: &str,
    mirror: &DbSkyboxAssetMirror,
    in_flight: &DbSkyboxSyncInFlight,
) -> bool {
    let key = MirrorKey {
        addr: connection_addr,
        skybox: skybox.to_string(),
    };
    if mirror.synced.as_ref() == Some(&key) {
        return false;
    }
    if in_flight.key.as_ref() == Some(&key) && in_flight.task.is_some() {
        return true;
    }
    if let Some((failed, _)) = &mirror.last_failed
        && failed == &key
    {
        return false;
    }
    true
}

pub fn desired_skybox_from_config(config: &DbConfig) -> Option<Option<String>> {
    config.skybox_active_desired()
}

/// Whether the DB mirror should re-assert its synced skybox onto the live
/// cache. It only does so for an *external* drift (`live != desired`). When the
/// drift matches a state the user just pushed locally (`pushed_locally`) — most
/// importantly a clear — the live `DbConfig` is simply stale and re-asserting
/// would resurrect the skybox the user just cleared, so we hold off until the
/// `SetDbConfig` echo catches up.
fn should_reassert_db_skybox(live: Option<&str>, desired: &str, pushed_locally: bool) -> bool {
    live != Some(desired) && !pushed_locally
}

#[allow(clippy::too_many_arguments)]
pub fn sync_db_skybox_assets_from_config(
    config: Res<DbConfig>,
    connection_addr: Option<Res<ConnectionAddr>>,
    settings: Option<Res<SkyboxAssetSettings>>,
    mut cache: Option<ResMut<SkyboxCache>>,
    mut mirror: ResMut<DbSkyboxAssetMirror>,
    mut in_flight: ResMut<DbSkyboxSyncInFlight>,
    mut skyboxes: MessageWriter<SetActiveSkybox>,
    locally_pushed: Option<Res<crate::skybox_generation::LocallyPushedSkyboxActive>>,
) {
    if let Some(task) = in_flight.task.as_mut() {
        if let Some(result) = future::block_on(future::poll_once(task)) {
            let started_key = in_flight.key.take();
            in_flight.task = None;
            if let (Some(key), Some(settings), Some(cache)) =
                (started_key, settings.as_ref(), cache.as_mut())
                && skybox_still_desired(&config, connection_addr.as_deref(), &key)
                // A newer local push (e.g. a clear) not yet echoed into `config`
                // supersedes this download; don't re-apply the stale skybox.
                && !locally_pushed
                    .as_ref()
                    .is_some_and(|pushed| pushed.latest_supersedes(Some(&key.skybox)))
            {
                match result {
                    Ok(payload) => match apply_db_skybox_download(&payload, settings, cache) {
                        Ok(()) => {
                            mirror.synced = Some(key.clone());
                            mirror.last_failed = None;
                            skyboxes.write(SetActiveSkybox::ByName(key.skybox));
                        }
                        Err(error) => {
                            tracing::warn!(
                                skybox = %key.skybox,
                                error = %error,
                                "failed to apply mirrored skybox assets from database"
                            );
                            mirror.last_failed = Some((key, Instant::now()));
                        }
                    },
                    Err(error) => {
                        // Transient while the editor is still uploading the
                        // cubemap to the db; the mirror retries and recovers,
                        // so this is info rather than a warning.
                        tracing::info!(
                            skybox = %key.skybox,
                            error = %error,
                            "skybox assets not yet available in database, will retry"
                        );
                        mirror.last_failed = Some((key, Instant::now()));
                    }
                }
                return;
            }
        } else if skybox_in_flight_still_desired(&config, connection_addr.as_deref(), &in_flight) {
            return;
        } else {
            in_flight.task = None;
            in_flight.key = None;
        }
    }

    let desired = match desired_skybox_from_config(&config) {
        None => return,
        Some(None) => {
            in_flight.task = None;
            in_flight.key = None;
            mirror.last_failed = None;
            if mirror.synced.take().is_some() {
                skyboxes.write(SetActiveSkybox::Clear);
            }
            return;
        }
        Some(Some(skybox)) => skybox,
    };
    // `config` may be stale relative to a local push still awaiting its echo
    // (e.g. the user just cleared or switched skybox). Don't start a download
    // for a skybox the user has already moved away from.
    if locally_pushed
        .as_ref()
        .is_some_and(|pushed| pushed.latest_supersedes(Some(&desired)))
    {
        return;
    }
    let Some(connection_addr) = connection_addr else {
        return;
    };
    if settings.is_none() || cache.is_none() {
        return;
    }
    let key = MirrorKey {
        addr: connection_addr.0,
        skybox: desired.clone(),
    };
    if mirror.synced.as_ref() == Some(&key) {
        let live = cache.as_ref().and_then(|cache| cache.active.as_deref());
        let pushed_locally = locally_pushed
            .as_ref()
            .is_some_and(|pushed| pushed.is_pending(live));
        if should_reassert_db_skybox(live, &desired, pushed_locally) {
            skyboxes.write(SetActiveSkybox::ByName(desired));
        }
        return;
    }
    if let Some((failed, at)) = &mirror.last_failed
        && failed == &key
        && at.elapsed() < RETRY_DELAY
    {
        return;
    }

    in_flight.key = Some(key);
    let connection_addr = connection_addr.0;
    in_flight.task = Some(IoTaskPool::get().spawn(async move {
        tokio::runtime::Runtime::new()
            .map_err(|err| err.to_string())?
            .block_on(download_db_skybox_assets(&desired, connection_addr))
    }));
}

/// Tracks the upload of the active skybox's local assets to the DB.
///
/// [`StoreAsset`] is fire-and-forget (the channel can drop on backpressure, the
/// server only logs store failures, and a mid-upload disconnect is silent), so
/// we don't trust the send alone: we mark an `(addr, skybox, content)` done only
/// once the DB asset server actually serves the bytes back, and otherwise
/// re-send with a backoff. The content fingerprint ensures that regenerating a
/// skybox under the same name re-uploads the new bytes.
#[derive(Resource, Default)]
pub struct DbSkyboxUploaded {
    /// `(addr, skybox, fingerprint)` confirmed queryable from the DB.
    confirmed: Option<(MirrorKey, UploadFingerprint)>,
    /// Last (re)send attempt, throttles retries until the upload is confirmed.
    last_attempt: Option<(MirrorKey, Instant)>,
    /// In-flight upload/verify state machine for the active skybox.
    in_flight: Option<DbSkyboxUpload>,
}

struct DbSkyboxUpload {
    key: MirrorKey,
    fingerprint: UploadFingerprint,
    phase: UploadPhase,
}

/// `(db asset key, bytes)` pairs ready to send as [`StoreAsset`] messages.
type UploadPayload = Vec<(String, Vec<u8>)>;

/// Output of the prepare step: the messages to send plus the digest of the
/// cubemap bytes, used to confirm the *fresh* bytes (not a stale same-named
/// file) actually landed in the DB.
type PrepareOutput = (UploadPayload, CubemapDigest);

enum UploadPhase {
    /// Fetching the DB manifest and merging our entry off the main thread.
    Preparing(Task<Result<PrepareOutput, String>>),
    /// Confirming the DB now serves the uploaded assets.
    Verifying(Task<Result<bool, String>>),
}

/// Size + content hash of a cubemap, to detect stale vs. freshly uploaded bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CubemapDigest {
    len: u64,
    hash: u64,
}

fn digest_bytes(bytes: &[u8]) -> CubemapDigest {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    CubemapDigest {
        len: bytes.len() as u64,
        hash: hasher.finish(),
    }
}

/// Identifies the on-disk cubemap content, so a same-name regeneration (new
/// bytes) invalidates a prior confirmed upload.
#[derive(Clone, Copy, PartialEq, Eq)]
struct UploadFingerprint {
    len: u64,
    modified: Option<Duration>,
}

fn cubemap_fingerprint(path: &Path) -> Option<UploadFingerprint> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
    Some(UploadFingerprint {
        len: meta.len(),
        modified,
    })
}

fn spawn_skybox_verify(
    skybox: String,
    addr: SocketAddr,
    digest: CubemapDigest,
) -> Task<Result<bool, String>> {
    IoTaskPool::get().spawn(async move {
        tokio::runtime::Runtime::new()
            .map_err(|err| err.to_string())?
            .block_on(verify_db_skybox_assets(&skybox, addr, digest))
    })
}

/// Counterpart to [`sync_db_skybox_assets_from_config`]: mirror the active
/// skybox's local `manifest.ron` + cubemap *up* to the DB via [`StoreAsset`].
///
/// Skyboxes added at runtime (default, command palette, AI generation) only
/// have their bytes on the editor's local cache; without this the DB asset
/// server 404s and the render-server/followers cannot mirror them.
pub fn upload_active_skybox_assets_to_db(
    config: Res<DbConfig>,
    connection_addr: Option<Res<ConnectionAddr>>,
    settings: Option<Res<SkyboxAssetSettings>>,
    cache: Option<Res<SkyboxCache>>,
    tx: Option<Res<PacketTx>>,
    mut uploaded: ResMut<DbSkyboxUploaded>,
) {
    let Some(connection_addr) = connection_addr else {
        return;
    };
    let (Some(settings), Some(cache), Some(tx)) = (settings, cache, tx) else {
        return;
    };

    let desired = match desired_skybox_from_config(&config) {
        Some(Some(name)) => name,
        Some(None) => {
            uploaded.confirmed = None;
            uploaded.last_attempt = None;
            uploaded.in_flight = None;
            return;
        }
        None => return,
    };

    let addr = connection_addr.0;
    let key = MirrorKey {
        addr,
        skybox: desired.clone(),
    };

    // Local entry + cubemap must be on disk before we can upload.
    let Some(entry) = cache.manifest.get(&desired).cloned() else {
        return;
    };
    let cubemap_path = settings.cache_dir.join(&entry.cubemap_file);
    let Some(fingerprint) = cubemap_fingerprint(&cubemap_path) else {
        return; // bytes not on disk yet (e.g. generation still running)
    };
    if !settings.manifest_path().exists() {
        return;
    }

    if uploaded
        .confirmed
        .as_ref()
        .is_some_and(|(confirmed, fp)| confirmed == &key && *fp == fingerprint)
    {
        return;
    }

    // Drive the upload/verify state machine.
    if let Some(up) = uploaded.in_flight.as_mut() {
        if up.key != key || up.fingerprint != fingerprint {
            uploaded.in_flight = None; // desired skybox or local content changed
        } else {
            match &mut up.phase {
                UploadPhase::Preparing(task) => match future::block_on(future::poll_once(task)) {
                    Some(Ok((uploads, digest))) => {
                        for (asset_key, bytes) in uploads {
                            tx.send_msg(StoreAsset {
                                key: asset_key,
                                bytes,
                            });
                        }
                        up.phase = UploadPhase::Verifying(spawn_skybox_verify(
                            desired.clone(),
                            addr,
                            digest,
                        ));
                        return;
                    }
                    Some(Err(error)) => {
                        tracing::warn!(
                            skybox = %desired,
                            error = %error,
                            "failed to prepare skybox assets for db upload"
                        );
                        uploaded.in_flight = None;
                        uploaded.last_attempt = Some((key, Instant::now()));
                        return;
                    }
                    None => return, // still preparing
                },
                UploadPhase::Verifying(task) => match future::block_on(future::poll_once(task)) {
                    Some(Ok(true)) => {
                        uploaded.confirmed = Some((key, fingerprint));
                        uploaded.last_attempt = None;
                        uploaded.in_flight = None;
                        return;
                    }
                    Some(Ok(false)) => {
                        uploaded.in_flight = None; // not present yet — re-send below
                    }
                    Some(Err(error)) => {
                        tracing::info!(
                            skybox = %desired,
                            error = %error,
                            "could not verify skybox upload in database, will retry"
                        );
                        uploaded.in_flight = None;
                    }
                    None => return, // still verifying
                },
            }
        }
    }

    // Throttle (re)attempts so a slow/failing upload doesn't resend every frame.
    if let Some((attempted, at)) = &uploaded.last_attempt
        && attempted == &key
        && at.elapsed() < RETRY_DELAY
    {
        return;
    }

    uploaded.last_attempt = Some((key.clone(), Instant::now()));
    uploaded.in_flight = Some(DbSkyboxUpload {
        key,
        fingerprint,
        phase: UploadPhase::Preparing(IoTaskPool::get().spawn(async move {
            tokio::runtime::Runtime::new()
                .map_err(|err| err.to_string())?
                .block_on(prepare_skybox_upload(addr, entry, cubemap_path))
        })),
    });
}

/// Prepares the `(db key, bytes)` pairs to upload for the active skybox.
///
/// The manifest is **merged** into the DB's current copy rather than sent
/// wholesale, so entries written by `persist_schematic_assets` or other clients
/// are preserved instead of being clobbered by the editor's local manifest.
async fn prepare_skybox_upload(
    connection_addr: SocketAddr,
    entry: ManifestEntry,
    cubemap_path: PathBuf,
) -> Result<PrepareOutput, String> {
    let (cubemap_name, cubemap_bytes, digest) =
        read_local_cubemap(&entry.cubemap_file, &cubemap_path)?;

    let mut manifest = fetch_db_skybox_manifest(connection_addr).await?;
    manifest.upsert(entry);
    let manifest_bytes = manifest.to_ron_bytes().map_err(|err| err.to_string())?;

    Ok((
        vec![
            (
                impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME.to_string(),
                manifest_bytes,
            ),
            (cubemap_name, cubemap_bytes),
        ],
        digest,
    ))
}

/// Local (no-network) portion of an upload: validates the cubemap file path,
/// reads its bytes, and returns the db asset key, bytes, and content digest.
fn read_local_cubemap(
    cubemap_file: &str,
    cubemap_path: &Path,
) -> Result<(String, Vec<u8>, CubemapDigest), String> {
    let cubemap_name = impeller2_kdl::skybox_cubemap_asset_name(cubemap_file)
        .ok_or_else(|| format!("invalid skybox cubemap file path `{cubemap_file}`"))?;
    let cubemap_bytes =
        std::fs::read(cubemap_path).map_err(|err| format!("{}: {err}", cubemap_path.display()))?;
    let digest = digest_bytes(&cubemap_bytes);
    Ok((cubemap_name, cubemap_bytes, digest))
}

/// Fetches the DB's current skybox manifest, or an empty one if absent.
async fn fetch_db_skybox_manifest(connection_addr: SocketAddr) -> Result<SkyboxManifest, String> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|err| err.to_string())?;
    let base = assets_http_base(connection_addr);
    let url = format!("{base}/{}", impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME);
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|err| format!("{url}: {err}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(SkyboxManifest::default());
    }
    let bytes = response
        .error_for_status()
        .map_err(|err| format!("{url}: {err}"))?
        .bytes()
        .await
        .map_err(|err| format!("{url}: {err}"))?;
    let text = std::str::from_utf8(&bytes).map_err(|err| err.to_string())?;
    SkyboxManifest::from_ron_str(text).map_err(|err| err.to_string())
}

async fn fetch_asset(client: &reqwest::Client, url: &str) -> Result<Vec<u8>, String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("{url}: {err}"))?
        .error_for_status()
        .map_err(|err| format!("{url}: {err}"))?;
    response
        .bytes()
        .await
        .map(|bytes| bytes.to_vec())
        .map_err(|err| format!("{url}: {err}"))
}

async fn download_db_skybox_assets(
    skybox: &str,
    connection_addr: SocketAddr,
) -> Result<SkyboxDownloadPayload, String> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|err| err.to_string())?;
    let base = assets_http_base(connection_addr);
    let manifest_url = format!("{base}/{}", impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME);
    let manifest_bytes = fetch_asset(&client, &manifest_url).await?;
    let manifest = std::str::from_utf8(&manifest_bytes).map_err(|err| err.to_string())?;
    let manifest = SkyboxManifest::from_ron_str(manifest).map_err(|err| err.to_string())?;
    let entry = manifest
        .get(skybox)
        .cloned()
        .ok_or_else(|| format!("skybox `{skybox}` is not present in database manifest"))?;

    let cubemap_name = impeller2_kdl::skybox_cubemap_asset_name(&entry.cubemap_file)
        .ok_or_else(|| format!("invalid skybox cubemap file path `{}`", entry.cubemap_file))?;
    let cubemap_url = format!("{base}/{cubemap_name}");
    let cubemap_bytes = fetch_asset(&client, &cubemap_url).await?;
    Ok(SkyboxDownloadPayload {
        entry,
        cubemap_bytes,
    })
}

/// Confirms the DB asset server actually serves the active skybox's manifest
/// entry and the *freshly uploaded* cubemap bytes (matched by digest, so a
/// stale same-named cubemap or a manifest that lands ahead of the new bytes
/// does not pass). `Ok(false)` means not ready yet (caller re-sends); `Err` is
/// an unexpected failure (caller retries).
async fn verify_db_skybox_assets(
    skybox: &str,
    connection_addr: SocketAddr,
    expected: CubemapDigest,
) -> Result<bool, String> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|err| err.to_string())?;
    let base = assets_http_base(connection_addr);

    let manifest_url = format!("{base}/{}", impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME);
    let manifest_resp = client
        .get(&manifest_url)
        .send()
        .await
        .map_err(|err| format!("{manifest_url}: {err}"))?;
    if manifest_resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(false);
    }
    let manifest_bytes = manifest_resp
        .error_for_status()
        .map_err(|err| format!("{manifest_url}: {err}"))?
        .bytes()
        .await
        .map_err(|err| format!("{manifest_url}: {err}"))?;
    let manifest = std::str::from_utf8(&manifest_bytes).map_err(|err| err.to_string())?;
    let manifest = SkyboxManifest::from_ron_str(manifest).map_err(|err| err.to_string())?;
    let Some(entry) = manifest.get(skybox) else {
        return Ok(false);
    };

    let cubemap_name = impeller2_kdl::skybox_cubemap_asset_name(&entry.cubemap_file)
        .ok_or_else(|| format!("invalid skybox cubemap file path `{}`", entry.cubemap_file))?;
    let cubemap_url = format!("{base}/{cubemap_name}");

    // Cheap precheck: a HEAD reveals presence + length without transferring the
    // (large) cubemap; a length mismatch means the fresh bytes haven't landed.
    // axum serves HEAD for `get` routes automatically.
    let head = client
        .head(&cubemap_url)
        .send()
        .await
        .map_err(|err| format!("{cubemap_url}: {err}"))?;
    if head.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(false);
    }
    let head = head
        .error_for_status()
        .map_err(|err| format!("{cubemap_url}: {err}"))?;
    if head.content_length().is_some_and(|len| len != expected.len) {
        return Ok(false);
    }

    // Confirm the served bytes are exactly the ones we uploaded.
    let cubemap_bytes = fetch_asset(&client, &cubemap_url).await?;
    Ok(digest_bytes(&cubemap_bytes) == expected)
}

fn cubemap_cache_path(cache_dir: &Path, cubemap_file: &str) -> Result<PathBuf, String> {
    let asset_name = impeller2_kdl::skybox_cubemap_asset_name(cubemap_file)
        .ok_or_else(|| format!("invalid skybox cubemap file path `{cubemap_file}`"))?;
    let rel = asset_name
        .strip_prefix("skyboxes/")
        .ok_or_else(|| format!("invalid skybox cubemap asset name `{asset_name}`"))?;
    Ok(cache_dir.join(rel))
}

fn write_cubemap(cache_dir: &Path, cubemap_file: &str, bytes: &[u8]) -> Result<(), String> {
    let path = cubemap_cache_path(cache_dir, cubemap_file)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| format!("{}: {err}", parent.display()))?;
    }
    let tmp = path.with_extension("ktx2.elodin-db-sync");
    std::fs::write(&tmp, bytes).map_err(|err| format!("{}: {err}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|err| format!("{}: {err}", path.display()))?;
    Ok(())
}

fn apply_db_skybox_download(
    payload: &SkyboxDownloadPayload,
    settings: &SkyboxAssetSettings,
    cache: &mut SkyboxCache,
) -> Result<(), String> {
    write_cubemap(
        &settings.cache_dir,
        &payload.entry.cubemap_file,
        &payload.cubemap_bytes,
    )?;
    cache.manifest.upsert(payload.entry.clone());
    cache
        .manifest
        .write_atomic(&settings.manifest_path())
        .map_err(|err| err.to_string())?;
    cache.handles.remove(&payload.entry.name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desired_skybox_from_config_reads_skybox_active() {
        let mut config = DbConfig::default();
        config.set_skybox_active(Some("desert_night"));

        assert_eq!(
            desired_skybox_from_config(&config),
            Some(Some("desert_night".to_string()))
        );
    }

    #[test]
    fn desired_skybox_from_config_honors_explicit_clear() {
        let mut config = DbConfig::default();
        config
            .metadata
            .insert("skybox.active".to_string(), String::new());
        config.set_schematic_active("schematics/main.kdl");

        assert_eq!(desired_skybox_from_config(&config), Some(None));
    }

    #[test]
    fn reasserts_db_skybox_only_for_external_drift() {
        // Steady state: live matches desired — nothing to do.
        assert!(!should_reassert_db_skybox(
            Some("desert_night"),
            "desert_night",
            false
        ));
        // External drift (e.g. a schematic reload dropped the skybox) — restore.
        assert!(should_reassert_db_skybox(None, "desert_night", false));
        // User just cleared locally; the config echo hasn't landed. The drift is
        // the user's own pending push, so we must not resurrect the skybox.
        assert!(!should_reassert_db_skybox(None, "desert_night", true));
    }

    #[test]
    fn cubemap_cache_path_rejects_traversal() {
        assert!(cubemap_cache_path(Path::new("cache"), "../bad.ktx2").is_err());
    }

    #[test]
    fn read_local_cubemap_reads_bytes_name_and_digest() {
        let dir = tempfile::tempdir().unwrap();
        let cubemap_path = dir.path().join("desert_night.cubemap.ktx2");
        std::fs::write(&cubemap_path, b"ktx2").unwrap();

        let (name, bytes, digest) =
            read_local_cubemap("desert_night.cubemap.ktx2", &cubemap_path).unwrap();
        assert_eq!(name, "skyboxes/desert_night.cubemap.ktx2");
        assert_eq!(bytes, b"ktx2".to_vec());
        assert_eq!(digest, digest_bytes(b"ktx2"));
    }

    #[test]
    fn read_local_cubemap_rejects_cubemap_traversal() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_local_cubemap("../escape.ktx2", &dir.path().join("escape.ktx2")).is_err());
    }

    #[test]
    fn cubemap_digest_changes_with_content() {
        assert_ne!(digest_bytes(b"old-cubemap"), digest_bytes(b"new-cubemap"));
        assert_eq!(digest_bytes(b"same"), digest_bytes(b"same"));
    }

    #[test]
    fn db_skybox_mirror_synced_matches_active_mirror_key() {
        let addr: SocketAddr = "127.0.0.1:2240".parse().unwrap();
        let mirror = DbSkyboxAssetMirror {
            synced: Some(MirrorKey {
                addr,
                skybox: "desert_night".to_string(),
            }),
            last_failed: None,
        };
        assert!(db_skybox_mirror_synced(addr, "desert_night", &mirror));
        assert!(!db_skybox_mirror_synced(addr, "grand_canyon", &mirror));
    }

    #[test]
    fn db_skybox_mirror_pending_tracks_sync_state() {
        let addr: SocketAddr = "127.0.0.1:2240".parse().unwrap();
        let key = MirrorKey {
            addr,
            skybox: "desert_night".to_string(),
        };
        let mirror = DbSkyboxAssetMirror::default();
        let in_flight = DbSkyboxSyncInFlight::default();
        assert!(db_skybox_mirror_pending(
            addr,
            "desert_night",
            &mirror,
            &in_flight
        ));

        let mirror = DbSkyboxAssetMirror {
            synced: Some(key.clone()),
            last_failed: None,
        };
        assert!(!db_skybox_mirror_pending(
            addr,
            "desert_night",
            &mirror,
            &in_flight
        ));

        let mirror = DbSkyboxAssetMirror {
            synced: None,
            last_failed: Some((key, Instant::now())),
        };
        assert!(!db_skybox_mirror_pending(
            addr,
            "desert_night",
            &mirror,
            &in_flight
        ));
    }
}
