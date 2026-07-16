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
    time::{Duration, Instant, SystemTime},
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
    /// Skybox name the user cleared *locally* (e.g. "Clear Schematic") without
    /// pushing the clear to the DB. While `skybox.active` still names it, the
    /// mirror must not re-assert or re-apply it — the drift is intentional.
    /// Lifted when the DB moves to a different skybox (external change wins),
    /// or when any skybox becomes live again.
    locally_cleared: Option<String>,
}

impl DbSkyboxAssetMirror {
    /// Record that the user cleared the rendered skybox locally while the DB's
    /// `skybox.active` (`desired`) still names it, so the mirror won't fight the
    /// clear by re-asserting that skybox. `None` (no DB skybox) clears any
    /// stale suppression.
    pub fn note_local_clear(&mut self, desired: Option<String>) {
        self.locally_cleared = desired;
    }
}

/// Whether a locally cleared skybox suppresses mirror activation of `desired`.
/// Only while nothing is live and the DB still names the skybox the user
/// cleared; a different desired name is external news and wins.
fn local_clear_suppresses(
    locally_cleared: Option<&str>,
    desired: &str,
    live: Option<&str>,
) -> bool {
    live.is_none() && locally_cleared == Some(desired)
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
                            // The user locally cleared this skybox while the
                            // download was in flight: keep the bytes mirrored
                            // but don't resurrect it on screen.
                            if mirror.locally_cleared.as_deref() != Some(key.skybox.as_str()) {
                                skyboxes.write(SetActiveSkybox::ByName(key.skybox));
                            }
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
            // The DB agrees the skybox is cleared; no local override needed.
            mirror.locally_cleared = None;
            if mirror.synced.take().is_some() {
                skyboxes.write(SetActiveSkybox::Clear);
            }
            return;
        }
        Some(Some(skybox)) => skybox,
    };
    // The DB moved to a different skybox than the one the user cleared locally:
    // external news wins, lift the suppression so it applies normally.
    if mirror
        .locally_cleared
        .as_deref()
        .is_some_and(|cleared| cleared != desired)
    {
        mirror.locally_cleared = None;
    }
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
        if live.is_some() {
            // A skybox is rendering again (user activated one, or a schematic
            // load re-applied it): the local clear is over.
            mirror.locally_cleared = None;
        }
        let pushed_locally = locally_pushed
            .as_ref()
            .is_some_and(|pushed| pushed.is_pending(live));
        let locally_cleared =
            local_clear_suppresses(mirror.locally_cleared.as_deref(), &desired, live);
        if should_reassert_db_skybox(live, &desired, pushed_locally || locally_cleared) {
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
/// re-send with a backoff. Content is fingerprinted by digest (not mtime) so a
/// DB→local mirror rewrite of identical bytes does not invalidate confirmation
/// and re-upload forever.
#[derive(Resource, Default)]
pub struct DbSkyboxUploaded {
    /// `(addr, skybox, digest)` confirmed queryable from the DB.
    confirmed: Option<(MirrorKey, CubemapDigest)>,
    /// Last (re)send attempt, throttles retries until the upload is confirmed.
    last_attempt: Option<(MirrorKey, Instant)>,
    /// In-flight upload/verify state machine for the active skybox.
    in_flight: Option<DbSkyboxUpload>,
    /// Memo of the last on-disk cubemap digest so the hot path can `stat`
    /// instead of re-reading multi-MB KTX2 bytes every frame.
    digest_cache: Option<(PathBuf, SystemTime, u64, CubemapDigest)>,
}

struct DbSkyboxUpload {
    key: MirrorKey,
    digest: CubemapDigest,
    phase: UploadPhase,
}

/// `(db asset key, bytes)` pairs ready to send as [`StoreAsset`] messages.
type UploadPayload = Vec<(String, Vec<u8>)>;

/// Result of the prepare step: either the DB already has matching bytes, or we
/// need to `StoreAsset` the (merged) manifest + cubemap and then verify.
enum PrepareOutput {
    /// DB already serves this skybox entry with matching cubemap digest.
    AlreadyPresent { digest: CubemapDigest },
    /// Messages to send, plus the digest used to confirm they landed.
    NeedsUpload {
        uploads: UploadPayload,
        digest: CubemapDigest,
    },
}

enum UploadPhase {
    /// Checking the DB / building the upload payload off the main thread.
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

/// Content digest of the on-disk cubemap. Same bytes ⇒ same digest regardless
/// of mtime (DB mirror rewrites must not look like a regeneration).
fn cubemap_digest(path: &Path) -> Option<CubemapDigest> {
    let bytes = std::fs::read(path).ok()?;
    Some(digest_bytes(&bytes))
}

/// Metadata-gated digest: `stat` first; only re-read when path/mtime/len change.
fn cubemap_digest_cached(
    cache: &mut Option<(PathBuf, SystemTime, u64, CubemapDigest)>,
    path: &Path,
) -> Option<CubemapDigest> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().ok()?;
    let len = meta.len();
    if let Some((cached_path, cached_mtime, cached_len, digest)) = cache.as_ref() {
        if cached_path == path && *cached_mtime == mtime && *cached_len == len {
            return Some(*digest);
        }
    }
    let digest = cubemap_digest(path)?;
    *cache = Some((path.to_path_buf(), mtime, len, digest));
    Some(digest)
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
    let Some(digest) = cubemap_digest_cached(&mut uploaded.digest_cache, &cubemap_path) else {
        return; // bytes not on disk yet (e.g. generation still running)
    };
    if !settings.manifest_path().exists() {
        return;
    }

    if uploaded
        .confirmed
        .as_ref()
        .is_some_and(|(confirmed, confirmed_digest)| {
            confirmed == &key && *confirmed_digest == digest
        })
    {
        return;
    }

    // Drive the upload/verify state machine.
    if let Some(up) = uploaded.in_flight.as_mut() {
        if up.key != key || up.digest != digest {
            uploaded.in_flight = None; // desired skybox or local content changed
        } else {
            match &mut up.phase {
                UploadPhase::Preparing(task) => match future::block_on(future::poll_once(task)) {
                    Some(Ok(PrepareOutput::AlreadyPresent { digest })) => {
                        uploaded.confirmed = Some((key, digest));
                        uploaded.last_attempt = None;
                        uploaded.in_flight = None;
                        return;
                    }
                    Some(Ok(PrepareOutput::NeedsUpload { uploads, digest })) => {
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
                        uploaded.confirmed = Some((key, digest));
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
        digest,
        phase: UploadPhase::Preparing(IoTaskPool::get().spawn(async move {
            tokio::runtime::Runtime::new()
                .map_err(|err| err.to_string())?
                .block_on(prepare_skybox_upload(addr, entry, cubemap_path))
        })),
    });
}

/// Prepares an upload for the active skybox, or reports that the DB already
/// serves matching bytes (so no `StoreAsset` is needed — common for recorded
/// DBs that ingested the skybox at sim time).
///
/// When uploading, the manifest is **merged** into the DB's current copy rather
/// than sent wholesale, so entries written by `persist_schematic_assets` or
/// other clients are preserved instead of being clobbered by the editor's local
/// manifest.
async fn prepare_skybox_upload(
    connection_addr: SocketAddr,
    entry: ManifestEntry,
    cubemap_path: PathBuf,
) -> Result<PrepareOutput, String> {
    let skybox_name = entry.name.clone();
    let (cubemap_name, cubemap_bytes, digest) =
        read_local_cubemap(&entry.cubemap_file, &cubemap_path)?;

    // Recorded / already-ingested DBs already hold the cubemap. Confirm first
    // so we do not re-StoreAsset ~16MB every few seconds on editor connect.
    if verify_db_skybox_assets(&skybox_name, connection_addr, digest).await? {
        return Ok(PrepareOutput::AlreadyPresent { digest });
    }

    let mut manifest = fetch_db_skybox_manifest(connection_addr).await?;
    manifest.upsert(entry);
    let manifest_bytes = manifest.to_ron_bytes().map_err(|err| err.to_string())?;

    Ok(PrepareOutput::NeedsUpload {
        uploads: vec![
            (
                impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME.to_string(),
                manifest_bytes,
            ),
            (cubemap_name, cubemap_bytes),
        ],
        digest,
    })
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

    // Do not use HEAD for length: axum's auto-HEAD for `get` handlers that
    // return an owned body reports `Content-Length: 0` to reqwest, which made
    // verify always fail and re-StoreAsset the cubemap every retry.
    // Confirm the served bytes are exactly the ones we uploaded / expect.
    let cubemap_bytes = match client.get(&cubemap_url).send().await {
        Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => return Ok(false),
        Ok(resp) => resp
            .error_for_status()
            .map_err(|err| format!("{cubemap_url}: {err}"))?
            .bytes()
            .await
            .map_err(|err| format!("{cubemap_url}: {err}"))?
            .to_vec(),
        Err(err) => return Err(format!("{cubemap_url}: {err}")),
    };
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
    // Skip rewrite when content already matches — avoids mtime churn that used
    // to invalidate upload confirmation under the old len+mtime fingerprint.
    if path.is_file()
        && std::fs::read(&path)
            .ok()
            .is_some_and(|existing| existing.as_slice() == bytes)
    {
        return Ok(());
    }
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
    fn local_clear_suppresses_only_matching_desired_while_nothing_live() {
        // "Clear Schematic" cleared the skybox locally while the DB still names
        // it: the mirror must not resurrect it.
        assert!(local_clear_suppresses(
            Some("desert_night"),
            "desert_night",
            None
        ));
        // The DB moved to another skybox: external news wins.
        assert!(!local_clear_suppresses(
            Some("desert_night"),
            "grand_canyon",
            None
        ));
        // A skybox is live again: the local clear is over.
        assert!(!local_clear_suppresses(
            Some("desert_night"),
            "desert_night",
            Some("desert_night")
        ));
        // No local clear recorded.
        assert!(!local_clear_suppresses(None, "desert_night", None));
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
    fn cubemap_digest_ignores_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sky.cubemap.ktx2");
        std::fs::write(&path, b"stable-bytes").unwrap();
        let first = cubemap_digest(&path).unwrap();
        // Re-write identical bytes (mtime changes; content does not).
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, b"stable-bytes").unwrap();
        let second = cubemap_digest(&path).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn cubemap_digest_cached_same_bytes_rewrite_returns_equal_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sky.cubemap.ktx2");
        std::fs::write(&path, b"stable-bytes").unwrap();
        let mut cache = None;
        let first = cubemap_digest_cached(&mut cache, &path).unwrap();
        assert!(cache.is_some());
        // Same-bytes rewrite bumps mtime/len match fails → re-read; digest equal.
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, b"stable-bytes").unwrap();
        let second = cubemap_digest_cached(&mut cache, &path).unwrap();
        assert_eq!(first, second);
        // Unchanged mtime/len hits the memo path (still equal).
        let third = cubemap_digest_cached(&mut cache, &path).unwrap();
        assert_eq!(first, third);
    }

    #[test]
    fn cubemap_digest_cached_changed_bytes_yields_new_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sky.cubemap.ktx2");
        std::fs::write(&path, b"old-cubemap").unwrap();
        let mut cache = None;
        let first = cubemap_digest_cached(&mut cache, &path).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        std::fs::write(&path, b"new-cubemap").unwrap();
        let second = cubemap_digest_cached(&mut cache, &path).unwrap();
        assert_ne!(first, second);
        assert_eq!(second, digest_bytes(b"new-cubemap"));
    }

    #[test]
    fn write_cubemap_skips_identical_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path();
        write_cubemap(cache, "desert_night.cubemap.ktx2", b"ktx2").unwrap();
        let path = cubemap_cache_path(cache, "desert_night.cubemap.ktx2").unwrap();
        let mtime_before = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(Duration::from_millis(20));
        write_cubemap(cache, "desert_night.cubemap.ktx2", b"ktx2").unwrap();
        let mtime_after = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(mtime_before, mtime_after);
    }

    #[test]
    fn db_skybox_mirror_synced_matches_active_mirror_key() {
        let addr: SocketAddr = "127.0.0.1:2240".parse().unwrap();
        let mirror = DbSkyboxAssetMirror {
            synced: Some(MirrorKey {
                addr,
                skybox: "desert_night".to_string(),
            }),
            ..Default::default()
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
            ..Default::default()
        };
        assert!(!db_skybox_mirror_pending(
            addr,
            "desert_night",
            &mirror,
            &in_flight
        ));

        let mirror = DbSkyboxAssetMirror {
            last_failed: Some((key, Instant::now())),
            ..Default::default()
        };
        assert!(!db_skybox_mirror_pending(
            addr,
            "desert_night",
            &mirror,
            &in_flight
        ));
    }
}
