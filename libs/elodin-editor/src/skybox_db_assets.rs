use bevy::prelude::*;
use bevy::tasks::{IoTaskPool, Task, futures_lite::future};
use bevy_ai_skybox::{
    ManifestEntry, SkyboxManifest,
    prelude::{SetActiveSkybox, SkyboxAssetSettings, SkyboxCache},
};
use impeller2_bevy::ConnectionAddr;
use impeller2_kdl::FromKdl;
use impeller2_wkt::{DbConfig, Schematic};
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
    if let Some(desired) = config.skybox_active_desired() {
        return Some(desired);
    }

    let content = config.schematic_content()?;
    let schematic = Schematic::from_kdl(content)
        .inspect_err(|e| tracing::warn!("Failed to parse schematic KDL while syncing skybox: {e}"))
        .ok()?;
    Some(schematic.skybox.as_ref().map(|skybox| skybox.name.clone()))
}

pub fn sync_db_skybox_assets_from_config(
    config: Res<DbConfig>,
    connection_addr: Option<Res<ConnectionAddr>>,
    settings: Option<Res<SkyboxAssetSettings>>,
    mut cache: Option<ResMut<SkyboxCache>>,
    mut mirror: ResMut<DbSkyboxAssetMirror>,
    mut in_flight: ResMut<DbSkyboxSyncInFlight>,
    mut skyboxes: MessageWriter<SetActiveSkybox>,
) {
    if let Some(task) = in_flight.task.as_mut() {
        if let Some(result) = future::block_on(future::poll_once(task)) {
            let started_key = in_flight.key.take();
            in_flight.task = None;
            if let (Some(key), Some(settings), Some(cache)) =
                (started_key, settings.as_ref(), cache.as_mut())
                && skybox_still_desired(&config, connection_addr.as_deref(), &key)
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
                        tracing::warn!(
                            skybox = %key.skybox,
                            error = %error,
                            "failed to mirror skybox assets from database"
                        );
                        mirror.last_failed = Some((key, Instant::now()));
                    }
                }
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
        if cache
            .as_ref()
            .is_some_and(|cache| cache.active.as_deref() != Some(desired.as_str()))
        {
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
    in_flight.task = Some(
        IoTaskPool::get()
            .spawn(async move { download_db_skybox_assets(&desired, connection_addr).await }),
    );
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
    fn desired_skybox_from_config_reads_schematic_content() {
        let mut config = DbConfig::default();
        config.set_schematic_content(r#"skybox name="mojave_desert""#.to_string());

        assert_eq!(
            desired_skybox_from_config(&config),
            Some(Some("mojave_desert".to_string()))
        );
    }

    #[test]
    fn desired_skybox_from_config_honors_explicit_clear() {
        let mut config = DbConfig::default();
        config
            .metadata
            .insert("skybox.active".to_string(), String::new());
        config.set_schematic_content(r#"skybox name="mojave_desert""#.to_string());

        assert_eq!(desired_skybox_from_config(&config), Some(None));
    }

    #[test]
    fn cubemap_cache_path_rejects_traversal() {
        assert!(cubemap_cache_path(Path::new("cache"), "../bad.ktx2").is_err());
    }

    #[test]
    fn db_skybox_mirror_synced_matches_active_mirror_key() {
        let addr: SocketAddr = "127.0.0.1:2240".parse().unwrap();
        let mirror = DbSkyboxAssetMirror {
            synced: Some(MirrorKey {
                addr,
                skybox: "mojave_desert".to_string(),
            }),
            last_failed: None,
        };
        assert!(db_skybox_mirror_synced(addr, "mojave_desert", &mirror));
        assert!(!db_skybox_mirror_synced(addr, "grand_canyon", &mirror));
    }

    #[test]
    fn db_skybox_mirror_pending_tracks_sync_state() {
        let addr: SocketAddr = "127.0.0.1:2240".parse().unwrap();
        let key = MirrorKey {
            addr,
            skybox: "mojave_desert".to_string(),
        };
        let mirror = DbSkyboxAssetMirror::default();
        let in_flight = DbSkyboxSyncInFlight::default();
        assert!(db_skybox_mirror_pending(
            addr,
            "mojave_desert",
            &mirror,
            &in_flight
        ));

        let mirror = DbSkyboxAssetMirror {
            synced: Some(key.clone()),
            last_failed: None,
        };
        assert!(!db_skybox_mirror_pending(
            addr,
            "mojave_desert",
            &mirror,
            &in_flight
        ));

        let mirror = DbSkyboxAssetMirror {
            synced: None,
            last_failed: Some((key, Instant::now())),
        };
        assert!(!db_skybox_mirror_pending(
            addr,
            "mojave_desert",
            &mirror,
            &in_flight
        ));
    }
}
