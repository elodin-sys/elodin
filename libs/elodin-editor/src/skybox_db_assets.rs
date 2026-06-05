use bevy::prelude::*;
use bevy_ai_skybox::{
    SkyboxManifest,
    prelude::{SetActiveSkybox, SkyboxAssetSettings, SkyboxCache},
};
use impeller2_bevy::ConnectionAddr;
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
    mut skyboxes: MessageWriter<SetActiveSkybox>,
) {
    let Some(skybox) = desired_skybox_from_config(&config).flatten() else {
        mirror.synced = None;
        mirror.last_failed = None;
        return;
    };
    let (Some(connection_addr), Some(settings), Some(cache)) =
        (connection_addr, settings, cache.as_mut())
    else {
        return;
    };
    let key = MirrorKey {
        addr: connection_addr.0,
        skybox: skybox.clone(),
    };
    if mirror.synced.as_ref() == Some(&key) {
        return;
    }
    if let Some((failed, at)) = &mirror.last_failed
        && failed == &key
        && at.elapsed() < RETRY_DELAY
    {
        return;
    }

    match sync_db_skybox_asset(&skybox, connection_addr.0, &settings, cache) {
        Ok(()) => {
            mirror.synced = Some(key);
            mirror.last_failed = None;
            skyboxes.write(SetActiveSkybox::ByName(skybox));
        }
        Err(error) => {
            tracing::warn!(
                skybox = %skybox,
                error = %error,
                "failed to mirror skybox assets from database"
            );
            mirror.last_failed = Some((key, Instant::now()));
        }
    }
}

fn fetch_asset(client: &reqwest::blocking::Client, url: &str) -> Result<Vec<u8>, String> {
    let response = client
        .get(url)
        .send()
        .map_err(|err| format!("{url}: {err}"))?
        .error_for_status()
        .map_err(|err| format!("{url}: {err}"))?;
    response
        .bytes()
        .map(|bytes| bytes.to_vec())
        .map_err(|err| format!("{url}: {err}"))
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

fn sync_db_skybox_asset(
    skybox: &str,
    connection_addr: SocketAddr,
    settings: &SkyboxAssetSettings,
    cache: &mut SkyboxCache,
) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|err| err.to_string())?;
    let base = assets_http_base(connection_addr);
    let manifest_url = format!("{base}/{}", impeller2_kdl::SKYBOX_MANIFEST_ASSET_NAME);
    let manifest_bytes = fetch_asset(&client, &manifest_url)?;
    let manifest = std::str::from_utf8(&manifest_bytes).map_err(|err| err.to_string())?;
    let manifest = SkyboxManifest::from_ron_str(manifest).map_err(|err| err.to_string())?;
    let entry = manifest
        .get(skybox)
        .cloned()
        .ok_or_else(|| format!("skybox `{skybox}` is not present in database manifest"))?;

    let cubemap_name = impeller2_kdl::skybox_cubemap_asset_name(&entry.cubemap_file)
        .ok_or_else(|| format!("invalid skybox cubemap file path `{}`", entry.cubemap_file))?;
    let cubemap_url = format!("{base}/{cubemap_name}");
    let cubemap_bytes = fetch_asset(&client, &cubemap_url)?;
    write_cubemap(&settings.cache_dir, &entry.cubemap_file, &cubemap_bytes)?;

    cache.manifest.upsert(entry);
    cache.handles.remove(skybox);
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
}
