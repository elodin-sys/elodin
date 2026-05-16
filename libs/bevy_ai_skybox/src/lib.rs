//! Asset-only skybox loading for Elodin.
//!
//! This vendored slice intentionally does not include the Blockade generation
//! client. It loads cached cubemap entries from `assets/skyboxes/manifest.ron`
//! and applies them to Bevy `Camera3d` entities.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use bevy::{
    asset::AssetEvent,
    core_pipeline::Skybox,
    image::Image,
    light::{EnvironmentMapLight, GeneratedEnvironmentMapLight},
    prelude::*,
    render::render_resource::{TextureViewDescriptor, TextureViewDimension},
};
use serde::{Deserialize, Serialize};

pub mod prelude {
    pub use crate::{
        ManifestReloaded, PrimarySkybox, SetActiveSkybox, SkyboxAssetPlugin, SkyboxAssetSettings,
        SkyboxCache, SkyboxFailed, SkyboxManifest, SkyboxReady, SkyboxStyle,
    };
}

pub type Result<T> = std::result::Result<T, SkyboxError>;

#[derive(Debug, thiserror::Error)]
pub enum SkyboxError {
    #[error("skybox `{0}` is not present in the manifest")]
    MissingSkybox(String),
    #[error("ron: {0}")]
    Ron(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl From<ron::error::SpannedError> for SkyboxError {
    fn from(value: ron::error::SpannedError) -> Self {
        Self::Ron(value.to_string())
    }
}

impl From<ron::Error> for SkyboxError {
    fn from(value: ron::Error) -> Self {
        Self::Ron(value.to_string())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum SkyboxStyle {
    #[default]
    M3Photoreal,
    M3UhdRender,
    M3Advanced,
    Custom(u32),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BlockadeMetadata {
    pub imagine_id: u64,
    pub obfuscated_id: Option<String>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ManifestEntry {
    pub name: String,
    pub prompt: String,
    pub style: SkyboxStyle,
    pub blockade: Option<BlockadeMetadata>,
    pub equirect_file: String,
    pub cubemap_file: String,
    pub face_size: u32,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SkyboxManifest {
    pub version: u32,
    pub entries: Vec<ManifestEntry>,
    pub default: Option<String>,
}

impl Default for SkyboxManifest {
    fn default() -> Self {
        Self {
            version: 2,
            entries: Vec::new(),
            default: None,
        }
    }
}

impl SkyboxManifest {
    pub fn read_or_create(path: &Path) -> Result<Self> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let manifest = Self::default();
            manifest.write_atomic(path)?;
            return Ok(manifest);
        }

        let contents = fs::read_to_string(path)?;
        Ok(ron::from_str(&contents)?)
    }

    pub fn write_atomic(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let contents = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        let mut tmp = path.to_path_buf();
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("tmp");
        tmp.set_extension(format!("{extension}.tmp"));
        fs::write(&tmp, contents)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&ManifestEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }
}

#[derive(Clone, Debug, Resource)]
pub struct SkyboxAssetSettings {
    pub cache_dir: PathBuf,
    pub asset_dir: PathBuf,
    pub manifest_file: PathBuf,
    pub default_skybox: Option<String>,
    pub apply_to_all_cameras: bool,
    pub env_lighting: bool,
}

impl Default for SkyboxAssetSettings {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from("assets/skyboxes"),
            asset_dir: PathBuf::from("skyboxes"),
            manifest_file: PathBuf::from("manifest.ron"),
            default_skybox: None,
            apply_to_all_cameras: true,
            env_lighting: true,
        }
    }
}

impl SkyboxAssetSettings {
    pub fn manifest_path(&self) -> PathBuf {
        self.cache_dir.join(&self.manifest_file)
    }

    pub fn asset_path_for(&self, file_name: &str) -> String {
        self.asset_dir
            .join(file_name)
            .to_string_lossy()
            .replace('\\', "/")
    }
}

#[derive(Clone, Debug, Resource)]
pub struct SkyboxCache {
    pub manifest: SkyboxManifest,
    pub active: Option<String>,
    pub handles: HashMap<String, Handle<Image>>,
    pub manifest_path: PathBuf,
    pub last_manifest_modified: Option<SystemTime>,
}

impl SkyboxCache {
    pub fn load(settings: &SkyboxAssetSettings) -> Result<Self> {
        let manifest_path = settings.manifest_path();
        let manifest = SkyboxManifest::read_or_create(&manifest_path)?;
        let last_manifest_modified = modified_at(&manifest_path);
        Ok(Self {
            active: manifest.default.clone(),
            manifest,
            handles: HashMap::new(),
            manifest_path,
            last_manifest_modified,
        })
    }

    pub fn empty(manifest_path: PathBuf) -> Self {
        Self {
            manifest: SkyboxManifest::default(),
            active: None,
            handles: HashMap::new(),
            manifest_path,
            last_manifest_modified: None,
        }
    }

    pub fn reload_from_disk(&mut self) -> Result<()> {
        self.manifest = SkyboxManifest::read_or_create(&self.manifest_path)?;
        self.last_manifest_modified = modified_at(&self.manifest_path);
        self.handles
            .retain(|name, _| self.manifest.get(name).is_some());
        Ok(())
    }

    pub fn entry(&self, name: &str) -> Result<&ManifestEntry> {
        self.manifest
            .get(name)
            .ok_or_else(|| SkyboxError::MissingSkybox(name.to_string()))
    }

    pub fn manifest_changed_on_disk(&self) -> bool {
        modified_at(&self.manifest_path) != self.last_manifest_modified
    }
}

fn modified_at(path: &Path) -> Option<SystemTime> {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

#[derive(Clone, Debug, Message)]
pub enum SetActiveSkybox {
    ByName(String),
    ByHandle(Handle<Image>),
}

#[derive(Clone, Debug, Message)]
pub struct SkyboxReady {
    pub name: String,
    pub handle: Handle<Image>,
    pub disk_bytes: u64,
}

#[derive(Debug, Message)]
pub struct SkyboxFailed {
    pub name: String,
    pub error: SkyboxError,
}

#[derive(Clone, Debug, Message)]
pub struct ManifestReloaded {
    pub entry_count: usize,
}

#[derive(Component, Clone, Debug, Default)]
pub struct PrimarySkybox;

#[derive(Clone, Debug)]
pub struct SkyboxAssetPlugin {
    pub cache_dir: PathBuf,
    pub asset_dir: PathBuf,
    pub manifest_file: PathBuf,
    pub default_skybox: Option<String>,
    pub apply_to_all_cameras: bool,
    pub env_lighting: bool,
}

impl Default for SkyboxAssetPlugin {
    fn default() -> Self {
        let settings = SkyboxAssetSettings::default();
        Self {
            cache_dir: settings.cache_dir,
            asset_dir: settings.asset_dir,
            manifest_file: settings.manifest_file,
            default_skybox: settings.default_skybox,
            apply_to_all_cameras: settings.apply_to_all_cameras,
            env_lighting: settings.env_lighting,
        }
    }
}

impl Plugin for SkyboxAssetPlugin {
    fn build(&self, app: &mut App) {
        let settings = SkyboxAssetSettings {
            cache_dir: if self.cache_dir.as_os_str().is_empty() {
                SkyboxAssetSettings::default().cache_dir
            } else {
                self.cache_dir.clone()
            },
            asset_dir: if self.asset_dir.as_os_str().is_empty() {
                SkyboxAssetSettings::default().asset_dir
            } else {
                self.asset_dir.clone()
            },
            manifest_file: if self.manifest_file.as_os_str().is_empty() {
                SkyboxAssetSettings::default().manifest_file
            } else {
                self.manifest_file.clone()
            },
            default_skybox: self.default_skybox.clone(),
            apply_to_all_cameras: self.apply_to_all_cameras,
            env_lighting: self.env_lighting,
        };

        let cache = match SkyboxCache::load(&settings) {
            Ok(cache) => cache,
            Err(error) => {
                warn!("failed to initialize skybox cache: {error}");
                SkyboxCache::empty(settings.manifest_path())
            }
        };

        app.insert_resource(settings)
            .insert_resource(cache)
            .add_message::<SetActiveSkybox>()
            .add_message::<SkyboxReady>()
            .add_message::<SkyboxFailed>()
            .add_message::<ManifestReloaded>()
            .add_systems(Startup, load_default_skybox)
            .add_systems(
                Update,
                (
                    apply_skybox_to_camera,
                    apply_active_skybox_to_new_cameras,
                    configure_loaded_cubemaps,
                    watch_manifest_changes,
                    observe_image_hot_reload,
                    log_skybox_outcomes,
                ),
            );
    }
}

fn load_default_skybox(
    cache: Res<SkyboxCache>,
    settings: Res<SkyboxAssetSettings>,
    mut writer: MessageWriter<SetActiveSkybox>,
) {
    let Some(name) = settings
        .default_skybox
        .clone()
        .or_else(|| cache.manifest.default.clone())
    else {
        return;
    };
    writer.write(SetActiveSkybox::ByName(name));
}

fn apply_skybox_to_camera(
    mut commands: Commands,
    mut cache: ResMut<SkyboxCache>,
    mut reader: MessageReader<SetActiveSkybox>,
    settings: Res<SkyboxAssetSettings>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut cameras: Query<(Entity, Option<&PrimarySkybox>, Option<&Skybox>), With<Camera3d>>,
    mut ready: MessageWriter<SkyboxReady>,
    mut failed: MessageWriter<SkyboxFailed>,
) {
    for message in reader.read() {
        let (name, handle, disk_bytes) = match message {
            SetActiveSkybox::ByName(name) => {
                let entry = match cache.entry(name) {
                    Ok(entry) => entry.clone(),
                    Err(error) => {
                        warn!("{error}");
                        failed.write(SkyboxFailed {
                            name: name.clone(),
                            error,
                        });
                        continue;
                    }
                };
                let cubemap_file = entry.cubemap_file.clone();
                let handle = cache
                    .handles
                    .entry(name.clone())
                    .or_insert_with(|| asset_server.load(settings.asset_path_for(&cubemap_file)))
                    .clone();
                cache.active = Some(name.clone());
                let disk_bytes = fs::metadata(settings.cache_dir.join(&entry.cubemap_file))
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);
                (Some(name.clone()), handle, disk_bytes)
            }
            SetActiveSkybox::ByHandle(handle) => (None, handle.clone(), 0),
        };

        configure_cubemap_image(&handle, &mut images);
        for (entity, primary, skybox) in &mut cameras {
            if !settings.apply_to_all_cameras && primary.is_none() {
                continue;
            }
            let brightness = skybox.map(|s| s.brightness).unwrap_or(1000.0);
            let mut entity_commands = commands.entity(entity);
            entity_commands.insert(Skybox {
                image: handle.clone(),
                brightness,
                ..default()
            });
            if settings.env_lighting {
                entity_commands.remove::<EnvironmentMapLight>();
                entity_commands.insert(GeneratedEnvironmentMapLight {
                    environment_map: handle.clone(),
                    intensity: 250.0,
                    ..default()
                });
            } else {
                entity_commands.remove::<GeneratedEnvironmentMapLight>();
            }
        }

        if let Some(name) = name {
            debug!("active skybox: {name}");
            ready.write(SkyboxReady {
                name,
                handle,
                disk_bytes,
            });
        }
    }
}

fn apply_active_skybox_to_new_cameras(
    cache: Res<SkyboxCache>,
    settings: Res<SkyboxAssetSettings>,
    cameras: Query<(Option<&PrimarySkybox>, Option<&Skybox>), With<Camera3d>>,
    mut writer: MessageWriter<SetActiveSkybox>,
) {
    let Some(active) = cache.active.as_ref() else {
        return;
    };

    let needs_skybox = cameras.iter().any(|(primary, skybox)| {
        skybox.is_none() && (settings.apply_to_all_cameras || primary.is_some())
    });

    if needs_skybox {
        writer.write(SetActiveSkybox::ByName(active.clone()));
    }
}

fn configure_loaded_cubemaps(cache: Res<SkyboxCache>, mut images: ResMut<Assets<Image>>) {
    for handle in cache.handles.values() {
        configure_cubemap_image(handle, &mut images);
    }
}

fn configure_cubemap_image(handle: &Handle<Image>, images: &mut Assets<Image>) {
    let Some(image) = images.get(handle) else {
        return;
    };
    if image.texture_descriptor.array_layer_count() != 1 || image.width() == 0 {
        return;
    }

    let layers = image.height() / image.width();
    if layers != 6 {
        return;
    }

    let Some(image) = images.get_mut(handle) else {
        return;
    };
    image.reinterpret_stacked_2d_as_array(layers);
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
}

fn watch_manifest_changes(
    mut timer: Local<Option<Timer>>,
    time: Res<Time>,
    mut cache: ResMut<SkyboxCache>,
    mut writer: MessageWriter<ManifestReloaded>,
) {
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(1.0, TimerMode::Repeating));
    timer.tick(time.delta());
    if !timer.just_finished() || !cache.manifest_changed_on_disk() {
        return;
    }

    match cache.reload_from_disk() {
        Ok(()) => {
            writer.write(ManifestReloaded {
                entry_count: cache.manifest.entries.len(),
            });
        }
        Err(error) => warn!("failed to reload skybox manifest: {error}"),
    }
}

fn observe_image_hot_reload(
    mut events: MessageReader<AssetEvent<Image>>,
    cache: Res<SkyboxCache>,
    mut writer: MessageWriter<SetActiveSkybox>,
) {
    for event in events.read() {
        let AssetEvent::Modified { id } = event else {
            continue;
        };
        let Some((name, _)) = cache.handles.iter().find(|(_, handle)| handle.id() == *id) else {
            continue;
        };
        if cache.active.as_deref() == Some(name) {
            writer.write(SetActiveSkybox::ByName(name.clone()));
        }
    }
}

fn log_skybox_outcomes(
    mut ready: MessageReader<SkyboxReady>,
    mut failed: MessageReader<SkyboxFailed>,
) {
    for event in ready.read() {
        info!(
            target: "bevy_ai_skybox",
            "skybox ready: name={:?} disk_bytes={}",
            event.name, event.disk_bytes
        );
    }
    for event in failed.read() {
        error!(
            target: "bevy_ai_skybox",
            "skybox failed for {:?}: {}",
            event.name, event.error
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundled_manifest() -> SkyboxManifest {
        ron::from_str(include_str!("../../../assets/skyboxes/manifest.ron")).unwrap()
    }

    #[test]
    fn bundled_manifest_parses() {
        let manifest = bundled_manifest();

        assert_eq!(manifest.version, 2);
        assert!(manifest.get("mojave_desert").is_some());
        assert!(manifest.get("alien_swamp").is_some());
    }

    #[test]
    fn rc_jet_uses_bundled_skybox() {
        let manifest = bundled_manifest();
        let rc_jet = include_str!("../../../examples/rc-jet/main.py");
        let skybox_name = rc_jet
            .lines()
            .map(str::trim)
            .find_map(|line| {
                line.strip_prefix("skybox name=\"")
                    .and_then(|name| name.strip_suffix('"'))
            })
            .expect("rc-jet schematic should request a skybox");

        assert_eq!(skybox_name, "alien_swamp");
        assert!(manifest.get(skybox_name).is_some());
    }
}
