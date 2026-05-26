//! Skybox loading and prompt-driven generation for Elodin.
//!
//! The lightweight asset plugin loads cached cubemap entries from
//! `assets/skyboxes/manifest.ron` and applies them to Bevy `Camera3d` entities.
//! The optional Blockade plugin adds prompt-driven generation.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant, SystemTime},
};

use bevy::{
    core_pipeline::Skybox,
    image::Image,
    light::{EnvironmentMapLight, GeneratedEnvironmentMapLight},
    prelude::*,
    render::render_resource::{TextureViewDescriptor, TextureViewDimension},
    tasks::{IoTaskPool, Task, futures_lite::future},
};
use chrono::{SecondsFormat, Utc};
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

pub mod prelude {
    pub use crate::{
        BlockadeSkyboxPlugin, GenerateSkybox, ManifestReloaded, PrimarySkybox, SetActiveSkybox,
        SkyboxAssetPlugin, SkyboxAssetSettings, SkyboxCache, SkyboxFailed,
        SkyboxGenerationSettings, SkyboxManifest, SkyboxReady, SkyboxResolution, SkyboxStyle,
    };
}

pub type Result<T> = std::result::Result<T, SkyboxError>;

#[derive(Debug, thiserror::Error)]
pub enum SkyboxError {
    #[error("missing API key (set BLOCKADE_API_KEY)")]
    MissingApiKey,
    #[error("skybox `{0}` is not present in the manifest")]
    MissingSkybox(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("generation timed out: {0}")]
    Timeout(String),
    #[error("unauthorized Blockade request: {0}")]
    Unauthorized(String),
    #[error("bad Blockade request: {0}")]
    BadRequest(String),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
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

impl SkyboxStyle {
    pub const fn id(self) -> u32 {
        match self {
            Self::M3Photoreal => 67,
            Self::M3UhdRender => 35,
            Self::M3Advanced => 82,
            Self::Custom(id) => id,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum SkyboxResolution {
    OneK,
    TwoK,
    #[default]
    FourK,
    EightK,
    SixteenK,
}

impl SkyboxResolution {
    pub const fn id(self) -> u32 {
        match self {
            Self::OneK => 1,
            Self::TwoK => 2,
            Self::FourK => 3,
            Self::EightK => 4,
            Self::SixteenK => 7,
        }
    }

    pub const fn face_size(self) -> u32 {
        match self {
            Self::OneK => 256,
            Self::TwoK => 512,
            Self::FourK => 1024,
            Self::EightK => 2048,
            Self::SixteenK => 4096,
        }
    }
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

    pub fn upsert(&mut self, entry: ManifestEntry) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|existing| existing.name == entry.name)
        {
            *existing = entry;
        } else {
            self.entries.push(entry);
        }
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
    /// Poll `manifest.ron` for changes (useful after AI generation).
    pub watch_manifest: bool,
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
            watch_manifest: true,
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
            active: None,
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

    pub fn write_manifest(&mut self) -> Result<()> {
        self.manifest.write_atomic(&self.manifest_path)?;
        self.last_manifest_modified = modified_at(&self.manifest_path);
        Ok(())
    }

    pub fn upsert_and_activate(&mut self, entry: ManifestEntry) -> Result<()> {
        self.active = Some(entry.name.clone());
        self.manifest.upsert(entry);
        self.write_manifest()
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
    Clear,
}

#[derive(Clone, Debug, Default, Message)]
pub struct GenerateSkybox {
    pub prompt: String,
    pub style: Option<SkyboxStyle>,
    pub negative_text: Option<String>,
    pub seed: Option<u64>,
    pub save_as: Option<String>,
    pub resolution: Option<SkyboxResolution>,
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

#[derive(Clone, Debug, Resource)]
pub struct SkyboxGenerationSettings {
    pub api_key: Option<String>,
    pub api_base_url: String,
    pub default_style: SkyboxStyle,
    pub default_resolution: SkyboxResolution,
    pub enhance_prompt: bool,
    pub poll_interval: Duration,
    pub request_timeout: Duration,
}

impl Default for SkyboxGenerationSettings {
    fn default() -> Self {
        Self {
            api_key: std::env::var("BLOCKADE_API_KEY").ok(),
            api_base_url: "https://backend.blockadelabs.com/api/v1".to_string(),
            default_style: SkyboxStyle::M3Photoreal,
            default_resolution: SkyboxResolution::FourK,
            enhance_prompt: true,
            poll_interval: Duration::from_secs(2),
            request_timeout: Duration::from_secs(180),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockadeSkyboxPlugin {
    pub api_key: Option<String>,
    pub api_base_url: String,
    pub default_style: SkyboxStyle,
    pub default_resolution: SkyboxResolution,
    pub enhance_prompt: bool,
    pub poll_interval: Duration,
    pub request_timeout: Duration,
}

impl Default for BlockadeSkyboxPlugin {
    fn default() -> Self {
        let settings = SkyboxGenerationSettings::default();
        Self {
            api_key: settings.api_key,
            api_base_url: settings.api_base_url,
            default_style: settings.default_style,
            default_resolution: settings.default_resolution,
            enhance_prompt: settings.enhance_prompt,
            poll_interval: settings.poll_interval,
            request_timeout: settings.request_timeout,
        }
    }
}

impl Plugin for BlockadeSkyboxPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SkyboxGenerationSettings {
            api_key: self
                .api_key
                .clone()
                .or_else(|| std::env::var("BLOCKADE_API_KEY").ok()),
            api_base_url: self.api_base_url.clone(),
            default_style: self.default_style,
            default_resolution: self.default_resolution,
            enhance_prompt: self.enhance_prompt,
            poll_interval: self.poll_interval,
            request_timeout: self.request_timeout,
        })
        .init_resource::<SkyboxGenerationState>()
        .add_message::<GenerateSkybox>()
        .add_systems(Update, (start_generation_jobs, finish_generation_jobs));
    }
}

#[derive(Resource, Default)]
struct SkyboxGenerationState {
    active: Option<ActiveGeneration>,
}

struct ActiveGeneration {
    prompt: String,
    task: Task<Result<GeneratedSkybox>>,
}

struct GeneratedSkybox {
    entry: ManifestEntry,
}

mod system_params {
    use bevy::{
        asset::{AssetServer, Assets},
        core_pipeline::Skybox,
        ecs::system::SystemParam,
        image::Image,
        prelude::*,
    };

    use super::{
        PrimarySkybox, SetActiveSkybox, SkyboxAssetSettings, SkyboxCache, SkyboxFailed, SkyboxReady,
    };

    type SkyboxCameraQuery<'w, 's> = Query<
        'w,
        's,
        (
            Entity,
            Option<&'static PrimarySkybox>,
            Option<&'static Skybox>,
        ),
        With<Camera3d>,
    >;

    #[derive(SystemParam)]
    pub(super) struct ApplySkyboxParams<'w, 's> {
        pub commands: Commands<'w, 's>,
        pub cache: ResMut<'w, SkyboxCache>,
        pub reader: MessageReader<'w, 's, SetActiveSkybox>,
        pub settings: Res<'w, SkyboxAssetSettings>,
        pub asset_server: Res<'w, AssetServer>,
        pub images: ResMut<'w, Assets<Image>>,
        pub cameras: SkyboxCameraQuery<'w, 's>,
        pub ready: MessageWriter<'w, SkyboxReady>,
        pub failed: MessageWriter<'w, SkyboxFailed>,
    }
}

use system_params::ApplySkyboxParams;

#[derive(Clone, Debug)]
pub struct SkyboxAssetPlugin {
    pub cache_dir: PathBuf,
    pub asset_dir: PathBuf,
    pub manifest_file: PathBuf,
    pub default_skybox: Option<String>,
    pub apply_to_all_cameras: bool,
    pub env_lighting: bool,
    pub watch_manifest: bool,
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
            watch_manifest: settings.watch_manifest,
        }
    }
}

#[derive(Resource, Default)]
struct ConfiguredCubemapIds(HashSet<AssetId<Image>>);

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
            watch_manifest: self.watch_manifest,
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
            .init_resource::<ConfiguredCubemapIds>()
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
                    watch_manifest_changes
                        .run_if(|settings: Res<SkyboxAssetSettings>| settings.watch_manifest),
                    log_skybox_outcomes,
                ),
            );
    }
}

fn load_default_skybox(
    settings: Res<SkyboxAssetSettings>,
    mut writer: MessageWriter<SetActiveSkybox>,
) {
    let Some(name) = settings.default_skybox.clone() else {
        return;
    };
    writer.write(SetActiveSkybox::ByName(name));
}

fn start_generation_jobs(
    mut reader: MessageReader<GenerateSkybox>,
    settings: Res<SkyboxGenerationSettings>,
    asset_settings: Res<SkyboxAssetSettings>,
    mut state: ResMut<SkyboxGenerationState>,
    mut failed: MessageWriter<SkyboxFailed>,
) {
    for request in reader.read() {
        let mut request = request.clone();
        request.prompt = request.prompt.trim().to_string();
        if request.prompt.is_empty() {
            failed.write(SkyboxFailed {
                name: "empty prompt".into(),
                error: SkyboxError::GenerationFailed("prompt cannot be empty".into()),
            });
            continue;
        }

        if settings.api_key.is_none() {
            failed.write(SkyboxFailed {
                name: request.prompt.clone(),
                error: SkyboxError::MissingApiKey,
            });
            continue;
        }

        let generation_settings = settings.clone();
        let asset_settings = asset_settings.clone();
        let prompt = request.prompt.clone();
        let task = IoTaskPool::get().spawn(async move {
            generate_skybox_blocking(generation_settings, asset_settings, request)
        });
        state.active = Some(ActiveGeneration { prompt, task });
    }
}

fn finish_generation_jobs(
    mut state: ResMut<SkyboxGenerationState>,
    mut cache: ResMut<SkyboxCache>,
    mut manifest_reloaded: MessageWriter<ManifestReloaded>,
    mut activate: MessageWriter<SetActiveSkybox>,
    mut failed: MessageWriter<SkyboxFailed>,
) {
    let Some(active) = state.active.as_mut() else {
        return;
    };
    let Some(result) = future::block_on(future::poll_once(&mut active.task)) else {
        return;
    };

    let prompt = active.prompt.clone();
    state.active = None;
    match result {
        Ok(generated) => {
            let name = generated.entry.name.clone();
            match cache.upsert_and_activate(generated.entry) {
                Ok(()) => {
                    manifest_reloaded.write(ManifestReloaded {
                        entry_count: cache.manifest.entries.len(),
                    });
                    activate.write(SetActiveSkybox::ByName(name));
                }
                Err(error) => {
                    failed.write(SkyboxFailed {
                        name: prompt,
                        error,
                    });
                }
            }
        }
        Err(error) => {
            failed.write(SkyboxFailed {
                name: prompt,
                error,
            });
        }
    }
}

fn camera_targets_skybox(
    settings: &SkyboxAssetSettings,
    primary: Option<&PrimarySkybox>,
    skybox: Option<&Skybox>,
    handle: &Handle<Image>,
) -> bool {
    if !settings.apply_to_all_cameras && primary.is_none() {
        return false;
    }
    skybox.is_none() || skybox.is_some_and(|skybox| skybox.image != *handle)
}

fn apply_skybox_to_camera(mut params: ApplySkyboxParams) {
    for message in params.reader.read() {
        let (name, handle, disk_bytes) = match message {
            SetActiveSkybox::Clear => {
                if params.cache.active.is_none()
                    && !params.cameras.iter().any(|(_, primary, skybox)| {
                        (params.settings.apply_to_all_cameras || primary.is_some())
                            && skybox.is_some()
                    })
                {
                    continue;
                }
                params.cache.active = None;
                for (entity, primary, skybox) in &mut params.cameras {
                    if !params.settings.apply_to_all_cameras && primary.is_none() {
                        continue;
                    }
                    if skybox.is_none() {
                        continue;
                    }
                    params
                        .commands
                        .entity(entity)
                        .remove::<Skybox>()
                        .remove::<GeneratedEnvironmentMapLight>();
                }
                debug!("skybox cleared");
                continue;
            }
            SetActiveSkybox::ByName(name) => {
                let entry = match params.cache.entry(name) {
                    Ok(entry) => entry.clone(),
                    Err(error) => {
                        warn!("{error}");
                        params.failed.write(SkyboxFailed {
                            name: name.clone(),
                            error,
                        });
                        continue;
                    }
                };
                let cubemap_file = entry.cubemap_file.clone();
                let handle = params
                    .cache
                    .handles
                    .entry(name.clone())
                    .or_insert_with(|| {
                        params
                            .asset_server
                            .load(params.settings.asset_path_for(&cubemap_file))
                    })
                    .clone();
                if params.cache.active.as_deref() == Some(name.as_str())
                    && !params.cameras.iter().any(|(_, primary, skybox)| {
                        camera_targets_skybox(&params.settings, primary, skybox, &handle)
                    })
                {
                    continue;
                }
                params.cache.active = Some(name.clone());
                let disk_bytes = fs::metadata(params.settings.cache_dir.join(&entry.cubemap_file))
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);
                (Some(name.clone()), handle, disk_bytes)
            }
            SetActiveSkybox::ByHandle(handle) => (None, handle.clone(), 0),
        };

        configure_cubemap_image(&handle, &mut params.images);
        let mut updated_any = false;
        for (entity, primary, skybox) in &mut params.cameras {
            if !camera_targets_skybox(&params.settings, primary, skybox, &handle) {
                continue;
            }
            updated_any = true;
            let brightness = skybox.map(|s| s.brightness).unwrap_or(1000.0);
            let mut entity_commands = params.commands.entity(entity);
            if skybox.is_some() {
                entity_commands.remove::<Skybox>();
            }
            entity_commands.insert(Skybox {
                image: handle.clone(),
                brightness,
                ..default()
            });
            if params.settings.env_lighting {
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

        if updated_any && let Some(name) = name {
            debug!("active skybox: {name}");
            params.ready.write(SkyboxReady {
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

fn configure_loaded_cubemaps(
    mut configured: ResMut<ConfiguredCubemapIds>,
    cache: Res<SkyboxCache>,
    mut images: ResMut<Assets<Image>>,
) {
    for handle in cache.handles.values() {
        if configured.0.contains(&handle.id()) {
            continue;
        }
        if configure_cubemap_image(handle, &mut images) {
            configured.0.insert(handle.id());
        }
    }
}

fn configure_cubemap_image(handle: &Handle<Image>, images: &mut Assets<Image>) -> bool {
    let Some(image) = images.get(handle) else {
        return false;
    };
    if image
        .texture_view_descriptor
        .as_ref()
        .is_some_and(|descriptor| descriptor.dimension == Some(TextureViewDimension::Cube))
    {
        return true;
    }
    if image.texture_descriptor.array_layer_count() != 1 || image.width() == 0 {
        return false;
    }

    let layers = image.height() / image.width();
    if layers != 6 {
        return false;
    }

    let Some(image) = images.get_mut(handle) else {
        return false;
    };
    let _ = image.reinterpret_stacked_2d_as_array(layers);
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    true
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

fn generate_skybox_blocking(
    settings: SkyboxGenerationSettings,
    asset_settings: SkyboxAssetSettings,
    request: GenerateSkybox,
) -> Result<GeneratedSkybox> {
    let client = BlockadeClient::new(settings.api_key.clone(), settings.api_base_url.clone())?;
    let style = request.style.unwrap_or(settings.default_style);
    let resolution = request.resolution.unwrap_or(settings.default_resolution);
    let created = client.submit(&GenerateSkyboxRequest {
        prompt: request.prompt.clone(),
        skybox_style_id: style.id(),
        negative_text: request.negative_text.clone(),
        enhance_prompt: Some(settings.enhance_prompt),
        seed: request.seed,
    })?;

    let started = Instant::now();
    let status = loop {
        if started.elapsed() > settings.request_timeout {
            return Err(SkyboxError::Timeout(format!(
                "request {} exceeded {:?}",
                created.id, settings.request_timeout
            )));
        }

        let status = client.poll(created.id)?;
        if status.is_complete() {
            break status;
        }
        if status.is_error() {
            return Err(SkyboxError::GenerationFailed(
                status
                    .error_message
                    .unwrap_or_else(|| format!("remote status `{}`", status.status)),
            ));
        }
        thread::sleep(settings.poll_interval);
    };

    let Some(file_url) = status.file_url else {
        return Err(SkyboxError::GenerationFailed(
            "complete status did not include file_url".into(),
        ));
    };
    let source_bytes = client.download(&file_url)?;
    let name = sanitize_name(
        request
            .save_as
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(&request.prompt),
    );
    let equirect_file = format!("{name}.equirect.png");
    let cubemap_file = format!("{name}.cubemap.png");
    let equirect_path = asset_settings.cache_dir.join(&equirect_file);
    let cubemap_path = asset_settings.cache_dir.join(&cubemap_file);

    fs::create_dir_all(&asset_settings.cache_dir)?;
    let equirect = image::load_from_memory(&source_bytes)?.to_rgba8();
    equirect.save(&equirect_path)?;
    equirect_to_stacked_cubemap(&equirect, resolution.face_size()).save(&cubemap_path)?;

    Ok(GeneratedSkybox {
        entry: ManifestEntry {
            name,
            prompt: request.prompt,
            style,
            blockade: Some(BlockadeMetadata {
                imagine_id: status.id,
                obfuscated_id: status.obfuscated_id.or(created.obfuscated_id),
                seed: request.seed,
            }),
            equirect_file,
            cubemap_file,
            face_size: resolution.face_size(),
            created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true),
        },
    })
}

#[derive(Clone, Debug, Serialize)]
struct GenerateSkyboxRequest {
    prompt: String,
    skybox_style_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    negative_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enhance_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct ImagineCreated {
    id: u64,
    #[serde(default)]
    obfuscated_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ImagineStatus {
    id: u64,
    #[serde(default)]
    obfuscated_id: Option<String>,
    status: String,
    #[serde(default)]
    file_url: Option<String>,
    #[serde(default)]
    error_message: Option<String>,
}

impl ImagineStatus {
    fn is_complete(&self) -> bool {
        self.status.eq_ignore_ascii_case("complete")
            && self
                .file_url
                .as_ref()
                .is_some_and(|url| !url.trim().is_empty())
    }

    fn is_error(&self) -> bool {
        matches!(self.status.as_str(), "error" | "abort" | "failed")
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PollEnvelope {
    Wrapped { request: ImagineStatus },
    Flat(ImagineStatus),
}

impl PollEnvelope {
    fn into_status(self) -> ImagineStatus {
        match self {
            Self::Wrapped { request } => request,
            Self::Flat(status) => status,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    error: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct MessageBody {
    #[serde(default)]
    success: Option<bool>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Clone, Debug)]
struct BlockadeClient {
    http: reqwest::blocking::Client,
    base_url: String,
    api_key: String,
}

impl BlockadeClient {
    fn new(api_key: Option<String>, base_url: impl Into<String>) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("BLOCKADE_API_KEY").ok())
            .ok_or(SkyboxError::MissingApiKey)?;

        Ok(Self {
            http: reqwest::blocking::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
        })
    }

    fn submit(&self, request: &GenerateSkyboxRequest) -> Result<ImagineCreated> {
        self.post_json("skybox", request)
    }

    fn poll(&self, id: u64) -> Result<ImagineStatus> {
        let envelope: PollEnvelope = self.get_json(&format!("imagine/requests/{id}"))?;
        Ok(envelope.into_status())
    }

    fn download(&self, url: &str) -> Result<Vec<u8>> {
        let response = self.http.get(url).send()?;
        let status = response.status();
        let bytes = response.bytes()?;
        if !status.is_success() {
            let body = String::from_utf8_lossy(&bytes);
            return Err(parse_blockade_error(status, &body));
        }
        Ok(bytes.to_vec())
    }

    fn post_json<T, B>(&self, path: &str, body: &B) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize + ?Sized,
    {
        let response = self
            .http
            .post(format!(
                "{}/{}",
                self.base_url,
                path.trim_start_matches('/')
            ))
            .header("x-api-key", &self.api_key)
            .json(body)
            .send()?;
        Self::parse_response(response)
    }

    fn get_json<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let response = self
            .http
            .get(format!(
                "{}/{}",
                self.base_url,
                path.trim_start_matches('/')
            ))
            .header("x-api-key", &self.api_key)
            .send()?;
        Self::parse_response(response)
    }

    fn parse_response<T>(response: reqwest::blocking::Response) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let status = response.status();
        let text = response.text()?;
        if !status.is_success() {
            return Err(parse_blockade_error(status, &text));
        }
        Ok(serde_json::from_str(&text)?)
    }
}

fn parse_blockade_error(status: reqwest::StatusCode, body: &str) -> SkyboxError {
    if status == reqwest::StatusCode::UNAUTHORIZED || body.contains("API key") {
        return SkyboxError::Unauthorized(body.trim().to_string());
    }

    if let Ok(parsed) = serde_json::from_str::<ErrorBody>(body) {
        let message = match parsed.error {
            serde_json::Value::String(value) => value,
            other => other.to_string(),
        };
        return SkyboxError::BadRequest(message);
    }

    if let Ok(parsed) = serde_json::from_str::<MessageBody>(body)
        && (parsed.success == Some(false) || parsed.message.is_some())
    {
        return SkyboxError::BadRequest(parsed.message.unwrap_or_else(|| body.to_string()));
    }

    SkyboxError::BadRequest(body.trim().to_string())
}

fn equirect_to_stacked_cubemap(source: &RgbaImage, face_size: u32) -> RgbaImage {
    let mut output = RgbaImage::new(face_size, face_size * 6);
    for face in 0..6 {
        for y in 0..face_size {
            for x in 0..face_size {
                let direction = cube_face_uv_to_direction(face, x, y, face_size);
                let (u, v) = direction_to_equirect_uv(direction);
                output.put_pixel(x, y + face * face_size, sample_equirect(source, u, v));
            }
        }
    }
    output
}

fn cube_face_uv_to_direction(face: u32, x: u32, y: u32, face_size: u32) -> Vec3 {
    let s = 2.0 * ((x as f32 + 0.5) / face_size as f32) - 1.0;
    let t = 2.0 * ((y as f32 + 0.5) / face_size as f32) - 1.0;
    match face {
        0 => Vec3::new(1.0, -t, -s).normalize(),
        1 => Vec3::new(-1.0, -t, s).normalize(),
        2 => Vec3::new(s, 1.0, t).normalize(),
        3 => Vec3::new(s, -1.0, -t).normalize(),
        4 => Vec3::new(s, -t, 1.0).normalize(),
        _ => Vec3::new(-s, -t, -1.0).normalize(),
    }
}

fn direction_to_equirect_uv(direction: Vec3) -> (f32, f32) {
    let direction = direction.normalize();
    let u = (0.5 + direction.z.atan2(direction.x) / std::f32::consts::TAU).fract();
    let v = (direction.y.acos() / std::f32::consts::PI).clamp(0.0, 1.0);
    (u, v)
}

fn sample_equirect(source: &RgbaImage, u: f32, v: f32) -> Rgba<u8> {
    let width = source.width();
    let height = source.height();
    let x = u * width as f32 - 0.5;
    let y = (v * (height.saturating_sub(1)) as f32).clamp(0.0, height.saturating_sub(1) as f32);
    let x_floor = x.floor();
    let y_floor = y.floor();
    let x0 = wrap_pixel_x(x_floor as i32, width);
    let x1 = wrap_pixel_x(x_floor as i32 + 1, width);
    let y0 = y_floor as u32;
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let tx = x - x_floor;
    let ty = y - y_floor;
    let p00 = source.get_pixel(x0, y0);
    let p10 = source.get_pixel(x1, y0);
    let p01 = source.get_pixel(x0, y1);
    let p11 = source.get_pixel(x1, y1);
    let mut rgba = [0u8; 4];
    for i in 0..4 {
        let top = lerp(p00[i] as f32, p10[i] as f32, tx);
        let bottom = lerp(p01[i] as f32, p11[i] as f32, tx);
        rgba[i] = lerp(top, bottom, ty).round().clamp(0.0, 255.0) as u8;
    }
    Rgba(rgba)
}

fn wrap_pixel_x(x: i32, width: u32) -> u32 {
    let width = width as i32;
    x.rem_euclid(width) as u32
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn sanitize_name(input: &str) -> String {
    let mut name: String = input
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    while name.contains("__") {
        name = name.replace("__", "_");
    }
    let name = name.trim_matches('_').to_string();
    if name.is_empty() {
        format!("skybox_{}", Utc::now().timestamp())
    } else {
        name.chars().take(64).collect()
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
