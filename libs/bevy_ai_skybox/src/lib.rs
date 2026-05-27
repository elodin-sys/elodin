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
    asset::LoadState,
    core_pipeline::Skybox,
    image::Image,
    light::{EnvironmentMapLight, GeneratedEnvironmentMapLight},
    prelude::*,
    render::render_resource::{TextureViewDescriptor, TextureViewDimension},
    tasks::{IoTaskPool, Task, futures_lite::future},
};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};

pub mod cubemap_convert;

pub mod prelude {
    pub use crate::{
        BlockadeSkyboxPlugin, GenerateSkybox, ManifestReloaded, PrimarySkybox, SetActiveSkybox,
        SkyboxAssetPlugin, SkyboxAssetSettings, SkyboxCache, SkyboxFailed, SkyboxGenerated,
        SkyboxGenerationComplete, SkyboxGenerationPhase, SkyboxGenerationSettings,
        SkyboxGenerationUi, SkyboxManifest, SkyboxReady, SkyboxResolution, SkyboxStyle,
    };
}

pub type Result<T> = std::result::Result<T, SkyboxError>;

#[derive(Debug, thiserror::Error)]
pub enum SkyboxError {
    #[error("missing API key (set BLOCKADE_API_KEY)")]
    MissingApiKey,
    #[error("skybox `{0}` is not present in the manifest")]
    MissingSkybox(String),
    #[error("skybox `{0}` already exists")]
    DuplicateSkybox(String),
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
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    pub equirect_file: Option<String>,
    pub cubemap_file: String,
    pub face_size: u32,
    pub created_at: String,
}

fn deserialize_optional_string<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OptionalString {
        String(String),
        Option(Option<String>),
    }

    Ok(match OptionalString::deserialize(deserializer)? {
        OptionalString::String(value) => Some(value),
        OptionalString::Option(value) => value,
    })
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

    pub fn insert_new(&mut self, entry: ManifestEntry) -> Result<()> {
        if self.get(&entry.name).is_some() {
            return Err(SkyboxError::DuplicateSkybox(entry.name));
        }
        self.entries.push(entry);
        Ok(())
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
    pub manifest_poll_secs: f32,
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
            manifest_poll_secs: 1.0,
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

    pub fn insert_and_activate(&mut self, entry: ManifestEntry) -> Result<()> {
        let name = entry.name.clone();
        self.reload_from_disk()?;
        self.handles.remove(&name);
        self.manifest.insert_new(entry)?;
        self.write_manifest()?;
        self.active = Some(name);
        Ok(())
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

/// Emitted after a prompt-generated skybox is written to disk and registered.
#[derive(Clone, Debug, Message)]
pub struct SkyboxGenerated {
    pub name: String,
}

/// Emitted after a prompt-generated skybox is loaded and applied to cameras.
#[derive(Clone, Debug, Message)]
pub struct SkyboxGenerationComplete {
    pub name: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SkyboxGenerationPhase {
    #[default]
    Idle,
    Generating,
    PendingApply,
    Failed,
    Ready,
}

/// Editor-facing generation state for status UI and revert.
#[derive(Resource, Clone, Debug)]
pub struct SkyboxGenerationUi {
    pub phase: SkyboxGenerationPhase,
    pub prompt: Option<String>,
    pub message: Option<String>,
    /// Skybox name being generated or loaded onto cameras.
    pub target_name: Option<String>,
    /// Active skybox name before the in-flight generation; used for revert.
    pub revert_name: Option<String>,
}

impl Default for SkyboxGenerationUi {
    fn default() -> Self {
        Self {
            phase: SkyboxGenerationPhase::Idle,
            prompt: None,
            message: None,
            target_name: None,
            revert_name: None,
        }
    }
}

impl SkyboxGenerationUi {
    pub fn is_busy(&self) -> bool {
        matches!(
            self.phase,
            SkyboxGenerationPhase::Generating | SkyboxGenerationPhase::PendingApply
        )
    }
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

/// Reads `BLOCKADE_API_KEY` from the editor process environment (must be set before launch).
pub fn blockade_api_key_from_env() -> Option<String> {
    std::env::var("BLOCKADE_API_KEY")
        .ok()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
}

impl SkyboxGenerationSettings {
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key.clone().or_else(blockade_api_key_from_env)
    }
}

impl Default for SkyboxGenerationSettings {
    fn default() -> Self {
        Self {
            api_key: blockade_api_key_from_env(),
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
            api_key: self.api_key.clone().or_else(blockade_api_key_from_env),
            api_base_url: self.api_base_url.clone(),
            default_style: self.default_style,
            default_resolution: self.default_resolution,
            enhance_prompt: self.enhance_prompt,
            poll_interval: self.poll_interval,
            request_timeout: self.request_timeout,
        })
        .init_resource::<SkyboxGenerationState>()
        .init_resource::<SkyboxGenerationUi>()
        .add_message::<GenerateSkybox>()
        .add_systems(
            Update,
            (
                start_generation_jobs,
                finish_generation_jobs,
                track_skybox_generation_failures,
            )
                .chain(),
        );
    }
}

#[derive(Resource, Default)]
struct SkyboxGenerationState {
    active: Option<ActiveGeneration>,
}

#[derive(Resource, Default)]
struct PendingSkyboxActivation {
    name: Option<String>,
    notify_generation_complete: bool,
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
        ConfiguredCubemapIds, PendingSkyboxActivation, PrimarySkybox, SetActiveSkybox,
        SkyboxAssetSettings, SkyboxCache, SkyboxFailed, SkyboxGenerationComplete,
        SkyboxGenerationUi, SkyboxReady,
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
        pub pending: ResMut<'w, PendingSkyboxActivation>,
        pub reader: MessageReader<'w, 's, SetActiveSkybox>,
        pub settings: Res<'w, SkyboxAssetSettings>,
        pub asset_server: Res<'w, AssetServer>,
        pub images: ResMut<'w, Assets<Image>>,
        pub cameras: SkyboxCameraQuery<'w, 's>,
        pub ready: MessageWriter<'w, SkyboxReady>,
    }

    #[derive(SystemParam)]
    pub(super) struct ApplyPendingSkyboxParams<'w> {
        pub pending: ResMut<'w, PendingSkyboxActivation>,
        pub cache: ResMut<'w, SkyboxCache>,
        pub settings: Res<'w, SkyboxAssetSettings>,
        pub asset_server: Res<'w, AssetServer>,
        pub images: ResMut<'w, Assets<Image>>,
        pub configured: ResMut<'w, ConfiguredCubemapIds>,
        pub activate: MessageWriter<'w, SetActiveSkybox>,
        pub complete: MessageWriter<'w, SkyboxGenerationComplete>,
        pub failed: MessageWriter<'w, SkyboxFailed>,
        pub ui: Option<ResMut<'w, SkyboxGenerationUi>>,
    }
}

use system_params::{ApplyPendingSkyboxParams, ApplySkyboxParams};

#[derive(Clone, Debug)]
pub struct SkyboxAssetPlugin {
    pub cache_dir: PathBuf,
    pub asset_dir: PathBuf,
    pub manifest_file: PathBuf,
    pub default_skybox: Option<String>,
    pub apply_to_all_cameras: bool,
    pub env_lighting: bool,
    pub watch_manifest: bool,
    pub manifest_poll_secs: f32,
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
            manifest_poll_secs: settings.manifest_poll_secs,
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
            manifest_poll_secs: self.manifest_poll_secs,
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
            .init_resource::<PendingSkyboxActivation>()
            .add_message::<SetActiveSkybox>()
            .add_message::<SkyboxReady>()
            .add_message::<SkyboxFailed>()
            .add_message::<SkyboxGenerated>()
            .add_message::<SkyboxGenerationComplete>()
            .add_message::<ManifestReloaded>()
            .add_systems(Startup, load_default_skybox)
            .add_systems(
                Update,
                (
                    strip_unconfigured_skybox_components,
                    apply_pending_skybox_activation,
                    apply_skybox_to_camera,
                    apply_active_skybox_to_new_cameras,
                    reapply_skybox_after_manifest_reload,
                    configure_loaded_cubemaps,
                    watch_manifest_changes
                        .run_if(|settings: Res<SkyboxAssetSettings>| settings.watch_manifest),
                    log_skybox_outcomes,
                )
                    .chain(),
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
    mut settings: ResMut<SkyboxGenerationSettings>,
    asset_settings: Res<SkyboxAssetSettings>,
    mut state: ResMut<SkyboxGenerationState>,
    cache: Res<SkyboxCache>,
    mut ui: ResMut<SkyboxGenerationUi>,
    mut failed: MessageWriter<SkyboxFailed>,
) {
    if let Some(key) = settings.resolved_api_key() {
        settings.api_key = Some(key);
    }

    for request in reader.read() {
        if state.active.is_some() || ui.is_busy() {
            ui.message = Some("A skybox generation is already in progress".into());
            continue;
        }

        let mut request = request.clone();
        request.prompt = request.prompt.trim().to_string();
        if request.prompt.is_empty() {
            error!(
                target: "bevy_ai_skybox",
                "skybox generation rejected: prompt cannot be empty"
            );
            failed.write(SkyboxFailed {
                name: "empty prompt".into(),
                error: SkyboxError::GenerationFailed("prompt cannot be empty".into()),
            });
            continue;
        }

        if settings.resolved_api_key().is_none() {
            error!(
                target: "bevy_ai_skybox",
                prompt = %request.prompt,
                "skybox generation rejected: missing BLOCKADE_API_KEY"
            );
            failed.write(SkyboxFailed {
                name: request.prompt.clone(),
                error: SkyboxError::MissingApiKey,
            });
            continue;
        }

        ui.revert_name = cache.active.clone();
        ui.target_name = None;
        ui.phase = SkyboxGenerationPhase::Generating;
        ui.prompt = Some(request.prompt.clone());
        ui.message = Some(format!(
            "Generating skybox… ({})",
            request.prompt.chars().take(48).collect::<String>()
        ));

        let mut generation_settings = settings.clone();
        generation_settings.api_key = settings.resolved_api_key();
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
    settings: Res<SkyboxAssetSettings>,
    asset_server: Res<AssetServer>,
    mut pending: ResMut<PendingSkyboxActivation>,
    mut ui: ResMut<SkyboxGenerationUi>,
    mut generated_writer: MessageWriter<SkyboxGenerated>,
    mut manifest_reloaded: MessageWriter<ManifestReloaded>,
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
            let cubemap_file = generated.entry.cubemap_file.clone();
            match cache.insert_and_activate(generated.entry) {
                Ok(()) => {
                    let handle = asset_server.load(settings.asset_path_for(&cubemap_file));
                    cache.handles.insert(name.clone(), handle);
                    generated_writer.write(SkyboxGenerated { name: name.clone() });
                    manifest_reloaded.write(ManifestReloaded {
                        entry_count: cache.manifest.entries.len(),
                    });
                    pending.name = Some(name.clone());
                    pending.notify_generation_complete = true;
                    ui.target_name = Some(name.clone());
                    ui.phase = SkyboxGenerationPhase::PendingApply;
                    ui.message = Some(format!("Loading skybox `{name}`…"));
                }
                Err(error) => {
                    error!(
                        target: "bevy_ai_skybox",
                        prompt = %prompt,
                        "failed to register generated skybox: {error}"
                    );
                    failed.write(SkyboxFailed {
                        name: prompt,
                        error,
                    });
                }
            }
        }
        Err(error) => {
            error!(
                target: "bevy_ai_skybox",
                prompt = %prompt,
                "skybox generation failed: {error}"
            );
            failed.write(SkyboxFailed {
                name: prompt,
                error,
            });
        }
    }
}

fn strip_unconfigured_skybox_components(
    images: Res<Assets<Image>>,
    cameras: Query<(Entity, &Skybox), With<Camera3d>>,
    mut commands: Commands,
) {
    for (entity, skybox) in &cameras {
        if !is_cubemap_configured(&skybox.image, &images) {
            commands.entity(entity).remove::<Skybox>();
        }
    }
}

fn is_cubemap_configured(handle: &Handle<Image>, images: &Assets<Image>) -> bool {
    images.get(handle).is_some_and(|image| {
        image
            .texture_view_descriptor
            .as_ref()
            .is_some_and(|descriptor| descriptor.dimension == Some(TextureViewDimension::Cube))
    })
}

fn apply_pending_skybox_activation(mut params: ApplyPendingSkyboxParams) {
    let Some(name) = params.pending.name.clone() else {
        return;
    };

    let entry = match params.cache.entry(&name) {
        Ok(entry) => entry.clone(),
        Err(error) => {
            warn!("pending skybox `{name}` missing from manifest: {error}");
            params.pending.name = None;
            params.pending.notify_generation_complete = false;
            params.failed.write(SkyboxFailed {
                name: name.clone(),
                error,
            });
            if let Some(mut ui) = params.ui {
                ui.target_name = None;
                ui.phase = SkyboxGenerationPhase::Failed;
                ui.message = Some(format!("Skybox `{name}` failed to register"));
            }
            return;
        }
    };

    let handle = params
        .cache
        .handles
        .entry(name.clone())
        .or_insert_with(|| {
            params
                .asset_server
                .load(params.settings.asset_path_for(&entry.cubemap_file))
        })
        .clone();

    if !is_cubemap_ready(&handle, &mut params.images) {
        if let Some(error) = pending_cubemap_load_error(&params.asset_server, &handle) {
            warn!("pending skybox `{name}` failed to load: {error}");
            params.pending.name = None;
            params.pending.notify_generation_complete = false;
            params.failed.write(SkyboxFailed {
                name: name.clone(),
                error: SkyboxError::GenerationFailed(format!(
                    "failed to load generated cubemap `{name}`: {error}"
                )),
            });
            if let Some(mut ui) = params.ui {
                ui.target_name = None;
                ui.phase = SkyboxGenerationPhase::Failed;
                ui.message = Some(format!("Skybox `{name}` failed to load"));
            }
        }
        return;
    }
    params.configured.0.insert(handle.id());

    let notify_generation = params.pending.notify_generation_complete;
    params.pending.name = None;
    params.pending.notify_generation_complete = false;
    params.activate.write(SetActiveSkybox::ByName(name.clone()));
    if notify_generation {
        params
            .complete
            .write(SkyboxGenerationComplete { name: name.clone() });
        if let Some(mut ui) = params.ui {
            ui.target_name = None;
            ui.phase = SkyboxGenerationPhase::Ready;
            ui.message = Some(format!("Skybox ready: {name}"));
        }
    }
}

fn is_cubemap_ready(handle: &Handle<Image>, images: &mut Assets<Image>) -> bool {
    configure_cubemap_image(handle, images)
}

fn pending_cubemap_load_error(
    asset_server: &AssetServer,
    handle: &Handle<Image>,
) -> Option<String> {
    match asset_server.load_state(handle) {
        LoadState::Failed(error) => Some(error.to_string()),
        LoadState::Loaded => Some("asset is not a cube texture".to_string()),
        _ => None,
    }
}

fn track_skybox_generation_failures(
    mut reader: MessageReader<SkyboxFailed>,
    mut ui: ResMut<SkyboxGenerationUi>,
    mut state: ResMut<SkyboxGenerationState>,
    mut pending: ResMut<PendingSkyboxActivation>,
) {
    for event in reader.read() {
        error!(
            target: "bevy_ai_skybox",
            skybox = %event.name,
            "skybox failure: {}",
            event.error
        );
        if !matches!(
            ui.phase,
            SkyboxGenerationPhase::Generating | SkyboxGenerationPhase::PendingApply
        ) && state.active.is_none()
        {
            continue;
        }
        state.active = None;
        pending.name = None;
        ui.target_name = None;
        ui.phase = SkyboxGenerationPhase::Failed;
        ui.message = Some(format!("Skybox failed: {}", event.error));
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
                params.pending.name = None;
                params.pending.notify_generation_complete = false;
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
                        params.pending.name = Some(name.clone());
                        debug!("skybox `{name}` queued until manifest contains entry: {error}");
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
                if !is_cubemap_ready(&handle, &mut params.images) {
                    params.pending.name = Some(name.clone());
                    params.pending.notify_generation_complete = false;
                    debug!("skybox `{name}` queued until cubemap asset is ready");
                    continue;
                }
                params.cache.active = Some(name.clone());
                let disk_bytes = fs::metadata(params.settings.cache_dir.join(&entry.cubemap_file))
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);
                (Some(name.clone()), handle, disk_bytes)
            }
            SetActiveSkybox::ByHandle(handle) => {
                if !is_cubemap_ready(handle, &mut params.images) {
                    warn!("SetActiveSkybox::ByHandle skipped: cubemap not ready");
                    continue;
                }
                (None, handle.clone(), 0)
            }
        };

        if !is_cubemap_ready(&handle, &mut params.images) {
            continue;
        }
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

fn reapply_skybox_after_manifest_reload(
    mut reader: MessageReader<ManifestReloaded>,
    cache: Res<SkyboxCache>,
    pending: Res<PendingSkyboxActivation>,
    mut writer: MessageWriter<SetActiveSkybox>,
) {
    if reader.read().next().is_none() {
        return;
    }
    if let Some(name) = pending.name.clone() {
        writer.write(SetActiveSkybox::ByName(name));
        return;
    }
    if let Some(active) = cache.active.clone() {
        writer.write(SetActiveSkybox::ByName(active));
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
    if image.width() == 0 {
        return false;
    }

    let array_layers = image.texture_descriptor.array_layer_count();
    if array_layers == 6 {
        let Some(image) = images.get_mut(handle) else {
            return false;
        };
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
        return true;
    }

    if array_layers == 1 {
        let layers = image.height() / image.width();
        if layers == 6 {
            let Some(image) = images.get_mut(handle) else {
                return false;
            };
            let _ = image.reinterpret_stacked_2d_as_array(layers);
            image.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..default()
            });
            return true;
        }
    }

    false
}

fn watch_manifest_changes(
    settings: Res<SkyboxAssetSettings>,
    mut timer: Local<Option<Timer>>,
    time: Res<Time>,
    mut cache: ResMut<SkyboxCache>,
    mut writer: MessageWriter<ManifestReloaded>,
) {
    let poll_secs = settings.manifest_poll_secs.max(0.1);
    let timer = timer.get_or_insert_with(|| Timer::from_seconds(poll_secs, TimerMode::Repeating));
    if timer.duration().as_secs_f32() != poll_secs {
        *timer = Timer::from_seconds(poll_secs, TimerMode::Repeating);
    }
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
    let client = BlockadeClient::new(
        settings.api_key.clone(),
        settings.api_base_url.clone(),
        settings.request_timeout,
    )?;
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
    let manifest = SkyboxManifest::read_or_create(&asset_settings.manifest_path())?;
    let base_name = sanitize_name(
        request
            .save_as
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or(&request.prompt),
    );
    let name = unique_generated_skybox_name(&base_name, &asset_settings.cache_dir, &manifest);
    let cubemap_file = cubemap_convert::cubemap_ktx2_filename(&name);
    let cubemap_path = asset_settings.cache_dir.join(&cubemap_file);

    fs::create_dir_all(&asset_settings.cache_dir)?;

    let equirect = image::load_from_memory(&source_bytes)?.to_rgba8();
    let face_size = resolution.face_size();
    cubemap_convert::write_cubemap_ktx2(
        &equirect,
        face_size,
        &cubemap_path,
        cubemap_convert::resolve_toktx_executable(),
    )
    .map_err(SkyboxError::GenerationFailed)?;

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
            equirect_file: None,
            cubemap_file,
            face_size,
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
    fn new(
        api_key: Option<String>,
        base_url: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var("BLOCKADE_API_KEY").ok())
            .ok_or(SkyboxError::MissingApiKey)?;

        let connect_timeout = timeout.min(Duration::from_secs(15));
        Ok(Self {
            http: reqwest::blocking::Client::builder()
                .timeout(timeout)
                .connect_timeout(connect_timeout)
                .build()?,
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

const MAX_SKYBOX_NAME_CHARS: usize = 64;

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
        name.chars().take(MAX_SKYBOX_NAME_CHARS).collect()
    }
}

fn unique_generated_skybox_name(
    base_name: &str,
    cache_dir: &Path,
    manifest: &SkyboxManifest,
) -> String {
    cubemap_convert::remove_legacy_png_assets(cache_dir, base_name);
    if !skybox_name_conflicts(base_name, cache_dir, manifest) {
        return base_name.to_string();
    }

    for suffix in 2.. {
        let candidate = skybox_name_with_suffix(base_name, suffix);
        cubemap_convert::remove_legacy_png_assets(cache_dir, &candidate);
        if !skybox_name_conflicts(&candidate, cache_dir, manifest) {
            return candidate;
        }
    }
    unreachable!("unbounded suffix search should always find a unique skybox name")
}

fn skybox_name_with_suffix(base_name: &str, suffix: u32) -> String {
    let suffix = format!("_{suffix}");
    let prefix_len = MAX_SKYBOX_NAME_CHARS.saturating_sub(suffix.chars().count());
    let mut name = base_name.chars().take(prefix_len).collect::<String>();
    name.push_str(&suffix);
    name
}

fn skybox_name_conflicts(name: &str, cache_dir: &Path, manifest: &SkyboxManifest) -> bool {
    manifest.get(name).is_some()
        || cache_dir
            .join(cubemap_convert::cubemap_ktx2_filename(name))
            .exists()
        || cache_dir
            .join(cubemap_convert::equirect_manifest_filename(name))
            .exists()
        || cache_dir.join(format!("{name}.cubemap.png")).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Excerpt of `examples/rc-jet/main.py` schematic (tabs layout only).
    const RC_JET_SCHEMATIC_FRAGMENT: &str = r#"
        tabs {
            hsplit name="Main View" {
                viewport name=Viewport pos="bdx.world_pos.translate_world(-8.0,-8.0,4.0)" look_at="bdx.world_pos" show_grid=#true show_frustums=#true active=#true
                vsplit share=0.4 {
                    vsplit {
                        graph "bdx.alpha" name="Angle of Attack (rad)"
                        viewport name=TGTViewport pos="target.world_pos.translate_world(1,1,0.2)" look_at="bdx.world_pos" show_grid=#true
                        hsplit {
                            viewport name=FPVViewport pos="bdx.world_pos.rotate_z(-90).translate_y(-2.0)" show_grid=#true
                            sensor_view "bdx.fpv_cam" name="FPV (sensor_camera)"
                        }
                    }
                }
            }
        }
    "#;

    const SCHEMATIC_WITH_SKYBOX_FRAGMENT: &str = r#"
        skybox name="mojave_desert"

        tabs {
            viewport name=Viewport active=#true
        }
    "#;

    fn bundled_manifest() -> SkyboxManifest {
        ron::from_str(include_str!("../../../assets/skyboxes/manifest.ron")).unwrap()
    }

    fn parse_skybox_name_from_kdl(kdl: &str) -> Option<String> {
        kdl.lines().map(str::trim).find_map(|line| {
            let name = line.strip_prefix("skybox name=\"")?;
            name.strip_suffix('"').map(str::to_string)
        })
    }

    fn schematic_declares_skybox(kdl: &str) -> bool {
        parse_skybox_name_from_kdl(kdl).is_some()
    }

    #[test]
    fn bundled_manifest_parses() {
        let manifest = bundled_manifest();
        assert_eq!(manifest.version, 2);
        assert_eq!(manifest.default.as_deref(), Some("mojave_desert"));
    }

    #[test]
    fn bundled_manifest_lists_shipped_skyboxes() {
        let manifest = bundled_manifest();
        for name in [
            "mojave_desert",
            "alien_swamp",
            "beach_sunset",
            "seaport",
            "coastal_beach",
            "grand_canyon",
        ] {
            let entry = manifest
                .get(name)
                .unwrap_or_else(|| panic!("missing bundled skybox `{name}`"));
            assert!(
                !entry.cubemap_file.is_empty(),
                "{name} must reference a cubemap asset"
            );
            assert!(
                entry.equirect_file.is_none(),
                "{name} must not reference an equirect source that is not shipped"
            );
            assert!(
                entry.cubemap_file.ends_with(".cubemap.ktx2"),
                "{name} must ship as KTX2 cubemap"
            );
            assert_eq!(
                entry.face_size,
                cubemap_convert::BUNDLED_CUBEMAP_FACE_SIZE,
                "{name} must use bundled face size"
            );
        }
    }

    #[test]
    fn bundled_manifest_entry_asset_names_are_unique() {
        let manifest = bundled_manifest();
        let cubemap_files: HashSet<_> = manifest
            .entries
            .iter()
            .map(|entry| entry.cubemap_file.as_str())
            .collect();
        assert_eq!(cubemap_files.len(), manifest.entries.len());
    }

    #[test]
    fn manifest_upsert_replaces_by_name() {
        let mut manifest = SkyboxManifest::default();
        manifest.upsert(ManifestEntry {
            name: "test_sky".into(),
            prompt: "first".into(),
            style: SkyboxStyle::default(),
            blockade: None,
            equirect_file: Some("a.equirect.png".into()),
            cubemap_file: "a.cubemap.ktx2".into(),
            face_size: 512,
            created_at: "2026-01-01T00:00:00Z".into(),
        });
        manifest.upsert(ManifestEntry {
            name: "test_sky".into(),
            prompt: "second".into(),
            style: SkyboxStyle::default(),
            blockade: None,
            equirect_file: Some("b.equirect.png".into()),
            cubemap_file: "b.cubemap.ktx2".into(),
            face_size: 1024,
            created_at: "2026-01-02T00:00:00Z".into(),
        });
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.get("test_sky").unwrap().prompt, "second");
        assert_eq!(manifest.get("test_sky").unwrap().face_size, 1024);
    }

    #[test]
    fn manifest_roundtrip_write_read() {
        let dir = std::env::temp_dir().join(format!("bevy_ai_skybox_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.ron");

        let original = bundled_manifest();
        original.write_atomic(&path).unwrap();
        let loaded = SkyboxManifest::read_or_create(&path).unwrap();
        assert_eq!(loaded.version, original.version);
        assert_eq!(loaded.default, original.default);
        assert_eq!(loaded.entries.len(), original.entries.len());
        assert_eq!(
            loaded.get("seaport").unwrap().cubemap_file,
            "seaport.cubemap.ktx2"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn insert_new_rejects_duplicate_names() {
        let mut manifest = SkyboxManifest::default();
        manifest
            .insert_new(ManifestEntry {
                name: "test_sky".into(),
                prompt: "first".into(),
                style: SkyboxStyle::default(),
                blockade: None,
                equirect_file: Some("a.equirect.png".into()),
                cubemap_file: "a.cubemap.ktx2".into(),
                face_size: 512,
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();

        let error = manifest
            .insert_new(ManifestEntry {
                name: "test_sky".into(),
                prompt: "second".into(),
                style: SkyboxStyle::default(),
                blockade: None,
                equirect_file: Some("b.equirect.png".into()),
                cubemap_file: "b.cubemap.ktx2".into(),
                face_size: 1024,
                created_at: "2026-01-02T00:00:00Z".into(),
            })
            .unwrap_err();
        assert!(matches!(error, SkyboxError::DuplicateSkybox(name) if name == "test_sky"));
    }

    #[test]
    fn generated_names_avoid_manifest_and_asset_collisions() {
        let dir = std::env::temp_dir().join(format!(
            "bevy_ai_skybox_collision_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("grand_canyon_2.cubemap.ktx2"), b"existing").unwrap();

        let mut manifest = SkyboxManifest::default();
        manifest
            .insert_new(ManifestEntry {
                name: "grand_canyon".into(),
                prompt: "grand canyon".into(),
                style: SkyboxStyle::default(),
                blockade: None,
                equirect_file: None,
                cubemap_file: "grand_canyon.cubemap.ktx2".into(),
                face_size: 2048,
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();

        assert_eq!(
            unique_generated_skybox_name("grand_canyon", &dir, &manifest),
            "grand_canyon_3"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn generated_names_ignore_and_clean_legacy_png_collisions() {
        let dir = std::env::temp_dir().join(format!(
            "bevy_ai_skybox_legacy_collision_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let stale = dir.join("grand_canyon_2.equirect.png");
        fs::write(&stale, b"old").unwrap();

        let mut manifest = SkyboxManifest::default();
        manifest
            .insert_new(ManifestEntry {
                name: "grand_canyon".into(),
                prompt: "grand canyon".into(),
                style: SkyboxStyle::default(),
                blockade: None,
                equirect_file: None,
                cubemap_file: "grand_canyon.cubemap.ktx2".into(),
                face_size: 2048,
                created_at: "2026-01-01T00:00:00Z".into(),
            })
            .unwrap();

        assert_eq!(
            unique_generated_skybox_name("grand_canyon", &dir, &manifest),
            "grand_canyon_2"
        );
        assert!(!stale.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_skybox_name_from_kdl_fragment() {
        assert_eq!(
            parse_skybox_name_from_kdl(SCHEMATIC_WITH_SKYBOX_FRAGMENT).as_deref(),
            Some("mojave_desert")
        );
        assert_eq!(parse_skybox_name_from_kdl(RC_JET_SCHEMATIC_FRAGMENT), None);
    }

    #[test]
    fn rc_jet_schematic_fragment_has_no_embedded_skybox() {
        assert!(
            !schematic_declares_skybox(RC_JET_SCHEMATIC_FRAGMENT),
            "rc-jet layout fragment must not declare a skybox; use the command palette"
        );
    }

    #[test]
    fn rc_jet_example_file_matches_fragment_policy() {
        let rc_jet = include_str!("../../../examples/rc-jet/main.py");
        assert_eq!(
            schematic_declares_skybox(rc_jet),
            schematic_declares_skybox(RC_JET_SCHEMATIC_FRAGMENT),
            "full rc-jet example and the checked fragment should agree on skybox policy"
        );
    }

    #[test]
    fn skybox_resolution_face_sizes() {
        assert_eq!(SkyboxResolution::OneK.face_size(), 256);
        assert_eq!(SkyboxResolution::TwoK.face_size(), 512);
        assert_eq!(SkyboxResolution::FourK.face_size(), 1024);
    }

    #[test]
    fn generated_assets_use_ktx2_cubemap_only() {
        assert_eq!(
            cubemap_convert::cubemap_ktx2_filename("test_sky"),
            "test_sky.cubemap.ktx2"
        );
    }

    #[test]
    fn legacy_png_assets_are_removed() {
        let dir = std::env::temp_dir().join(format!(
            "bevy_ai_skybox_legacy_png_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let equirect = dir.join("test_sky.equirect.png");
        let cubemap_png = dir.join("test_sky.cubemap.png");
        let cubemap_ktx2 = dir.join("test_sky.cubemap.ktx2");
        fs::write(&equirect, b"old").unwrap();
        fs::write(&cubemap_png, b"old").unwrap();
        fs::write(&cubemap_ktx2, b"new").unwrap();

        cubemap_convert::remove_legacy_png_assets(&dir, "test_sky");

        assert!(!equirect.exists());
        assert!(!cubemap_png.exists());
        assert!(cubemap_ktx2.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn skybox_style_ids_match_blockade_styles() {
        assert_eq!(SkyboxStyle::M3Photoreal.id(), 67);
        assert_eq!(SkyboxStyle::M3UhdRender.id(), 35);
        assert_eq!(SkyboxStyle::M3Advanced.id(), 82);
        assert_eq!(SkyboxStyle::Custom(99).id(), 99);
    }
}
