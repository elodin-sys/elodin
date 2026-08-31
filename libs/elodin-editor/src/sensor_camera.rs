use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};

use bevy::{
    app::{App, Plugin},
    asset::{Assets, RenderAssetUsages, embedded_asset},
    camera::{ClearColorConfig, Exposure, Hdr, RenderTarget, visibility::RenderLayers},
    core_pipeline::{
        Core3d, Core3dSystems, FullscreenShader,
        tonemapping::{Tonemapping, tonemapping},
    },
    image::Image,
    light::AmbientLight,
    math::{DVec3, Vec3},
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TextureDimension, TextureFormat,
            TextureUsages,
            binding_types::{sampler, texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        view::{ViewDepthTexture, ViewTarget},
    },
};
use bevy_ai_skybox::prelude::PrimarySkybox;
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, GeoRotation};
use impeller2::types::{ComponentId, Timestamp};
pub use impeller2_wkt::SensorCameraConfig;
use impeller2_wkt::{CurrentTimestamp, DbConfig, ThermalTagConfig};

use crate::object_3d::{ComponentArrayExt, ELLIPSOID_RENDER_LAYER, Object3DState};
use crate::plugins::render_layer_alloc::{CINEMATIC_EARTH_RENDER_LAYER, THERMAL_MASK_RENDER_LAYER};
use crate::plugins::scene_environment::CinematicViewport;
use crate::ui::tiles::bloom_from_config;

#[derive(Resource, Default, Debug, Clone)]
pub struct SensorCameraConfigs(pub Vec<SensorCameraConfig>);

#[derive(Resource, Default, Debug, Clone)]
pub struct ThermalTagConfigs(pub Vec<ThermalTagConfig>);

// ---------------------------------------------------------------------------
// ECS components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct SensorCamera {
    pub config_index: usize,
}

#[derive(Component)]
struct ThermalMaskCamera {
    config_index: usize,
}

#[derive(Component)]
struct ThermalMaskProxy;

#[derive(Component)]
pub struct SensorCameraFrustumSource {
    pub config_index: usize,
}

/// Complete sensor output model, extracted as one dynamic uniform per camera.
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct SensorOutputSettings {
    pub mode: UVec4,
    pub viewport: Vec4,
    pub view_rotation: Vec4,
    pub lens: Vec4,
    pub legacy: Vec4,
    pub thermal: Vec4,
    pub agc: Vec4,
    pub sensor: Vec4,
    pub range: Vec4,
}

#[derive(Component, Clone, ExtractComponent)]
struct SensorOutputTarget {
    output_image: Handle<Image>,
    palette_lut: Handle<Image>,
    thermal_mask: Handle<Image>,
    /// AGC statistics side channel (R = temperature, G = sky flag), written
    /// by the second MRT attachment of the sensor output pass.
    temp_map: Handle<Image>,
}

/// Double-buffered GPU readback state for a single sensor camera.
/// One buffer can be mapped for CPU read while the other receives the next frame.
/// The active buffer index is tracked in the render-world `BufferToggle` resource
/// so it persists across extract rebuilds.
#[derive(Clone, Component)]
struct ImageCopier {
    buffers: Vec<Buffer>,
    in_flight: Vec<Arc<AtomicBool>>,
    src_image: Handle<Image>,
    camera_name: String,
    width: u32,
    height: u32,
    is_active: bool,
}

/// Readback of the AGC temperature map for LWIR cameras. Same double-buffered
/// machinery as [`ImageCopier`]; its frames are consumed by `update_auto_agc`
/// and never pushed to the DB.
#[derive(Clone, Component)]
struct TempMapCopier(ImageCopier);

/// Suffix distinguishing temperature-map readbacks from DB camera frames in
/// the shared frame channel.
pub const TEMP_MAP_SUFFIX: &str = "#temp";

pub fn temp_map_msg_name(camera_name: &str) -> String {
    format!("{camera_name}{TEMP_MAP_SUFFIX}")
}

const READBACK_BUFFER_COUNT: usize = 4;
const READBACK_WORKER_COUNT: usize = 2;
const LWIR_AGC_READBACK_INTERVAL_US: i64 = 100_000;

#[derive(Clone, Default, Resource)]
struct ImageCopiers(pub Vec<ImageCopier>, pub Timestamp);

#[derive(Resource)]
pub struct SensorFrameReceiver(pub flume::Receiver<(String, Timestamp, Vec<u8>, u32, u32)>);

#[derive(Resource, Clone)]
struct SensorFrameSender(flume::Sender<(String, Timestamp, Vec<u8>, u32, u32)>);

#[derive(Clone, Default, Resource)]
pub struct SensorReadbackStatus(Arc<AtomicUsize>);

impl SensorReadbackStatus {
    pub fn pending(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }
}

struct PendingReadback(SensorReadbackStatus);

impl PendingReadback {
    fn new(status: SensorReadbackStatus) -> Self {
        status.0.fetch_add(1, Ordering::AcqRel);
        Self(status)
    }
}

impl Drop for PendingReadback {
    fn drop(&mut self) {
        self.0.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Render-world resource that persists the next readback-buffer index.
#[derive(Resource, Default)]
struct BufferToggle(Vec<usize>);

#[derive(Resource, Default)]
pub struct SensorCamerasSpawned(pub bool);

#[derive(Resource, Default)]
pub struct SensorCameraFrustumSourcesSpawned(pub bool);

/// Controls whether GPU readback is active for this sensor camera.
/// Separate from Camera.is_active: cameras stay rendering (pipeline warm),
/// but GPU readback only happens when this is true.
#[derive(Component, Default)]
pub struct ReadbackArmed(pub bool);

/// Marker resource set only in the headless render app. When present, readback is controlled
/// solely by ReadbackArmed (cameras stay active for pipeline warm). When absent (editor),
/// readback runs when ReadbackArmed or Camera.is_active is true.
#[derive(Resource, Default)]
pub struct HeadlessMode;

#[derive(Clone, Copy, Debug, Default, Resource)]
pub struct SensorCameraRenderMetrics {
    pub image_copy_driver_ms: f64,
    pub receive_image_poll_wait_ms: f64,
    pub receive_image_from_buffer_ms: f64,
}

// ---------------------------------------------------------------------------
// Post-process render pass
// ---------------------------------------------------------------------------

/// Final sensor-output pass shared by normal and cinematic cameras.
fn sensor_output_pass(
    view: ViewQuery<(
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static SensorOutputTarget,
        &'static SensorOutputSettings,
        &'static DynamicUniformIndex<SensorOutputSettings>,
    )>,
    pipeline_res: Res<SensorOutputPipeline>,
    pipeline_cache: Res<PipelineCache>,
    settings_uniforms: Res<ComponentUniforms<SensorOutputSettings>>,
    gpu_images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
    mut render_context: RenderContext,
) {
    let (view_target, depth, target, _settings, settings_index) = view.into_inner();
    let Some(output) = gpu_images.get(&target.output_image) else {
        return;
    };
    let Some(palette_lut) = gpu_images.get(&target.palette_lut) else {
        return;
    };
    let Some(thermal_mask) = gpu_images.get(&target.thermal_mask) else {
        return;
    };
    let Some(temp_map) = gpu_images.get(&target.temp_map) else {
        return;
    };

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_res.pipeline_id) else {
        return;
    };

    let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
        return;
    };

    let bind_group = render_context.render_device().create_bind_group(
        "sensor_output_bind_group",
        &pipeline_res.bind_group_layout,
        &BindGroupEntries::sequential((
            view_target.main_texture_view(),
            depth.view(),
            &pipeline_res.sampler,
            settings_binding.clone(),
            &palette_lut.texture_view,
            &thermal_mask.texture_view,
        )),
    );

    let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("sensor_output_pass"),
        color_attachments: &[
            Some(RenderPassColorAttachment {
                view: &output.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            }),
            Some(RenderPassColorAttachment {
                view: &temp_map.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            }),
        ],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        ..default()
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[settings_index.index()]);
    render_pass.draw(0..3, 0..1);
}

#[derive(Resource)]
struct SensorOutputPipeline {
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

fn init_sensor_output_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let layout_entries = BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_2d(TextureSampleType::Depth),
            sampler(SamplerBindingType::Filtering),
            uniform_buffer::<SensorOutputSettings>(true),
            texture_2d(TextureSampleType::Float { filterable: true }),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    );
    let layout_descriptor =
        BindGroupLayoutDescriptor::new("sensor_output_bind_group_layout", &layout_entries);
    let bind_group_layout =
        render_device.create_bind_group_layout("sensor_output_bind_group_layout", &layout_entries);

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let shader = asset_server.load("embedded://elodin_editor/assets/shaders/sensor_output.wgsl");
    let vertex_state = fullscreen_shader.to_vertex_state();

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sensor_output_pipeline".into()),
        layout: vec![layout_descriptor.clone()],
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader,
            targets: vec![
                Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                }),
                // Linear (not sRGB) so the temperature normalization written
                // by the shader survives readback unchanged.
                Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                }),
            ],
            ..default()
        }),
        ..default()
    });

    commands.insert_resource(SensorOutputPipeline {
        bind_group_layout,
        sampler,
        pipeline_id,
    });
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct SensorCameraPlugin;

impl Plugin for SensorCameraPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = flume::unbounded();
        let readback_status = SensorReadbackStatus::default();

        embedded_asset!(app, "assets/shaders/sensor_output.wgsl");

        app.init_resource::<SensorCameraConfigs>()
            .init_resource::<ThermalTagConfigs>()
            .init_resource::<SensorCamerasSpawned>()
            .insert_resource(readback_status.clone())
            .insert_resource(SensorFrameReceiver(rx))
            .add_plugins((
                ExtractComponentPlugin::<SensorOutputSettings>::default(),
                UniformComponentPlugin::<SensorOutputSettings>::default(),
                ExtractComponentPlugin::<SensorOutputTarget>::default(),
            ))
            // Headless path only; editor registers load_sensor_configs_from_db in lib.rs.
            .add_systems(
                PreUpdate,
                (load_sensor_configs_from_db, load_thermal_tags_from_db),
            )
            .add_systems(
                PreUpdate,
                spawn_sensor_cameras.run_if(should_spawn_sensor_cameras),
            )
            .add_systems(
                PreUpdate,
                update_sensor_camera_transforms.after(crate::PositionSync),
            )
            .add_systems(
                Update,
                (
                    update_sensor_camera_render_layers,
                    update_sensor_output_seed,
                    spawn_thermal_mask_proxies,
                ),
            );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(SensorFrameSender(tx))
                .insert_resource(readback_status)
                .init_resource::<BufferToggle>()
                .init_resource::<SensorCameraRenderMetrics>()
                .add_systems(ExtractSchedule, image_copy_extract)
                .add_systems(Render, image_copy_driver.after(RenderSystems::Render))
                .add_systems(RenderStartup, init_sensor_output_pipeline)
                .add_systems(
                    Core3d,
                    sensor_output_pass
                        .in_set(Core3dSystems::PostProcess)
                        .after(tonemapping),
                );
        }
    }
}

// ---------------------------------------------------------------------------
// Main-world systems
// ---------------------------------------------------------------------------

pub fn load_sensor_configs_from_db(
    db_config: Res<DbConfig>,
    mut configs: ResMut<SensorCameraConfigs>,
    spawned: Res<SensorCamerasSpawned>,
) {
    if spawned.0 || !configs.0.is_empty() {
        return;
    }

    if let Some(json) = db_config.metadata.get("sensor_cameras") {
        match serde_json::from_str::<Vec<SensorCameraConfig>>(json) {
            Ok(camera_configs) if !camera_configs.is_empty() => {
                bevy::log::debug!(
                    "Loaded {} sensor camera configs from DB metadata",
                    camera_configs.len()
                );
                configs.0 = camera_configs;
            }
            Ok(_) => {}
            Err(e) => {
                bevy::log::warn!("Failed to parse sensor_cameras from DB config: {e}");
            }
        }
    }
}

fn load_thermal_tags_from_db(db_config: Res<DbConfig>, mut tags: ResMut<ThermalTagConfigs>) {
    if !tags.0.is_empty() {
        return;
    }
    let Some(json) = db_config.metadata.get("thermal_tags") else {
        return;
    };
    match serde_json::from_str::<Vec<ThermalTagConfig>>(json) {
        Ok(loaded) => tags.0 = loaded,
        Err(error) => bevy::log::warn!("Failed to parse thermal_tags from DB config: {error}"),
    }
}

fn should_spawn_sensor_cameras(
    configs: Res<SensorCameraConfigs>,
    spawned: Res<SensorCamerasSpawned>,
) -> bool {
    !configs.0.is_empty() && !spawned.0
}

pub fn should_spawn_sensor_camera_frustum_sources(
    configs: Res<SensorCameraConfigs>,
    spawned: Res<SensorCameraFrustumSourcesSpawned>,
) -> bool {
    !configs.0.is_empty() && !spawned.0
}

pub fn spawn_sensor_camera_frustum_sources(
    mut commands: Commands,
    configs: Res<SensorCameraConfigs>,
    mut spawned: ResMut<SensorCameraFrustumSourcesSpawned>,
    #[cfg(feature = "big_space")] root: Option<Res<crate::spatial::BigSpaceRootEntity>>,
) {
    for (i, config) in configs.0.iter().enumerate() {
        let mut perspective = PerspectiveProjection {
            fov: config.fov_degrees.to_radians(),
            near: config.near,
            far: config.far,
            near_clip_plane: crate::plugins::frustum_common::near_clip_plane(config.near),
            ..default()
        };
        if config.height > 0 {
            perspective.aspect_ratio = config.width as f32 / config.height as f32;
        }

        let mut entity = commands.spawn((
            Transform::default(),
            GlobalTransform::default(),
            Projection::Perspective(perspective),
            #[cfg(feature = "big_space")]
            crate::spatial::GridCell::default(),
            SensorCameraFrustumSource { config_index: i },
            Name::new(format!("sensor_camera_frustum_{}", config.camera_name)),
        ));
        #[cfg(feature = "big_space")]
        crate::spatial::parent_under_big_space(&mut entity, root.as_deref());
    }

    spawned.0 = true;
}

#[derive(Clone, Copy)]
enum SensorPalette {
    WhiteHot,
    BlackHot,
    Ironbow,
}

fn palette_image(palette: SensorPalette) -> Image {
    let mut data = Vec::with_capacity(256 * 4);
    for value in 0..=255 {
        let t = value as f32 / 255.0;
        let rgb = match palette {
            SensorPalette::WhiteHot => [t, t, t],
            SensorPalette::BlackHot => [1.0 - t, 1.0 - t, 1.0 - t],
            SensorPalette::Ironbow => {
                let r = (1.5 * t - 0.25).clamp(0.0, 1.0);
                let g = (2.0 * t - 0.75).clamp(0.0, 1.0);
                let b = (3.0 * (t - 0.1) * (0.6 - t)).clamp(0.0, 1.0)
                    + ((t - 0.85).clamp(0.0, 1.0) * 3.0);
                [r, g, b.clamp(0.0, 1.0)]
            }
        };
        data.extend(rgb.map(|channel| (channel * 255.0).round() as u8));
        data.push(255);
    }
    Image::new(
        Extent3d {
            width: 256,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn empty_thermal_mask_image() -> Image {
    Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0, 0],
        TextureFormat::R16Float,
        RenderAssetUsages::default(),
    )
}

fn thermal_mask_target_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_target_texture(width, height, TextureFormat::R16Float, None);
    image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
    image
}

fn sensor_output_settings(config: &SensorCameraConfig) -> SensorOutputSettings {
    let value = |path: &[&str], legacy: &str, default: f32| {
        config.effect_param_f32(path, config.effect_param_f32(&[legacy], default))
    };
    let (effect_type, param_a, param_b) = match config.effect.as_str() {
        "thermal" => (
            1,
            value(&["contrast"], "contrast", 1.5),
            value(&["noise_sigma"], "noise_sigma", 0.02),
        ),
        "night_vision" => (
            2,
            value(&["gain"], "gain", 2.0),
            value(&["noise_sigma"], "noise_sigma", 0.04),
        ),
        "depth" => (3, 0.0, 0.0),
        "lwir" => (4, 0.0, 0.0),
        _ => (0, 0.0, 0.0),
    };
    let palette = match config.effect_param_str(
        &["palette"],
        if config.effect == "thermal" {
            "ironbow"
        } else {
            "white_hot"
        },
    ) {
        "black_hot" => SensorPalette::BlackHot,
        "ironbow" => SensorPalette::Ironbow,
        _ => SensorPalette::WhiteHot,
    };
    let palette_index = match palette {
        SensorPalette::WhiteHot => 0,
        SensorPalette::BlackHot => 1,
        SensorPalette::Ironbow => 2,
    };
    SensorOutputSettings {
        mode: UVec4::new(effect_type, palette_index, 0, 0),
        viewport: Vec4::new(
            config.width as f32,
            config.height as f32,
            config.near.max(1.0e-6),
            config.far.max(config.near + 1.0e-6),
        ),
        view_rotation: Vec4::new(0.0, 0.0, 0.0, 1.0),
        lens: Vec4::new(
            config.fov_degrees.to_radians(),
            config.width as f32 / config.height.max(1) as f32,
            0.0,
            0.0,
        ),
        legacy: Vec4::new(
            param_a,
            param_b,
            1.0,
            value(&["sky_offset_dn"], "sky_offset_dn", 2.0).clamp(0.0, 32.0) / 255.0,
        ),
        thermal: Vec4::new(
            value(&["t_air_c"], "t_air_c", 30.0),
            value(&["t_sky_zenith_c"], "t_sky_zenith_c", -50.0),
            value(&["t_base_c"], "t_base_c", 45.0),
            value(&["sun_gain"], "sun_gain", 10.0),
        ),
        agc: Vec4::new(
            value(&["agc", "min_c"], "agc_min_c", 20.0),
            value(&["agc", "max_c"], "agc_max_c", 60.0),
            value(&["agc", "smoothing"], "agc_smoothing", 0.9),
            value(&["dde"], "dde", 0.6),
        ),
        sensor: Vec4::new(
            value(&["mtf_blur_px"], "mtf_blur_px", 0.65),
            value(
                &["temporal_noise_sigma_dn"],
                "temporal_noise_sigma_dn",
                2.526,
            ),
            value(&["column_fpn_sigma_dn"], "column_fpn_sigma_dn", 0.25),
            value(&["vignette"], "vignette", 0.1),
        ),
        range: Vec4::new(
            value(&["transmission_km"], "transmission_km", 8.0),
            value(&["dead_pixel_ppm"], "dead_pixel_ppm", 0.0),
            value(&["agc", "low"], "agc_low_percentile", 0.01),
            value(&["agc", "high"], "agc_high_percentile", 0.99),
        ),
    }
}

fn spawn_sensor_cameras(
    mut commands: Commands,
    configs: Res<SensorCameraConfigs>,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    mut spawned: ResMut<SensorCamerasSpawned>,
    #[cfg(feature = "big_space")] root: Option<Res<crate::spatial::BigSpaceRootEntity>>,
) {
    for (i, config) in configs.0.iter().enumerate() {
        // LDR clipping erases the radiance variation used for LWIR terrain contrast.
        let hdr_input = config.cinematic || config.effect == "lwir";
        let size = Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let render_target_handle = if hdr_input {
            let mut hdr_image = Image::new_target_texture(
                size.width,
                size.height,
                TextureFormat::Rgba16Float,
                None,
            );
            hdr_image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
            images.add(hdr_image)
        } else {
            let mut render_target_image = Image::new_target_texture(
                size.width,
                size.height,
                TextureFormat::Rgba8UnormSrgb,
                None,
            );
            render_target_image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
            images.add(render_target_image)
        };

        let mut output_image =
            Image::new_target_texture(size.width, size.height, TextureFormat::Rgba8UnormSrgb, None);
        output_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let output_image = images.add(output_image);
        let mut output_settings = sensor_output_settings(config);
        let palette = match output_settings.mode.y {
            1 => SensorPalette::BlackHot,
            2 => SensorPalette::Ironbow,
            _ => SensorPalette::WhiteHot,
        };
        let palette_lut = images.add(palette_image(palette));
        let has_thermal_mask = config.effect == "lwir";
        let thermal_mask = if has_thermal_mask {
            output_settings.mode.z |= 1;
            images.add(thermal_mask_target_image(size.width, size.height))
        } else {
            images.add(empty_thermal_mask_image())
        };
        let thermal_mask_target = thermal_mask.clone();
        let mut temp_map_image =
            Image::new_target_texture(size.width, size.height, TextureFormat::Rgba8Unorm, None);
        temp_map_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let temp_map = images.add(temp_map_image);
        let output_target = SensorOutputTarget {
            output_image: output_image.clone(),
            palette_lut,
            thermal_mask,
            temp_map: temp_map.clone(),
        };

        let padded_bytes_per_row =
            RenderDevice::align_copy_bytes_per_row((size.width as usize) * 4);
        let buffer_size = padded_bytes_per_row as u64 * size.height as u64;
        let readback_buffer = |label: &str| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: buffer_size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let copier = ImageCopier {
            buffers: (0..READBACK_BUFFER_COUNT)
                .map(|_| readback_buffer("sensor_camera_readback"))
                .collect(),
            in_flight: (0..READBACK_BUFFER_COUNT)
                .map(|_| Arc::new(AtomicBool::new(false)))
                .collect(),
            src_image: output_image,
            camera_name: config.camera_name.clone(),
            width: config.width,
            height: config.height,
            is_active: false,
        };
        let temp_map_copier = (config.effect == "lwir").then(|| {
            TempMapCopier(ImageCopier {
                buffers: (0..READBACK_BUFFER_COUNT)
                    .map(|_| readback_buffer("sensor_camera_temp_readback"))
                    .collect(),
                in_flight: (0..READBACK_BUFFER_COUNT)
                    .map(|_| Arc::new(AtomicBool::new(false)))
                    .collect(),
                src_image: temp_map,
                camera_name: temp_map_msg_name(&config.camera_name),
                width: config.width,
                height: config.height,
                is_active: false,
            })
        });

        let perspective = PerspectiveProjection {
            fov: config.fov_degrees.to_radians(),
            near: config.near,
            far: config.far,
            near_clip_plane: crate::plugins::frustum_common::near_clip_plane(config.near),
            ..default()
        };
        let mask_perspective = perspective.clone();
        let camera_order = -(10 + i as isize);

        let camera_3d = Camera3d {
            depth_texture_usages: (TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING)
                .into(),
            ..default()
        };

        let mut entity = commands.spawn((
            (
                camera_3d,
                Camera {
                    order: camera_order,
                    is_active: false,
                    ..default()
                },
                RenderTarget::Image(render_target_handle.into()),
                Projection::Perspective(perspective),
                if hdr_input {
                    Tonemapping::TonyMcMapface
                } else {
                    Tonemapping::None
                },
                bevy::render::view::Msaa::Off,
            ),
            Transform::from_xyz(0.0, 5.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            GlobalTransform::default(),
            #[cfg(feature = "big_space")]
            crate::spatial::GridCell::default(),
            SensorCamera { config_index: i },
            output_settings,
            output_target,
            copier,
            ReadbackArmed(false),
            sensor_camera_render_layers(config),
            Name::new(format!("sensor_camera_{}", config.camera_name)),
        ));
        #[cfg(feature = "big_space")]
        crate::spatial::parent_under_big_space(&mut entity, root.as_deref());
        if let Some(temp_map_copier) = temp_map_copier {
            entity.insert(temp_map_copier);
        }

        // Earth owns the cinematic cubemap (`CinematicSkybox`). Tagging this
        // camera as `PrimarySkybox` makes the render-server wait forever for
        // that cubemap to disappear before emitting frames.
        if !config.cinematic {
            entity.insert(PrimarySkybox);
        }

        if hdr_input {
            entity.insert((
                Hdr,
                Exposure {
                    ev100: config.cinematic_ev100(),
                },
            ));
        }

        if config.cinematic {
            entity.insert((
                CinematicViewport,
                AmbientLight {
                    brightness: 0.0,
                    ..default()
                },
                EnvironmentMapLight {
                    diffuse_map: asset_server.load("embedded://elodin_editor/assets/diffuse.ktx2"),
                    specular_map: asset_server
                        .load("embedded://elodin_editor/assets/specular.ktx2"),
                    intensity: 2000.0,
                    ..Default::default()
                },
            ));
            if config.effect != "lwir" {
                entity.insert(bloom_from_config(config.bloom.as_ref(), true));
            }
        }
        let sensor_entity = entity.id();

        if has_thermal_mask {
            commands.spawn((
                Camera3d::default(),
                Camera {
                    order: camera_order - 1000,
                    is_active: false,
                    clear_color: ClearColorConfig::Custom(Color::BLACK),
                    ..default()
                },
                RenderTarget::Image(thermal_mask_target.into()),
                Projection::Perspective(mask_perspective),
                Tonemapping::None,
                bevy::render::view::Msaa::Off,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                ThermalMaskCamera { config_index: i },
                RenderLayers::layer(THERMAL_MASK_RENDER_LAYER),
                ChildOf(sensor_entity),
                Name::new(format!("thermal_mask_camera_{}", config.camera_name)),
            ));
        }

        bevy::log::debug!(
            "Spawned sensor camera '{}' ({}x{}, effect={}, cinematic={})",
            config.camera_name,
            config.width,
            config.height,
            config.effect,
            config.cinematic,
        );
    }

    spawned.0 = true;
}

fn sensor_camera_render_layers(config: &SensorCameraConfig) -> RenderLayers {
    let mut layers = RenderLayers::default();
    if config.cinematic {
        layers = layers.with(CINEMATIC_EARTH_RENDER_LAYER);
    }
    if config.show_ellipsoids {
        layers = layers.with(ELLIPSOID_RENDER_LAYER);
    }
    layers
}

pub(crate) fn update_sensor_camera_render_layers(
    configs: Res<SensorCameraConfigs>,
    mut sensor_cameras: Query<(&SensorCamera, &mut RenderLayers)>,
) {
    for (sensor_cam, mut layers) in &mut sensor_cameras {
        let Some(config) = configs.0.get(sensor_cam.config_index) else {
            continue;
        };
        let desired_layers = sensor_camera_render_layers(config);
        if *layers != desired_layers {
            *layers = desired_layers;
        }
    }
}

fn update_sensor_output_seed(
    timestamp: Option<Res<CurrentTimestamp>>,
    mut query: Query<&mut SensorOutputSettings>,
) {
    let timestamp = timestamp.map_or(0, |timestamp| timestamp.0.0 as u64);
    let seed = (timestamp ^ (timestamp >> 32)) as u32;
    for mut settings in &mut query {
        settings.mode.w = seed;
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_thermal_mask_proxies(
    mut commands: Commands,
    tags: Res<ThermalTagConfigs>,
    source_meshes: Query<(Entity, &Mesh3d), Without<ThermalMaskProxy>>,
    parents: Query<&ChildOf>,
    objects: Query<&Object3DState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut inspected: Local<HashSet<Entity>>,
    mut material_cache: Local<HashMap<(u32, u32), Handle<StandardMaterial>>>,
) {
    if tags.0.is_empty() {
        return;
    }
    for (mesh_entity, mesh) in &source_meshes {
        if inspected.contains(&mesh_entity) {
            continue;
        }
        let mut ancestor = mesh_entity;
        let object = loop {
            if let Ok(object) = objects.get(ancestor) {
                break Some(object);
            }
            let Ok(parent) = parents.get(ancestor) else {
                break None;
            };
            ancestor = parent.parent();
        };
        let tag = object.and_then(|object| {
            tags.0.iter().find(|tag| {
                object
                    .data
                    .eql
                    .strip_prefix(&tag.entity_name)
                    .is_some_and(|suffix| suffix.starts_with(".world_pos"))
            })
        });
        if let Some(tag) = tag {
            let key = (tag.temperature_c.to_bits(), tag.emissivity.to_bits());
            let material = material_cache
                .entry(key)
                .or_insert_with(|| {
                    let apparent_temperature = 20.0 + tag.emissivity * (tag.temperature_c - 20.0);
                    let encoded = ((apparent_temperature + 100.0) / 500.0).clamp(f32::EPSILON, 1.0);
                    materials.add(StandardMaterial {
                        base_color: Color::linear_rgba(encoded, 0.0, 0.0, 1.0),
                        unlit: true,
                        ..default()
                    })
                })
                .clone();
            commands.spawn((
                Mesh3d(mesh.0.clone()),
                MeshMaterial3d(material),
                Transform::IDENTITY,
                RenderLayers::layer(THERMAL_MASK_RENDER_LAYER),
                bevy::light::NotShadowCaster,
                bevy::light::NotShadowReceiver,
                ThermalMaskProxy,
                ChildOf(mesh_entity),
                Name::new("thermal mask proxy"),
            ));
        }
        inspected.insert(mesh_entity);
    }
}

/// Split an absolute translation across the floating-origin grid.
///
/// `Transform` alone is not the position. `big_space` moves a translation past
/// `maximum_distance_from_origin` into `GridCell` in `PostUpdate`, so writing
/// the absolute value back the next frame while leaving the cell where it
/// landed makes the two add up, and the entity walks a further cell out on
/// every frame after that.
#[cfg(feature = "big_space")]
fn place_on_grid(
    settings: &crate::spatial::FloatingOriginSettings,
    cell: &mut crate::spatial::GridCell,
    transform: &mut Transform,
    translation: DVec3,
) {
    let (new_cell, local_translation) = settings.translation_to_grid(translation);
    *cell = new_cell;
    transform.translation = local_translation;
}

#[allow(clippy::too_many_arguments)]
fn update_sensor_camera_transforms(
    configs: Res<SensorCameraConfigs>,
    mut sensor_cameras: Query<(
        Entity,
        &SensorCamera,
        &mut Transform,
        &mut SensorOutputSettings,
    )>,
    #[cfg(feature = "big_space")] mut cells: Query<&mut crate::spatial::GridCell>,
    #[cfg(feature = "big_space")] settings: Res<crate::spatial::FloatingOriginSettings>,
    cache: Res<impeller2_bevy::TelemetryCache>,
    current_ts: Res<impeller2_wkt::CurrentTimestamp>,
    coordinate: Res<crate::Coordinate>,
    geo_context: Res<GeoContext>,
) {
    let ts = current_ts.0;
    let frame = coordinate.0.unwrap_or_default();
    for (_entity, sensor_cam, mut transform, mut output) in &mut sensor_cameras {
        let Some(config) = configs.0.get(sensor_cam.config_index) else {
            continue;
        };

        let Some(pose) = sensor_camera_transform(config, &cache, ts, frame, &geo_context) else {
            continue;
        };
        transform.rotation = pose.rotation;
        output.view_rotation = Vec4::from_array(pose.rotation.to_array());

        #[cfg(feature = "big_space")]
        if let Ok(mut cell) = cells.get_mut(_entity) {
            place_on_grid(&settings, &mut cell, &mut transform, pose.translation);
        }
        #[cfg(not(feature = "big_space"))]
        {
            transform.translation = pose.translation.as_vec3();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_sensor_camera_frustum_source_transforms(
    configs: Res<SensorCameraConfigs>,
    mut sources: Query<(
        Entity,
        &SensorCameraFrustumSource,
        &mut Transform,
        &mut Projection,
    )>,
    #[cfg(feature = "big_space")] mut cells: Query<&mut crate::spatial::GridCell>,
    #[cfg(feature = "big_space")] settings: Res<crate::spatial::FloatingOriginSettings>,
    cache: Res<impeller2_bevy::TelemetryCache>,
    current_ts: Res<impeller2_wkt::CurrentTimestamp>,
    coordinate: Res<crate::Coordinate>,
    geo_context: Res<GeoContext>,
) {
    let ts = current_ts.0;
    let frame = coordinate.0.unwrap_or_default();
    for (_entity, source, mut transform, mut projection) in &mut sources {
        let Some(config) = configs.0.get(source.config_index) else {
            continue;
        };

        if let Some(pose) = sensor_camera_transform(config, &cache, ts, frame, &geo_context) {
            transform.rotation = pose.rotation;
            #[cfg(feature = "big_space")]
            if let Ok(mut cell) = cells.get_mut(_entity) {
                place_on_grid(&settings, &mut cell, &mut transform, pose.translation);
            }
            #[cfg(not(feature = "big_space"))]
            {
                transform.translation = pose.translation.as_vec3();
            }
        }

        if let Projection::Perspective(perspective) = projection.as_mut() {
            perspective.fov = config.fov_degrees.to_radians();
            perspective.near = config.near;
            perspective.far = config.far;
            if config.height > 0 {
                perspective.aspect_ratio = config.width as f32 / config.height as f32;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SensorCameraPose {
    translation: DVec3,
    rotation: Quat,
}

fn sensor_camera_transform(
    config: &SensorCameraConfig,
    cache: &impeller2_bevy::TelemetryCache,
    ts: impeller2::types::Timestamp,
    frame: GeoFrame,
    geo_context: &GeoContext,
) -> Option<SensorCameraPose> {
    let world_pos_id = ComponentId::new(&format!("{}.world_pos", config.entity_name));
    let value = cache.get_at_or_before(&world_pos_id, ts)?;
    let world_pos = value.as_world_pos()?;
    sensor_camera_pose(config, world_pos, frame, geo_context)
}

fn sensor_camera_pose(
    config: &SensorCameraConfig,
    world_pos: impeller2_wkt::WorldPos,
    frame: GeoFrame,
    geo_context: &GeoContext,
) -> Option<SensorCameraPose> {
    let entity_pos: DVec3 = {
        let [x, y, z] = world_pos.pos.parts().map(nox::Tensor::into_buf);
        DVec3::new(x, y, z)
    };
    let entity_att: bevy::math::DQuat = {
        let [i, j, k, w] = world_pos.att.parts().map(nox::Tensor::into_buf);
        bevy::math::DQuat::from_xyzw(i, j, k, w)
    };

    let pos_offset_body = DVec3::new(
        config.pos_offset[0],
        config.pos_offset[1],
        config.pos_offset[2],
    );
    let [roll_deg, pitch_deg, yaw_deg] = config.rot_offset;
    let rot_offset_body = bevy::math::DQuat::from_axis_angle(DVec3::X, roll_deg.to_radians())
        * bevy::math::DQuat::from_axis_angle(DVec3::Y, pitch_deg.to_radians())
        * bevy::math::DQuat::from_axis_angle(DVec3::Z, yaw_deg.to_radians());

    let cam_forward_body = rot_offset_body * DVec3::X;
    let cam_up_body = rot_offset_body * DVec3::Z;
    let cam_pos = entity_pos + entity_att * pos_offset_body;
    let cam_forward = entity_att * cam_forward_body;
    let cam_up = entity_att * cam_up_body;

    if !cam_pos.is_finite()
        || !cam_forward.is_finite()
        || !cam_up.is_finite()
        || cam_forward.length_squared() <= 1e-12
    {
        return None;
    }

    Some(SensorCameraPose {
        translation: GeoPosition(frame, cam_pos).to_bevy(geo_context),
        rotation: GeoRotation::look_at(frame, cam_forward, Some(cam_up), geo_context)
            .to_bevy(geo_context)
            .as_quat(),
    })
}

// ---------------------------------------------------------------------------
// Render-world systems (GPU readback)
// ---------------------------------------------------------------------------

type ImageCopierQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static ImageCopier,
        Option<&'static TempMapCopier>,
        &'static ReadbackArmed,
        &'static Camera,
    ),
>;

fn image_copy_extract(
    mut commands: Commands,
    headless_mode: Option<Res<HeadlessMode>>,
    image_copiers: Extract<ImageCopierQuery>,
    current_timestamp: Extract<Res<CurrentTimestamp>>,
    mut last_temp_readback: Local<HashMap<String, Timestamp>>,
) {
    let headless = headless_mode.is_some();
    let mut copiers = Vec::new();
    for (copier, temp_map_copier, readback_armed, camera) in image_copiers.iter() {
        let is_active = if headless {
            readback_armed.0
        } else {
            readback_armed.0 || camera.is_active
        };
        let mut main = copier.clone();
        main.is_active = is_active;
        copiers.push(main);
        if let Some(TempMapCopier(temp)) = temp_map_copier {
            let mut temp = temp.clone();
            let last = last_temp_readback.get(&temp.camera_name).copied();
            temp.is_active = is_active
                && last.is_none_or(|last| {
                    current_timestamp.0.0.saturating_sub(last.0) >= LWIR_AGC_READBACK_INTERVAL_US
                });
            if temp.is_active {
                last_temp_readback.insert(temp.camera_name.clone(), current_timestamp.0);
            }
            copiers.push(temp);
        }
    }
    commands.insert_resource(ImageCopiers(copiers, current_timestamp.0));
}

struct ReadbackJob {
    buffer: Buffer,
    in_flight: Arc<AtomicBool>,
    _pending: PendingReadback,
    camera_name: String,
    timestamp: Timestamp,
    width: u32,
    height: u32,
}

struct SensorReadbackWorker {
    sender: Option<std::sync::mpsc::Sender<ReadbackJob>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl SensorReadbackWorker {
    fn new(
        index: usize,
        render_device: RenderDevice,
        frame_sender: SensorFrameSender,
    ) -> Result<Self, std::io::Error> {
        let (sender, receiver) = std::sync::mpsc::channel::<ReadbackJob>();
        let handle = std::thread::Builder::new()
            .name(format!("sensor-readback-{index}"))
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    readback_sensor_frame(&render_device, &frame_sender, job);
                }
            })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    fn send(&self, job: ReadbackJob) {
        if let Some(sender) = &self.sender
            && let Err(err) = sender.send(job)
        {
            err.0.in_flight.store(false, Ordering::Release);
        }
    }
}

impl Drop for SensorReadbackWorker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn readback_sensor_frame(
    render_device: &RenderDevice,
    frame_sender: &SensorFrameSender,
    job: ReadbackJob,
) {
    let buffer_slice = job.buffer.slice(..);
    let (sender, receiver) = crossbeam_channel::bounded(1);
    buffer_slice.map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let mapped = loop {
        std::thread::sleep(std::time::Duration::from_millis(2));
        let _ = render_device.poll(PollType::Poll);
        match receiver.try_recv() {
            Ok(result) => break result.is_ok(),
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => break false,
        }
    };
    if !mapped {
        job.buffer.unmap();
        job.in_flight.store(false, Ordering::Release);
        return;
    }

    let data = buffer_slice.get_mapped_range();
    let row_bytes = job.width as usize * 4;
    let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);
    let required_len = job.height as usize * row_bytes;
    let mut frame = vec![0; required_len];
    if row_bytes == aligned_row_bytes {
        let copy_len = required_len.min(data.len());
        frame[..copy_len].copy_from_slice(&data[..copy_len]);
    } else {
        for (row_idx, chunk) in data
            .chunks(aligned_row_bytes)
            .take(job.height as usize)
            .enumerate()
        {
            let len = row_bytes.min(chunk.len());
            let start = row_idx * row_bytes;
            if start + len <= frame.len() {
                frame[start..start + len].copy_from_slice(&chunk[..len]);
            }
        }
    }
    drop(data);
    job.buffer.unmap();
    job.in_flight.store(false, Ordering::Release);
    let _ = frame_sender
        .0
        .send((job.camera_name, job.timestamp, frame, job.width, job.height));
}

fn image_copy_driver(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
    frame_sender: Res<SensorFrameSender>,
    readback_status: Res<SensorReadbackStatus>,
    mut buffer_toggle: ResMut<BufferToggle>,
    mut workers: Local<Vec<SensorReadbackWorker>>,
    mut metrics: ResMut<SensorCameraRenderMetrics>,
) {
    let _span = tracing::info_span!("sensor_camera_image_copy_driver").entered();
    let copy_start = Instant::now();
    if buffer_toggle.0.len() < image_copiers.0.len() {
        buffer_toggle.0.resize(image_copiers.0.len(), 0);
    }
    if workers.is_empty() {
        for index in 0..READBACK_WORKER_COUNT {
            match SensorReadbackWorker::new(index, render_device.clone(), frame_sender.clone()) {
                Ok(readback_worker) => workers.push(readback_worker),
                Err(err) => {
                    tracing::error!("failed to start sensor readback worker: {err}");
                    return;
                }
            }
        }
    }

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    let mut jobs = Vec::new();

    for (i, image_copier) in image_copiers.0.iter().enumerate() {
        if !image_copier.is_active {
            continue;
        }
        let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
            continue;
        };

        let block_dimensions = src_image.texture_descriptor.format.block_dimensions();
        let block_size = src_image
            .texture_descriptor
            .format
            .block_copy_size(None)
            .unwrap();

        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (src_image.texture_descriptor.size.width as usize / block_dimensions.0 as usize)
                * block_size as usize,
        );

        let preferred = buffer_toggle.0[i];
        let buffer_count = image_copier.buffers.len();
        let buf_idx = loop {
            if let Some(index) = (0..buffer_count)
                .map(|offset| (preferred + offset) % buffer_count)
                .find(|index| !image_copier.in_flight[*index].swap(true, Ordering::AcqRel))
            {
                break index;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        };
        encoder.copy_texture_to_buffer(
            src_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &image_copier.buffers[buf_idx],
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
                            .unwrap()
                            .into(),
                    ),
                    rows_per_image: None,
                },
            },
            src_image.texture_descriptor.size,
        );
        buffer_toggle.0[i] = (buf_idx + 1) % buffer_count;
        jobs.push((
            i,
            ReadbackJob {
                buffer: image_copier.buffers[buf_idx].clone(),
                in_flight: image_copier.in_flight[buf_idx].clone(),
                _pending: PendingReadback::new(readback_status.clone()),
                camera_name: image_copier.camera_name.clone(),
                timestamp: image_copiers.1,
                width: image_copier.width,
                height: image_copier.height,
            },
        ));
    }

    if !jobs.is_empty() {
        render_queue.submit(std::iter::once(encoder.finish()));
    }
    for (index, job) in jobs {
        workers[index % workers.len()].send(job);
    }
    metrics.image_copy_driver_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
}

// ---------------------------------------------------------------------------
// Patch sensor_view panel dimensions once configs arrive
// ---------------------------------------------------------------------------

/// The sensor_view panels may be spawned before SensorCameraConfigs are loaded.
pub fn patch_sensor_view_dims(
    configs: Res<SensorCameraConfigs>,
    mut streams: Query<(
        &mut crate::ui::video_stream::VideoStream,
        &mut crate::ui::video_stream::VideoFrameCache,
    )>,
) {
    if configs.0.is_empty() {
        return;
    }
    for (mut stream, mut cache) in streams.iter_mut() {
        if let Some(config) = configs.0.iter().find(|c| c.camera_name == stream.msg_name) {
            if config.format == "h264" {
                if stream.raw_dims.take().is_some() || !cache.is_h264 {
                    *cache = crate::ui::video_stream::VideoFrameCache::default();
                }
            } else {
                let format = if config.format == "gray8" {
                    crate::ui::video_stream::RawPixelFormat::Gray8
                } else {
                    crate::ui::video_stream::RawPixelFormat::Rgba8
                };
                let dimensions = (config.width, config.height, format);
                if stream.raw_dims != Some(dimensions) || cache.is_h264 {
                    stream.raw_dims = Some(dimensions);
                    *cache = crate::ui::video_stream::VideoFrameCache::for_raw();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Headless render server helpers
// ---------------------------------------------------------------------------

/// Arm or disarm GPU readback for specific cameras by name.
/// Called by the headless render server to control which cameras perform
/// expensive GPU texture-to-buffer copies.
pub fn set_readback_armed(world: &mut World, camera_names: &[String], armed: bool) {
    let configs = world.resource::<SensorCameraConfigs>();
    let target_indices: Vec<usize> = camera_names
        .iter()
        .filter_map(|name| configs.0.iter().position(|c| &c.camera_name == name))
        .collect();

    let mut query = world.query::<(&SensorCamera, &mut ReadbackArmed)>();
    for (sensor, mut readback) in query.iter_mut(world) {
        if target_indices.contains(&sensor.config_index) {
            readback.0 = armed;
        }
    }
}

/// Toggle `Camera.is_active` for specific sensor cameras by name. Bevy's
/// render extract skips inactive cameras, so flipping this off between
/// scheduled frames prevents the GPU from rendering an unread scene on every
/// polling iteration. Headless steady state is "all sensor cameras inactive";
/// the render-server flips the due set on for each `render_and_emit` cycle
/// and back off afterwards.
pub fn set_cameras_active(world: &mut World, camera_names: &[String], active: bool) {
    let configs = world.resource::<SensorCameraConfigs>();
    let target_indices: Vec<usize> = camera_names
        .iter()
        .filter_map(|name| configs.0.iter().position(|c| &c.camera_name == name))
        .collect();

    {
        let mut query = world.query::<(&SensorCamera, &mut Camera)>();
        for (sensor, mut camera) in query.iter_mut(world) {
            if target_indices.contains(&sensor.config_index) {
                camera.is_active = active;
            }
        }
    }
    let mut masks = world.query::<(&ThermalMaskCamera, &mut Camera)>();
    for (mask, mut camera) in masks.iter_mut(world) {
        if target_indices.contains(&mask.config_index) {
            camera.is_active = active;
        }
    }
}

fn percentile_bin(histogram: &[u32; 256], fraction: f32) -> u8 {
    let total: u32 = histogram.iter().sum();
    let threshold = ((total as f32 * fraction.clamp(0.0, 1.0)).ceil() as u32).max(1);
    let mut seen = 0;
    for (value, count) in histogram.iter().enumerate() {
        seen += count;
        if seen >= threshold {
            return value as u8;
        }
    }
    255
}

/// Fixed physical range of the temperature-map attachment. Must match the
/// `TEMP_MAP_MIN_C` / `TEMP_MAP_MAX_C` constants in `sensor_output.wgsl`.
const TEMP_MAP_MIN_C: f32 = -60.0;
const TEMP_MAP_MAX_C: f32 = 140.0;
/// Below this fraction of non-sky pixels the AGC freezes instead of adapting
/// (Boson-like plateau behavior); it recovers as soon as ground returns.
const AGC_MIN_GROUND_FRACTION: f32 = 0.02;
const AGC_MIN_SPAN_C: f32 = 2.0;
const AGC_MAX_SPAN_C: f32 = 120.0;
const AGC_GAMMA_RANGE: (f32, f32) = (0.25, 4.0);

fn temp_map_bin_to_c(bin: u8) -> f32 {
    TEMP_MAP_MIN_C + f32::from(bin) / 255.0 * (TEMP_MAP_MAX_C - TEMP_MAP_MIN_C)
}

struct AgcUpdate {
    config_index: usize,
    camera_name: String,
    low_c: f32,
    high_c: f32,
    median_c: f32,
    target_median: f32,
    ground_fraction: f32,
}

/// Adapt LWIR level/span/gamma from the pre-AGC temperature-map readback.
///
/// Statistics come from the scene temperature field with sky masked out, so
/// they are independent of the current AGC state: a maneuver through
/// sky-only views can freeze adaptation but can never latch it.
pub fn update_auto_agc(world: &mut World, temp_frames: &[(String, Vec<u8>)]) {
    let configs = world.resource::<SensorCameraConfigs>();
    let updates: Vec<AgcUpdate> = temp_frames
        .iter()
        .filter_map(|(name, bytes)| {
            let camera_name = name.strip_suffix(TEMP_MAP_SUFFIX)?;
            let config_index = configs
                .0
                .iter()
                .position(|config| config.camera_name == camera_name)?;
            let config = &configs.0[config_index];
            if config.effect != "lwir"
                || config.effect_param_str(&["agc", "mode"], "manual") != "auto"
                || bytes.len() < 4
            {
                return None;
            }
            let mut histogram = [0u32; 256];
            let mut sky = 0u32;
            for pixel in bytes.as_chunks::<4>().0 {
                if pixel[1] >= 128 {
                    sky += 1;
                } else {
                    histogram[pixel[0] as usize] += 1;
                }
            }
            let ground: u32 = histogram.iter().sum();
            let ground_fraction = ground as f32 / (ground + sky).max(1) as f32;
            if ground_fraction < AGC_MIN_GROUND_FRACTION {
                tracing::debug!(
                    camera = camera_name,
                    ground_fraction,
                    "lwir agc frozen: sky-dominant frame"
                );
                return None;
            }
            let low_fraction = config.effect_param_f32(&["agc", "low"], 0.01);
            let high_fraction = config.effect_param_f32(&["agc", "high"], 0.99);
            Some(AgcUpdate {
                config_index,
                camera_name: camera_name.to_owned(),
                low_c: temp_map_bin_to_c(percentile_bin(&histogram, low_fraction)),
                high_c: temp_map_bin_to_c(percentile_bin(&histogram, high_fraction)),
                median_c: temp_map_bin_to_c(percentile_bin(&histogram, 0.5)),
                target_median: config.effect_param_f32(&["agc", "target_median"], 0.35),
                ground_fraction,
            })
        })
        .collect();
    if updates.is_empty() {
        return;
    }

    let mut cameras = world.query::<(&SensorCamera, &mut SensorOutputSettings)>();
    for (camera, mut settings) in cameras.iter_mut(world) {
        let Some(update) = updates
            .iter()
            .find(|update| update.config_index == camera.config_index)
        else {
            continue;
        };
        let candidate_min = update.low_c;
        let candidate_max = update
            .high_c
            .max(candidate_min + AGC_MIN_SPAN_C)
            .min(candidate_min + AGC_MAX_SPAN_C);
        let history = settings.agc.z.clamp(0.0, 0.999);
        settings.agc.x = settings.agc.x * history + candidate_min * (1.0 - history);
        settings.agc.y = settings.agc.y * history + candidate_max * (1.0 - history);

        let span = (settings.agc.y - settings.agc.x).max(1.0e-3);
        let median_signal = (update.median_c - settings.agc.x) / span;
        if (0.02..=0.98).contains(&median_signal) {
            let candidate_gamma = (update.target_median.clamp(0.05, 0.95).ln()
                / median_signal.ln())
            .clamp(AGC_GAMMA_RANGE.0, AGC_GAMMA_RANGE.1);
            settings.legacy.z = settings.legacy.z * history + candidate_gamma * (1.0 - history);
        }
        tracing::debug!(
            camera = update.camera_name,
            min_c = settings.agc.x,
            max_c = settings.agc.y,
            gamma = settings.legacy.z,
            ground_fraction = update.ground_fraction,
            "lwir agc update"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_geo_frames::GeoOrigin;

    fn camera_config(pos_offset: [f64; 3]) -> SensorCameraConfig {
        SensorCameraConfig {
            entity_name: "vehicle".into(),
            camera_name: "camera".into(),
            width: 640,
            height: 480,
            fov_degrees: 90.0,
            near: 0.1,
            far: 100_000.0,
            pos_offset,
            rot_offset: [0.0; 3],
            format: "rgba8".into(),
            effect: String::new(),
            effect_params: Default::default(),
            create_frustum: false,
            show_ellipsoids: false,
            frustums_color: Default::default(),
            projection_color: Default::default(),
            frustums_thickness: 0.006,
            fps: 30.0,
            ..Default::default()
        }
    }

    #[test]
    fn pending_readback_tracks_job_lifetime() {
        let status = SensorReadbackStatus::default();
        let pending = PendingReadback::new(status.clone());
        assert_eq!(status.pending(), 1);
        drop(pending);
        assert_eq!(status.pending(), 0);
    }

    fn world_pos(pos: DVec3, att: bevy::math::DQuat) -> impeller2_wkt::WorldPos {
        impeller2_wkt::WorldPos {
            att: nox::Quaternion::new(att.w, att.x, att.y, att.z),
            pos: nox::Vector3::new(pos.x, pos.y, pos.z),
        }
    }

    #[test]
    fn enu_sensor_camera_preserves_legacy_plane_mapping() {
        let context = GeoContext::default();
        let config = camera_config([1.0, 2.0, 3.0]);
        let pose = sensor_camera_pose(
            &config,
            world_pos(DVec3::new(10.0, 20.0, 30.0), bevy::math::DQuat::IDENTITY),
            GeoFrame::ENU,
            &context,
        )
        .expect("valid sensor camera pose");

        let expected_translation = DVec3::new(11.0, 33.0, -22.0);
        assert!(
            pose.translation.abs_diff_eq(expected_translation, 1e-12),
            "got {:?}, expected {expected_translation:?}",
            pose.translation
        );
        let expected_rotation = Transform::from_translation(expected_translation.as_vec3())
            .looking_at((expected_translation + DVec3::X).as_vec3(), Vec3::Y)
            .rotation;
        assert!(
            pose.rotation.dot(expected_rotation).abs() > 1.0 - 1e-6,
            "got {:?}, expected {expected_rotation:?}",
            pose.rotation
        );
    }

    #[test]
    fn ecef_sensor_camera_uses_schematic_origin_rebase() {
        let context =
            GeoContext::from(GeoOrigin::new_from_degrees(35.350664, -117.809027, 589.274));
        let ecef_r_enu = GeoFrame::ecef_R_(&GeoFrame::ENU, &context.origin);
        let entity_att = bevy::math::DQuat::from_mat3(&ecef_r_enu);
        let entity_pos = GeoFrame::ecef_M_(&GeoFrame::ENU, &context)
            .transform_point3(DVec3::new(100.0, 200.0, 300.0));
        let offset = DVec3::new(1.2, 0.0, 0.1);
        let config = camera_config(offset.to_array());

        let pose = sensor_camera_pose(
            &config,
            world_pos(entity_pos, entity_att),
            GeoFrame::ECEF,
            &context,
        )
        .expect("valid ECEF sensor camera pose");

        let camera_ecef = entity_pos + entity_att * offset;
        let expected_translation = GeoPosition(GeoFrame::ECEF, camera_ecef).to_bevy(&context);
        assert!(
            pose.translation.abs_diff_eq(expected_translation, 1e-8),
            "got {:?}, expected {expected_translation:?}",
            pose.translation
        );

        let expected_forward =
            GeoFrame::bevy_R_(&GeoFrame::ECEF, &context) * (entity_att * DVec3::X);
        let expected_up = GeoFrame::bevy_R_(&GeoFrame::ECEF, &context) * (entity_att * DVec3::Z);
        let actual_forward = pose.rotation.as_dquat() * DVec3::NEG_Z;
        let actual_up = pose.rotation.as_dquat() * DVec3::Y;
        assert!(actual_forward.abs_diff_eq(expected_forward, 1e-6));
        assert!(actual_up.abs_diff_eq(expected_up, 1e-6));

        let legacy_unrebased = DVec3::new(camera_ecef.x, camera_ecef.z, -camera_ecef.y);
        assert!(
            pose.translation.distance(legacy_unrebased) > 1.0e6,
            "ECEF pose still looks like an unrebased flat swizzle"
        );
    }

    #[test]
    fn depth_effect_uses_output_pipeline_settings() {
        let mut config = camera_config([0.0; 3]);
        config.effect = "depth".into();
        let settings = sensor_output_settings(&config);
        assert_eq!(settings.mode.x, 3);
        assert_eq!(settings.viewport, Vec4::new(640.0, 480.0, 0.1, 100_000.0));
        assert_eq!(settings.view_rotation, Vec4::new(0.0, 0.0, 0.0, 1.0));
        assert!((settings.lens.x - 90.0_f32.to_radians()).abs() < f32::EPSILON);
        assert!((settings.lens.y - 640.0 / 480.0).abs() < f32::EPSILON);
    }

    fn lwir_agc_world(smoothing: f64) -> (World, Entity) {
        let mut config = camera_config([0.0; 3]);
        config.effect = "lwir".into();
        config.camera_name = "vehicle.ir".into();
        config.effect_params = serde_json::json!({
            "agc": {"mode": "auto", "low": 0.01, "high": 0.99, "smoothing": smoothing}
        });
        let settings = sensor_output_settings(&config);
        let mut world = World::new();
        world.insert_resource(SensorCameraConfigs(vec![config]));
        let camera = world
            .spawn((SensorCamera { config_index: 0 }, settings))
            .id();
        (world, camera)
    }

    /// RGBA temperature-map pixels: R = temperature bin, G = sky flag.
    fn temp_map_frame(ground_bins: &[u8], sky_pixels: usize) -> (String, Vec<u8>) {
        let mut bytes = Vec::with_capacity((ground_bins.len() + sky_pixels) * 4);
        for bin in ground_bins {
            bytes.extend_from_slice(&[*bin, 0, 0, 255]);
        }
        for _ in 0..sky_pixels {
            bytes.extend_from_slice(&[0, 255, 0, 255]);
        }
        (temp_map_msg_name("vehicle.ir"), bytes)
    }

    #[test]
    fn auto_agc_converges_toward_ground_percentiles() {
        let (mut world, camera) = lwir_agc_world(0.0);
        // Uniform ground temperatures across bins 100..=200.
        let bins: Vec<u8> = (100..=200).collect();
        update_auto_agc(&mut world, &[temp_map_frame(&bins, 0)]);
        let settings = world.get::<SensorOutputSettings>(camera).unwrap();
        assert!((settings.agc.x - temp_map_bin_to_c(101)).abs() < 1.0);
        assert!((settings.agc.y - temp_map_bin_to_c(199)).abs() < 1.0);
        assert!(settings.agc.y > settings.agc.x);
    }

    #[test]
    fn auto_agc_freezes_on_sky_dominant_frames() {
        let (mut world, camera) = lwir_agc_world(0.0);
        let before = *world.get::<SensorOutputSettings>(camera).unwrap();
        update_auto_agc(&mut world, &[temp_map_frame(&[150; 4], 996)]);
        let after = world.get::<SensorOutputSettings>(camera).unwrap();
        assert_eq!(after.agc, before.agc);
        assert_eq!(after.legacy, before.legacy);
    }

    #[test]
    fn auto_agc_recovers_after_sky_exposure() {
        // The latch regression: sky-only frames must freeze (not corrupt) the
        // state, and ground statistics must then re-converge through the EMA.
        let (mut world, camera) = lwir_agc_world(0.9);
        for _ in 0..30 {
            update_auto_agc(&mut world, &[temp_map_frame(&[], 1000)]);
        }
        let frozen = *world.get::<SensorOutputSettings>(camera).unwrap();
        assert_eq!(frozen.agc.x, 20.0);
        assert_eq!(frozen.agc.y, 60.0);
        let bins: Vec<u8> = (100..=200).collect();
        for _ in 0..80 {
            update_auto_agc(&mut world, &[temp_map_frame(&bins, 0)]);
        }
        let settings = world.get::<SensorOutputSettings>(camera).unwrap();
        assert!((settings.agc.x - temp_map_bin_to_c(101)).abs() < 1.0);
        assert!((settings.agc.y - temp_map_bin_to_c(199)).abs() < 1.0);
    }

    #[test]
    fn auto_agc_enforces_minimum_span() {
        let (mut world, camera) = lwir_agc_world(0.0);
        update_auto_agc(&mut world, &[temp_map_frame(&[150; 1000], 0)]);
        let settings = world.get::<SensorOutputSettings>(camera).unwrap();
        assert!(settings.agc.y - settings.agc.x >= AGC_MIN_SPAN_C - 1.0e-3);
    }
}
