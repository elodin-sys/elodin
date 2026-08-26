use std::{
    collections::{HashMap, HashSet},
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
use impeller2::types::ComponentId;
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
}

/// Double-buffered GPU readback state for a single sensor camera.
/// One buffer can be mapped for CPU read while the other receives the next frame.
/// The active buffer index is tracked in the render-world `BufferToggle` resource
/// so it persists across extract rebuilds.
#[derive(Clone, Component)]
struct ImageCopier {
    buffers: [Buffer; 2],
    src_image: Handle<Image>,
    camera_name: String,
    width: u32,
    height: u32,
    is_active: bool,
}

#[derive(Clone, Default, Resource)]
struct ImageCopiers(pub Vec<ImageCopier>);

#[derive(Resource)]
pub struct SensorFrameReceiver(pub flume::Receiver<(String, Vec<u8>, u32, u32)>);

#[derive(Resource, Clone)]
struct SensorFrameSender(flume::Sender<(String, Vec<u8>, u32, u32)>);

/// Per-camera reusable buffers for GPU readback to avoid per-frame allocation.
/// Indexed by camera position in the `ImageCopiers` list.
#[derive(Resource, Default)]
struct ReusableFrameBuffer(Vec<Vec<u8>>);

/// Render-world resource that persists buffer toggle state across frames.
/// The extract system rebuilds `ImageCopiers` each frame (resetting `write_buffer_idx`),
/// so we track the actual ping-pong indices here.
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
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &output.texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations::default(),
        })],
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
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
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

        embedded_asset!(app, "assets/shaders/sensor_output.wgsl");

        app.init_resource::<SensorCameraConfigs>()
            .init_resource::<ThermalTagConfigs>()
            .init_resource::<SensorCamerasSpawned>()
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
                .init_resource::<ReusableFrameBuffer>()
                .init_resource::<BufferToggle>()
                .init_resource::<SensorCameraRenderMetrics>()
                .add_systems(ExtractSchedule, image_copy_extract)
                .add_systems(
                    Render,
                    (image_copy_driver, receive_image_from_buffer)
                        .chain()
                        .after(RenderSystems::Render),
                )
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
        legacy: Vec4::new(param_a, param_b, 1.0, 0.0),
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
        let size = Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let render_target_handle = if config.cinematic {
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
        let output_target = SensorOutputTarget {
            output_image: output_image.clone(),
            palette_lut,
            thermal_mask,
        };

        let padded_bytes_per_row =
            RenderDevice::align_copy_bytes_per_row((size.width as usize) * 4);
        let buffer_size = padded_bytes_per_row as u64 * size.height as u64;
        let cpu_buffer_0 = render_device.create_buffer(&BufferDescriptor {
            label: Some("sensor_camera_readback_0"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cpu_buffer_1 = render_device.create_buffer(&BufferDescriptor {
            label: Some("sensor_camera_readback_1"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let copier = ImageCopier {
            buffers: [cpu_buffer_0, cpu_buffer_1],
            src_image: output_image,
            camera_name: config.camera_name.clone(),
            width: config.width,
            height: config.height,
            is_active: false,
        };

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
                if config.cinematic && config.effect != "lwir" {
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

        // Earth owns the cinematic cubemap (`CinematicSkybox`). Tagging this
        // camera as `PrimarySkybox` makes the render-server wait forever for
        // that cubemap to disappear before emitting frames.
        if !config.cinematic {
            entity.insert(PrimarySkybox);
        }

        if config.cinematic {
            entity.insert((
                Hdr,
                CinematicViewport,
                Exposure {
                    ev100: config.cinematic_ev100(),
                },
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
    mut sensor_cameras: Query<(Entity, &SensorCamera, &mut Transform)>,
    #[cfg(feature = "big_space")] mut cells: Query<&mut crate::spatial::GridCell>,
    #[cfg(feature = "big_space")] settings: Res<crate::spatial::FloatingOriginSettings>,
    cache: Res<impeller2_bevy::TelemetryCache>,
    current_ts: Res<impeller2_wkt::CurrentTimestamp>,
    coordinate: Res<crate::Coordinate>,
    geo_context: Res<GeoContext>,
) {
    let ts = current_ts.0;
    let frame = coordinate.0.unwrap_or_default();
    for (_entity, sensor_cam, mut transform) in &mut sensor_cameras {
        let Some(config) = configs.0.get(sensor_cam.config_index) else {
            continue;
        };

        let Some(pose) = sensor_camera_transform(config, &cache, ts, frame, &geo_context) else {
            continue;
        };
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

fn image_copy_extract(
    mut commands: Commands,
    headless_mode: Option<Res<HeadlessMode>>,
    image_copiers: Extract<Query<(&ImageCopier, &ReadbackArmed, &Camera)>>,
) {
    let headless = headless_mode.is_some();
    let copiers: Vec<ImageCopier> = image_copiers
        .iter()
        .map(|(copier, readback_armed, camera)| {
            let mut c = copier.clone();
            c.is_active = if headless {
                readback_armed.0
            } else {
                readback_armed.0 || camera.is_active
            };
            c
        })
        .collect();
    commands.insert_resource(ImageCopiers(copiers));
}

fn image_copy_driver(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
    buffer_toggle: Res<BufferToggle>,
    mut metrics: ResMut<SensorCameraRenderMetrics>,
) {
    let _span = tracing::info_span!("sensor_camera_image_copy_driver").entered();
    let copy_start = Instant::now();

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    let mut any_copies = false;

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

        let buf_idx = buffer_toggle.0.get(i).copied().unwrap_or(0);
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
        any_copies = true;
    }

    if any_copies {
        render_queue.submit(std::iter::once(encoder.finish()));
    }
    metrics.image_copy_driver_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<SensorFrameSender>,
    mut reusable: ResMut<ReusableFrameBuffer>,
    mut buffer_toggle: ResMut<BufferToggle>,
    mut metrics: ResMut<SensorCameraRenderMetrics>,
) {
    let receive_span = tracing::info_span!(
        "sensor_camera_receive_image_from_buffer",
        receive_image_poll_wait_ms = tracing::field::Empty,
        receive_image_from_buffer_ms = tracing::field::Empty,
    );
    let _receive_span = receive_span.enter();
    let cam_count = image_copiers.0.len();
    if buffer_toggle.0.len() < cam_count {
        buffer_toggle.0.resize(cam_count, 0);
    }
    if reusable.0.len() < cam_count {
        reusable.0.resize_with(cam_count, Vec::new);
    }

    metrics.receive_image_poll_wait_ms = 0.0;
    metrics.receive_image_from_buffer_ms = 0.0;
    let receive_start = Instant::now();

    // Phase 1: request async map for all active cameras.
    let mut pending: Vec<(usize, crossbeam_channel::Receiver<()>)> = Vec::new();
    for (i, image_copier) in image_copiers.0.iter().enumerate() {
        if !image_copier.is_active {
            continue;
        }
        let buf_idx = buffer_toggle.0[i];
        let buffer = &image_copier.buffers[buf_idx];
        let buffer_slice = buffer.slice(..);

        let (s, r) = crossbeam_channel::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |result| match result {
            Ok(()) => {
                let _ = s.send(());
            }
            Err(err) => tracing::warn!("Failed to map sensor camera buffer: {err}"),
        });
        pending.push((i, r));
    }

    if pending.is_empty() {
        return;
    }

    // Phase 2: single blocking poll for all pending copies.
    {
        let _span = tracing::info_span!("sensor_camera_poll_wait").entered();
        let poll_start = Instant::now();
        if render_device.poll(PollType::wait_indefinitely()).is_err() {
            for (i, _) in &pending {
                let buf_idx = buffer_toggle.0[*i];
                image_copiers.0[*i].buffers[buf_idx].unmap();
            }
            return;
        }
        metrics.receive_image_poll_wait_ms = poll_start.elapsed().as_secs_f64() * 1000.0;
    }

    // Phase 3: read all mapped buffers.
    for (i, r) in pending {
        let image_copier = &image_copiers.0[i];
        let buf_idx = buffer_toggle.0[i];
        let buffer = &image_copier.buffers[buf_idx];

        if r.recv().is_ok() {
            let buffer_slice = buffer.slice(..);
            let data = buffer_slice.get_mapped_range();
            let width = image_copier.width;
            let height = image_copier.height;
            let row_bytes = width as usize * 4;
            let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);
            let required_len = (height as usize) * row_bytes;

            reusable.0[i].resize(required_len, 0);
            let buf = &mut reusable.0[i][..required_len];
            if row_bytes == aligned_row_bytes {
                let copy_len = required_len.min(data.len());
                buf[..copy_len].copy_from_slice(&data[..copy_len]);
            } else {
                for (row_idx, chunk) in data
                    .chunks(aligned_row_bytes)
                    .take(height as usize)
                    .enumerate()
                {
                    let len = row_bytes.min(chunk.len());
                    let start = row_idx * row_bytes;
                    if start + len <= buf.len() {
                        buf[start..start + len].copy_from_slice(&chunk[..len]);
                    }
                }
            }

            drop(data);
            buffer.unmap();

            let _ = sender.0.send((
                image_copier.camera_name.clone(),
                std::mem::take(&mut reusable.0[i]),
                width,
                height,
            ));
        } else {
            buffer.unmap();
        }

        buffer_toggle.0[i] = 1 - buf_idx;
    }

    metrics.receive_image_from_buffer_ms = receive_start.elapsed().as_secs_f64() * 1000.0;
    receive_span.record(
        "receive_image_poll_wait_ms",
        metrics.receive_image_poll_wait_ms,
    );
    receive_span.record(
        "receive_image_from_buffer_ms",
        metrics.receive_image_from_buffer_ms,
    );
}

// ---------------------------------------------------------------------------
// Patch sensor_view panel dimensions once configs arrive
// ---------------------------------------------------------------------------

/// The sensor_view panels may be spawned before SensorCameraConfigs are loaded
/// from the DB. This system patches their `raw_rgba_dims` once configs arrive.
pub fn patch_sensor_view_dims(
    configs: Res<SensorCameraConfigs>,
    mut streams: Query<&mut crate::ui::video_stream::VideoStream>,
) {
    if configs.0.is_empty() {
        return;
    }
    for mut stream in streams.iter_mut() {
        if stream.raw_rgba_dims.is_some() {
            continue;
        }
        // Only set dims when msg_name matches a sensor camera config; H.264 video_stream names (e.g. obs_stream) never match.
        if let Some(config) = configs.0.iter().find(|c| c.camera_name == stream.msg_name) {
            stream.raw_rgba_dims = Some((config.width, config.height));
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

/// Feed post-AGC output percentiles back into the next LWIR frame's level/span.
pub fn update_auto_agc(world: &mut World, frames: &[(String, Vec<u8>)]) {
    let configs = world.resource::<SensorCameraConfigs>();
    let updates: Vec<_> = frames
        .iter()
        .filter_map(|(name, bytes)| {
            let config_index = configs
                .0
                .iter()
                .position(|config| &config.camera_name == name)?;
            let config = &configs.0[config_index];
            if config.effect != "lwir"
                || config.effect_param_str(&["agc", "mode"], "manual") != "auto"
                || bytes.len() < 4
            {
                return None;
            }
            let mut histogram = [0u32; 256];
            let black_hot = config.effect_param_str(&["palette"], "white_hot") == "black_hot";
            for pixel in bytes.as_chunks::<4>().0 {
                let value = if black_hot { 255 - pixel[0] } else { pixel[0] };
                if value >= 2 {
                    histogram[value as usize] += 1;
                }
            }
            let low_fraction = config.effect_param_f32(&["agc", "low"], 0.01);
            let high_fraction = config.effect_param_f32(&["agc", "high"], 0.99);
            Some((
                config_index,
                percentile_bin(&histogram, low_fraction),
                percentile_bin(&histogram, high_fraction),
                percentile_bin(&histogram, 0.5),
                config.effect_param_f32(&["agc", "target_median"], 0.35),
            ))
        })
        .collect();
    if updates.is_empty() {
        return;
    }

    let mut cameras = world.query::<(&SensorCamera, &mut SensorOutputSettings)>();
    for (camera, mut settings) in cameras.iter_mut(world) {
        let Some((_, low, high, median, target_median)) = updates
            .iter()
            .find(|(config_index, ..)| *config_index == camera.config_index)
        else {
            continue;
        };
        let saturated_highlights = *high >= 254;
        if high.saturating_sub(*low) < 4 && !saturated_highlights {
            continue;
        }
        let old_min = settings.agc.x;
        let old_max = settings.agc.y;
        let span = (old_max - old_min).max(1.0e-3);
        let candidate_min = if saturated_highlights {
            old_min
        } else {
            old_min + f32::from(*low) / 255.0 * span
        };
        let candidate_max = if saturated_highlights {
            old_max + span * 0.1
        } else {
            old_min + f32::from(*high) / 255.0 * span
        };
        if candidate_max - candidate_min < 0.5 {
            continue;
        }
        let history = settings.agc.z.clamp(0.0, 0.999);
        settings.agc.x = old_min * history + candidate_min * (1.0 - history);
        settings.agc.y = old_max * history + candidate_max * (1.0 - history);
        let observed_median = f32::from(*median) / 255.0;
        if (0.01..0.99).contains(&observed_median) {
            let old_gamma = settings.legacy.z.max(0.1);
            let candidate_gamma = (old_gamma * (*target_median).clamp(0.05, 0.95).ln()
                / observed_median.ln())
            .clamp(0.1, 64.0);
            settings.legacy.z = old_gamma * history + candidate_gamma * (1.0 - history);
        }
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
    }

    #[test]
    fn auto_agc_converges_toward_observed_span() {
        let mut config = camera_config([0.0; 3]);
        config.effect = "lwir".into();
        config.camera_name = "vehicle.ir".into();
        config.effect_params = serde_json::json!({
            "agc": {"mode": "auto", "low": 0.01, "high": 0.99, "smoothing": 0.0}
        });
        let settings = sensor_output_settings(&config);
        let mut world = World::new();
        world.insert_resource(SensorCameraConfigs(vec![config]));
        let camera = world
            .spawn((SensorCamera { config_index: 0 }, settings))
            .id();
        let frame: Vec<u8> = (0..256)
            .flat_map(|value| [value as u8, value as u8, value as u8, 255])
            .collect();
        update_auto_agc(&mut world, &[("vehicle.ir".into(), frame)]);
        let settings = world.get::<SensorOutputSettings>(camera).unwrap();
        assert!(settings.agc.x > 20.0);
        assert!(settings.agc.y < 60.0);
        assert!(settings.agc.y > settings.agc.x);
    }

    #[test]
    fn auto_agc_recovers_from_saturated_highlights() {
        let mut config = camera_config([0.0; 3]);
        config.effect = "lwir".into();
        config.camera_name = "vehicle.ir".into();
        config.effect_params = serde_json::json!({
            "agc": {"mode": "auto", "low": 0.01, "high": 0.99, "smoothing": 0.0}
        });
        let settings = sensor_output_settings(&config);
        let mut world = World::new();
        world.insert_resource(SensorCameraConfigs(vec![config]));
        let camera = world
            .spawn((SensorCamera { config_index: 0 }, settings))
            .id();
        let frame = [255, 255, 255, 255].repeat(256);
        update_auto_agc(&mut world, &[("vehicle.ir".into(), frame)]);
        let settings = world.get::<SensorOutputSettings>(camera).unwrap();
        assert_eq!(settings.agc.x, 20.0);
        assert!(settings.agc.y > 60.0);
    }
}
