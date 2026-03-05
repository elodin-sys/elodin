use bevy::{
    app::{App, Plugin},
    asset::{Assets, embedded_asset},
    camera::RenderTarget,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        tonemapping::Tonemapping,
        FullscreenShader,
    },
    ecs::query::QueryItem,
    image::Image,
    math::{DVec3, Vec3},
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
            binding_types::{sampler, texture_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::ViewTarget,
    },
};
use big_space::GridCell;
use impeller2::types::ComponentId;
use impeller2_wkt::{DbConfig, LastUpdated};
use serde::{Deserialize, Serialize};

use crate::object_3d::ComponentArrayExt;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

fn default_format() -> String {
    "rgba".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorCameraConfig {
    pub entity_name: String,
    pub camera_name: String,
    pub width: u32,
    pub height: u32,
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    pub pos_offset: [f64; 3],
    pub look_at_offset: [f64; 3],
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub effect: String,
    #[serde(default)]
    pub effect_params: std::collections::HashMap<String, f64>,
}

#[derive(Resource, Default, Debug, Clone)]
pub struct SensorCameraConfigs(pub Vec<SensorCameraConfig>);

// ---------------------------------------------------------------------------
// ECS components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct SensorCamera {
    pub config_index: usize,
}

/// GPU post-process settings extracted to the render world.
/// The `effect_type` field selects the shader pipeline:
///   0 = normal (no post-process), 1 = thermal, 2 = night vision, 3 = depth
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
pub struct SensorEffectSettings {
    pub effect_type: u32,
    pub param_a: f32,
    pub param_b: f32,
    pub time: f32,
}

#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    src_image: Handle<Image>,
    camera_name: String,
    width: u32,
    height: u32,
}

#[derive(Clone, Default, Resource)]
struct ImageCopiers(pub Vec<ImageCopier>);

#[derive(Resource)]
pub struct SensorFrameReceiver(pub flume::Receiver<(String, Vec<u8>, u32, u32)>);

#[derive(Resource, Clone)]
struct SensorFrameSender(flume::Sender<(String, Vec<u8>, u32, u32)>);

#[derive(Resource, Default)]
struct SensorCamerasSpawned(bool);

// ---------------------------------------------------------------------------
// Post-process render graph
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SensorPostProcessLabel;

#[derive(Default)]
struct SensorPostProcessNode;

impl ViewNode for SensorPostProcessNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static SensorEffectSettings,
        &'static DynamicUniformIndex<SensorEffectSettings>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, _settings, settings_index): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_res = world.resource::<SensorPostProcessPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_res.pipeline_id) else {
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<SensorEffectSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "sensor_post_process_bind_group",
            &pipeline_res.layout,
            &BindGroupEntries::sequential((
                post_process.source,
                &pipeline_res.sampler,
                settings_binding.clone(),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("sensor_post_process_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[settings_index.index()]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct SensorPostProcessPipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

fn init_sensor_post_process_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let layout = render_device.create_bind_group_layout(
        "sensor_post_process_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<SensorEffectSettings>(true),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let shader =
        asset_server.load("embedded://elodin_editor/assets/shaders/sensor_post_process.wgsl");
    let vertex_state = fullscreen_shader.to_vertex_state();

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("sensor_post_process_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: vertex_state,
        fragment: Some(FragmentState {
            shader,
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });

    commands.insert_resource(SensorPostProcessPipeline {
        layout,
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

        embedded_asset!(app, "assets/shaders/sensor_post_process.wgsl");

        app.init_resource::<SensorCameraConfigs>()
            .init_resource::<SensorCamerasSpawned>()
            .insert_resource(SensorFrameReceiver(rx))
            .add_plugins((
                ExtractComponentPlugin::<SensorEffectSettings>::default(),
                UniformComponentPlugin::<SensorEffectSettings>::default(),
            ))
            .add_systems(PreUpdate, load_sensor_configs_from_db)
            .add_systems(
                PreUpdate,
                spawn_sensor_cameras.run_if(should_spawn_sensor_cameras),
            )
            .add_systems(
                PreUpdate,
                update_sensor_camera_transforms.after(crate::PositionSync),
            )
            .add_systems(Update, tick_effect_time)
            .add_systems(Update, patch_sensor_view_dims);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(SensorFrameSender(tx))
                .add_systems(ExtractSchedule, image_copy_extract)
                .add_systems(
                    Render,
                    (image_copy_driver, receive_image_from_buffer)
                        .chain()
                        .after(RenderSystems::Render),
                )
                .add_systems(RenderStartup, init_sensor_post_process_pipeline)
                .add_render_graph_node::<ViewNodeRunner<SensorPostProcessNode>>(
                    Core3d,
                    SensorPostProcessLabel,
                )
                .add_render_graph_edges(
                    Core3d,
                    (
                        Node3d::Tonemapping,
                        SensorPostProcessLabel,
                        Node3d::EndMainPassPostProcessing,
                    ),
                );
        }
    }
}

// ---------------------------------------------------------------------------
// Main-world systems
// ---------------------------------------------------------------------------

fn load_sensor_configs_from_db(
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
                bevy::log::info!(
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

fn should_spawn_sensor_cameras(
    configs: Res<SensorCameraConfigs>,
    spawned: Res<SensorCamerasSpawned>,
) -> bool {
    !configs.0.is_empty() && !spawned.0
}

fn spawn_sensor_cameras(
    mut commands: Commands,
    configs: Res<SensorCameraConfigs>,
    mut images: ResMut<Assets<Image>>,
    render_device: Res<RenderDevice>,
    mut spawned: ResMut<SensorCamerasSpawned>,
) {
    for (i, config) in configs.0.iter().enumerate() {
        let size = Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let mut render_target_image =
            Image::new_target_texture(size.width, size.height, TextureFormat::bevy_default());
        render_target_image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let render_target_handle = images.add(render_target_image);

        let padded_bytes_per_row =
            RenderDevice::align_copy_bytes_per_row((size.width as usize) * 4);
        let cpu_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("sensor_camera_readback"),
            size: padded_bytes_per_row as u64 * size.height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let copier = ImageCopier {
            buffer: cpu_buffer,
            src_image: render_target_handle.clone(),
            camera_name: config.camera_name.clone(),
            width: config.width,
            height: config.height,
        };

        let perspective = PerspectiveProjection {
            fov: config.fov_degrees.to_radians(),
            near: config.near,
            far: config.far,
            ..default()
        };

        let (effect_type, param_a, param_b) = match config.effect.as_str() {
            "thermal" => (
                1u32,
                *config.effect_params.get("contrast").unwrap_or(&1.5) as f32,
                *config.effect_params.get("noise_sigma").unwrap_or(&0.02) as f32,
            ),
            "night_vision" => (
                2u32,
                *config.effect_params.get("gain").unwrap_or(&2.0) as f32,
                *config.effect_params.get("noise_sigma").unwrap_or(&0.04) as f32,
            ),
            "depth" => (3u32, 0.0, 0.0),
            _ => (0u32, 0.0, 0.0),
        };

        commands.spawn((
            Camera3d::default(),
            Camera {
                target: RenderTarget::Image(render_target_handle.into()),
                order: -(10 + i as isize),
                is_active: false,
                ..default()
            },
            Projection::Perspective(perspective),
            Tonemapping::None,
            Transform::from_xyz(0.0, 5.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
            GlobalTransform::default(),
            GridCell::<i128>::default(),
            SensorCamera { config_index: i },
            SensorEffectSettings {
                effect_type,
                param_a,
                param_b,
                time: 0.0,
            },
            copier,
            Name::new(format!("sensor_camera_{}", config.camera_name)),
        ));

        bevy::log::info!(
            "Spawned sensor camera '{}' ({}x{}, effect={})",
            config.camera_name,
            config.width,
            config.height,
            config.effect,
        );
    }

    spawned.0 = true;
}

fn tick_effect_time(time: Res<Time>, mut query: Query<&mut SensorEffectSettings>) {
    let t = time.elapsed_secs();
    for mut settings in &mut query {
        settings.time = t;
    }
}

fn update_sensor_camera_transforms(
    configs: Res<SensorCameraConfigs>,
    mut sensor_cameras: Query<(&SensorCamera, &mut Transform)>,
    cache: Res<impeller2_bevy::TelemetryCache>,
    last_updated: Res<LastUpdated>,
) {
    let ts = last_updated.0;
    for (sensor_cam, mut transform) in &mut sensor_cameras {
        let Some(config) = configs.0.get(sensor_cam.config_index) else {
            continue;
        };

        let world_pos_id = ComponentId::new(&format!("{}.world_pos", config.entity_name));
        let Some(value) = cache.get_at_or_before(&world_pos_id, ts) else {
            continue;
        };

        let Some(world_pos) = value.as_world_pos() else {
            continue;
        };

        let entity_pos: DVec3 = {
            let [x, y, z] = world_pos.pos.parts().map(nox::Tensor::into_buf);
            DVec3::new(x, y, z)
        };
        let entity_att: bevy::math::DQuat = {
            let [i, j, k, w] = world_pos.att.parts().map(nox::Tensor::into_buf);
            bevy::math::DQuat::from_xyzw(i, j, k, w)
        };

        let offset = DVec3::new(
            config.pos_offset[0],
            config.pos_offset[1],
            config.pos_offset[2],
        );
        let look_at_offset = DVec3::new(
            config.look_at_offset[0],
            config.look_at_offset[1],
            config.look_at_offset[2],
        );

        let rotated_offset = entity_att * offset;
        let cam_pos = entity_pos + rotated_offset;

        let rotated_look_at = entity_att * look_at_offset;
        let look_at_pos = entity_pos + rotated_look_at;

        // Z-up (sim) to Y-up (Bevy) coordinate conversion
        let cam_pos_bevy = Vec3::new(cam_pos.x as f32, cam_pos.z as f32, -cam_pos.y as f32);
        let look_at_bevy = Vec3::new(
            look_at_pos.x as f32,
            look_at_pos.z as f32,
            -look_at_pos.y as f32,
        );

        if cam_pos_bevy.distance(look_at_bevy) > 1e-6 {
            *transform =
                Transform::from_translation(cam_pos_bevy).looking_at(look_at_bevy, Vec3::Y);
        }
    }
}

// ---------------------------------------------------------------------------
// Render-world systems (GPU readback)
// ---------------------------------------------------------------------------

fn image_copy_extract(mut commands: Commands, image_copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(
        image_copiers.iter().cloned().collect::<Vec<ImageCopier>>(),
    ));
}

fn image_copy_driver(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<bevy::render::texture::GpuImage>>,
) {
    for image_copier in image_copiers.0.iter() {
        let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
            continue;
        };

        let mut encoder =
            render_device.create_command_encoder(&CommandEncoderDescriptor::default());

        let block_dimensions = src_image.texture_format.block_dimensions();
        let block_size = src_image.texture_format.block_copy_size(None).unwrap();

        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (src_image.size.width as usize / block_dimensions.0 as usize) * block_size as usize,
        );

        encoder.copy_texture_to_buffer(
            src_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &image_copier.buffer,
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
            src_image.size,
        );

        render_queue.submit(std::iter::once(encoder.finish()));
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<SensorFrameSender>,
) {
    for image_copier in image_copiers.0.iter() {
        let buffer_slice = image_copier.buffer.slice(..);

        let (s, r) = crossbeam_channel::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |result| match result {
            Ok(()) => {
                let _ = s.send(());
            }
            Err(err) => tracing::warn!("Failed to map sensor camera buffer: {err}"),
        });

        if render_device.poll(PollType::wait()).is_err() {
            continue;
        }

        if r.recv().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let width = image_copier.width;
            let height = image_copier.height;
            let row_bytes = width as usize * 4;
            let aligned_row_bytes = RenderDevice::align_copy_bytes_per_row(row_bytes);

            let frame_data: Vec<u8> = if row_bytes == aligned_row_bytes {
                data.to_vec()
            } else {
                data.chunks(aligned_row_bytes)
                    .take(height as usize)
                    .flat_map(|row| &row[..row_bytes.min(row.len())])
                    .cloned()
                    .collect()
            };

            drop(data);
            image_copier.buffer.unmap();

            let _ = sender.0.send((
                image_copier.camera_name.clone(),
                frame_data,
                width,
                height,
            ));
        } else {
            image_copier.buffer.unmap();
        }
    }
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
        if let Some(config) = configs.0.iter().find(|c| c.camera_name == stream.msg_name) {
            stream.raw_rgba_dims = Some((config.width, config.height));
        }
    }
}

