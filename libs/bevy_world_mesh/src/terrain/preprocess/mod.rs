use crate::terrain::{
    formats::tiff::TiffLoader,
    preprocess::{
        gpu_preprocessor::{
            create_downsample_layout, create_split_layout, create_stitch_layout, GpuPreprocessor,
        },
        preprocessor::{preprocessor_load_tile, select_ready_tasks, PreprocessTaskType},
    },
    shaders::{load_preprocess_shaders, DOWNSAMPLE_SHADER, SPLIT_SHADER, STITCH_SHADER},
    terrain::TerrainComponents,
    terrain_data::gpu_tile_atlas::{create_attachment_layout, GpuTileAtlas},
};
use bevy::{
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems},
        Render, RenderApp, RenderSystems,
    },
};

pub mod gpu_preprocessor;
pub mod preprocessor;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct TerrainPreprocessPipelineKey: u32 {
        const NONE       = 1 << 0;
        const SPLIT      = 1 << 1;
        const STITCH     = 1 << 2;
        const DOWNSAMPLE = 1 << 3;
    }
}

pub(crate) struct TerrainPreprocessItem {
    split_pipeline: CachedComputePipelineId,
    stitch_pipeline: CachedComputePipelineId,
    downsample_pipeline: CachedComputePipelineId,
}

impl TerrainPreprocessItem {
    fn pipelines<'a>(
        &'a self,
        pipeline_cache: &'a PipelineCache,
    ) -> Option<(
        &'a ComputePipeline,
        &'a ComputePipeline,
        &'a ComputePipeline,
    )> {
        Some((
            pipeline_cache.get_compute_pipeline(self.split_pipeline)?,
            pipeline_cache.get_compute_pipeline(self.stitch_pipeline)?,
            pipeline_cache.get_compute_pipeline(self.downsample_pipeline)?,
        ))
    }

    pub(crate) fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipelines(pipeline_cache).is_some()
    }
}

#[derive(Resource)]
pub struct TerrainPreprocessPipelines {
    attachment_layout: BindGroupLayoutDescriptor,
    split_layout: BindGroupLayoutDescriptor,
    stitch_layout: BindGroupLayoutDescriptor,
    downsample_layout: BindGroupLayoutDescriptor,
    split_shader: Handle<Shader>,
    stitch_shader: Handle<Shader>,
    downsample_shader: Handle<Shader>,
}

impl FromWorld for TerrainPreprocessPipelines {
    fn from_world(world: &mut World) -> Self {
        let _device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();

        let attachment_layout = create_attachment_layout();
        let split_layout = create_split_layout();
        let stitch_layout = create_stitch_layout();
        let downsample_layout = create_downsample_layout();

        let split_shader = asset_server.load(SPLIT_SHADER);
        let stitch_shader = asset_server.load(STITCH_SHADER);
        let downsample_shader = asset_server.load(DOWNSAMPLE_SHADER);

        Self {
            attachment_layout,
            split_layout,
            stitch_layout,
            downsample_layout,
            split_shader,
            stitch_shader,
            downsample_shader,
        }
    }
}

impl SpecializedComputePipeline for TerrainPreprocessPipelines {
    type Key = TerrainPreprocessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut layout = default();
        let mut shader = default();
        let mut entry_point = default();

        let shader_defs = vec![];

        if key.contains(TerrainPreprocessPipelineKey::SPLIT) {
            layout = vec![self.attachment_layout.clone(), self.split_layout.clone()];
            shader = self.split_shader.clone();
            entry_point = Some("split".into());
        }
        if key.contains(TerrainPreprocessPipelineKey::STITCH) {
            layout = vec![self.attachment_layout.clone(), self.stitch_layout.clone()];
            shader = self.stitch_shader.clone();
            entry_point = Some("stitch".into());
        }
        if key.contains(TerrainPreprocessPipelineKey::DOWNSAMPLE) {
            layout = vec![
                self.attachment_layout.clone(),
                self.downsample_layout.clone(),
            ];
            shader = self.downsample_shader.clone();
            entry_point = Some("downsample".into());
        }

        ComputePipelineDescriptor {
            label: Some("terrain_preprocess_pipeline".into()),
            layout,
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
            shader,
            shader_defs,
            entry_point,
        }
    }
}

/// Encodes the terrain preprocessing compute passes. Runs in the root
/// [`bevy::render::renderer::RenderGraph`] schedule before
/// [`camera_driver`] so the tile atlas is updated before any camera samples
/// it (previously a render-graph node with an edge to `CameraDriverLabel`).
pub(crate) fn dispatch_terrain_preprocess(
    preprocess_items: Res<TerrainComponents<TerrainPreprocessItem>>,
    pipeline_cache: Res<PipelineCache>,
    preprocess_data: Res<TerrainComponents<GpuPreprocessor>>,
    gpu_tile_atlases: Res<TerrainComponents<GpuTileAtlas>>,
    mut context: RenderContext,
) {
    let command_encoder = context.command_encoder();

    for (&terrain, preprocess_item) in preprocess_items.iter() {
        let Some((split_pipeline, stitch_pipeline, downsample_pipeline)) =
            preprocess_item.pipelines(&pipeline_cache)
        else {
            continue;
        };

        let preprocess_data = preprocess_data.get(&terrain).unwrap();
        let gpu_tile_atlas = gpu_tile_atlases.get(&terrain).unwrap();

        for attachment in &gpu_tile_atlas.attachments {
            attachment.copy_tiles_to_write_section(command_encoder);
        }

        if !preprocess_data.processing_tasks.is_empty() {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&ComputePassDescriptor::default());

            for task in &preprocess_data.processing_tasks {
                let attachment =
                    &gpu_tile_atlas.attachments[task.task.tile.attachment_index as usize];

                let pipeline = match task.task.task_type {
                    PreprocessTaskType::Split { .. } => split_pipeline,
                    PreprocessTaskType::Stitch { .. } => stitch_pipeline,
                    PreprocessTaskType::Downsample { .. } => downsample_pipeline,
                    _ => continue,
                };

                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &attachment.bind_group, &[]);
                compute_pass.set_bind_group(1, task.bind_group.as_ref().unwrap(), &[]);
                compute_pass.dispatch_workgroups(
                    attachment.buffer_info.workgroup_count.x,
                    attachment.buffer_info.workgroup_count.y,
                    attachment.buffer_info.workgroup_count.z,
                );
            }
        }

        for attachment in &gpu_tile_atlas.attachments {
            attachment.copy_tiles_from_write_section(command_encoder);

            attachment.download_tiles(command_encoder);
        }
    }
}

pub(crate) fn queue_terrain_preprocess(
    pipeline_cache: Res<PipelineCache>,
    preprocess_pipelines: ResMut<TerrainPreprocessPipelines>,
    mut pipelines: ResMut<SpecializedComputePipelines<TerrainPreprocessPipelines>>,
    mut preprocess_items: ResMut<TerrainComponents<TerrainPreprocessItem>>,
    gpu_tile_atlas: Res<TerrainComponents<GpuTileAtlas>>,
) {
    for &terrain in gpu_tile_atlas.keys() {
        let split_pipeline = pipelines.specialize(
            &pipeline_cache,
            &preprocess_pipelines,
            TerrainPreprocessPipelineKey::SPLIT,
        );
        let stitch_pipeline = pipelines.specialize(
            &pipeline_cache,
            &preprocess_pipelines,
            TerrainPreprocessPipelineKey::STITCH,
        );
        let downsample_pipeline = pipelines.specialize(
            &pipeline_cache,
            &preprocess_pipelines,
            TerrainPreprocessPipelineKey::DOWNSAMPLE,
        );

        preprocess_items.insert(
            terrain,
            TerrainPreprocessItem {
                split_pipeline,
                stitch_pipeline,
                downsample_pipeline,
            },
        );
    }
}

pub struct TerrainPreprocessPlugin;

impl Plugin for TerrainPreprocessPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<TiffLoader>()
            .add_systems(Update, (select_ready_tasks, preprocessor_load_tile));

        app.sub_app_mut(RenderApp)
            .init_resource::<TerrainComponents<GpuPreprocessor>>()
            .init_resource::<TerrainComponents<TerrainPreprocessItem>>()
            .add_systems(
                ExtractSchedule,
                (
                    GpuPreprocessor::initialize,
                    GpuPreprocessor::extract.after(GpuPreprocessor::initialize),
                ),
            )
            .add_systems(
                Render,
                (
                    queue_terrain_preprocess.in_set(RenderSystems::Queue),
                    GpuPreprocessor::prepare
                        .in_set(RenderSystems::PrepareAssets)
                        .before(GpuTileAtlas::prepare),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        load_preprocess_shaders(app);

        app.sub_app_mut(RenderApp)
            .init_resource::<SpecializedComputePipelines<TerrainPreprocessPipelines>>()
            .init_resource::<TerrainPreprocessPipelines>()
            // Update the tile atlas before any camera samples it (previously
            // a render-graph node edge to `CameraDriverLabel`).
            .add_systems(
                RenderGraph,
                dispatch_terrain_preprocess
                    .in_set(RenderGraphSystems::Render)
                    .before(camera_driver),
            );
    }
}
