use bevy::asset::{AssetApp, Assets, weak_handle};
use bevy::color::ColorToComponents;
use bevy::core_pipeline::core_2d::CORE_2D_DEPTH_FORMAT;
use bevy::ecs::bundle::Bundle;
use bevy::ecs::entity::Entity;
use bevy::ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy::ecs::system::{Commands, Query, Res, ResMut, SystemState};
use bevy::math::{FloatOrd, Vec4};
use bevy::prelude::Deref;
use bevy::render::extract_component::{ComponentUniforms, DynamicUniformIndex};
use bevy::render::render_phase::{
    DrawFunctions, PhaseItemExtraIndex, RenderCommandResult, SetItemPipeline,
    ViewSortedRenderPhases,
};

use bevy::image::BevyDefault;
use bevy::render::renderer::RenderQueue;
use bevy::render::view::{ExtractedView, Msaa, RenderLayers};
use bevy::render::{ExtractSchedule, MainWorld, Render, RenderSet};
use bevy::sprite::{Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dViewBindGroup};
use bevy::{
    app::Plugin,
    asset::{Handle, load_internal_asset},
    core_pipeline::core_2d::Transparent2d,
    ecs::{
        component::Component,
        system::lifetimeless::{Read, SRes},
        world::FromWorld,
    },
    prelude::{Color, Resource},
    render::{
        RenderApp,
        extract_component::UniformComponentPlugin,
        render_phase::{AddRenderCommand, PhaseItem, RenderCommand},
        render_resource::{binding_types::uniform_buffer, *},
        renderer::RenderDevice,
        view::ViewTarget,
    },
};
use bevy_render::extract_component::ExtractComponent;
use bevy_render::sync_world::{MainEntity, SyncToRenderWorld, TemporaryRenderEntity};
use binding_types::storage_buffer_read_only_sized;
use impeller2::types::Timestamp;
use std::num::NonZeroU64;
use std::ops::Range;

use crate::ui::widgets::plot::{CHUNK_COUNT, CHUNK_LEN, Line};

const LINE_SHADER_HANDLE: Handle<Shader> = weak_handle!("e44f3b60-cb86-42a2-b7d8-d8dbf1f0299a");

pub const VALUE_BUFFER_SIZE: NonZeroU64 =
    NonZeroU64::new((CHUNK_COUNT * CHUNK_LEN * size_of::<f32>()) as u64).unwrap();
pub const INDEX_BUFFER_LEN: usize = 1024 * 4;
pub const INDEX_BUFFER_SIZE: NonZeroU64 =
    NonZeroU64::new((INDEX_BUFFER_LEN * size_of::<u32>()) as u64).unwrap();

#[derive(SystemSet, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlotSystem {
    QueueLine,
}

pub struct PlotGpuPlugin;

impl Plugin for PlotGpuPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(UniformComponentPlugin::<LineUniform>::default())
            .init_asset::<Line>();

        load_internal_asset!(app, LINE_SHADER_HANDLE, "./line.wgsl", Shader::from_wgsl);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_command::<Transparent2d, DrawLine2d>()
            .init_resource::<SpecializedRenderPipelines<LinePipeline>>()
            .configure_sets(
                Render,
                PlotSystem::QueueLine
                    .in_set(RenderSet::Queue)
                    .ambiguous_with(
                        bevy::pbr::queue_material_meshes::<bevy::pbr::StandardMaterial>,
                    ),
            )
            .add_systems(ExtractSchedule, extract_lines)
            .add_systems(
                Render,
                prepare_uniform_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_systems(
                Render,
                queue_line.in_set(PlotSystem::QueueLine), //.after(prepare_gpu_line),
            );
    }

    fn finish(&self, app: &mut bevy::prelude::App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_device = render_app.world().resource::<RenderDevice>();
        let single = BindGroupLayoutEntries::single(
            ShaderStages::VERTEX,
            uniform_buffer::<LineUniform>(true),
        );
        let uniform_layout = render_device.create_bind_group_layout("LineUniform layout", &single);

        let layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(INDEX_BUFFER_SIZE)),
            ),
        );
        let values_layout =
            render_device.create_bind_group_layout("LineValues layout", &layout_entries);

        render_app.insert_resource(LineValuesLayout {
            layout: values_layout,
        });
        render_app.insert_resource(UniformLayout {
            layout: uniform_layout,
        });
        render_app.init_resource::<LinePipeline>();
    }
}

#[derive(Component, Deref, Debug, Clone, ExtractComponent)]
#[require(SyncToRenderWorld)]
pub struct LineHandle(pub Handle<Line>);

#[derive(Component, Deref, Debug, Clone, ExtractComponent)]
#[require(SyncToRenderWorld)]
pub struct LineVisibleRange(pub Range<Timestamp>);

#[derive(Bundle)]
pub struct LineBundle {
    pub line: LineHandle,
    pub uniform: LineUniform,
    pub config: LineConfig,
    pub line_visible_range: LineVisibleRange,
}

#[derive(Component, ShaderType, Clone, Copy, ExtractComponent)]
pub struct LineUniform {
    pub line_width: f32,
    pub color: Vec4,
    pub chunk_size: f32,
    #[cfg(target_arch = "wasm32")]
    _padding: bevy::math::Vec2,
}

impl LineUniform {
    pub fn new(line_width: f32, color: Color) -> Self {
        Self {
            line_width,
            color: Vec4::from_array(color.to_linear().to_f32_array()),
            chunk_size: 1.0,
            #[cfg(target_arch = "wasm32")]
            _padding: Default::default(),
        }
    }
}

#[derive(Resource)]
struct LineValuesLayout {
    layout: BindGroupLayout,
}

#[derive(Resource)]
struct UniformLayout {
    layout: BindGroupLayout,
}

#[derive(Resource)]
pub struct UniformBindGroup {
    bindgroup: BindGroup,
}

fn prepare_uniform_bind_group(
    mut commands: Commands,
    line_uniform_layout: Res<UniformLayout>,
    render_device: Res<RenderDevice>,
    line_uniforms: Res<ComponentUniforms<LineUniform>>,
) {
    if let Some(binding) = line_uniforms.uniforms().binding() {
        commands.insert_resource(UniformBindGroup {
            bindgroup: render_device.create_bind_group(
                "LineUniform bindgroup",
                &line_uniform_layout.layout,
                &BindGroupEntries::single(binding),
            ),
        });
    }
}

#[derive(Resource)]
pub struct LinePipeline {
    mesh_pipeline: Mesh2dPipeline,
    uniform_layout: BindGroupLayout,
    storage_layout: BindGroupLayout,
}

impl FromWorld for LinePipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        Self {
            mesh_pipeline: world.resource::<Mesh2dPipeline>().clone(),
            uniform_layout: world.resource::<UniformLayout>().layout.clone(),
            storage_layout: world.resource::<LineValuesLayout>().layout.clone(),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct LinePipelineKey {
    view_key: Mesh2dPipelineKey,
}

impl SpecializedRenderPipeline for LinePipeline {
    type Key = LinePipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
    ) -> bevy::render::render_resource::RenderPipelineDescriptor {
        let shader_defs = vec![
            #[cfg(target_arch = "wasm32")]
            "SIXTEEN_BYTE_ALIGNMENT".into(),
        ];

        let view_layout = self.mesh_pipeline.view_layout.clone();

        let layout = vec![
            view_layout,
            self.uniform_layout.clone(),
            self.storage_layout.clone(),
        ];

        let format = if key.view_key.contains(Mesh2dPipelineKey::HDR) {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: LINE_SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: line_vertex_buffer_layouts(),
            },
            fragment: Some(FragmentState {
                shader: LINE_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            layout,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_2D_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState {
                count: key.view_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("Plot Line Pipeline".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

fn line_vertex_buffer_layouts() -> Vec<VertexBufferLayout> {
    vec![]
}

#[derive(Component, Clone, ExtractComponent)]
pub struct LineConfig {
    pub render_layers: RenderLayers,
}

#[derive(Component, Clone, ExtractComponent)]
pub struct LineWidgetWidth(pub usize);

#[derive(Clone, Component, ExtractComponent)]
pub struct GpuLine {
    values_bind_group: BindGroup,
    index_buffer: Buffer,
    count: u32,
}

pub struct SetLineBindGroup;

impl<P: PhaseItem> RenderCommand<P> for SetLineBindGroup {
    type Param = SRes<UniformBindGroup>;
    type ViewQuery = ();
    type ItemQuery = Read<DynamicUniformIndex<LineUniform>>;

    fn render<'w>(
        _item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        uniform_index: Option<bevy::ecs::query::ROQueryItem<'w, Self::ItemQuery>>,
        bind_group: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(uniform_index) = uniform_index else {
            return RenderCommandResult::Failure("no uniform index");
        };
        pass.set_bind_group(
            1,
            &bind_group.into_inner().bindgroup,
            &[uniform_index.index()],
        );
        RenderCommandResult::Success
    }
}

pub struct DrawLine;

impl<P: PhaseItem> RenderCommand<P> for DrawLine {
    type Param = ();

    type ViewQuery = ();

    type ItemQuery = Read<GpuLine>;

    fn render<'w>(
        _item: &P,
        _view: bevy::ecs::query::ROQueryItem<'w, Self::ViewQuery>,
        handle: Option<bevy::ecs::query::ROQueryItem<'w, Self::ItemQuery>>,
        _param: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(gpu_line) = handle else {
            return RenderCommandResult::Failure("no gpu line");
        };
        pass.set_bind_group(2, &gpu_line.values_bind_group, &[]);
        let instances = gpu_line.count.saturating_sub(1);
        pass.draw(0..4, 0..instances);
        RenderCommandResult::Success
    }
}

type DrawLine2d = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetLineBindGroup,
    DrawLine,
);

type LineQueryMut = (
    Entity,
    &'static LineHandle,
    &'static LineConfig,
    &'static mut LineUniform,
    &'static mut LineVisibleRange,
    &'static mut LineWidgetWidth,
    Option<&'static mut GpuLine>,
);

fn extract_lines(
    mut main_world: ResMut<MainWorld>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    values_layout: Res<LineValuesLayout>,
) {
    let mut state = SystemState::<(
        Query<'static, 'static, LineQueryMut>,
        ResMut<'static, Assets<Line>>,
    )>::new(&mut main_world);
    let (mut lines, mut line_assets) = state.get_mut(&mut main_world);
    for (entity, line_handle, config, uniform, line_visible_range, width, gpu_line) in
        lines.iter_mut()
    {
        let Some(line) = line_assets.get_mut(&line_handle.0) else {
            continue;
        };
        let line = &mut line.data;
        line.queue_load_range(line_visible_range.0.clone(), &render_queue, &render_device);
        let index_buffer = if let Some(ref gpu_line) = gpu_line {
            gpu_line.index_buffer.clone()
        } else {
            render_device.create_buffer(
                &(BufferDescriptor {
                    label: Some("Line index Buffer"),
                    size: (INDEX_BUFFER_LEN * size_of::<u32>()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            )
        };

        let values_bind_group = if let Some(ref gpu_line) = gpu_line {
            gpu_line.values_bind_group.clone()
        } else {
            let size = Some(VALUE_BUFFER_SIZE);
            render_device.create_bind_group(
                "line values",
                &values_layout.layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: line
                                .timestamp_buffer_shard_alloc()
                                .expect("no timestamp buf")
                                .buffer(),
                            offset: 0,
                            size,
                        }),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: line
                                .data_buffer_shard_alloc()
                                .expect("no data buf")
                                .buffer(),
                            offset: 0,
                            size,
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &index_buffer,
                            offset: 0,
                            size: Some(INDEX_BUFFER_SIZE),
                        }),
                    },
                ],
            )
        };
        let count = line.write_to_index_buffer(
            &index_buffer,
            &render_queue,
            line_visible_range.0.clone(),
            width.0,
        );
        let gpu_line = GpuLine {
            values_bind_group,
            index_buffer,
            count,
        };

        commands.spawn((
            MainEntity::from(entity),
            LineBundle {
                line: line_handle.clone(),
                config: config.clone(),
                uniform: *uniform,
                line_visible_range: line_visible_range.clone(),
            },
            gpu_line,
            TemporaryRenderEntity,
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_line(
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    pipeline: Res<LinePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<LinePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    lines: Query<(Entity, &MainEntity, &LineConfig)>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent2d>>,
    mut views: Query<(&ExtractedView, &Msaa, Option<&RenderLayers>)>,
) {
    let draw_function = draw_functions.read().get_id::<DrawLine2d>().unwrap();

    for (view, msaa, render_layers) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr);

        let render_layers = render_layers.unwrap_or_default();
        for (entity, main_entity, config) in &lines {
            if !config.render_layers.intersects(render_layers) {
                continue;
            }

            let pipeline = pipelines.specialize(
                &pipeline_cache,
                &pipeline,
                LinePipelineKey { view_key: mesh_key },
            );

            transparent_phase.add(Transparent2d {
                entity: (entity, *main_entity),
                draw_function,
                pipeline,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                sort_key: FloatOrd(0.0),
                extracted_index: 0,
                indexed: true,
            });
        }
    }
}
