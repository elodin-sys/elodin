use std::mem;

use bevy::app::PostUpdate;
use bevy::asset::{AssetApp, Assets};
use bevy::color::ColorToComponents;
use bevy::core_pipeline::prepass::{
    DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
};
use bevy::ecs::bundle::Bundle;
use bevy::ecs::entity::Entity;
use bevy::ecs::query::Has;
use bevy::ecs::schedule::{IntoSystemConfigs, IntoSystemSetConfigs, SystemSet};
use bevy::ecs::system::{Commands, Query, Res, ResMut, SystemState};
use bevy::ecs::world::{Mut, World};
use bevy::math::{Mat4, Vec4};
use bevy::pbr::SetMeshViewBindGroup;
use bevy::render::extract_component::{ComponentUniforms, DynamicUniformIndex};
use bevy::render::render_phase::{
    DrawFunctions, PhaseItemExtraIndex, RenderCommandResult, SetItemPipeline,
    ViewSortedRenderPhases,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::view::{ExtractedView, Msaa, RenderLayers};
use bevy::render::{ExtractSchedule, MainWorld, Render, RenderSet};
use bevy::transform::components::{GlobalTransform, Transform};
use bevy::{
    app::Plugin,
    asset::{load_internal_asset, Handle},
    core_pipeline::core_3d::{Transparent3d, CORE_3D_DEPTH_FORMAT},
    ecs::{
        component::Component,
        system::{
            lifetimeless::{Read, SRes},
            Resource,
        },
        world::FromWorld,
    },
    pbr::{MeshPipeline, MeshPipelineKey},
    prelude::Color,
    render::{
        extract_component::UniformComponentPlugin,
        render_phase::{AddRenderCommand, PhaseItem, RenderCommand},
        render_resource::{binding_types::uniform_buffer, *},
        renderer::RenderDevice,
        texture::BevyDefault,
        view::ViewTarget,
        RenderApp,
    },
};
use big_space::GridCell;

use crate::ui::widgets::plot_3d::data::LineData;

const LINE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(267882706676311365151377673216596804695);

#[derive(SystemSet, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlotSystem {
    QueueLine,
}

pub struct Plot3dGpuPlugin;

impl Plugin for Plot3dGpuPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(UniformComponentPlugin::<LineUniform>::default())
            .init_resource::<CachedSystemState>()
            .init_asset::<LineData>()
            .add_systems(PostUpdate, update_uniform_model);

        load_internal_asset!(app, LINE_SHADER_HANDLE, "./line.wgsl", Shader::from_wgsl);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_command::<Transparent3d, DrawLineData>()
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
        let line_layout = render_device.create_bind_group_layout("LineUniform layout", &single);

        render_app.insert_resource(UniformLayout {
            layout: line_layout,
        });
        render_app.init_resource::<LinePipeline>();
    }
}

fn update_uniform_model(mut query: Query<(&mut LineUniform, &GlobalTransform)>) {
    for (mut uniform, transform) in query.iter_mut() {
        uniform.model = transform.compute_matrix();
    }
}

#[derive(Bundle)]
pub struct LineBundle {
    pub line: Handle<LineData>,
    pub uniform: LineUniform,
    pub config: LineConfig,
    pub global_transform: GlobalTransform,
    pub transform: Transform,
    pub grid_cell: GridCell<i128>,
}

#[derive(Component, ShaderType, Clone, Copy)]
pub struct LineUniform {
    pub line_width: f32,
    pub color: Vec4,
    pub depth_bias: f32,
    pub model: Mat4,
    pub perspective: u32,
    #[cfg(target_arch = "wasm32")]
    pub _padding: f32,
}

impl LineUniform {
    pub fn new(line_width: f32, color: Color) -> Self {
        Self {
            line_width,
            color: Vec4::from_array(color.to_linear().to_f32_array()),
            depth_bias: 0.0,
            model: Mat4::IDENTITY,
            perspective: 0,
            #[cfg(target_arch = "wasm32")]
            _padding: Default::default(),
        }
    }
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
    mesh_pipeline: MeshPipeline,
    uniform_layout: BindGroupLayout,
}

impl FromWorld for LinePipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            uniform_layout: world.resource::<UniformLayout>().layout.clone(),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct LinePipelineKey {
    view_key: MeshPipelineKey,
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

        let view_layout = self
            .mesh_pipeline
            .get_view_layout(key.view_key.into())
            .clone();

        let layout = vec![view_layout, self.uniform_layout.clone()];

        let format = if key.view_key.contains(MeshPipelineKey::HDR) {
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
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.view_key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("Plot Line Pipeline 3d".into()),
            push_constant_ranges: vec![],
        }
    }
}

fn line_vertex_buffer_layouts() -> Vec<VertexBufferLayout> {
    let pos_layout = VertexBufferLayout {
        array_stride: VertexFormat::Float32x3.size(),
        step_mode: VertexStepMode::Instance,
        attributes: vec![
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 4 * 3,
                shader_location: 1,
            },
        ],
    };
    vec![pos_layout]
}

#[derive(Component, Clone)]
pub struct LineConfig {
    pub render_layers: RenderLayers,
}

#[derive(Clone, Component)]
pub struct GpuLine {
    position_buffer: Buffer,
    position_count: u32,
    buffer_len: usize,
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
            return RenderCommandResult::Failure;
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
            return RenderCommandResult::Failure;
        };
        pass.set_vertex_buffer(0, gpu_line.position_buffer.slice(..));
        pass.set_vertex_buffer(1, gpu_line.position_buffer.slice(..));
        let instances = u32::max(gpu_line.position_count, 1) - 1;
        pass.draw(0..6, 0..instances);
        RenderCommandResult::Success
    }
}

type DrawLineData = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetLineBindGroup,
    DrawLine,
);

#[derive(Resource)]
struct CachedSystemState {
    state: SystemState<(
        Query<'static, 'static, LineQueryMut>,
        ResMut<'static, Assets<LineData>>,
        Commands<'static, 'static>,
    )>,
}

impl FromWorld for CachedSystemState {
    fn from_world(world: &mut World) -> Self {
        Self {
            state: SystemState::new(world),
        }
    }
}

type LineQueryMut = (
    Entity,
    &'static Handle<LineData>,
    &'static LineConfig,
    &'static mut LineUniform,
    Option<&'static mut GpuLine>,
);

fn extract_lines(
    mut main_world: ResMut<MainWorld>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    const BUFFER_CHUNK_SIZE: usize = 0x4000;
    main_world.resource_scope(|world, mut cached_state: Mut<CachedSystemState>| {
        let (mut lines, mut line_assets, mut main_commands) = cached_state.state.get_mut(world);
        for (entity, line_handle, config, uniform, gpu_line) in lines.iter_mut() {
            let Some(line) = line_assets.get_mut(line_handle) else {
                continue;
            };
            let buffer_len =
                line.processed_data.len().div_ceil(BUFFER_CHUNK_SIZE) * BUFFER_CHUNK_SIZE;
            let mut new_buffer = |buffer_len: usize| {
                let buffer_descriptor = BufferDescriptor {
                    label: Some("Line Vertex Buffer"),
                    size: (mem::size_of::<f64>() * 3 * buffer_len) as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                };
                let buffer = render_device.create_buffer(&buffer_descriptor);
                let gpu_line = GpuLine {
                    position_buffer: buffer,
                    position_count: line.processed_data.len() as u32,
                    buffer_len,
                };
                main_commands.get_or_spawn(entity).insert(gpu_line.clone());
                (0..line.processed_data.len(), gpu_line)
            };
            let (range, gpu_line) = if let Some(mut gpu_line) = gpu_line {
                if buffer_len != gpu_line.buffer_len {
                    new_buffer(buffer_len)
                } else {
                    gpu_line.position_count = line.processed_data.len() as u32;
                    let range = line
                        .invalidated_range
                        .take()
                        .unwrap_or_else(|| 0..line.processed_data.len());
                    (range, gpu_line.clone())
                }
            } else {
                new_buffer(buffer_len)
            };

            let buffer = gpu_line.position_buffer.clone();
            commands.get_or_spawn(entity).insert((
                LineBundle {
                    line: line_handle.clone(),
                    config: config.clone(),
                    uniform: *uniform,
                    global_transform: GlobalTransform::default(),
                    transform: Transform::default(),
                    grid_cell: GridCell::default(),
                },
                gpu_line,
            ));
            let index = (range.start * std::mem::size_of::<f64>() * 3) as u64;
            let data = &line.processed_data.buf()[range];
            let data = &data[..(data.len().min(buffer_len))];
            let data = bytemuck::cast_slice(data);
            render_queue.0.write_buffer(&buffer, index, data);
        }
        cached_state.state.apply(world)
    })
}

type ViewQuery = (
    Entity,
    &'static ExtractedView,
    //&'static mut ViewSortedRenderPhases<Transparent3d>,
    Option<&'static RenderLayers>,
    (
        Has<NormalPrepass>,
        Has<DepthPrepass>,
        Has<MotionVectorPrepass>,
        Has<DeferredPrepass>,
    ),
);

#[allow(clippy::too_many_arguments)]
fn queue_line(
    draw_functions: Res<DrawFunctions<Transparent3d>>,
    pipeline: Res<LinePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<LinePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    msaa: Res<Msaa>,
    lines: Query<(Entity, &Handle<LineData>, &LineConfig)>,
    mut views: Query<ViewQuery>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
) {
    let draw_function = draw_functions.read().get_id::<DrawLineData>().unwrap();

    for (
        view_entity,
        view,
        //mut transparent_phase,
        render_layers,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
    ) in &mut views
    {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };
        let render_layers = render_layers.cloned().unwrap_or_default();

        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if deferred_prepass {
            view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        }

        for (entity, _handle, config) in &lines {
            if !config.render_layers.intersects(&render_layers) {
                continue;
            }

            let pipeline =
                pipelines.specialize(&pipeline_cache, &pipeline, LinePipelineKey { view_key });

            transparent_phase.add(Transparent3d {
                entity,
                draw_function,
                pipeline,
                distance: 0.,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
            });
        }
    }
}
