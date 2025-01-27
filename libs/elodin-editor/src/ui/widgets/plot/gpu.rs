use bevy::asset::{AssetApp, Assets};
use bevy::color::ColorToComponents;
use bevy::core_pipeline::core_2d::CORE_2D_DEPTH_FORMAT;
use bevy::ecs::bundle::Bundle;
use bevy::ecs::entity::Entity;
use bevy::ecs::schedule::{IntoSystemConfigs, IntoSystemSetConfigs, SystemSet};
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
    asset::{load_internal_asset, Handle},
    core_pipeline::core_2d::Transparent2d,
    ecs::{
        component::Component,
        system::{
            lifetimeless::{Read, SRes},
            Resource,
        },
        world::FromWorld,
    },
    prelude::Color,
    render::{
        extract_component::UniformComponentPlugin,
        render_phase::{AddRenderCommand, PhaseItem, RenderCommand},
        render_resource::{binding_types::uniform_buffer, *},
        renderer::RenderDevice,
        view::ViewTarget,
        RenderApp,
    },
};
use bevy_render::extract_component::ExtractComponent;
use bevy_render::sync_world::{MainEntity, SyncToRenderWorld, TemporaryRenderEntity};
use std::mem;

use crate::ui::widgets::plot::Line;

const LINE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(175745314079092880743018103868034362817);

#[derive(SystemSet, Clone, Debug, Hash, PartialEq, Eq)]
pub enum PlotSystem {
    QueueLine,
}

pub struct PlotGpuPlugin;

impl Plugin for PlotGpuPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(UniformComponentPlugin::<LineUniform>::default())
            //.add_plugins(ExtractComponentPlugin::<LineHandle>::default())
            //.add_plugins(ExtractComponentPlugin::<LineUniform>::default())
            //.add_plugins(ExtractComponentPlugin::<LineConfig>::default())
            //.add_plugins(RenderAssetPlugin::<LineData>)
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
        let line_layout = render_device.create_bind_group_layout("LineUniform layout", &single);

        render_app.insert_resource(UniformLayout {
            layout: line_layout,
        });
        render_app.init_resource::<LinePipeline>();
    }
}

#[derive(Component, Deref, Debug, Clone, ExtractComponent)]
#[require(SyncToRenderWorld)]
pub struct LineHandle(pub Handle<Line>);

#[derive(Bundle)]
pub struct LineBundle {
    pub line: LineHandle,
    pub uniform: LineUniform,
    pub config: LineConfig,
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
}

impl FromWorld for LinePipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        Self {
            mesh_pipeline: world.resource::<Mesh2dPipeline>().clone(),
            uniform_layout: world.resource::<UniformLayout>().layout.clone(),
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

        let layout = vec![view_layout, self.uniform_layout.clone()];

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
    let pos_a_layout = VertexBufferLayout {
        array_stride: VertexFormat::Float32.size(),
        step_mode: VertexStepMode::Instance,
        attributes: vec![VertexAttribute {
            format: VertexFormat::Float32,
            offset: 0,
            shader_location: 0,
        }],
    };
    let pos_b_layout = VertexBufferLayout {
        array_stride: VertexFormat::Float32.size(),
        step_mode: VertexStepMode::Instance,
        attributes: vec![VertexAttribute {
            format: VertexFormat::Float32,
            offset: 0,
            shader_location: 1,
        }],
    };

    vec![pos_a_layout, pos_b_layout]
}

#[derive(Component, Clone, ExtractComponent)]
pub struct LineConfig {
    pub render_layers: RenderLayers,
}

#[derive(Clone, Component, ExtractComponent)]
pub struct GpuLine {
    position_buffer: Buffer,
    position_count: u32,
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
            println!("no gpu line");
            return RenderCommandResult::Failure("no gpu line");
        };
        pass.set_vertex_buffer(0, gpu_line.position_buffer.slice(..));
        pass.set_vertex_buffer(1, gpu_line.position_buffer.slice(4..));
        let instances = u32::max(gpu_line.position_count, 1) - 1;
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
    Option<&'static mut GpuLine>,
);

fn extract_lines(
    mut main_world: ResMut<MainWorld>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let mut state = SystemState::<(
        Query<'static, 'static, LineQueryMut>,
        ResMut<'static, Assets<Line>>,
        Commands<'static, 'static>,
    )>::new(&mut main_world);
    let (mut lines, mut line_assets, mut main_commands) = state.get_mut(&mut main_world);
    for (entity, line_handle, config, mut uniform, gpu_line) in lines.iter_mut() {
        let Some(line) = line_assets.get_mut(&line_handle.0) else {
            continue;
        };
        let line = &mut line.data;
        let (range, uniform, gpu_line) = if let Some(mut gpu_line) = gpu_line {
            gpu_line.position_count = line.current_range.len() as u32;
            let range = line
                .invalidated_range
                .take()
                .unwrap_or_else(|| line.current_range.clone());
            uniform.chunk_size = 1.0;
            (range, *uniform, gpu_line.clone())
        } else {
            let buffer_descriptor = BufferDescriptor {
                label: Some("Line Vertex Buffer"),
                size: (mem::size_of::<f32>() * 2_000_000) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            };
            let buffer = render_device.create_buffer(&buffer_descriptor);
            let gpu_line = GpuLine {
                position_buffer: buffer,
                position_count: line.current_range.len() as u32,
            };
            let mut uniform = *uniform;
            uniform.chunk_size = 1.0;
            main_commands.entity(entity).insert(gpu_line.clone());
            (line.current_range.clone(), uniform, gpu_line)
        };

        let buffer = gpu_line.position_buffer.clone();
        commands.spawn((
            MainEntity::from(entity),
            LineBundle {
                line: line_handle.clone(),
                config: config.clone(),
                uniform,
            },
            gpu_line,
            TemporaryRenderEntity,
        ));

        for chunk in line.data.chunks_range(range.clone()) {
            let data_index = range.start.saturating_sub(chunk.range.start);
            let buffer_index = if range.start > chunk.range.start {
                range.start
            } else {
                chunk.range.start
            } - line.current_range.start;
            let buffer_index = buffer_index * size_of::<f32>();
            let len = (range.end.saturating_sub(chunk.range.start)).min(chunk.data.len());
            let data = bytemuck::cast_slice(&chunk.data[data_index..len]);
            render_queue
                .0
                .write_buffer(&buffer, buffer_index as u64, data);
        }
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
    mut views: Query<(Entity, &ExtractedView, &Msaa, Option<&RenderLayers>)>,
) {
    let draw_function = draw_functions.read().get_id::<DrawLine2d>().unwrap();

    for (view_entity, view, msaa, render_layers) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
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
                extra_index: PhaseItemExtraIndex::NONE,
                sort_key: FloatOrd(0.0),
            });
        }
    }
}
