use std::mem;

use crate::{
    ui::widgets::plot::{
        gpu::{LineHandle, INDEX_BUFFER_LEN, INDEX_BUFFER_SIZE, VALUE_BUFFER_SIZE},
        Line,
    },
    SelectedTimeRange,
};
use bevy::{
    app::{Plugin, PostUpdate},
    asset::{load_internal_asset, AssetApp, Assets, Handle},
    color::ColorToComponents,
    core_pipeline::{
        core_3d::{Transparent3d, CORE_3D_DEPTH_FORMAT},
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    ecs::{
        bundle::Bundle,
        component::Component,
        entity::Entity,
        query::Has,
        schedule::{IntoSystemConfigs, IntoSystemSetConfigs, SystemSet},
        system::{
            lifetimeless::{Read, SRes},
            Commands, Query, Res, ResMut, Resource, SystemState,
        },
        world::{FromWorld, Mut, World},
    },
    image::BevyDefault,
    math::{Mat4, Vec4},
    pbr::{MeshPipeline, MeshPipelineKey, SetMeshViewBindGroup},
    prelude::{Color, Deref},
    render::{
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, ViewSortedRenderPhases,
        },
        render_resource::{binding_types::uniform_buffer, *},
        renderer::{RenderDevice, RenderQueue},
        view::{ExtractedView, Msaa, RenderLayers, ViewTarget},
        ExtractSchedule, MainWorld, Render, RenderApp, RenderSet,
    },
    transform::components::{GlobalTransform, Transform},
};
use bevy_render::{
    extract_component::ExtractComponent,
    sync_world::{MainEntity, SyncToRenderWorld, TemporaryRenderEntity},
};
use big_space::GridCell;
use binding_types::storage_buffer_read_only_sized;

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
            .init_asset::<Line>()
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

        let layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
            ),
        );
        let values_layout =
            render_device.create_bind_group_layout("LineValues layout", &layout_entries);

        let index_layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, Some(INDEX_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(INDEX_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(INDEX_BUFFER_SIZE)),
            ),
        );
        let index_layout =
            render_device.create_bind_group_layout("LineIndex layout", &index_layout_entries);

        let line_layout = render_device.create_bind_group_layout("LineUniform Layout", &single);

        render_app.insert_resource(UniformLayout {
            layout: line_layout,
        });

        render_app.insert_resource(LineValuesLayout {
            layout: values_layout,
        });

        render_app.insert_resource(LineIndexLayout {
            layout: index_layout,
        });

        render_app.init_resource::<LinePipeline>();
    }
}

fn update_uniform_model(mut query: Query<(&mut LineUniform, &GlobalTransform)>) {
    for (mut uniform, transform) in query.iter_mut() {
        uniform.model = transform.compute_matrix();
    }
}
#[derive(Component, Debug, Clone, ExtractComponent)]
#[require(SyncToRenderWorld)]
pub struct LineHandles(pub [Handle<Line>; 3]);

#[derive(Bundle)]
pub struct LineBundle {
    pub line: LineHandles,
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
struct LineValuesLayout {
    layout: BindGroupLayout,
}

#[derive(Resource)]
struct LineIndexLayout {
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
    mesh_pipeline: MeshPipeline,
    uniform_layout: BindGroupLayout,
    index_layout: BindGroupLayout,
    values_layout: BindGroupLayout,
}

impl FromWorld for LinePipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            uniform_layout: world.resource::<UniformLayout>().layout.clone(),
            index_layout: world.resource::<LineIndexLayout>().layout.clone(),
            values_layout: world.resource::<LineValuesLayout>().layout.clone(),
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

        let layout = vec![
            view_layout,
            self.uniform_layout.clone(),
            self.values_layout.clone(),
            self.index_layout.clone(),
        ];

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
            zero_initialize_workgroup_memory: false,
        }
    }
}

fn line_vertex_buffer_layouts() -> Vec<VertexBufferLayout> {
    vec![]
}

#[derive(Component, Clone)]
pub struct LineConfig {
    pub render_layers: RenderLayers,
}

#[derive(Clone, Component)]
pub struct GpuLine {
    values_bind_group: BindGroup,
    index_bind_group: BindGroup,
    index_buffers: [Buffer; 3],
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
        pass.set_bind_group(3, &gpu_line.index_bind_group, &[]);
        let instances = gpu_line.count.saturating_sub(1);
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
        ResMut<'static, Assets<Line>>,
        Commands<'static, 'static>,
        Res<'static, SelectedTimeRange>,
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
    &'static LineHandles,
    &'static LineConfig,
    &'static mut LineUniform,
    Option<&'static mut GpuLine>,
);

fn extract_lines(
    mut main_world: ResMut<MainWorld>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    values_layout: Res<LineValuesLayout>,
    index_layout: Res<LineIndexLayout>,
) {
    main_world.resource_scope(|world, mut cached_state: Mut<CachedSystemState>| {
        let (mut lines, mut line_assets, mut main_commands, selected_time_range) =
            cached_state.state.get_mut(world);
        'outer: for (entity, line_handles, config, uniform, gpu_line) in lines.iter_mut() {
            for line in &line_handles.0 {
                let Some(line) = line_assets.get_mut(line) else {
                    continue 'outer;
                };
                line.data.queue_load_range(
                    selected_time_range.0.clone(),
                    &render_queue,
                    &render_device,
                );
            }

            let index_buffers = if let Some(ref gpu_line) = gpu_line {
                gpu_line.index_buffers.clone()
            } else {
                ['x', 'y', 'z'].map(|axis| {
                    render_device.create_buffer(
                        &(BufferDescriptor {
                            label: Some(&format!("Line {} Index Buffer", axis)),
                            size: (INDEX_BUFFER_LEN * size_of::<u32>()) as u64,
                            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    )
                })
            };

            let values_bind_group = if let Some(ref gpu_line) = gpu_line {
                gpu_line.values_bind_group.clone()
            } else {
                let entries = [0, 1, 2].map(|i| {
                    let line = &line_handles.0[i];
                    let line = line_assets.get(line).expect("line missing");
                    BindGroupEntry {
                        binding: i as u32,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: line
                                .data
                                .data_buffer_shard_alloc()
                                .expect("no data buf")
                                .buffer(),
                            offset: 0,
                            size: Some(VALUE_BUFFER_SIZE),
                        }),
                    }
                });
                render_device.create_bind_group("line values", &values_layout.layout, &entries)
            };

            let index_bind_group = if let Some(ref gpu_line) = gpu_line {
                gpu_line.index_bind_group.clone()
            } else {
                let entries = [0, 1, 2].map(|i| {
                    let buffer = &index_buffers[i];
                    let entry = BindGroupEntry {
                        binding: i as u32,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &buffer,
                            offset: 0,
                            size: Some(INDEX_BUFFER_SIZE),
                        }),
                    };

                    entry
                });
                render_device.create_bind_group("line indexes", &index_layout.layout, &entries)
            };
            let counts = [0, 1, 2].map(|i| {
                let line = &line_handles.0[i];
                let line = line_assets.get(line).expect("line missing");
                let index_buffer = &index_buffers[i];
                line.data.write_to_index_buffer(
                    index_buffer,
                    &render_queue,
                    selected_time_range.0.clone(),
                    INDEX_BUFFER_LEN,
                )
            });
            let count = counts.into_iter().min().unwrap_or_default();
            let gpu_line = GpuLine {
                values_bind_group,
                index_bind_group,
                index_buffers,
                count,
            };

            commands.spawn((
                MainEntity::from(entity),
                LineBundle {
                    line: line_handles.clone(),
                    config: config.clone(),
                    uniform: *uniform,
                    global_transform: GlobalTransform::default(),
                    transform: Transform::default(),
                    grid_cell: GridCell::default(),
                },
                gpu_line,
                TemporaryRenderEntity,
            ));
        }
        cached_state.state.apply(world)
    })
}

type ViewQuery = (
    Entity,
    &'static ExtractedView,
    &'static Msaa,
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
    lines: Query<(Entity, &MainEntity, &LineHandles, &LineConfig)>,
    mut views: Query<ViewQuery>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
) {
    let draw_function = draw_functions.read().get_id::<DrawLineData>().unwrap();

    for (
        view_entity,
        view,
        msaa,
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

        for (entity, main_entity, _handle, config) in &lines {
            if !config.render_layers.intersects(&render_layers) {
                continue;
            }

            let pipeline =
                pipelines.specialize(&pipeline_cache, &pipeline, LinePipelineKey { view_key });

            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                draw_function,
                pipeline,
                distance: 0.,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
            });
        }
    }
}
