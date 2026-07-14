use crate::{
    SelectedTimeRange,
    ui::plot::{
        Line,
        gpu::{INDEX_BUFFER_LEN, INDEX_BUFFER_SIZE, VALUE_BUFFER_SIZE},
    },
    ui::timeline::TimelineSettings,
};
use bevy::camera::visibility::RenderLayers;
use bevy::shader::Shader;
use bevy::{
    app::{Plugin, PostUpdate},
    asset::{AssetApp, Assets, Handle, load_internal_asset, uuid_handle},
    color::ColorToComponents,
    core_pipeline::{
        core_3d::{CORE_3D_DEPTH_FORMAT, Transparent3d},
        prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    },
    ecs::{
        component::Component,
        entity::Entity,
        query::Has,
        schedule::{IntoScheduleConfigs, SystemSet},
        system::{
            Commands, Query, Res, ResMut, SystemState,
            lifetimeless::{Read, SRes},
        },
        world::{FromWorld, Mut, World},
    },
    image::BevyDefault,
    math::{Mat4, Vec4},
    mesh::VertexBufferLayout,
    pbr::{MeshPipeline, MeshPipelineKey, SetMeshViewBindGroup},
    prelude::{Color, Reflect, Resource},
    render::{
        ExtractSchedule, MainWorld, Render, RenderApp, RenderSystems,
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, ViewSortedRenderPhases,
        },
        render_resource::{binding_types::uniform_buffer, *},
        renderer::{RenderDevice, RenderQueue},
        view::{ExtractedView, Msaa, ViewTarget},
    },
    transform::TransformSystems,
    transform::components::{GlobalTransform, Transform},
};
use bevy_render::{
    extract_component::ExtractComponent,
    sync_world::{MainEntity, SyncToRenderWorld, TemporaryRenderEntity},
};
use binding_types::storage_buffer_read_only_sized;
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp, LastUpdated};

const LINE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("bfffa3c4-9401-4b6e-b3ab-3564180352f1");

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
            .add_systems(
                PostUpdate,
                update_uniform_model.after(TransformSystems::Propagate),
            );

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
                    .in_set(RenderSystems::Queue)
                    .ambiguous_with(bevy::pbr::queue_material_meshes),
            )
            .add_systems(ExtractSchedule, extract_lines)
            .add_systems(
                Render,
                prepare_uniform_bind_group.in_set(RenderSystems::PrepareBindGroups),
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
        let uniform_descriptor = BindGroupLayoutDescriptor::new("LineUniform Layout", &single);

        let layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(VALUE_BUFFER_SIZE)),
            ),
        );
        let values_descriptor =
            BindGroupLayoutDescriptor::new("LineValues layout", &layout_entries);
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
        let index_descriptor =
            BindGroupLayoutDescriptor::new("LineIndex layout", &index_layout_entries);
        let index_layout =
            render_device.create_bind_group_layout("LineIndex layout", &index_layout_entries);

        let line_layout = render_device.create_bind_group_layout("LineUniform Layout", &single);

        render_app.insert_resource(UniformLayout {
            layout: line_layout,
            descriptor: uniform_descriptor,
        });

        render_app.insert_resource(LineValuesLayout {
            layout: values_layout,
            descriptor: values_descriptor,
        });

        render_app.insert_resource(LineIndexLayout {
            layout: index_layout,
            descriptor: index_descriptor,
        });

        render_app.init_resource::<LinePipeline>();
    }
}

fn update_uniform_model(mut query: Query<(&mut LineUniform, &GlobalTransform)>) {
    for (mut uniform, transform) in query.iter_mut() {
        uniform.model = transform.to_matrix();
    }
}

#[derive(Component, Debug, Clone, ExtractComponent)]
#[require(SyncToRenderWorld)]
pub struct LineHandles(pub [Handle<Line>; 3]);

/// Default opacity applied to the future (not-yet-played) trail segment when a
/// line does not set its own `future_color`, so the future reads as dimmer than
/// the played segment. A per-line `future_color` overrides this with its own
/// alpha.
pub const DEFAULT_FUTURE_TRAIL_ALPHA: f32 = 0.35;

/// Per-line trail colors resolved from the KDL `color`/`future_color`.
///
/// Each is linear RGBA; `None` falls back to the timeline trail colors. The
/// played and future segments are independent: a line with only `color` set
/// keeps the timeline future color (faded) for its future segment.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct LineTrailColors {
    pub played: Option<Vec4>,
    pub future: Option<Vec4>,
}

impl LineTrailColors {
    /// Resolve the played/future segment colors against the timeline fallbacks.
    ///
    /// - played: explicit `played`, else the timeline played color.
    /// - future: explicit `future` is authoritative (its alpha is the per-line
    ///   opacity, used as-is); otherwise the timeline future color, faded by
    ///   `future_alpha` so the not-yet-played segment reads dimmer. The future
    ///   does not inherit the played color.
    fn resolve(
        &self,
        played_timeline: Vec4,
        future_timeline: Vec4,
        future_alpha: f32,
    ) -> (Vec4, Vec4) {
        let played = self.played.unwrap_or(played_timeline);
        let future = match self.future {
            Some(future) => future,
            None => {
                let mut fallback = future_timeline;
                fallback.w *= future_alpha;
                fallback
            }
        };
        (played, future)
    }
}

/// Linearize a schematic (sRGB) color for the line shader, preserving alpha.
fn wkt_color_linear(color: impeller2_wkt::Color) -> Vec4 {
    Vec4::from_array(
        Color::srgba(color.r, color.g, color.b, color.a)
            .to_linear()
            .to_f32_array(),
    )
}

#[derive(Component, ShaderType, Clone, Copy, Reflect)]
pub struct LineUniform {
    pub line_width: f32,
    pub color: Vec4,
    pub depth_bias: f32,
    pub model: Mat4,
    /// Reference point (raw frame coords) the shader subtracts from every vertex
    /// before the f32 `model` multiply. Kept at zero now that the rebase happens
    /// in f64 at ingestion (`PlotDataComponent::value_offset`), which also
    /// recovers the ~0.5 m the vertices used to lose to the `f32` cast at ECEF
    /// magnitudes. Retained (subtracting zero) so the uniform layout is stable.
    /// `w` is unused. See [`LineFrameOrigin`].
    pub world_origin: Vec4,
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
            world_origin: Vec4::ZERO,
            perspective: 0,
            #[cfg(target_arch = "wasm32")]
            _padding: Default::default(),
        }
    }
}

/// Per-`line_3d` reference point, in the line's frame coordinates, used to
/// rebase the raw vertices in f64 at ingestion (`PlotDataComponent::value_offset`)
/// so ECEF trails keep mm precision instead of the ~0.5 m lost to the `f32` cast.
///
/// Set at spawn from the schematic's `GeoContext` origin so it equals the
/// entity's `GeoPosition` reference: for an ECEF line this is the geo origin in
/// ECEF (~6.4e6 m), for ENU/NED it is zero (those vertices are already
/// launch-relative). Mirroring the `GeoPosition` reference is what keeps the
/// rendered coordinates small without shifting the line. Kept as `f64` so the
/// rebase offset itself is exact.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct LineFrameOrigin(pub bevy::math::DVec3);

#[derive(Resource)]
struct LineValuesLayout {
    layout: BindGroupLayout,
    descriptor: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
struct LineIndexLayout {
    layout: BindGroupLayout,
    descriptor: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
struct UniformLayout {
    layout: BindGroupLayout,
    descriptor: BindGroupLayoutDescriptor,
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
    uniform_layout: BindGroupLayoutDescriptor,
    index_layout: BindGroupLayoutDescriptor,
    values_layout: BindGroupLayoutDescriptor,
}

impl FromWorld for LinePipeline {
    fn from_world(world: &mut bevy::prelude::World) -> Self {
        Self {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            uniform_layout: world.resource::<UniformLayout>().descriptor.clone(),
            index_layout: world.resource::<LineIndexLayout>().descriptor.clone(),
            values_layout: world.resource::<LineValuesLayout>().descriptor.clone(),
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
            .main_layout
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
                entry_point: Some("vertex".into()),
                shader_defs: shader_defs.clone(),
                buffers: line_vertex_buffer_layouts(),
            },
            fragment: Some(FragmentState {
                shader: LINE_SHADER_HANDLE,
                shader_defs,
                entry_point: Some("fragment".into()),
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
    /// Strip indices (decimated by a uniform sampling step to fit
    /// `INDEX_BUFFER_LEN`) backing `index_bind_group`. This is the GPU-side
    /// point reduction, distinct from the upstream Hamann-Chen (1994) in-memory
    /// simplification done on the `LineTree`. Never read directly, but owned
    /// here so the buffers outlive the bind group / draw rather than relying on
    /// wgpu's internal reference counting.
    #[allow(dead_code)]
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
        _view: bevy::ecs::query::ROQueryItem<'w, '_, Self::ViewQuery>,
        uniform_index: Option<bevy::ecs::query::ROQueryItem<'w, '_, Self::ItemQuery>>,
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
        _view: bevy::ecs::query::ROQueryItem<'w, '_, Self::ViewQuery>,
        handle: Option<bevy::ecs::query::ROQueryItem<'w, '_, Self::ItemQuery>>,
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

type ExtractLinesParams = (
    Query<'static, 'static, LineQueryMut>,
    ResMut<'static, Assets<Line>>,
    Commands<'static, 'static>,
    Res<'static, SelectedTimeRange>,
    Res<'static, EarliestTimestamp>,
    Res<'static, LastUpdated>,
    Res<'static, CurrentTimestamp>,
    Res<'static, TimelineSettings>,
    Res<'static, crate::ui::timeline::LatestFollow>,
);

#[derive(Resource)]
struct CachedSystemState {
    state: SystemState<ExtractLinesParams>,
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
    Option<&'static LineTrailColors>,
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
        let replay_mode = world.contains_resource::<crate::ReplayMode>();
        let (
            mut lines,
            mut line_assets,
            mut _main_commands,
            selected_time_range,
            earliest_timestamp,
            latest_timestamp,
            current_timestamp,
            timeline_settings,
            latest_follow,
        ) = cached_state.state.get_mut(world);
        let selected_range = selected_time_range.0.clone();
        let sampling_range = if replay_mode && earliest_timestamp.0 < latest_timestamp.0 {
            earliest_timestamp.0..latest_timestamp.0
        } else {
            selected_range.clone()
        };
        let future_trail_alpha = DEFAULT_FUTURE_TRAIL_ALPHA;
        // Fallback colors for lines without explicit KDL colors. Kept unfaded
        // here; the default future fade is applied only to fallback futures.
        let played_timeline_color = wkt_color_linear(timeline_settings.played_color);
        let future_timeline_color = wkt_color_linear(timeline_settings.future_color);

        // Live-follow mode: the whole trail is "already played", so render
        // everything in the played color (yolk) and skip the future pass
        // entirely. Without this, Table packets racing ahead of
        // LastUpdated put `latest_sample_ts > current_ts` for one frame,
        // the snap-back below manufactures a 1-sample future range, and
        // the white trail overdraws the tail of the yellow trail.
        let live_follow = latest_follow.0;
        let played_range = if live_follow {
            selected_range.clone()
        } else {
            selected_range.start..selected_range.end.min(current_timestamp.0)
        };
        // Future segment must contain >= 2 samples or the shader draws only
        // sentinel(NaN)-to-point instances and nothing shows up. When
        // `current_timestamp` falls between sim ticks (the common case in live
        // streaming), the naive split leaves a single index in the future
        // range, which blinks at the render framerate near the rocket. Snap
        // the split back onto the previous sample boundary instead.
        let split = selected_range.start.max(current_timestamp.0);

        'outer: for (entity, line_handles, config, uniform, trail_colors, gpu_line) in
            lines.iter_mut()
        {
            let (played_color, future_color) = trail_colors.copied().unwrap_or_default().resolve(
                played_timeline_color,
                future_timeline_color,
                future_trail_alpha,
            );
            for line in &line_handles.0 {
                let Some(line) = line_assets.get_mut(line) else {
                    continue 'outer;
                };
                line.data
                    .queue_load_range(selected_range.clone(), &render_queue, &render_device);
            }

            // Replay grows the revealed prefix every frame. If the decimation
            // step is derived from only that prefix, the full trail gets
            // resampled whenever it crosses a threshold, which shows up as
            // flicker. Keep the reveal clipped by CurrentTimestamp, but derive
            // the stride from the fixed recording extent.
            let line_stats = [0, 1, 2].map(|i| {
                let line = &line_handles.0[i];
                let line = line_assets.get(line).expect("line missing");
                line.data.range_index_stats(sampling_range.clone())
            });
            let sampling_chunk_count = line_stats
                .iter()
                .map(|(chunks, _)| *chunks)
                .max()
                .unwrap_or(0);
            let sampling_index_count = line_stats
                .iter()
                .map(|(_, count)| *count)
                .max()
                .unwrap_or(0);
            let sampling_step = crate::ui::plot::data::index_sampling_step(
                sampling_chunk_count,
                sampling_index_count,
                INDEX_BUFFER_LEN,
            );

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

            let build_gpu_line = |range: std::ops::Range<impeller2::types::Timestamp>| {
                if range.start >= range.end {
                    return None;
                }
                let mut step = sampling_step.max(1);
                const MAX_INDEX_U32: u32 = INDEX_BUFFER_LEN as u32;
                for _ in 0..26 {
                    let mut max_needed = 0u32;
                    for i in 0..3 {
                        let line = &line_handles.0[i];
                        let line = line_assets.get(line).expect("line missing");
                        max_needed =
                            max_needed.max(line.data.count_strip_index_u32s(range.clone(), step));
                    }
                    if max_needed <= MAX_INDEX_U32 {
                        break;
                    }
                    step = step.saturating_mul(2).max(2);
                }
                let index_buffers = ['x', 'y', 'z'].map(|axis| {
                    render_device.create_buffer(
                        &(BufferDescriptor {
                            label: Some(&format!("Line {} Index Buffer", axis)),
                            size: (INDEX_BUFFER_LEN * size_of::<u32>()) as u64,
                            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }),
                    )
                });
                let entries = [0, 1, 2].map(|i| BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &index_buffers[i],
                        offset: 0,
                        size: Some(INDEX_BUFFER_SIZE),
                    }),
                });
                let index_bind_group =
                    render_device.create_bind_group("line indexes", &index_layout.layout, &entries);
                let counts = [0, 1, 2].map(|i| {
                    let line = &line_handles.0[i];
                    let line = line_assets.get(line).expect("line missing");
                    line.data.write_to_index_buffer_with_step(
                        &index_buffers[i],
                        &render_queue,
                        range.clone(),
                        step,
                    )
                });
                let count = counts.into_iter().min().unwrap_or_default();
                if count < 2 {
                    return None;
                }
                Some(GpuLine {
                    values_bind_group: values_bind_group.clone(),
                    index_bind_group,
                    index_buffers,
                    count,
                })
            };

            if let Some(gpu_line) = build_gpu_line(played_range.clone()) {
                let mut played_uniform = *uniform;
                played_uniform.color = played_color;
                commands.spawn((
                    MainEntity::from(entity),
                    line_handles.clone(),
                    config.clone(),
                    played_uniform,
                    GlobalTransform::default(),
                    Transform::default(),
                    #[cfg(feature = "big_space")]
                    crate::spatial::GridCell::default(),
                    gpu_line,
                    TemporaryRenderEntity,
                ));
            }

            // Live-follow: played covers everything, nothing is "future".
            // Otherwise snap the start back to the previous sample so the
            // future segment always has >= 2 indices (single-index segments
            // collapse to a NaN draw and blink at framerate).
            let future_range = if live_follow {
                split..split
            } else {
                let future_start = line_assets
                    .get(&line_handles.0[0])
                    .and_then(|l| l.data.last_timestamp_strictly_before(split))
                    .map(|ts| selected_range.start.max(ts))
                    .unwrap_or(split);
                future_start..selected_range.end
            };

            if let Some(gpu_line) = build_gpu_line(future_range.clone()) {
                let mut future_uniform = *uniform;
                future_uniform.color = future_color;
                commands.spawn((
                    MainEntity::from(entity),
                    line_handles.clone(),
                    config.clone(),
                    future_uniform,
                    GlobalTransform::default(),
                    Transform::default(),
                    #[cfg(feature = "big_space")]
                    crate::spatial::GridCell::default(),
                    gpu_line,
                    TemporaryRenderEntity,
                ));
            }
        }
        cached_state.state.apply(world)
    })
}

type ViewQuery = (
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
        view,
        msaa,
        render_layers,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
    ) in &mut views
    {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
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
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAYED_TL: Vec4 = Vec4::new(1.0, 1.0, 0.0, 1.0); // timeline played (yalk-ish)
    const FUTURE_TL: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0); // timeline future (white)
    const G: Vec4 = Vec4::new(0.0, 1.0, 0.0, 1.0); // KDL `color green`
    const W: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0); // KDL `future_color white`
    const ALPHA: f32 = 0.5;

    fn resolve(trail: LineTrailColors) -> (Vec4, Vec4) {
        trail.resolve(PLAYED_TL, FUTURE_TL, ALPHA)
    }

    fn faded(mut c: Vec4) -> Vec4 {
        c.w *= ALPHA;
        c
    }

    #[test]
    fn no_kdl_colors_use_timeline() {
        let (played, future) = resolve(LineTrailColors::default());
        assert_eq!(played, PLAYED_TL);
        assert_eq!(future, faded(FUTURE_TL));
    }

    #[test]
    fn color_only_keeps_timeline_future() {
        // A lone played color leaves the future on the timeline future color
        // (faded); the future does not inherit the played color.
        let (played, future) = resolve(LineTrailColors {
            played: Some(G),
            future: None,
        });
        assert_eq!(played, G);
        assert_eq!(future, faded(FUTURE_TL));
    }

    #[test]
    fn color_and_future_color_are_independent() {
        // An explicit future color is authoritative: its alpha is used as-is.
        let (played, future) = resolve(LineTrailColors {
            played: Some(G),
            future: Some(W),
        });
        assert_eq!(played, G);
        assert_eq!(future, W);
    }

    #[test]
    fn future_color_only_keeps_timeline_played() {
        // Explicit future color keeps its own alpha (no global fade applied).
        let (played, future) = resolve(LineTrailColors {
            played: None,
            future: Some(W),
        });
        assert_eq!(played, PLAYED_TL);
        assert_eq!(future, W);
    }

    #[test]
    fn explicit_future_alpha_is_not_faded() {
        // A half-opaque future color renders at exactly that opacity.
        let half = Vec4::new(1.0, 1.0, 1.0, 0.5);
        let (_, future) = resolve(LineTrailColors {
            played: Some(G),
            future: Some(half),
        });
        assert_eq!(future, half);
    }

    /// Reproduces issue #735 numerically: the line_3d vertex pipeline is
    /// `clip = clip_from_world · model · point`. The `model · point` product is
    /// evaluated in f32. This emulates that product for an ECEF vertex, before
    /// and after the `world_origin` rebase, while the big_space floating origin
    /// drifts a few cm per frame (as it does when the camera settles even while
    /// playback is paused). It asserts the *temporal variation* of the render
    /// error — i.e. the visible flicker — collapses once the operands are small.
    /// The trail's paused shimmer was the ~0.5 m the ECEF vertices lose when a
    /// ~6.4e6 m `f64` position is cast straight to `f32` at ingestion: adjacent
    /// samples snap to the same 0.5 m grid, so the polyline is permanently
    /// jagged and shimmers under any camera motion. Rebasing against the frame
    /// origin **in f64 before the cast** (what `value_offset` now does) stores
    /// small launch-relative values that keep millimetre precision.
    #[test]
    fn ecef_ingest_rebase_preserves_precision() {
        use bevy::math::{DVec3, Vec3};
        use bevy_geo_frames::{GeoContext, GeoFrame, GeoOrigin, GeoPosition};

        // Mojave launch site, matching shane/line_3d.shane.kdl.
        let ctx: GeoContext = GeoOrigin::new_from_degrees(35.3506640, -117.80902, 589.2740).into();

        // ECEF reference used as the per-element rebase offset (~6.4e6 m).
        let reference = GeoPosition::from_bevy(GeoFrame::ECEF, DVec3::ZERO, &ctx).1;
        assert!(
            reference.length() > 6.0e6,
            "ECEF reference should be Earth-radius scale, got {}",
            reference.length()
        );

        // Walk a few hundred trail samples out from the launch site and track
        // the worst-case error of each stored vertex against the f64 truth.
        let mut old_err = 0.0f32;
        let mut new_err = 0.0f32;
        for k in 0..400i32 {
            let d = k as f64 * 7.31; // irregular metre-scale spacing along the trail
            let vertex = reference + DVec3::new(d, -0.5 * d, 0.25 * d);
            // Ground truth (full f64 subtraction, then cast).
            let truth = (vertex - reference).as_vec3();

            // OLD: cast each f64 element straight to f32, then subtract (what the
            // shader used to do) — both operands are quantised to the ~0.5 m f32
            // grid at ECEF scale, so the difference carries that grid noise.
            let old_stored = vertex.as_vec3() - reference.as_vec3();
            // NEW: subtract the reference in f64, then cast — small values, so
            // the f32 grid is millimetric.
            let new_stored = Vec3::new(
                (vertex.x - reference.x) as f32,
                (vertex.y - reference.y) as f32,
                (vertex.z - reference.z) as f32,
            );

            old_err = old_err.max((old_stored - truth).length());
            new_err = new_err.max((new_stored - truth).length());
        }

        // The straight cast loses decimetres; the f64 rebase is sub-millimetre.
        assert!(
            old_err > 0.05,
            "expected the straight f32 cast to lose >5 cm, got {old_err}"
        );
        assert!(
            new_err < 1.0e-3,
            "f64 rebase should keep mm precision, got {new_err}"
        );
        assert!(
            new_err < old_err / 50.0,
            "f64 rebase should be orders of magnitude tighter: old={old_err} new={new_err}"
        );
    }
}
