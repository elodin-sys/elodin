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
    /// Last range + LineTree `content_gen`s written into `index_buffers`.
    /// Range alone is not enough: SeriesStore sync can clear/rebuild trees while
    /// the quantized visible range stays fixed.
    last_index_key: Option<(i64, i64, u64, u64, u64)>,
}

/// Per-entity cache of played/future GPU index state so TemporaryRenderEntity
/// rebuilds can skip `write_to_index_buffer` when the quantized range is unchanged.
#[derive(Component, Clone, Default)]
struct GpuLineIndexCache {
    played: Option<GpuLine>,
    future: Option<GpuLine>,
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
    Option<&'static mut GpuLineIndexCache>,
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
        let selected_range = if crate::is_short_accuracy_window(&selected_time_range.0) {
            selected_time_range.0.clone()
        } else {
            crate::quantize_visible_range(
                selected_time_range.0.clone(),
                crate::TRAILING_RANGE_QUANTUM_MICROS,
            )
        };
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
        let quantized_playhead = if crate::is_short_accuracy_window(&selected_range) {
            current_timestamp.0
        } else {
            crate::floor_timestamp_quantum(
                current_timestamp.0,
                crate::TRAILING_RANGE_QUANTUM_MICROS,
            )
        };
        let played_range = if live_follow {
            selected_range.clone()
        } else {
            selected_range.start..selected_range.end.min(quantized_playhead)
        };
        // Future segment must contain >= 2 samples or the shader draws only
        // sentinel(NaN)-to-point instances and nothing shows up. When
        // `current_timestamp` falls between sim ticks (the common case in live
        // streaming), the naive split leaves a single index in the future
        // range, which blinks at the render framerate near the rocket. Snap
        // the split back onto the previous sample boundary instead.
        let split = selected_range.start.max(quantized_playhead);

        'outer: for (entity, line_handles, config, uniform, trail_colors, mut index_cache) in
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

            let cached_values = index_cache
                .as_ref()
                .and_then(|c| c.played.as_ref().or(c.future.as_ref()))
                .map(|g| g.values_bind_group.clone());
            let values_bind_group = if let Some(bg) = cached_values {
                bg
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

            let build_gpu_line = |range: std::ops::Range<impeller2::types::Timestamp>,
                                  cached: Option<&GpuLine>| {
                if range.start >= range.end {
                    return None;
                }
                let content_gens = [0, 1, 2].map(|i| {
                    line_assets
                        .get(&line_handles.0[i])
                        .map(|l| l.data.content_gen())
                        .unwrap_or(0)
                });
                let index_key = (
                    range.start.0,
                    range.end.0,
                    content_gens[0],
                    content_gens[1],
                    content_gens[2],
                );
                if let Some(prev) = cached
                    && prev.last_index_key == Some(index_key)
                {
                    return Some(prev.clone());
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
                    last_index_key: Some(index_key),
                })
            };

            let mut next_cache = GpuLineIndexCache::default();
            let played_cached = index_cache.as_ref().and_then(|c| c.played.clone());
            if let Some(gpu_line) = build_gpu_line(played_range.clone(), played_cached.as_ref()) {
                let mut played_uniform = *uniform;
                played_uniform.color = played_color;
                next_cache.played = Some(gpu_line.clone());
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

            let future_cached = index_cache.as_ref().and_then(|c| c.future.clone());
            if let Some(gpu_line) = build_gpu_line(future_range.clone(), future_cached.as_ref()) {
                let mut future_uniform = *uniform;
                future_uniform.color = future_color;
                next_cache.future = Some(gpu_line.clone());
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

            if let Some(ref mut cache) = index_cache {
                **cache = next_cache;
            } else {
                _main_commands.entity(entity).insert(next_cache);
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
}
