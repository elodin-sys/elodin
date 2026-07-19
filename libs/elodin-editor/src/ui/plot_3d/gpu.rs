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
        hierarchy::ChildOf,
        query::Has,
        schedule::{IntoScheduleConfigs, SystemSet},
        system::{
            Commands, Query, Res, ResMut, SystemState,
            lifetimeless::{Read, SRes},
        },
        world::{FromWorld, Mut, World},
    },
    image::BevyDefault,
    math::{DVec3, Mat4, Quat, Vec3, Vec4},
    mesh::VertexBufferLayout,
    pbr::{MeshPipeline, MeshPipelineKey, SetMeshViewBindGroup},
    prelude::{Color, Reflect, Resource, With},
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
    transform::{
        TransformSystems,
        components::{GlobalTransform, Transform},
    },
};
use bevy_geo_frames::{GeoContext, GeoFrame, GeoPosition, Present};
use bevy_render::{
    extract_component::ExtractComponent,
    sync_world::{MainEntity, SyncToRenderWorld, TemporaryRenderEntity},
};
use binding_types::storage_buffer_read_only_sized;
use impeller2_wkt::{CurrentTimestamp, EarliestTimestamp, LastUpdated, Line3d};
use std::num::NonZeroU64;
use zerocopy::IntoBytes;

const LINE_SHADER_HANDLE: Handle<Shader> = uuid_handle!("bfffa3c4-9401-4b6e-b3ab-3564180352f1");

/// Dense line-local XYZ buffers: one `f32` per strip sample plus a leading NaN sentinel.
/// Sized to the index budget so a full-fidelity short window still fits.
const LOCAL_VALUE_BUFFER_LEN: usize = INDEX_BUFFER_LEN;
const LOCAL_VALUE_BUFFER_SIZE: NonZeroU64 =
    NonZeroU64::new((LOCAL_VALUE_BUFFER_LEN * size_of::<f32>()) as u64).unwrap();

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
                (
                    // After geo rotation so we own Transform; before propagate
                    // so big_space sees the first-point GridCell pose.
                    place_line_3d_at_first_point
                        .after(bevy_geo_frames::apply_geo_rotation)
                        .before(TransformSystems::Propagate),
                    update_uniform_model.after(TransformSystems::Propagate),
                ),
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
                storage_buffer_read_only_sized(false, Some(LOCAL_VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(LOCAL_VALUE_BUFFER_SIZE)),
                storage_buffer_read_only_sized(false, Some(LOCAL_VALUE_BUFFER_SIZE)),
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
    // Entity sits at the line's first sample under BigSpaceRoot; big_space
    // puts that in FO-local GlobalTransform. Vertices are (p - first).
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
    /// Dense line-local XYZ value buffers (NaN at index 0, samples at 1..n),
    /// stored relative to the line's first sample (entity world pose).
    #[allow(dead_code)]
    value_buffers: [Buffer; 3],
    /// Strip indices into `value_buffers` (leading/trailing NaN sentinels).
    #[allow(dead_code)]
    index_buffers: [Buffer; 3],
    count: u32,
    /// Last range + LineTree `content_gen`s + GeoContext/frame hash written
    /// into the GPU buffers. Placement comes from the entity GlobalTransform.
    #[allow(clippy::type_complexity)]
    last_index_key: Option<(i64, i64, u64, u64, u64, u64)>,
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
    Option<&'static GeoPosition>,
    Option<&'static Line3d>,
    Option<&'static mut GpuLineIndexCache>,
);

/// Hash of GeoContext fields that affect frame→Bevy conversion, plus the
/// resolved line frame. Buffers built under a different context (e.g. before
/// the schematic origin lands) must not be reused.
fn geo_context_cache_key(ctx: &GeoContext, frame: GeoFrame) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ctx.origin.latitude.to_bits().hash(&mut hasher);
    ctx.origin.longitude.to_bits().hash(&mut hasher);
    ctx.origin.altitude.to_bits().hash(&mut hasher);
    match ctx.present {
        Present::Plane => 0u8,
        Present::Sphere => 1u8,
    }
    .hash(&mut hasher);
    match frame {
        GeoFrame::ENU => 0u8,
        GeoFrame::NED => 1u8,
        GeoFrame::ECEF => 2u8,
    }
    .hash(&mut hasher);
    hasher.finish()
}

fn resolve_line_frame(geo_pos: Option<&GeoPosition>, line: Option<&Line3d>) -> GeoFrame {
    if let Some(geo) = geo_pos {
        return geo.0;
    }
    line.and_then(|l| l.frame).unwrap_or_default()
}

/// Convert a frame-space sample to absolute Bevy coordinates (f64).
fn frame_point_to_bevy(frame: GeoFrame, x: f32, y: f32, z: f32, geo_ctx: &GeoContext) -> DVec3 {
    GeoPosition(frame, DVec3::new(x as f64, y as f64, z as f64)).to_bevy(geo_ctx)
}

/// First sample of the line (full recording), in Bevy space. Stable geographic
/// anchor for both the entity pose and relative vertex buffers.
fn line_first_point_bevy(
    line_assets: &Assets<Line>,
    handles: &[Handle<Line>; 3],
    frame: GeoFrame,
    geo_ctx: &GeoContext,
) -> Option<DVec3> {
    let full = impeller2::types::Timestamp(i64::MIN)..impeller2::types::Timestamp(i64::MAX);
    // Huge step still keeps the first sample (and last); we only need the first.
    let step = usize::MAX;
    let xs = line_assets
        .get(&handles[0])?
        .data
        .collect_strip_values(full.clone(), step);
    let ys = line_assets
        .get(&handles[1])?
        .data
        .collect_strip_values(full.clone(), step);
    let zs = line_assets
        .get(&handles[2])?
        .data
        .collect_strip_values(full, step);
    let x = *xs.first()?;
    let y = *ys.first()?;
    let z = *zs.first()?;
    Some(frame_point_to_bevy(frame, x, y, z, geo_ctx))
}

/// Place each `line_3d` entity at its first sample under [`BigSpaceRoot`] so
/// big_space owns FO-local `GlobalTransform`. Vertices are stored relative to
/// that same first point.
#[allow(clippy::type_complexity)]
fn place_line_3d_at_first_point(
    mut commands: Commands,
    mut lines: Query<(
        Entity,
        &LineHandles,
        Option<&GeoPosition>,
        Option<&Line3d>,
        &mut Transform,
        Option<&ChildOf>,
    )>,
    line_assets: Res<Assets<Line>>,
    geo_ctx: Res<GeoContext>,
    #[cfg(feature = "big_space")] settings: Res<crate::spatial::FloatingOriginSettings>,
    #[cfg(feature = "big_space")] roots: Query<Entity, With<crate::spatial::BigSpaceRoot>>,
) {
    #[cfg(feature = "big_space")]
    let Some(root) = roots.iter().next() else {
        return;
    };

    for (entity, handles, geo_pos, line_3d, mut transform, child_of) in &mut lines {
        // Vertices are already converted frame→Bevy; a GeoRotation basis on
        // this entity would double-apply the frame change.
        commands
            .entity(entity)
            .remove::<bevy_geo_frames::GeoRotation>();

        let frame = resolve_line_frame(geo_pos, line_3d);
        let Some(anchor) = line_first_point_bevy(&line_assets, &handles.0, frame, &geo_ctx) else {
            continue;
        };

        #[cfg(feature = "big_space")]
        {
            let (grid_cell, translation) = settings.translation_to_grid(anchor);
            *transform = Transform {
                translation,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            };
            commands
                .entity(entity)
                .insert(grid_cell)
                .insert(GlobalTransform::default());
            if child_of.map(|c| c.parent()) != Some(root) {
                commands.entity(entity).insert(ChildOf(root));
            }
        }
        #[cfg(not(feature = "big_space"))]
        {
            let _ = child_of;
            *transform = Transform {
                translation: anchor.as_vec3(),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            };
        }
    }
}

/// Build dense first-point-relative XYZ value buffers + remapped strip indices.
///
/// `anchor` is the line's first sample in Bevy space (entity world pose).
/// Vertices are `p - anchor`. Index layout matches the historical NaN-sentinel
/// strip: leading/trailing `0` (NaN slot), samples at `1..n`.
#[allow(clippy::too_many_arguments)]
fn write_anchor_local_line_buffers(
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    frame: GeoFrame,
    geo_ctx: &GeoContext,
    anchor: DVec3,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) -> Option<([Buffer; 3], [Buffer; 3], u32)> {
    let n = xs.len().min(ys.len()).min(zs.len());
    if n < 2 {
        return None;
    }
    // Slot 0 = NaN sentinel; samples occupy 1..=n (capped to buffer length).
    let max_samples = LOCAL_VALUE_BUFFER_LEN.saturating_sub(1);
    let n = n.min(max_samples);
    let mut x_local = vec![f32::NAN; n + 1];
    let mut y_local = vec![f32::NAN; n + 1];
    let mut z_local = vec![f32::NAN; n + 1];
    for i in 0..n {
        let p = frame_point_to_bevy(frame, xs[i], ys[i], zs[i], geo_ctx);
        let local = (p - anchor).as_vec3();
        x_local[i + 1] = local.x;
        y_local[i + 1] = local.y;
        z_local[i + 1] = local.z;
    }

    // Single contiguous strip with NaN sentinels (one logical chunk).
    let mut indices: Vec<u32> = Vec::with_capacity(n + 2);
    indices.push(0);
    for i in 0..n {
        indices.push((i + 1) as u32);
    }
    indices.push(0);
    if indices.len() > INDEX_BUFFER_LEN {
        indices.truncate(INDEX_BUFFER_LEN);
    }
    let count = indices.len() as u32;
    if count < 2 {
        return None;
    }

    let value_bufs = [x_local, y_local, z_local].map(|data| {
        let mut bytes = vec![0u8; LOCAL_VALUE_BUFFER_SIZE.get() as usize];
        let src = data.as_bytes();
        bytes[..src.len()].copy_from_slice(src);
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("line_3d anchor-local values"),
            contents: &bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        })
    });

    let index_bufs = ['x', 'y', 'z'].map(|_| {
        let mut bytes = vec![0u8; INDEX_BUFFER_LEN * size_of::<u32>()];
        let src = indices.as_bytes();
        bytes[..src.len()].copy_from_slice(src);
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("line_3d anchor-local indices"),
            contents: &bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        })
    });

    let _ = render_queue;

    Some((value_bufs, index_bufs, count))
}

fn extract_lines(
    mut main_world: ResMut<MainWorld>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    values_layout: Res<LineValuesLayout>,
    index_layout: Res<LineIndexLayout>,
) {
    main_world.resource_scope(|world, mut cached_state: Mut<CachedSystemState>| {
        let geo_ctx = world.resource::<GeoContext>().clone();
        let replay_mode = world.contains_resource::<crate::ReplayMode>();
        let (
            mut lines,
            line_assets,
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
        let selected_span_micros = selected_range.end.0.saturating_sub(selected_range.start.0);
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

        'outer: for (
            entity,
            line_handles,
            config,
            uniform,
            trail_colors,
            geo_pos,
            line_3d,
            mut index_cache,
        ) in lines.iter_mut()
        {
            let frame = resolve_line_frame(geo_pos, line_3d);
            let geo_key = geo_context_cache_key(&geo_ctx, frame);
            let (played_color, future_color) = trail_colors.copied().unwrap_or_default().resolve(
                played_timeline_color,
                future_timeline_color,
                future_trail_alpha,
            );
            for line in &line_handles.0 {
                if line_assets.get(line).is_none() {
                    continue 'outer;
                }
            }

            // Shared geographic anchor = first sample of the full line. Entity
            // pose (GlobalTransform → model) is placed there in main world.
            let Some(line_anchor) =
                line_first_point_bevy(&line_assets, &line_handles.0, frame, &geo_ctx)
            else {
                continue 'outer;
            };

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
            let sampling_step = crate::ui::plot::data::index_sampling_step_for_selection(
                selected_span_micros,
                sampling_chunk_count,
                sampling_index_count,
                INDEX_BUFFER_LEN,
            );

            // let cached_values = index_cache
            //     .as_ref()
            //     .and_then(|c| c.played.as_ref().or(c.future.as_ref()))
            //     .map(|g| g.values_bind_group.clone());
            // XXX: This is not used. Delete?
            // let _values_bind_group = if let Some(bg) = cached_values {
            //     bg
            // } else {
            //     let entries = [0, 1, 2].map(|i| {
            //         let line = &line_handles.0[i];
            //         let line = line_assets.get(line).expect("line missing");
            //         BindGroupEntry {
            //             binding: i as u32,
            //             resource: BindingResource::Buffer(BufferBinding {
            //                 buffer: line
            //                     .data
            //                     .data_buffer_shard_alloc()
            //                     .expect("no data buf")
            //                     .buffer(),
            //                 offset: 0,
            //                 size: Some(VALUE_BUFFER_SIZE),
            //             }),
            //         }
            //     });
            //     render_device.create_bind_group("line values", &values_layout.layout, &entries)
            // };

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
                    geo_key,
                );
                if let Some(prev) = cached
                    && prev.last_index_key == Some(index_key)
                {
                    return Some(prev.clone());
                }
                // Always start from the selection-derived step (1 for short
                // windows). Double until the strip fits the index budget so we
                // never silently truncate the newest tip when a short window
                // somehow exceeds LOCAL_VALUE_BUFFER_LEN (~4.4 kHz x 30 s).
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

                let xs = line_assets
                    .get(&line_handles.0[0])
                    .map(|l| l.data.collect_strip_values(range.clone(), step))
                    .unwrap_or_default();
                let ys = line_assets
                    .get(&line_handles.0[1])
                    .map(|l| l.data.collect_strip_values(range.clone(), step))
                    .unwrap_or_default();
                let zs = line_assets
                    .get(&line_handles.0[2])
                    .map(|l| l.data.collect_strip_values(range.clone(), step))
                    .unwrap_or_default();

                let (value_buffers, index_buffers, count) = write_anchor_local_line_buffers(
                    &xs,
                    &ys,
                    &zs,
                    frame,
                    &geo_ctx,
                    line_anchor,
                    &render_device,
                    &render_queue,
                )?;

                let value_entries = [0, 1, 2].map(|i| BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &value_buffers[i],
                        offset: 0,
                        size: Some(LOCAL_VALUE_BUFFER_SIZE),
                    }),
                });
                let values_bind_group = render_device.create_bind_group(
                    "line_3d anchor-local values",
                    &values_layout.layout,
                    &value_entries,
                );

                let index_entries = [0, 1, 2].map(|i| BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &index_buffers[i],
                        offset: 0,
                        size: Some(INDEX_BUFFER_SIZE),
                    }),
                });
                let index_bind_group = render_device.create_bind_group(
                    "line_3d anchor-local indexes",
                    &index_layout.layout,
                    &index_entries,
                );

                Some(GpuLine {
                    values_bind_group,
                    index_bind_group,
                    value_buffers,
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
                // `uniform.model` is the entity GlobalTransform (first-point pose).
                next_cache.played = Some(gpu_line.clone());
                commands.spawn((
                    MainEntity::from(entity),
                    line_handles.clone(),
                    config.clone(),
                    played_uniform,
                    GlobalTransform::default(),
                    Transform::default(),
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

    #[test]
    fn frame_point_to_bevy_matches_geoposition() {
        let ctx = GeoContext::default();
        let p_frame = DVec3::new(10.0, 20.0, 30.0);
        let expected = GeoPosition(GeoFrame::ENU, p_frame).to_bevy(&ctx);
        let actual = frame_point_to_bevy(
            GeoFrame::ENU,
            p_frame.x as f32,
            p_frame.y as f32,
            p_frame.z as f32,
            &ctx,
        );
        assert!((actual - expected).length() < 1e-6);
    }

    #[test]
    fn relative_vertices_plus_entity_pose_recover_bevy_point() {
        // Entity at first sample; vertex = p - first; model*vertex ≈ p - fo
        // when model is the FO-local entity pose (first - fo).
        let ctx = GeoContext::default();
        let start_frame = DVec3::new(1.0, 2.0, 3.0);
        let tip_frame = DVec3::new(11.0, 22.0, 33.0);
        let anchor = frame_point_to_bevy(
            GeoFrame::ENU,
            start_frame.x as f32,
            start_frame.y as f32,
            start_frame.z as f32,
            &ctx,
        );
        let tip = frame_point_to_bevy(
            GeoFrame::ENU,
            tip_frame.x as f32,
            tip_frame.y as f32,
            tip_frame.z as f32,
            &ctx,
        );
        let tip_local = (tip - anchor).as_vec3();
        let fo = DVec3::new(100.0, 200.0, 300.0);
        let model = Mat4::from_translation((anchor - fo).as_vec3());
        let placed_tip = model.transform_point3(tip_local);
        assert!((placed_tip.as_dvec3() + fo - tip).length() < 1e-3);
        let placed_start = model.transform_point3(Vec3::ZERO);
        assert!((placed_start.as_dvec3() + fo - anchor).length() < 1e-3);
    }

    #[test]
    fn geo_context_cache_key_changes_with_origin() {
        let a = GeoContext::default();
        let mut b = a.clone();
        b.origin.latitude += 1e-6;
        assert_ne!(
            geo_context_cache_key(&a, GeoFrame::ENU),
            geo_context_cache_key(&b, GeoFrame::ENU)
        );
        assert_eq!(
            geo_context_cache_key(&a, GeoFrame::ENU),
            geo_context_cache_key(&a, GeoFrame::ENU)
        );
        assert_ne!(
            geo_context_cache_key(&a, GeoFrame::ENU),
            geo_context_cache_key(&a, GeoFrame::ECEF)
        );
    }
}
