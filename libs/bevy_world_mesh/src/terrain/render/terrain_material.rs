use crate::terrain::{
    debug::DebugTerrain,
    render::{
        terrain_bind_group::{create_terrain_layout, SetTerrainBindGroup},
        terrain_view_bind_group::{
            create_terrain_view_layout, DrawTerrainCommand, SetTerrainViewBindGroup,
        },
    },
    shaders::{DEFAULT_FRAGMENT_SHADER, DEFAULT_VERTEX_SHADER},
    terrain::TerrainComponents,
    terrain_data::gpu_tile_atlas::GpuTileAtlas,
};
use bevy::{
    core_pipeline::core_3d::{Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey},
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    pbr::{
        MaterialBindGroupAllocators, MaterialPlugin, MeshMaterial3d, MeshPipeline,
        MeshPipelineViewLayoutKey, PreparedMaterial, SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        erased_render_asset::{prepare_erased_assets, ErasedRenderAssets},
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, PhaseItem,
            RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass,
            ViewBinnedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        sync_world::{MainEntity, MainEntityHashMap},
        view::{ExtractedView, Msaa, ViewTarget},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
    shader::{ShaderDefVal, ShaderRef},
};
use std::{any::TypeId, hash::Hash, marker::PhantomData};

/// The specialization key for our terrain pipeline. In 0.15 this carried an
/// `M::Data` copy mirroring bevy_pbr's `PreparedMaterial::key`, but 0.16's
/// `PreparedMaterial` no longer exposes `key` and we don't actually read
/// per-material data during specialize, so this is now just a flag wrapper.
///
/// `SpecializerKey` is derived so Bevy 0.17's `Variants<RenderPipeline, _>`
/// cache can use it as its primary lookup key. We get the default "canonical"
/// implementation (`IS_CANONICAL = true`, `Canonical = Self`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, SpecializerKey)]
pub struct TerrainPipelineKey {
    pub flags: TerrainPipelineFlags,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct TerrainPipelineFlags: u32 {
        const NONE               = 0;
        const SPHERICAL          = 1 <<  0;
        const WIREFRAME          = 1 <<  1;
        const SHOW_DATA_LOD      = 1 <<  2;
        const SHOW_GEOMETRY_LOD  = 1 <<  3;
        const SHOW_TILE_TREE     = 1 <<  4;
        const SHOW_PIXELS        = 1 <<  5;
        const SHOW_UV            = 1 <<  6;
        const SHOW_NORMALS       = 1 <<  7;
        const MORPH              = 1 <<  8;
        const BLEND              = 1 <<  9;
        const TILE_TREE_LOD      = 1 << 10;
        const LIGHTING           = 1 << 11;
        const SAMPLE_GRAD        = 1 << 12;
        const HIGH_PRECISION     = 1 << 13;
        const TEST1              = 1 << 14;
        const TEST2              = 1 << 15;
        const TEST3              = 1 << 16;
        const HDR                = 1 << 17;
        const MSAA_RESERVED_BITS = TerrainPipelineFlags::MSAA_MASK_BITS << TerrainPipelineFlags::MSAA_SHIFT_BITS;
    }
}

impl TerrainPipelineFlags {
    const MSAA_MASK_BITS: u32 = 0b111111;
    const MSAA_SHIFT_BITS: u32 = 32 - 6;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits = ((msaa_samples - 1) & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        TerrainPipelineFlags::from_bits(msaa_bits).unwrap()
    }

    pub fn from_debug(debug: &DebugTerrain) -> Self {
        let mut key = TerrainPipelineFlags::NONE;

        if debug.wireframe {
            key |= TerrainPipelineFlags::WIREFRAME;
        }
        if debug.show_data_lod {
            key |= TerrainPipelineFlags::SHOW_DATA_LOD;
        }
        if debug.show_geometry_lod {
            key |= TerrainPipelineFlags::SHOW_GEOMETRY_LOD;
        }
        if debug.show_tile_tree {
            key |= TerrainPipelineFlags::SHOW_TILE_TREE;
        }
        if debug.show_pixels {
            key |= TerrainPipelineFlags::SHOW_PIXELS;
        }
        if debug.show_uv {
            key |= TerrainPipelineFlags::SHOW_UV;
        }
        if debug.show_normals {
            key |= TerrainPipelineFlags::SHOW_NORMALS;
        }
        if debug.morph {
            key |= TerrainPipelineFlags::MORPH;
        }
        if debug.blend {
            key |= TerrainPipelineFlags::BLEND;
        }
        if debug.tile_tree_lod {
            key |= TerrainPipelineFlags::TILE_TREE_LOD;
        }
        if debug.lighting {
            key |= TerrainPipelineFlags::LIGHTING;
        }
        if debug.sample_grad {
            key |= TerrainPipelineFlags::SAMPLE_GRAD;
        }
        if debug.high_precision {
            key |= TerrainPipelineFlags::HIGH_PRECISION;
        }
        if debug.test1 {
            key |= TerrainPipelineFlags::TEST1;
        }
        if debug.test2 {
            key |= TerrainPipelineFlags::TEST2;
        }
        if debug.test3 {
            key |= TerrainPipelineFlags::TEST3;
        }

        key
    }

    pub fn msaa_samples(&self) -> u32 {
        ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS) + 1
    }

    pub fn polygon_mode(&self) -> PolygonMode {
        match self.contains(TerrainPipelineFlags::WIREFRAME) {
            true => PolygonMode::Line,
            false => PolygonMode::Fill,
        }
    }

    pub fn shader_defs(&self) -> Vec<ShaderDefVal> {
        let mut shader_defs = Vec::new();

        if self.contains(TerrainPipelineFlags::SPHERICAL) {
            shader_defs.push("SPHERICAL".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_DATA_LOD) {
            shader_defs.push("SHOW_DATA_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_GEOMETRY_LOD) {
            shader_defs.push("SHOW_GEOMETRY_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_TILE_TREE) {
            shader_defs.push("SHOW_TILE_TREE".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_PIXELS) {
            shader_defs.push("SHOW_PIXELS".into())
        }
        if self.contains(TerrainPipelineFlags::SHOW_UV) {
            shader_defs.push("SHOW_UV".into());
        }
        if self.contains(TerrainPipelineFlags::SHOW_NORMALS) {
            shader_defs.push("SHOW_NORMALS".into())
        }
        if self.contains(TerrainPipelineFlags::MORPH) {
            shader_defs.push("MORPH".into());
        }
        if self.contains(TerrainPipelineFlags::BLEND) {
            shader_defs.push("BLEND".into());
        }
        if self.contains(TerrainPipelineFlags::TILE_TREE_LOD) {
            shader_defs.push("TILE_TREE_LOD".into());
        }
        if self.contains(TerrainPipelineFlags::LIGHTING) {
            shader_defs.push("LIGHTING".into());
        }
        if self.contains(TerrainPipelineFlags::SAMPLE_GRAD) {
            shader_defs.push("SAMPLE_GRAD".into());
        }
        if self.contains(TerrainPipelineFlags::HIGH_PRECISION) {
            shader_defs.push("HIGH_PRECISION".into());
        }
        if self.contains(TerrainPipelineFlags::TEST1) {
            shader_defs.push("TEST1".into());
        }
        if self.contains(TerrainPipelineFlags::TEST2) {
            shader_defs.push("TEST2".into());
        }
        if self.contains(TerrainPipelineFlags::TEST3) {
            shader_defs.push("TEST3".into());
        }

        shader_defs
    }
}

/// The specializer that configures per-variant state on the base
/// `RenderPipelineDescriptor` stored in [`TerrainRenderPipeline::variants`].
///
/// Bevy 0.17 replaced `SpecializedRenderPipeline` / `SpecializedRenderPipelines<P>`
/// (migration guide PR#17373) with `Specializer<RenderPipeline>` +
/// `Variants<RenderPipeline, S>`. In the new model, specializers MUTATE a
/// base descriptor, not build it from scratch. The constant parts of the
/// pipeline (vertex/fragment shaders, depth stencil, topology, color target)
/// live on `base_descriptor`; only the MSAA sample count, the view bind
/// group layout (swapped between msaa and non-msaa), and the dynamic shader
/// defs get touched here.
pub struct TerrainSpecializer<M: Material> {
    view_layout: BindGroupLayoutDescriptor,
    view_layout_multisampled: BindGroupLayoutDescriptor,
    /// Snapshot of the bevy_pbr `MeshPipeline`'s runtime feature detection,
    /// cached at construction so `specialize` can push the same conditional
    /// shader_defs that `bevy_pbr::mesh_view_bindings.wgsl` expects. Without
    /// these, the view bind group layout (borrowed from `MeshPipeline`) and
    /// the naga-composed shader get out of sync and pipeline validation
    /// panics during shader module creation with an empty-source naga span
    /// error.
    binding_arrays_are_usable: bool,
    clustered_decals_are_usable: bool,
    marker: PhantomData<M>,
}

impl<M: Material> Specializer<RenderPipeline> for TerrainSpecializer<M> {
    type Key = TerrainPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        descriptor: &mut RenderPipelineDescriptor,
    ) -> Result<Canonical<Self::Key>, BevyError> {
        // Shader defs carried by the key (debug flags, spherical, lighting, etc.).
        let mut shader_defs = key.flags.shader_defs();

        // Mirror the conditional shader_defs that bevy_pbr's mesh pipeline
        // specializer pushes for the view bind group layout. The layout we
        // borrowed from `MeshPipeline` was generated with these defs in
        // mind; shader and layout must agree or composition fails.
        if self.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
            shader_defs.push("MULTIPLE_LIGHTMAPS_IN_ARRAY".into());
        }
        // `IRRADIANCE_VOLUMES_ARE_USABLE` is a crate-private const in bevy_pbr
        // evaluated as `cfg!(not(target_arch = "wasm32"))`. Mirror for native.
        #[cfg(not(target_arch = "wasm32"))]
        shader_defs.push("IRRADIANCE_VOLUMES_ARE_USABLE".into());
        if self.clustered_decals_are_usable {
            shader_defs.push("CLUSTERED_DECALS_ARE_USABLE".into());
        }

        // Swap the view layout at index 0 depending on MSAA. Terrain/
        // terrain-view/material layouts at indices 1/2/3 stay as the base
        // descriptor set them.
        if key.flags.msaa_samples() > 1 {
            shader_defs.push("MULTISAMPLED".into());
            descriptor.layout[0] = self.view_layout_multisampled.clone();
        } else {
            descriptor.layout[0] = self.view_layout.clone();
        }

        descriptor.primitive.polygon_mode = key.flags.polygon_mode();
        descriptor.multisample.count = key.flags.msaa_samples();
        if let Some(fragment) = &mut descriptor.fragment {
            if let Some(Some(target)) = fragment.targets.first_mut() {
                target.format = if key.flags.contains(TerrainPipelineFlags::HDR) {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::bevy_default()
                };
            }
        }

        descriptor.vertex.shader_defs = shader_defs.clone();
        let mut fragment_shader_defs = shader_defs;
        fragment_shader_defs.push("FRAGMENT".into());
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs = fragment_shader_defs;
        }

        Ok(key)
    }
}

/// The pipeline used to render the terrain entities.
///
/// In 0.17 this wraps a [`Variants<RenderPipeline, TerrainSpecializer<M>>`]
/// cache (PR#17373). `Variants` owns both the specializer and the
/// base descriptor, and memoizes `CachedRenderPipelineId` per key. The
/// `specialize` call on `Variants` mutates the base descriptor on-demand
/// through our [`TerrainSpecializer`].
#[derive(Resource)]
pub struct TerrainRenderPipeline<M: Material> {
    pub variants: Variants<RenderPipeline, TerrainSpecializer<M>>,
    _marker: PhantomData<M>,
}

impl<M: Material> FromWorld for TerrainRenderPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let mesh_pipeline = world.resource::<MeshPipeline>();

        // `get_view_layout` in 0.18 returns `MeshPipelineViewLayout`, which
        // bundles the primary view layout (as a `BindGroupLayoutDescriptor`)
        // plus separate binding-array and empty layouts (wgpu 25 moved
        // binding-array resources into a separate bind group). We only need
        // the primary view layout at slot 0 -- we don't participate in
        // bevy_pbr's binding array group.
        let view_layout = mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::empty())
            .main_layout
            .clone();
        let view_layout_multisampled = mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::MULTISAMPLED)
            .main_layout
            .clone();
        let terrain_layout = create_terrain_layout();
        let terrain_view_layout = create_terrain_view_layout();
        // 0.18 provides `bind_group_layout_descriptor(...)` alongside the
        // older `bind_group_layout(...)` for materials that need to feed
        // the new descriptor-based pipeline layout.
        let material_layout = M::bind_group_layout_descriptor(device);

        let vertex_shader = match M::vertex_shader() {
            ShaderRef::Default => asset_server.load(DEFAULT_VERTEX_SHADER),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        };

        let fragment_shader = match M::fragment_shader() {
            ShaderRef::Default => asset_server.load(DEFAULT_FRAGMENT_SHADER),
            ShaderRef::Handle(handle) => handle,
            ShaderRef::Path(path) => asset_server.load(path),
        };

        let base_descriptor = RenderPipelineDescriptor {
            label: Some("terrain_pipeline".into()),
            layout: vec![
                view_layout.clone(),
                terrain_layout,
                terrain_view_layout,
                material_layout,
            ],
            push_constant_ranges: default(),
            zero_initialize_workgroup_memory: false,
            vertex: VertexState {
                shader: vertex_shader,
                entry_point: Some("vertex".into()),
                shader_defs: Vec::new(),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
            },
            fragment: Some(FragmentState {
                shader: fragment_shader,
                shader_defs: Vec::new(),
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Greater,
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
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        };

        let specializer = TerrainSpecializer::<M> {
            view_layout,
            view_layout_multisampled,
            binding_arrays_are_usable: mesh_pipeline.binding_arrays_are_usable,
            clustered_decals_are_usable: mesh_pipeline.clustered_decals_are_usable,
            marker: PhantomData,
        };

        Self {
            variants: Variants::new(specializer, base_descriptor),
            _marker: PhantomData,
        }
    }
}

/// The draw function of the terrain. It sets the pipeline and the bind groups and then issues the
/// draw call.
///
/// In 0.15 slot 3 used bevy_pbr's `SetMaterialBindGroup<M, 3>`. In 0.16
/// bevy_pbr reworked material binding around `MaterialBindGroupAllocator<M>`
/// and `MaterialBindingId`, and only specializes pipelines for entities that
/// carry `Mesh3d` -- which our `TileAtlas` entity does not. We therefore
/// bypass bevy_pbr's `SetMaterialBindGroup` and read the allocator directly
/// via our own [`SetTerrainMaterialBindGroup`] below.
pub(crate) type DrawTerrain<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetTerrainBindGroup<1>,
    SetTerrainViewBindGroup<2>,
    SetTerrainMaterialBindGroup<M, 3>,
    DrawTerrainCommand,
);

/// Our parallel map from terrain `MainEntity` to the `AssetId<M>` of the
/// material they carry. In 0.16 `bevy_pbr::RenderMaterialInstances` is no
/// longer generic (a single shared map across all material types) and
/// `RenderMaterialInstance::asset_id` is private, so we can't read it from
/// outside bevy_pbr. We maintain our own keyed-by-MainEntity resource and let
/// bevy_pbr's own machinery run alongside us via the registered
/// [`MaterialPlugin::<M>`].
#[derive(Resource)]
pub struct TerrainMaterialInstances<M: Material> {
    pub instances: MainEntityHashMap<AssetId<M>>,
}

impl<M: Material> Default for TerrainMaterialInstances<M> {
    fn default() -> Self {
        Self {
            instances: default(),
        }
    }
}

/// Extract terrain material handles from the main world into
/// [`TerrainMaterialInstances`]. Mirrors bevy_pbr's `extract_mesh_materials`
/// but keyed by the terrain's `MainEntity`.
///
/// 0.16's `check_visibility` no longer takes a `QueryFilter` generic, so we
/// can't run the old `check_visibility::<With<TileAtlas>>` system that used
/// to set `ViewVisibility::get()` to `true` for our terrain each frame. Our
/// `TerrainBundle` carries `NoFrustumCulling` and is unconditionally visible,
/// so extract every entity that has the material regardless of the (now
/// unchecked) `ViewVisibility` component.
pub(crate) fn extract_terrain_materials<M: Material>(
    mut material_instances: ResMut<TerrainMaterialInstances<M>>,
    query: Extract<Query<(Entity, &MeshMaterial3d<M>)>>,
) {
    material_instances.instances.clear();
    for (entity, material) in &query {
        material_instances
            .instances
            .insert(entity.into(), material.id());
    }
}

/// Queues all terrain entities for rendering via the terrain pipeline.
///
/// 0.17 routes pipeline specialization through [`Variants::specialize`] (not
/// a separate `SpecializedRenderPipelines<P>` resource), which requires
/// `&mut TerrainRenderPipeline<M>` so `Variants` can grow its cache. Material
/// lookups moved to the single non-generic `ErasedRenderAssets<PreparedMaterial>`
/// and `MaterialBindGroupAllocators` (indexed by `TypeId::of::<M>()`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn queue_terrain<M: Material>(
    change_tick: bevy::ecs::system::SystemChangeTick,
    draw_functions: Res<DrawFunctions<Opaque3d>>,
    debug: Option<Res<DebugTerrain>>,
    render_materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    pipeline_cache: Res<PipelineCache>,
    mut terrain_pipeline: ResMut<TerrainRenderPipeline<M>>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    gpu_tile_atlases: Res<TerrainComponents<GpuTileAtlas>>,
    material_instances: Res<TerrainMaterialInstances<M>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let change_tick = change_tick.this_run();
    let draw_function = draw_functions.read().get_id::<DrawTerrain<M>>().unwrap();

    for (view, msaa) in &views {
        let Some(phase) = opaque_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        for (main_entity, material_id) in &material_instances.instances {
            // Our `TerrainComponents` / terrain bind group are keyed by the
            // raw `Entity` of the main-world terrain; keep using that shape
            // so the render-command path stays consistent with 0.15.
            let render_entity = main_entity.id();
            let Some(gpu_tile_atlas) = gpu_tile_atlases.get(&render_entity) else {
                continue;
            };
            let Some(material) = render_materials.get(*material_id) else {
                continue;
            };

            let mut flags = TerrainPipelineFlags::from_msaa_samples(msaa.samples());
            if view.hdr {
                flags |= TerrainPipelineFlags::HDR;
            }
            if gpu_tile_atlas.is_spherical {
                flags |= TerrainPipelineFlags::SPHERICAL;
            }
            if let Some(debug) = &debug {
                flags |= TerrainPipelineFlags::from_debug(debug);
            } else {
                flags |= TerrainPipelineFlags::LIGHTING
                    | TerrainPipelineFlags::MORPH
                    | TerrainPipelineFlags::BLEND
                    | TerrainPipelineFlags::SAMPLE_GRAD;
            }

            let key = TerrainPipelineKey { flags };
            let Ok(pipeline) = terrain_pipeline.variants.specialize(&pipeline_cache, key) else {
                continue;
            };

            let batch_set_key = Opaque3dBatchSetKey {
                pipeline,
                draw_function,
                material_bind_group_index: Some(material.binding.group.0),
                vertex_slab: default(),
                index_slab: None,
                lightmap_slab: None,
            };
            let bin_key = Opaque3dBinKey {
                asset_id: material_id.untyped(),
            };
            phase.add(
                batch_set_key,
                bin_key,
                (render_entity, *main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
                change_tick,
            );
        }
    }
}

/// Binds the material group for our terrain at descriptor-set slot `I`.
///
/// In Bevy 0.17 the standard `bevy_pbr::SetMaterialBindGroup<I>` expects
/// that the entity has gone through `specialize_material_meshes::<M>`, which
/// only iterates entities with `Mesh3d`. Our terrain isn't a mesh, so we
/// bypass that path and walk `MaterialBindGroupAllocators` (keyed by the
/// material's `TypeId`) + `ErasedRenderAssets<PreparedMaterial>` directly.
/// Mirrors the body of bevy_pbr's `SetMaterialBindGroup` but sources the
/// `AssetId` from our `TerrainMaterialInstances<M>` instead of bevy_pbr's
/// `RenderMaterialInstances`.
pub struct SetTerrainMaterialBindGroup<M: Material, const I: usize>(PhantomData<M>);

impl<P: PhaseItem, M: Material, const I: usize> RenderCommand<P>
    for SetTerrainMaterialBindGroup<M, I>
{
    type Param = (
        SRes<MaterialBindGroupAllocators>,
        SRes<ErasedRenderAssets<PreparedMaterial>>,
        SRes<TerrainMaterialInstances<M>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _item_query: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        (allocators, render_materials, instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let main_entity: MainEntity = item.main_entity();
        let Some(asset_id) = instances.into_inner().instances.get(&main_entity) else {
            return RenderCommandResult::Skip;
        };
        let Some(prepared) = render_materials.into_inner().get(*asset_id) else {
            return RenderCommandResult::Skip;
        };
        // 0.17 centralizes allocators by `TypeId` of the material.
        let Some(allocator) = allocators.into_inner().get(&TypeId::of::<M>()) else {
            return RenderCommandResult::Skip;
        };
        let Some(slab) = allocator.get(prepared.binding.group) else {
            return RenderCommandResult::Skip;
        };
        let Some(bind_group) = slab.bind_group() else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

/// This plugin adds a custom material for a terrain.
///
/// It can be used to render the terrain using a custom vertex and fragment shader.
pub struct TerrainMaterialPlugin<M: Material>(PhantomData<M>);

impl<M: Material> Default for TerrainMaterialPlugin<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<M: Material> Plugin for TerrainMaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        // `MaterialPlugin::<M>` in 0.17 schedules `prepare_erased_assets::<MeshMaterial3d<M>>`
        // (which feeds `ErasedRenderAssets<PreparedMaterial>`) and populates
        // `MaterialBindGroupAllocators` keyed by `TypeId::of::<M>()`. We need
        // both for our own `queue_terrain` and `SetTerrainMaterialBindGroup`
        // to work. Disabling prepass / shadows keeps the rest of bevy_pbr's
        // material systems out of the way for a renderer that has no `Mesh3d`.
        // Bevy 0.18 (PR#20999) moved `prepass_enabled` / `shadows_enabled`
        // off `MaterialPlugin` onto the `Material` trait via
        // `enable_prepass()` / `enable_shadows()`. Concrete material impls
        // override those to return `false` for terrain.
        app.add_plugins(MaterialPlugin::<M>::default());

        app.sub_app_mut(RenderApp)
            .init_resource::<TerrainMaterialInstances<M>>()
            .add_render_command::<Opaque3d, DrawTerrain<M>>()
            .add_systems(ExtractSchedule, extract_terrain_materials::<M>)
            .add_systems(
                Render,
                queue_terrain::<M>
                    .in_set(RenderSystems::QueueMeshes)
                    .after(prepare_erased_assets::<MeshMaterial3d<M>>),
            );
    }

    fn finish(&self, app: &mut App) {
        // Bevy 0.17 moved many render resources (including `MaterialPipeline`)
        // from `Plugin::finish` to `RenderStartup` (migration guide PRs
        // #19841..#20210). Our `TerrainRenderPipeline::<M>::from_world` only
        // needs `MeshPipeline` + `RenderDevice` + `AssetServer`, all of which
        // are still available during `finish`, so the resource init stays
        // here. `Variants` owns the specialization cache internally, so
        // there's no more `SpecializedRenderPipelines<_>` to register.
        app.sub_app_mut(RenderApp)
            .init_resource::<TerrainRenderPipeline<M>>();
    }
}
