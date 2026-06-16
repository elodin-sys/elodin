use crate::terrain::{
    math::{TerrainModelApproximation, TileCoordinate},
    render::instantiate_layout,
    terrain_data::{gpu_tile_tree::GpuTileTree, tile_tree::TileTree},
    terrain_view::TerrainViewComponents,
    util::StaticBuffer,
};
use bevy::{
    ecs::{
        query::ROQueryItem,
        system::{lifetimeless::SRes, SystemParamItem},
    },
    prelude::*,
    render::{
        render_phase::{PhaseItem, RenderCommand, RenderCommandResult, TrackedRenderPass},
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
        sync_world::MainEntity,
        Extract,
    },
};

pub(crate) fn create_prepare_indirect_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "prepare_indirect_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            storage_buffer::<Indirect>(false), // indirect buffer
        ),
    )
}

pub(crate) fn create_refine_tiles_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "refine_tiles_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<TerrainViewConfigUniform>(false), // terrain view config
                uniform_buffer::<TerrainModelApproximation>(false), // model view approximation
                storage_buffer_read_only_sized(false, None),       // tile_tree
                storage_buffer_read_only_sized(false, None),       // origins
                storage_buffer_sized(false, None),                 // final tiles
                storage_buffer_sized(false, None),                 // temporary tiles
                storage_buffer::<Parameters>(false),               // parameters
            ),
        ),
    )
}

pub(crate) fn create_terrain_view_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "terrain_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<TerrainViewConfigUniform>(false), // terrain view config
                uniform_buffer::<TerrainModelApproximation>(false), // model view approximation
                storage_buffer_read_only_sized(false, None),       // tile_tree
                storage_buffer_read_only_sized(false, None),       // origins
                storage_buffer_read_only_sized(false, None),       // tiles
            ),
        ),
    )
}

#[derive(Default, ShaderType)]
pub(crate) struct Indirect {
    x_or_vertex_count: u32,
    y_or_instance_count: u32,
    z_or_base_vertex: u32,
    base_instance: u32,
}

#[derive(Default, ShaderType)]
struct Parameters {
    tile_count: u32,
    counter: i32,
    child_index: i32,
    final_index: i32,
}

#[derive(Default, ShaderType)]
struct TerrainViewConfigUniform {
    tree_size: u32,
    geometry_tile_count: u32,
    refinement_count: u32,
    grid_size: f32,
    vertices_per_row: u32,
    vertices_per_tile: u32,
    morph_distance: f32,
    blend_distance: f32,
    load_distance: f32,
    subdivision_distance: f32,
    morph_range: f32,
    blend_range: f32,
    precision_threshold_distance: f32,
}

impl TerrainViewConfigUniform {
    fn from_tile_tree(tile_tree: &TileTree) -> Self {
        TerrainViewConfigUniform {
            tree_size: tile_tree.tree_size,
            geometry_tile_count: tile_tree.geometry_tile_count,
            refinement_count: tile_tree.refinement_count,
            grid_size: tile_tree.grid_size as f32,
            vertices_per_row: 2 * (tile_tree.grid_size + 2),
            vertices_per_tile: 2 * tile_tree.grid_size * (tile_tree.grid_size + 2),
            morph_distance: tile_tree.morph_distance as f32,
            blend_distance: tile_tree.blend_distance as f32,
            load_distance: tile_tree.load_distance as f32,
            subdivision_distance: tile_tree.subdivision_distance as f32,
            precision_threshold_distance: tile_tree.precision_threshold_distance as f32,
            morph_range: tile_tree.morph_range,
            blend_range: tile_tree.blend_range,
        }
    }
}

pub struct TerrainViewData {
    view_config_buffer: StaticBuffer<TerrainViewConfigUniform>,
    terrain_model_approximation_buffer: StaticBuffer<TerrainModelApproximation>,
    pub(super) indirect_buffer: StaticBuffer<Indirect>,
    pub(super) prepare_indirect_bind_group: BindGroup,
    pub(super) refine_tiles_bind_group: BindGroup,
    pub(super) terrain_view_bind_group: BindGroup,
}

impl TerrainViewData {
    fn new(device: &RenderDevice, tile_tree: &TileTree, gpu_tile_tree: &GpuTileTree) -> Self {
        // Todo: figure out a better way of limiting the tile buffer size
        let tile_buffer_size =
            TileCoordinate::min_size().get() * tile_tree.geometry_tile_count as BufferAddress;

        let view_config_buffer =
            StaticBuffer::empty(None, device, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let indirect_buffer =
            StaticBuffer::empty(None, device, BufferUsages::STORAGE | BufferUsages::INDIRECT);
        let parameter_buffer =
            StaticBuffer::<Parameters>::empty(None, device, BufferUsages::STORAGE);
        let temporary_tile_buffer =
            StaticBuffer::<()>::empty_sized(None, device, tile_buffer_size, BufferUsages::STORAGE);
        let final_tile_buffer =
            StaticBuffer::<()>::empty_sized(None, device, tile_buffer_size, BufferUsages::STORAGE);
        let terrain_model_approximation_buffer = StaticBuffer::<TerrainModelApproximation>::empty(
            None,
            device,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );

        let prepare_indirect_bind_group = device.create_bind_group(
            "prepare_indirect_bind_group",
            &instantiate_layout(device, &create_prepare_indirect_layout()),
            &BindGroupEntries::single(&indirect_buffer),
        );
        let refine_tiles_bind_group = device.create_bind_group(
            "refine_tiles_bind_group",
            &instantiate_layout(device, &create_refine_tiles_layout()),
            &BindGroupEntries::sequential((
                &view_config_buffer,
                &terrain_model_approximation_buffer,
                &gpu_tile_tree.tile_tree_buffer,
                &gpu_tile_tree.origins_buffer,
                &final_tile_buffer,
                &temporary_tile_buffer,
                &parameter_buffer,
            )),
        );
        let terrain_view_bind_group = device.create_bind_group(
            "terrain_view_bind_group",
            &instantiate_layout(device, &create_terrain_view_layout()),
            &BindGroupEntries::sequential((
                &view_config_buffer,
                &terrain_model_approximation_buffer,
                &gpu_tile_tree.tile_tree_buffer,
                &gpu_tile_tree.origins_buffer,
                &final_tile_buffer,
            )),
        );

        Self {
            view_config_buffer,
            terrain_model_approximation_buffer,
            indirect_buffer,
            prepare_indirect_bind_group,
            refine_tiles_bind_group,
            terrain_view_bind_group,
        }
    }

    pub(super) fn refinement_count(&self) -> u32 {
        self.view_config_buffer.value().refinement_count
    }

    pub(crate) fn initialize(
        device: Res<RenderDevice>,
        mut terrain_view_data: ResMut<TerrainViewComponents<TerrainViewData>>,
        gpu_tile_trees: Res<TerrainViewComponents<GpuTileTree>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            if terrain_view_data.contains_key(&(terrain, view)) {
                continue;
            }

            let Some(gpu_tile_tree) = gpu_tile_trees.get(&(terrain, view)) else {
                continue;
            };

            terrain_view_data.insert(
                (terrain, view),
                TerrainViewData::new(&device, tile_tree, gpu_tile_tree),
            );
        }
    }

    pub(crate) fn extract(
        mut terrain_view_data: ResMut<TerrainViewComponents<TerrainViewData>>,
        tile_trees: Extract<Res<TerrainViewComponents<TileTree>>>,
        terrain_model_approximations: Extract<
            Res<TerrainViewComponents<TerrainModelApproximation>>,
        >,
    ) {
        for (&(terrain, view), tile_tree) in tile_trees.iter() {
            let Some(terrain_view_data) = terrain_view_data.get_mut(&(terrain, view)) else {
                continue;
            };
            let Some(terrain_model_approximation) =
                terrain_model_approximations.get(&(terrain, view))
            else {
                continue;
            };

            terrain_view_data
                .view_config_buffer
                .set_value(TerrainViewConfigUniform::from_tile_tree(tile_tree));

            terrain_view_data
                .terrain_model_approximation_buffer
                .set_value(terrain_model_approximation.clone());
        }
    }

    pub(crate) fn prepare(
        queue: Res<RenderQueue>,
        mut terrain_view_data: ResMut<TerrainViewComponents<TerrainViewData>>,
    ) {
        for data in &mut terrain_view_data.values_mut() {
            data.view_config_buffer.update(&queue);
            data.terrain_model_approximation_buffer.update(&queue);
        }
    }
}

pub struct SetTerrainViewBindGroup<const I: usize>;

impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetTerrainViewBindGroup<I> {
    type Param = SRes<TerrainViewComponents<TerrainViewData>>;
    // Bevy 0.15 split main- and render-world entity IDs. Our resources are
    // keyed by main-world entities, so we query the render-world view's
    // `MainEntity` component to translate back at draw time.
    type ViewQuery = &'static MainEntity;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        terrain_view_data: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let main_view = view.id();
        let Some(data) = terrain_view_data
            .into_inner()
            .get(&(item.main_entity().id(), main_view))
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, &data.terrain_view_bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawTerrainCommand;

impl<P: PhaseItem> RenderCommand<P> for DrawTerrainCommand {
    type Param = SRes<TerrainViewComponents<TerrainViewData>>;
    type ViewQuery = &'static MainEntity;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        _: Option<ROQueryItem<'w, '_, Self::ItemQuery>>,
        terrain_view_data: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let main_view = view.id();
        let Some(data) = terrain_view_data
            .into_inner()
            .get(&(item.main_entity().id(), main_view))
        else {
            return RenderCommandResult::Skip;
        };

        pass.draw_indirect(&data.indirect_buffer, 0);

        RenderCommandResult::Success
    }
}
