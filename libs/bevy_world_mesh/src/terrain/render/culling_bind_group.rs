use crate::terrain::{
    render::instantiate_layout, terrain_data::gpu_tile_tree::GpuTileTree,
    terrain_view::TerrainViewComponents, util::StaticBuffer,
};
use bevy::{
    prelude::*,
    render::{
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::ExtractedView,
    },
};
use std::collections::HashMap;
use std::ops::Deref;

pub(crate) fn create_culling_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "culling_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::COMPUTE,
            uniform_buffer::<CullingUniform>(false), // culling data
        ),
    )
}

pub fn planes(view_projection: &Mat4) -> [Vec4; 5] {
    let row3 = view_projection.row(3);
    let mut planes = [default(); 5];
    for (i, plane) in planes.iter_mut().enumerate() {
        let row = view_projection.row(i / 2);
        *plane = if (i & 1) == 0 && i != 4 {
            row3 + row
        } else {
            row3 - row
        };
    }

    planes
}

#[derive(Default, ShaderType)]
pub struct CullingUniform {
    world_position: Vec3,
    view_proj: Mat4,
    planes: [Vec4; 5],
}

impl From<&ExtractedView> for CullingUniform {
    fn from(view: &ExtractedView) -> Self {
        Self {
            world_position: view.world_from_view.translation(),
            view_proj: view.world_from_view.to_matrix().inverse(),
            planes: default(),
        }
    }
}

#[derive(Component)]
pub struct CullingBindGroup(BindGroup);

impl Deref for CullingBindGroup {
    type Target = BindGroup;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl CullingBindGroup {
    fn new(device: &RenderDevice, culling_uniform: CullingUniform) -> Self {
        let culling_buffer = StaticBuffer::<CullingUniform>::create(
            None,
            device,
            &culling_uniform,
            BufferUsages::UNIFORM,
        );

        let bind_group = device.create_bind_group(
            None,
            &instantiate_layout(device, &create_culling_layout()),
            &BindGroupEntries::single(&culling_buffer),
        );

        Self(bind_group)
    }

    pub(crate) fn prepare(
        device: Res<RenderDevice>,
        gpu_tile_trees: Res<TerrainViewComponents<GpuTileTree>>,
        extracted_views: Query<(&ExtractedView, &MainEntity)>,
        mut culling_bind_groups: ResMut<TerrainViewComponents<CullingBindGroup>>,
    ) {
        // Bevy 0.15 split main-world and render-world entity IDs. Our terrain
        // resources are keyed by main-world entities, but `ExtractedView` lives
        // on render-world entities. Build a one-shot lookup so we can map back.
        let by_main: HashMap<Entity, &ExtractedView> = extracted_views
            .iter()
            .map(|(view, main_entity)| (main_entity.id(), view))
            .collect();

        for &(terrain, view) in gpu_tile_trees.keys() {
            let Some(extracted_view) = by_main.get(&view) else {
                continue;
            };

            culling_bind_groups.insert(
                (terrain, view),
                CullingBindGroup::new(&device, (*extracted_view).into()),
            );
        }
    }
}
