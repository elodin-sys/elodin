#define_import_path bevy_terrain::bindings

#import bevy_terrain::types::{TerrainViewConfig, TerrainConfig, TileTreeEntry, TileCoordinate, AttachmentConfig, TerrainModelApproximation, CullingData, IndirectBuffer, Parameters}
#import bevy_pbr::mesh_types::Mesh

// terrain bindings
@group(1) @binding(0)
var<storage> mesh: array<Mesh>;
@group(1) @binding(1)
var<uniform> config: TerrainConfig;
@group(1) @binding(2)
var<uniform> attachments: array<AttachmentConfig, 8u>;
@group(1) @binding(3)
var atlas_sampler: sampler;
@group(1) @binding(4)
var attachment0_atlas: texture_2d_array<f32>;
@group(1) @binding(5)
var attachment1_atlas: texture_2d_array<f32>;
@group(1) @binding(6)
var attachment2_atlas: texture_2d_array<f32>;
@group(1) @binding(7)
var attachment3_atlas: texture_2d_array<f32>;
@group(1) @binding(8)
var attachment4_atlas: texture_2d_array<f32>;
@group(1) @binding(9)
var attachment5_atlas: texture_2d_array<f32>;
@group(1) @binding(10)
var attachment6_atlas: texture_2d_array<f32>;
@group(1) @binding(11)
var attachment7_atlas: texture_2d_array<f32>;

// terrain view bindings
@group(2) @binding(0)
var<uniform> view_config: TerrainViewConfig;
@group(2) @binding(1)
var<uniform> terrain_model_approximation: TerrainModelApproximation;
@group(2) @binding(2)
var<storage> tile_tree: array<TileTreeEntry>;
@group(2) @binding(3)
var<storage> origins: array<vec2<u32>>;
@group(2) @binding(4)
var<storage> geometry_tiles: array<TileCoordinate>;

// refine geometry_tiles bindings
@group(2) @binding(4)
var<storage, read_write> final_tiles: array<TileCoordinate>;
@group(2) @binding(5)
var<storage, read_write> temporary_tiles: array<TileCoordinate>;
@group(2) @binding(6)
var<storage, read_write> parameters: Parameters;

@group(3) @binding(0)
var<storage, read_write> indirect_buffer: IndirectBuffer;

// culling bindings
@group(0) @binding(0)
var<uniform> culling_view: CullingData;