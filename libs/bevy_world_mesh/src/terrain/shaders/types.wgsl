#define_import_path bevy_terrain::types

struct TerrainConfig {
    lod_count: u32,
    min_height: f32,
    max_height: f32,
    scale: f32,
}

struct TerrainViewConfig {
    tree_size: u32,
    tile_count: u32,
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

struct TileCoordinate {
    side: u32,
    lod: u32,
    xy: vec2<u32>,
}

struct Coordinate {
    side: u32,
    lod: u32,
    xy: vec2<u32>,
    uv: vec2<f32>,
#ifdef FRAGMENT
    uv_dx: vec2<f32>,
    uv_dy: vec2<f32>,
#endif
}

struct Parameters {
    tile_count: u32,
    counter: i32,
    child_index: atomic<i32>,
    final_index: atomic<i32>,
}

struct Blend {
    lod: u32,
    ratio: f32,
}

struct TileTreeEntry {
    atlas_index: u32,
    atlas_lod: u32,
}

// A tile inside the tile atlas, looked up based on the view of a tile tree.
struct AtlasTile {
    index: u32,
    coordinate: Coordinate,
}

struct BestLookup {
    tile: AtlasTile,
    tile_tree_uv: vec2<f32>,
}

struct AttachmentConfig {
    size: f32,
    scale: f32,
    offset: f32,
    _padding: u32,
}

struct SideParameter {
    view_xy: vec2<i32>,
    view_uv: vec2<f32>,
    c: vec3<f32>,
    c_s: vec3<f32>,
    c_t: vec3<f32>,
    c_ss: vec3<f32>,
    c_st: vec3<f32>,
    c_tt: vec3<f32>,
}

struct TerrainModelApproximation {
    origin_lod: u32,
    approximate_height: f32,
    sides: array<SideParameter, 6>,
}

struct IndirectBuffer {
    workgroup_count: vec3<u32>,
}

struct CullingData {
    world_position: vec3<f32>,
    view_proj: mat4x4<f32>,
    planes: array<vec4<f32>, 5>,
}
