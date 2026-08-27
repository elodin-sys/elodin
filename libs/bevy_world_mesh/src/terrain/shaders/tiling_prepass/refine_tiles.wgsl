#import bevy_terrain::types::{TileCoordinate, Coordinate}
#import bevy_terrain::bindings::{config, culling_view, view_config, final_tiles, temporary_tiles, parameters, terrain_model_approximation}
#import bevy_terrain::functions::{approximate_view_distance, compute_relative_position, position_local_to_world, normal_local_to_world, tile_count, compute_subdivision_coordinate}

fn in_tile_buffer(index: i32) -> bool {
    return index >= 0 && index < i32(view_config.tile_count);
}

fn parent_index(id: u32) -> i32 {
    return i32(view_config.tile_count - 1u) * clamp(parameters.counter, 0, 1) - i32(id) * parameters.counter;
}

fn final_index() -> i32 {
    return atomicAdd(&parameters.final_index, 1);
}

fn should_be_divided(tile: TileCoordinate) -> bool {
    let coordinate    = compute_subdivision_coordinate(Coordinate(tile.side, tile.lod, tile.xy, vec2<f32>(0.0)));
    let view_distance = approximate_view_distance(coordinate, culling_view.world_position);

    return view_distance < view_config.subdivision_distance / tile_count(tile.lod);
}

fn emit_final(tile: TileCoordinate) {
    let index = final_index();
    if (in_tile_buffer(index)) {
        final_tiles[index] = tile;
    }
}

fn subdivide(tile: TileCoordinate) -> bool {
    let first = atomicAdd(&parameters.child_index, 4 * parameters.counter);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if (!in_tile_buffer(first + i32(i) * parameters.counter)) {
            return false;
        }
    }

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let index = first + i32(i) * parameters.counter;
        let child_xy  = vec2<u32>((tile.xy.x << 1u) + (i & 1u), (tile.xy.y << 1u) + (i >> 1u & 1u));
        let child_lod = tile.lod + 1u;
        temporary_tiles[index] = TileCoordinate(tile.side, child_lod, child_xy);
    }

    return true;
}

@compute @workgroup_size(64, 1, 1)
fn refine_tiles(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    if (invocation_id.x >= parameters.tile_count) { return; }

    let parent = parent_index(invocation_id.x);
    if (!in_tile_buffer(parent)) { return; }

    let tile = temporary_tiles[parent];

    if (should_be_divided(tile)) {
        if (!subdivide(tile)) {
            emit_final(tile);
        }
    } else {
        emit_final(tile);
    }
}
