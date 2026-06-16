#import bevy_terrain::types::TileCoordinate
#import bevy_terrain::bindings::{view_config, temporary_tiles, parameters, indirect_buffer}

@compute @workgroup_size(1, 1, 1)
fn prepare_root() {
    parameters.counter = -1;
    atomicStore(&parameters.child_index, i32(view_config.tile_count - 1u));
    atomicStore(&parameters.final_index, 0);

#ifdef SPHERICAL
    parameters.tile_count = 6u;

    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        temporary_tiles[i] = TileCoordinate(i, 0u, vec2<u32>(0u));
    }
#else
    parameters.tile_count = 1u;

    temporary_tiles[0] = TileCoordinate(0u, 0u, vec2<u32>(0u));
#endif

    indirect_buffer.workgroup_count = vec3<u32>(1u, 1u, 1u);
}

@compute @workgroup_size(1, 1, 1)
fn prepare_next() {
    if (parameters.counter == 1) {
        parameters.tile_count = u32(atomicExchange(&parameters.child_index, i32(view_config.tile_count - 1u)));
    }
    else {
        parameters.tile_count = view_config.tile_count - 1u - u32(atomicExchange(&parameters.child_index, 0));
    }

    parameters.counter = -parameters.counter;
    indirect_buffer.workgroup_count.x = (parameters.tile_count + 63u) / 64u;
}

@compute @workgroup_size(1, 1, 1)
fn prepare_render() {
    let tile_count = u32(atomicLoad(&parameters.final_index));
    let vertex_count = view_config.vertices_per_tile * tile_count;

    indirect_buffer.workgroup_count = vec3<u32>(vertex_count, 1u, 0u);
}