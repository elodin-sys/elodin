# Editor Plugins

This folder contains Bevy plugins used by the Elodin editor.

## `osm_world` plugin

`osm_world` adds lightweight geospatial context around the current simulated vehicle:

- building meshes (extruded from OSM polygons),
- roads and land/water areas,
- hover metadata on buildings,
- tile streaming with local cache and incremental frame-by-frame rendering.

### Prototype scope (important)

The use of **OpenStreetMap + Overpass** in this plugin is intentionally for a
**prototype** workflow.

It is meant to quickly validate editor UX and visual context for demos, not to be a
final geodata pipeline.

### How it works

1. Determine the tracked center (`bdx.world_pos`, `drone.world_pos`, `target.world_pos`, or configured component).
2. Convert center to local tiles.
3. Load tile data from cache when available.
4. Fetch missing tiles from Overpass, then persist JSON locally.
5. Build renderables and apply them progressively over frames.

### Default behavior

- Enabled by default in editor builds (non-wasm).
- Default origin is fixed near Century City:
  - lat: `34.054085661510506`
  - lon: `-118.42558289792434`
- Local cache defaults to: `~/.cache/elodin/osm_tiles`

### Main tuning environment variables

- `ELODIN_OSM_ENABLED`
- `ELODIN_OSM_ORIGIN_LAT`, `ELODIN_OSM_ORIGIN_LON`
- `ELODIN_OSM_TILE_SIZE_M`
- `ELODIN_OSM_TILE_RADIUS`
- `ELODIN_OSM_PREFETCH_TILE_RADIUS`
- `ELODIN_OSM_MAX_INFLIGHT_FETCHES`
- `ELODIN_OSM_SPAWN_PER_FRAME`
- `ELODIN_OSM_CACHE_DIR`
- `ELODIN_OSM_OVERPASS_URLS`
- `ELODIN_OSM_TRACK_COMPONENT`

### Quick run

From repository root:

```bash
cargo run -p elodin -- editor examples/drone/main.py
```

or:

```bash
cargo run -p elodin -- editor examples/rc-jet/main.py
```
