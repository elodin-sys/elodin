# `osm_world` (prototype geodata pipeline)

`osm_world` exists to validate one thing quickly: using real geodata polygons and
their `key/value` tags to enrich simulation behavior.

The added value is not only rendering buildings/roads/water/land. The key point is
that each polygon keeps metadata that systems can use (example: schools, piers,
roads, landuse) to drive simulation logic and editor interactions.

This is intentionally a **POC pipeline**:

- source: OpenStreetMap + Overpass,
- goal: fast iteration on UX and polygon-tag-driven simulation ideas,
- non-goal: final production geodata architecture.

Runtime flow is cache-first and incremental: when a tile is needed it is loaded from
local cache if present, otherwise fetched then persisted locally, and entities are
spawned progressively frame-by-frame.

For production, a **GeoParquet-based pipeline** is a likely next step to get stronger
schemas, reproducible offline datasets, and better scaling.

Activation rule:

- `main_osm_world.py` entrypoints: OSM world overlay is enabled.

This keeps default examples unchanged and isolates geodata behavior to dedicated
OSM entrypoints.

Entrypoints (drone):

```bash
# Default example, no geographic overlay
cargo run -p elodin -- editor examples/drone/main.py

# OSM world overlay enabled
cargo run -p elodin -- editor examples/drone/main_osm_world.py
```

Entrypoints (rc-jet):

```bash
# Default example, no geographic overlay
cargo run -p elodin -- editor examples/rc-jet/main.py

# OSM world overlay enabled
cargo run -p elodin -- editor examples/rc-jet/main_osm_world.py
```
