# Terrain capture tools

Developer tool for inspecting and exporting an existing native cache. For normal simulation and editor use, follow the [terrain CLI guide](../public/content/reference/terrain.md). Prepare declared geographic coverage with `elodin terrain prepare sim.py`; the exporter only covers views already present in the cache.

## Exporting a cached Photo 3D snapshot

For a local offline prototype, first visit the required views in the native editor
with Photo 3D enabled. Then export the tiles already present in Cesium Native's
SQLite cache, from the repository root inside `nix develop`:

```bash
uv run python scripts/export_photo3d_cache.py examples/falcon9/assets/terrain/Falcon9
FALCON9_TERRAIN_ASSET=terrain/Falcon9/tileset.json elodin editor examples/falcon9/main.py
```

Use a fresh recording so the DB ingests this folder. The exporter makes no network
requests and leaves the source cache unchanged. It copies self-contained GLBs,
rewrites tileset links to relative paths, preserves attribution, and excludes
access keys and session parameters from resource URLs. `--cache` selects a
different `requests.sqlite`; the output directory must not already exist.

`cache-manifest.json` reports tile counts, bytes and missing references. This is
only the reachable cached subset: uncached content links are removed while tile
bounds and hierarchy are preserved, so new camera positions may reveal gaps.
It does not guarantee coverage of a whole region or
flight. Implicit tilesets and GLBs with external dependencies are rejected.
This prototype is separate from geographic preparation and heightmap generation.
