#!/usr/bin/env bash
# Download and preprocess a planar world-mesh terrain region for the top-level
# Elodin Editor assets directory.
#
# This is the editor-data variant of libs/bevy_world_mesh/scripts/render_region.sh:
# it runs from the repository root, writes into ./assets, rebuilds the region's
# atlas, and intentionally does not launch the renderer or capture a screenshot.
#
# Outputs:
#   assets/terrains/planar/<region>/source/{height,albedo}.png
#   assets/terrains/planar/<region>/region.toml
#   assets/terrains/planar/<region>/data/...
#   assets/terrains/planar/<region>/config.tc
#
# Usage:
#   ./scripts/prepare_editor_terrain_region.sh brienz
#   ./scripts/prepare_editor_terrain_region.sh death_valley
#   ./scripts/prepare_editor_terrain_region.sh mojave_desert
#
# Environment overrides:
#   WORLD_MESH_FETCH_WORKERS    rayon worker count for each provider fetch pool

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

usage() {
    cat >&2 <<'EOF'
usage: ./scripts/prepare_editor_terrain_region.sh <region>
  known regions: brienz, death_valley, mojave_desert
EOF
}

if [ "${1:-}" = "" ]; then
    usage
    exit 2
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi

region="$1"
if [[ ! "$region" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "invalid region name: $region" >&2
    echo "region names may only contain letters, numbers, '_' and '-'" >&2
    exit 2
fi

# Force the asset root expected by the editor's default configuration. The
# fetcher writes under ./assets relative to cwd, while the atlas writer honors
# ELODIN_ASSETS; setting both cwd and env keeps them aligned.
export ELODIN_ASSETS="$repo_root/assets"
export BEVY_ASSET_ROOT="$repo_root"
export WORLD_MESH_REGION="$region"

echo "==> Fetching real terrain for editor region: $region"
cargo run -p bevy_world_mesh --release --bin fetch_real_terrain \
    --features "fetch,regions,scenes" -- --region "$region"

echo "==> Rebuilding planar atlas in top-level assets for: $region"
rm -rf -- "assets/terrains/planar/$region/data" \
    "assets/terrains/planar/$region/config.tc"

cargo run -p bevy_world_mesh --release --bin preprocess --features "scenes"

terrain_dir="assets/terrains/planar/$region"
if [ ! -s "$terrain_dir/source/height.png" ] || [ ! -s "$terrain_dir/source/albedo.png" ]; then
    echo "error: fetch did not write source height/albedo PNGs under $terrain_dir/source" >&2
    exit 1
fi
if [ ! -s "$terrain_dir/region.toml" ]; then
    echo "error: fetch did not write $terrain_dir/region.toml" >&2
    exit 1
fi
if [ ! -s "$terrain_dir/config.tc" ]; then
    echo "error: preprocess did not write $terrain_dir/config.tc" >&2
    exit 1
fi
if [ ! -d "$terrain_dir/data/height" ] || [ ! -d "$terrain_dir/data/albedo" ]; then
    echo "error: preprocess did not write both height and albedo atlas directories under $terrain_dir/data" >&2
    exit 1
fi
height_tiles="$(find "$terrain_dir/data/height" -type f -name '*.bin' | wc -l)"
albedo_tiles="$(find "$terrain_dir/data/albedo" -type f -name '*.bin' | wc -l)"
if [ "$height_tiles" -eq 0 ] || [ "$height_tiles" -ne "$albedo_tiles" ]; then
    echo "error: unexpected atlas tile counts: height=$height_tiles albedo=$albedo_tiles" >&2
    exit 1
fi

echo "==> Terrain ready: $terrain_dir ($height_tiles height + $albedo_tiles albedo tiles)"
