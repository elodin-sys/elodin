#!/usr/bin/env bash
# Download and preprocess spherical world-mesh terrain data for the top-level
# Elodin Editor assets directory.
#
# This is the editor-data variant of libs/bevy_world_mesh/scripts/render_globe.sh:
# it runs from the repository root, writes into ./assets, rebuilds the spherical
# atlas, and intentionally does not launch the renderer or capture a screenshot.
#
# Outputs:
#   assets/terrains/spherical/source/height/face{0..5}.tif
#   assets/terrains/spherical/source/albedo/face{0..5}.png
#   assets/terrains/spherical/globe.toml
#   assets/terrains/spherical/data/...
#   assets/terrains/spherical/config.tc
#
# Usage:
#   ./scripts/prepare_editor_spherical_terrain.sh
#   ./scripts/prepare_editor_spherical_terrain.sh --zoom 6
#   ./scripts/prepare_editor_spherical_terrain.sh --face-size 1024
#
# Environment overrides:
#   FORCE_REFETCH                set to 1 to wipe source faces before fetching
#   WORLD_MESH_FETCH_WORKERS     rayon worker count for each provider fetch pool

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

usage() {
    cat >&2 <<'EOF'
usage: ./scripts/prepare_editor_spherical_terrain.sh [fetch_global_spherical args]
  common args: --zoom N, --face-size N
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

# Force the asset root expected by the editor's default configuration. The
# fetcher writes under ./assets relative to cwd, while the atlas writer honors
# ELODIN_ASSETS_DIR; setting both cwd and env keeps them aligned.
export ELODIN_ASSETS_DIR="$repo_root/assets"
export BEVY_ASSET_ROOT="$repo_root"

terrain_dir="assets/terrains/spherical"

if [ "${FORCE_REFETCH:-0}" = "1" ]; then
    echo "==> FORCE_REFETCH=1 - wiping spherical source faces"
    rm -rf -- "$terrain_dir/source"
fi

echo "==> Fetching global spherical terrain for editor assets"
cargo run -p bevy_world_mesh --release --bin fetch_global_spherical \
    --features "fetch,scenes" -- "$@"

echo "==> Rebuilding spherical atlas in top-level assets"
rm -rf -- "$terrain_dir/data" "$terrain_dir/config.tc"

cargo run -p bevy_world_mesh --release --bin preprocess_global --features "scenes"

if [ ! -s "$terrain_dir/globe.toml" ]; then
    echo "error: fetch did not write $terrain_dir/globe.toml" >&2
    exit 1
fi
for face in 0 1 2 3 4 5; do
    if [ ! -s "$terrain_dir/source/height/face${face}.tif" ]; then
        echo "error: missing height source face $terrain_dir/source/height/face${face}.tif" >&2
        exit 1
    fi
    if [ ! -s "$terrain_dir/source/albedo/face${face}.png" ]; then
        echo "error: missing albedo source face $terrain_dir/source/albedo/face${face}.png" >&2
        exit 1
    fi
done
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

echo "==> Spherical terrain ready: $terrain_dir ($height_tiles height + $albedo_tiles albedo tiles)"
