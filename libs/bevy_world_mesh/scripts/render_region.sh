#!/usr/bin/env bash
# Render a real-world region end-to-end:
#   1. fetch real DEM + imagery tiles for the named region preset
#      (writes assets/terrains/planar/<region>/source/{height,albedo}.png +
#      assets/terrains/planar/<region>/region.toml)
#   2. wipe and rebuild the per-region tile atlas
#      (assets/terrains/planar/<region>/data + .../config.tc)
#   3. launch the renderer with WORLD_MESH_REGION=<region> and
#      WORLD_MESH_SCREENSHOT pointed at screenshots/world_mesh_<region>.png
#
# Each region writes to its own subdirectory so multiple regions coexist on
# disk; running this script for one region never touches another's data.
#
# Usage:
#   ./scripts/render_region.sh brienz
#   ./scripts/render_region.sh death_valley
#   ./scripts/render_region.sh mojave_desert
#
# Environment overrides:
#   SCREENSHOT_DELAY            (default: 8)
#   SCREENSHOT_TIMEOUT          (default: 90)
#   WORLD_MESH_FETCH_WORKERS    rayon worker count for *each* of the two
#                               per-provider fetch pools in
#                               `fetch_real_terrain` (default:
#                               `std::thread::available_parallelism()`,
#                               i.e. the machine's CPU count). The terrain
#                               and imagery phases run alongside each other
#                               in phase 1, so the live worker total is
#                               `2 * WORLD_MESH_FETCH_WORKERS`. Network-bound
#                               work scales fine beyond CPU count; lower
#                               this (e.g. 4) only if a tile provider starts
#                               rate-limiting you.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" = "" ]; then
    echo "usage: $0 <region>" >&2
    echo "  known regions: brienz, death_valley, mojave_desert" >&2
    exit 2
fi
region="$1"
export WORLD_MESH_REGION="$region"

echo "==> Fetching real terrain for region: $region"
cargo run --release --bin fetch_real_terrain --features "fetch,regions,scenes" -- --region "$region"

echo "==> Rebuilding planar atlas for the new source data"
rm -rf "assets/terrains/planar/$region/data" "assets/terrains/planar/$region/config.tc"
cargo run --release --bin preprocess --features "scenes"

mkdir -p screenshots
out="screenshots/world_mesh_${region}.png"
rm -f "$out"

echo "==> Capturing $out"
SCREENSHOT_OUT="$out" \
SCREENSHOT_DELAY="${SCREENSHOT_DELAY:-8}" \
SCREENSHOT_TIMEOUT="${SCREENSHOT_TIMEOUT:-90}" \
    ./scripts/phase_screenshot.sh
