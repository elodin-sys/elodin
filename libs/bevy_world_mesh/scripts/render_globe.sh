#!/usr/bin/env bash
# Render the real-world spherical Earth end-to-end:
#   1. fetch global cube-face TIFFs (height + albedo) from AWS Terrain Tiles
#      and EOX Sentinel-2 Cloudless. Writes
#      assets/terrains/spherical/source/{height,albedo}/face{0..5}.tif and
#      assets/terrains/spherical/globe.toml. The fetcher is incremental — it
#      skips faces that already exist on disk — so re-running is cheap.
#   2. wipe and rebuild the spherical tile atlas via preprocess_global.
#   3. launch world_mesh_globe with WORLD_MESH_SCREENSHOT pointed at
#      screenshots/world_mesh_globe.png and exit on capture.
#
# Usage:
#   ./scripts/render_globe.sh
#   ./scripts/render_globe.sh --zoom 6                       # cheaper fetch
#   ./scripts/render_globe.sh --face-size 1024               # smaller atlas
#
# Environment overrides:
#   SCREENSHOT_DELAY    (default: 15 — globe needs a few extra seconds for
#                        the tile tree to warm up around the first view)
#   SCREENSHOT_TIMEOUT  (default: 180)
#   FORCE_REFETCH       (default: 0; set to 1 to wipe source TIFFs before
#                        running so the fetcher re-downloads from scratch)

set -uo pipefail

cd "$(dirname "$0")/.."

DELAY="${SCREENSHOT_DELAY:-15}"
TIMEOUT="${SCREENSHOT_TIMEOUT:-180}"
OUT="screenshots/world_mesh_globe.png"
LOG="$(mktemp -t world_mesh_globe.XXXXXX)"

mkdir -p screenshots

if [ "${FORCE_REFETCH:-0}" = "1" ]; then
    echo "==> FORCE_REFETCH=1 — wiping spherical source TIFFs"
    rm -rf assets/terrains/spherical/source
fi

echo "==> Fetching global spherical Earth (z=7 default; pass --zoom N to override)"
cargo run --release --bin fetch_global_spherical --features "fetch,scenes" -- "$@"

echo "==> Rebuilding spherical atlas"
rm -rf assets/terrains/spherical/data assets/terrains/spherical/config.tc
cargo run --release --bin preprocess_global --features "scenes"

rm -f "$OUT"
echo "==> Launching world_mesh_globe with WORLD_MESH_SCREENSHOT=$OUT (delay=${DELAY}s, timeout=${TIMEOUT}s)"
WORLD_MESH_SCREENSHOT="$OUT" \
WORLD_MESH_SCREENSHOT_DELAY="$DELAY" \
WORLD_MESH_SCREENSHOT_EXIT=1 \
    cargo run --release --bin world_mesh_globe --features "scenes" > "$LOG" 2>&1 &
PID=$!

cleanup() {
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for _ in $(seq 1 "$TIMEOUT"); do
    if [ -s "$OUT" ]; then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

sleep 2

echo
echo "==== last 30 lines of log ===="
tail -30 "$LOG"

echo
echo "==== fatal-error scan ===="
FATAL_HITS=$(grep -E "thread .* panicked|wgpu error|Validation Error" "$LOG" | head -10 || true)
if [ -n "$FATAL_HITS" ]; then
    echo "$FATAL_HITS"
    echo
    echo "FAIL: detected panics or wgpu errors"
    exit 2
fi
echo "(no fatal panics or wgpu errors)"

if [ -s "$OUT" ]; then
    SIZE=$(wc -c < "$OUT" | tr -d ' ')
    echo
    echo "PASS: globe screenshot saved to $OUT ($SIZE bytes)"
    exit 0
else
    echo
    echo "FAIL: globe screenshot was not produced at $OUT"
    exit 3
fi
