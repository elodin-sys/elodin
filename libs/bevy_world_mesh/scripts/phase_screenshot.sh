#!/usr/bin/env bash
# Phase 1+ screenshot harness. Uses Bevy's in-process ScreenshotManager
# (no macOS Screen Recording permission required).
#
#   1. preprocess into a tile atlas (if missing for the active region)
#   2. launch the renderer with WORLD_MESH_REGION + WORLD_MESH_SCREENSHOT set;
#      the in-app harness writes a PNG after WORLD_MESH_SCREENSHOT_DELAY
#      seconds and exits.
#
# Region selection: WORLD_MESH_REGION env var (default: death_valley). The
# upstream caller (render_region.sh) sets this; running this script standalone
# uses the death_valley default. Source PNGs are expected at
# assets/terrains/planar/<region>/source/{height,albedo}.png — populate them
# via fetch_real_terrain (real data) or synthesize_height (synthetic, writes
# to the flat terrains/planar/source/ path which doesn't match the per-region
# layout; standalone synth runs are not supported here today).
#
# Environment overrides:
#   SCREENSHOT_DELAY  - seconds to wait before capturing (default: 8)
#   SCREENSHOT_OUT    - output path (default: screenshots/phase1_planar.png)
#   SCREENSHOT_TIMEOUT - hard kill after this many seconds (default: 60)
#   WORLD_MESH_REGION - active region (default: death_valley)

set -uo pipefail

cd "$(dirname "$0")/.."

DELAY="${SCREENSHOT_DELAY:-8}"
OUT="${SCREENSHOT_OUT:-screenshots/phase1_planar.png}"
TIMEOUT="${SCREENSHOT_TIMEOUT:-60}"
REGION="${WORLD_MESH_REGION:-death_valley}"
export WORLD_MESH_REGION="$REGION"
LOG="$(mktemp -t world_mesh_phase.XXXXXX)"

mkdir -p "$(dirname "$OUT")"

if [ ! -f "assets/terrains/planar/$REGION/source/height.png" ]; then
    echo "==> Missing height PNG at assets/terrains/planar/$REGION/source/height.png"
    echo "    run ./scripts/render_region.sh $REGION (or fetch_real_terrain --region $REGION)" >&2
    exit 4
fi

if [ ! -f "assets/terrains/planar/$REGION/config.tc" ]; then
    echo "==> Preprocessing terrain atlas for region $REGION"
    cargo run --release --bin preprocess --features "scenes"
fi

echo "==> Launching renderer for region=$REGION with WORLD_MESH_SCREENSHOT=$OUT (delay=${DELAY}s, timeout=${TIMEOUT}s)"
WORLD_MESH_SCREENSHOT="$OUT" \
WORLD_MESH_SCREENSHOT_DELAY="$DELAY" \
WORLD_MESH_SCREENSHOT_EXIT=1 \
    cargo run --release --features "scenes" > "$LOG" 2>&1 &
PID=$!

cleanup() {
    if kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Poll for the PNG to appear, with a hard timeout.
for _ in $(seq 1 "$TIMEOUT"); do
    if [ -s "$OUT" ]; then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        # process exited
        break
    fi
    sleep 1
done

# Give the process a couple more seconds to flush + exit gracefully.
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
    echo "PASS: screenshot saved to $OUT ($SIZE bytes)"
    exit 0
else
    echo
    echo "FAIL: screenshot was not produced at $OUT"
    exit 3
fi
