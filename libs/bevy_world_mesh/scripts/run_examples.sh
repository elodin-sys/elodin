#!/usr/bin/env bash
# Drive all five upstream `bevy_terrain` examples plus the top-level
# `world_mesh` app and capture a screenshot of each via the
# `EnvScreenshotPlugin` env-var harness shipped with the crate.
#
# Outputs:
#   screenshots/example_minimal.png
#   screenshots/example_planar.png
#   screenshots/example_spherical.png
#   screenshots/world_mesh_thesis_map.png
#
# (preprocess_planar / preprocess_spherical have no visible output; they
#  build the on-disk atlases that the other three examples consume, so
#  this script runs them but doesn't screenshot them.)

set -uo pipefail

cd "$(dirname "$0")/.."

DELAY="${SCREENSHOT_DELAY:-12}"
TIMEOUT="${SCREENSHOT_TIMEOUT:-90}"
LOG_DIR="$(mktemp -d -t world_mesh_examples.XXXXXX)"

mkdir -p screenshots

run_with_screenshot() {
    local label="$1"
    local out="screenshots/$2"
    shift 2
    local log="$LOG_DIR/$label.log"

    echo "==> $label  (out=$out)"
    rm -f "$out"
    WORLD_MESH_SCREENSHOT="$out" \
        WORLD_MESH_SCREENSHOT_DELAY="$DELAY" \
        WORLD_MESH_SCREENSHOT_EXIT=1 \
        "$@" >"$log" 2>&1 &
    local pid=$!

    local elapsed=0
    while [ "$elapsed" -lt "$TIMEOUT" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        if [ -s "$out" ]; then
            sleep 2
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$pid" 2>/dev/null || true
    fi

    if grep -E "thread .* panicked|wgpu error|Validation Error" "$log" > /dev/null; then
        echo "    FAIL: panic detected (log: $log)"
        tail -8 "$log"
        return 1
    fi

    if [ -s "$out" ]; then
        local sz
        sz=$(wc -c < "$out" | tr -d ' ')
        echo "    PASS: $out ($sz bytes)"
    else
        echo "    WARN: no screenshot saved (log: $log)"
        tail -8 "$log"
    fi
}

run_until_exit() {
    local label="$1"
    shift
    local log="$LOG_DIR/$label.log"
    local timeout="${TIMEOUT}"

    echo "==> $label  (no screenshot — atlas builder)"
    "$@" >"$log" 2>&1 &
    local pid=$!

    local elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$pid" 2>/dev/null || true
    fi

    if grep -E "thread .* panicked|wgpu error|Validation Error" "$log" > /dev/null; then
        echo "    FAIL: panic detected (log: $log)"
        tail -8 "$log"
        return 1
    fi

    if grep -E "Preprocessing took" "$log" > /dev/null; then
        echo "    PASS: $(grep -E 'Preprocessing took' "$log" | head -1)"
    else
        echo "    WARN: no \"Preprocessing took\" line (log: $log)"
        tail -8 "$log"
    fi
}

# Make sure the source assets exist before any of the preprocessing examples
# try to find them.
if [ ! -f assets/terrains/planar/source/height.png ] \
    || [ ! -f assets/terrains/planar/source/albedo.png ]; then
    echo "==> Synthesizing source PNGs (height, albedo, gradients)"
    cargo run --release --bin synthesize_height --features "synth" >/dev/null
fi

if [ ! -f assets/terrains/spherical/source/height/face0.tif ]; then
    echo "==> Synthesizing spherical cube-face heightmaps"
    cargo run --release --bin synthesize_spherical_faces --features "synth" >/dev/null
fi

# 1. Build the planar atlas (consumed by minimal + planar).
rm -rf assets/terrains/planar/data assets/terrains/planar/config.tc
run_until_exit "preprocess_planar" \
    cargo run --release --example preprocess_planar \
        --features "bevy/embedded_watcher"

# 2. Render the planar atlas with the debug material.
run_with_screenshot "minimal" "example_minimal.png" \
    cargo run --release --example minimal \
        --features "high_precision,bevy/embedded_watcher"

# 3. Render the planar atlas with the gradient terrain material.
run_with_screenshot "planar" "example_planar.png" \
    cargo run --release --example planar \
        --features "high_precision,bevy/embedded_watcher"

# 4. Build the spherical atlas (consumed by spherical).
rm -rf assets/terrains/spherical/data assets/terrains/spherical/config.tc
run_until_exit "preprocess_spherical" \
    cargo run --release --example preprocess_spherical \
        --features "bevy/embedded_watcher"

# 5. Render the spherical atlas.
run_with_screenshot "spherical" "example_spherical.png" \
    cargo run --release --example spherical \
        --features "high_precision,bevy/embedded_watcher"

# 6. The top-level world_mesh app rendering its real-data atlas.
run_with_screenshot "world_mesh" "world_mesh_thesis_map.png" \
    cargo run --release --features "scenes"

echo
echo "==> Done. Logs in $LOG_DIR; screenshots in screenshots/"
ls -la screenshots/*.png 2>/dev/null || true
