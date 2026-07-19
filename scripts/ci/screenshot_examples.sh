#!/usr/bin/env bash
# Capture editor screenshots for a list of examples using the env-gated
# EnvScreenshotPlugin (ELODIN_SCREENSHOT*). One editor process at a time;
# each run is bounded by a watchdog since a hung capture would block forever.
#
# Usage: scripts/ci/screenshot_examples.sh <output-dir> [example ...]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:?usage: screenshot_examples.sh <output-dir> [example ...]}"
shift
EXAMPLES=("$@")
if [ ${#EXAMPLES[@]} -eq 0 ]; then
    EXAMPLES=(ball three-body drone rc-jet apollo-lander video-stream sensor-camera cube-sat voyager geo-frames)
fi

ELODIN_BIN="${ELODIN_BIN:-$REPO_ROOT/target/release/elodin}"
DELAY="${ELODIN_SCREENSHOT_DELAY:-20}"
WATCHDOG="${SCREENSHOT_WATCHDOG:-180}"

mkdir -p "$OUT_DIR"

run_with_watchdog() {
    local secs="$1"
    shift
    "$@" &
    local pid=$!
    (
        sleep "$secs"
        kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
        sleep 5
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
    ) &
    local watchdog=$!
    wait "$pid"
    local status=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    return "$status"
}

for example in "${EXAMPLES[@]}"; do
    out="$OUT_DIR/$example.png"
    log="$OUT_DIR/$example.log"
    echo "=== $example -> $out"
    rm -f "$out"
    ELODIN_SCREENSHOT="$out" \
    ELODIN_SCREENSHOT_DELAY="$DELAY" \
    ELODIN_SCREENSHOT_EXIT=1 \
    run_with_watchdog "$WATCHDOG" \
        "$ELODIN_BIN" editor "$REPO_ROOT/examples/$example/main.py" >"$log" 2>&1
    if [ -s "$out" ]; then
        echo "    ok"
    else
        echo "    FAILED (see $log)"
    fi
done
