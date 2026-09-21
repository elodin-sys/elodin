#!/usr/bin/env bash
# Capture one branch's regression pass: run each example in the editor,
# record a screenshot, the full log, and the exit code.
#
# Usage (repo root, inside `nix develop`, on the branch you want to capture):
#   bash .cursor/skills/branch-regression/capture_branch.sh <out-dir> [example ...]
#
# Produces per example: <out-dir>/<example>.png, .log, .exit
# Headless examples (no screenshot) are run with `elodin run` instead when
# listed in HEADLESS_EXAMPLES below.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="${1:?usage: capture_branch.sh <out-dir> [example ...]}"
shift
EXAMPLES=("$@")
if [ ${#EXAMPLES[@]} -eq 0 ]; then
    # Known-good editor gallery set (same as scripts/ci/screenshot_examples.sh).
    EXAMPLES=(ball three-body drone rc-jet apollo-lander video-stream sensor-camera cube-sat voyager geo-frames)
fi

# Examples that cannot render an editor scene; exercised headless (log + exit only).
HEADLESS_EXAMPLES="frames linalg stablehlo cube-sat-pysim"

ELODIN_BIN="${ELODIN_BIN:-$REPO_ROOT/target/release/elodin}"
DELAY="${ELODIN_SCREENSHOT_DELAY:-20}"
WATCHDOG="${SCREENSHOT_WATCHDOG:-180}"
HEADLESS_SECS="${HEADLESS_SECS:-30}"

if [ ! -x "$ELODIN_BIN" ]; then
    echo "missing elodin binary: $ELODIN_BIN (run: just install)" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
git -C "$REPO_ROOT" rev-parse HEAD >"$OUT_DIR/commit.txt"
git -C "$REPO_ROOT" branch --show-current >>"$OUT_DIR/commit.txt"

run_with_watchdog() {
    local secs="$1"
    shift
    setsid "$@" &
    local pid=$!
    (
        sleep "$secs"
        kill -0 "$pid" 2>/dev/null && kill -TERM -- -"$pid" 2>/dev/null
        sleep 5
        kill -0 "$pid" 2>/dev/null && kill -KILL -- -"$pid" 2>/dev/null
    ) &
    local watchdog=$!
    wait "$pid"
    local status=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    # s10 children restart on plain kill; make sure the group is gone.
    kill -9 -- -"$pid" 2>/dev/null
    return "$status"
}

wait_port_free() {
    for _ in $(seq 1 40); do
        ss -ltn 2>/dev/null | grep -q ":2240 " || return 0
        sleep 0.5
    done
    echo "warning: port 2240 still bound" >&2
}

for example in "${EXAMPLES[@]}"; do
    out="$OUT_DIR/$example.png"
    log="$OUT_DIR/$example.log"
    rm -f "$out"

    # Stale DBs that commonly break re-runs.
    case "$example" in
        video-stream) rm -rf "$REPO_ROOT/video-stream-db" ;;
        voyager) rm -rf "$REPO_ROOT/examples/voyager/dbs/voyager" "$REPO_ROOT/dbs/voyager" ;;
    esac

    wait_port_free
    if echo " $HEADLESS_EXAMPLES " | grep -q " $example "; then
        echo "=== $example (headless, ${HEADLESS_SECS}s)"
        run_with_watchdog "$HEADLESS_SECS" \
            "$ELODIN_BIN" run "$REPO_ROOT/examples/$example/main.py" >"$log" 2>&1
        status=$?
        # A watchdog TERM/KILL of a healthy long-running sim is a pass; record 0.
        case "$status" in
            124|137|143) status=0 ;;
        esac
    else
        echo "=== $example -> $out (delay=${DELAY}s)"
        ELODIN_SCREENSHOT="$out" \
        ELODIN_SCREENSHOT_DELAY="$DELAY" \
        ELODIN_SCREENSHOT_EXIT=1 \
        run_with_watchdog "$WATCHDOG" \
            "$ELODIN_BIN" editor "$REPO_ROOT/examples/$example/main.py" >"$log" 2>&1
        status=$?
        [ -s "$out" ] || echo "    missing/empty screenshot" >&2
    fi
    echo "$status" >"$OUT_DIR/$example.exit"
    echo "    exit=$status"
done
wait_port_free
