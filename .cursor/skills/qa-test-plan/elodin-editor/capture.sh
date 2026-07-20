#!/usr/bin/env bash
# Capture one editor screenshot via EnvScreenshotPlugin.
# Usage: capture.sh <example-name> <out.png> [delay_secs]
# Env: ELODIN_BIN (default: $REPO_ROOT/target/release/elodin), SCREENSHOT_WATCHDOG (default 180)
set -euo pipefail

# elodin-editor/ → qa-test-plan/ → skills/ → .cursor/ → <repo root>
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXAMPLE="${1:?usage: capture.sh <example> <out.png> [delay]}"
OUT="${2:?usage: capture.sh <example> <out.png> [delay]}"
DELAY="${3:-${ELODIN_SCREENSHOT_DELAY:-15}}"
ELODIN_BIN="${ELODIN_BIN:-$REPO_ROOT/target/release/elodin}"
WATCHDOG="${SCREENSHOT_WATCHDOG:-180}"
LOG="${OUT%.png}.log"

if [[ ! -x "$ELODIN_BIN" ]]; then
  echo "missing elodin binary: $ELODIN_BIN (build with: cargo build -p elodin --release)" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

# Stale DBs that commonly break re-runs
case "$EXAMPLE" in
  video-stream) rm -rf "$REPO_ROOT/video-stream-db" ;;
  voyager) rm -rf "$REPO_ROOT/examples/voyager/dbs/voyager" "$REPO_ROOT/dbs/voyager" ;;
esac

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
  kill "$watchdog" 2>/dev/null || true
  wait "$watchdog" 2>/dev/null || true
  return "$status"
}

echo "=== $EXAMPLE -> $OUT (delay=${DELAY}s)"
set +e
ELODIN_SCREENSHOT="$OUT" \
ELODIN_SCREENSHOT_DELAY="$DELAY" \
ELODIN_SCREENSHOT_EXIT=1 \
  run_with_watchdog "$WATCHDOG" \
    "$ELODIN_BIN" editor "$REPO_ROOT/examples/$EXAMPLE/main.py" >"$LOG" 2>&1
status=$?
set -e

if [[ -s "$OUT" ]]; then
  bytes=$(wc -c <"$OUT" | tr -d ' ')
  echo "ok bytes=$bytes exit=$status log=$LOG"
  exit 0
fi
echo "FAILED: empty/missing screenshot (exit=$status). See $LOG" >&2
tail -40 "$LOG" >&2 || true
exit 1
