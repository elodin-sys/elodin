#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# The scene references this repo's assets (mojave atlas, drone GLB); an
# inherited ELODIN_ASSETS (e.g. from the fsw dev shell) would point elsewhere.
export ELODIN_ASSETS="$PWD/assets"

ATLAS="assets/terrains/planar/mojave_rc_field/config.tc"
if [[ ! -f "$ATLAS" ]]; then
  echo "missing terrain atlas: $ATLAS" >&2
  echo "run ./scripts/prepare_editor_terrain_region.sh mojave_rc_field first" >&2
  exit 2
fi

OUT="${ELODIN_LWIR_VISUAL_OUT:-/tmp/rc-jet-lwir-visual-check}"
case "$OUT" in
  /tmp/*) ;;
  *)
    echo "ELODIN_LWIR_VISUAL_OUT must be under /tmp: $OUT" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cargo build -p elodin --release
fi
ELODIN_BIN="${ELODIN_BIN:-./target/release/elodin}"

rm -rf -- "$OUT"
mkdir -p -- "$OUT"
LOG="$OUT/run.log"
: > "$LOG"

ELODIN_LWIR_VISUAL_OUT="$OUT" \
ELODIN_DB_PATH="$OUT/db" \
RUST_LOG="${RUST_LOG:-elodin_editor=info,wgpu=error}" \
  "$ELODIN_BIN" run examples/rc-jet/visual_check.py 2>&1 | tee "$LOG"

if ! rg -q '\[VISUAL\] PASS$' "$LOG"; then
  echo "RC-JET LWIR VISUAL CHECK: FAIL"
  exit 1
fi

echo "RC-JET LWIR VISUAL CHECK: PASS"
echo "captures + report: $OUT"
