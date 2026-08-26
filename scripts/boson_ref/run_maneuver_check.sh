#!/usr/bin/env bash
# LWIR maneuver visual check: mountains -> sky -> loop -> ground return.
# Run from the repo root inside `nix develop`:
#   scripts/boson_ref/run_maneuver_check.sh
# Env: SKIP_BUILD=1 reuses target/release/elodin; ELODIN_LWIR_MANEUVER_OUT
# overrides the output directory (default /tmp/lwir-maneuver).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
OUT="${ELODIN_LWIR_MANEUVER_OUT:-/tmp/lwir-maneuver}"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  cargo build -p elodin --release
fi

rm -rf "$OUT"
mkdir -p "$OUT"
LOG="$OUT/run.log"

ELODIN_LWIR_MANEUVER_OUT="$OUT" \
ELODIN_DB_PATH="$OUT/db" \
RUST_LOG=elodin_editor=info,wgpu=error \
  ./target/release/elodin run scripts/boson_ref/lwir_maneuver_sim.py 2>&1 | tee "$LOG"

echo
echo "captures + report: $OUT"
if grep -q "\[MANEUVER\] FAIL" "$LOG"; then
  echo "MANEUVER CHECK: FAIL"
  exit 1
fi
if ! grep -q "\[MANEUVER\] PASS" "$LOG"; then
  echo "MANEUVER CHECK: NO VERDICT"
  exit 2
fi
echo "MANEUVER CHECK: PASS"
