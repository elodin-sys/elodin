#!/usr/bin/env bash
#
# Fast smoke-test elodin monte-carlo on the Falcon 9 example for CI.
# Runs one truncated deterministic campaign (~40 s of flight at 1000 Hz):
# plant + sensors + UDP FSW bridge + hooks, not a landing validation.
# CI failure is enforced by the post_campaign ci_gate.py hook.
#
# Usage (from repo root, inside nix develop):
#   scripts/test-falcon9-monte-carlo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${ELODIN_BIN:-}" ]]; then
  if [[ -x "$ROOT/target/release/elodin" ]]; then
    ELODIN_BIN="$ROOT/target/release/elodin"
  elif command -v elodin >/dev/null 2>&1; then
    ELODIN_BIN=elodin
  elif [[ -n "${CARGO_HOME:-}" && -x "${CARGO_HOME}/bin/elodin" ]]; then
    ELODIN_BIN="${CARGO_HOME}/bin/elodin"
  elif [[ -x "$HOME/.cargo/bin/elodin" ]]; then
    ELODIN_BIN="$HOME/.cargo/bin/elodin"
  else
    echo "error: elodin not found; run \`just install editor\` first" >&2
    exit 127
  fi
fi

OUT_DIR="${FALCON9_MC_CI_OUT:-$ROOT/target/falcon9-mc-ci}"
rm -rf "$OUT_DIR"

# ~40 s of flight at 1000 Hz: liftoff, pitch kick, early gravity turn.
export ELODIN_FALCON9_MAX_TICKS="${ELODIN_FALCON9_MAX_TICKS:-40000}"

"$ELODIN_BIN" monte-carlo run examples/falcon9/main.py \
  --campaign examples/falcon9/campaign.ci.toml \
  --spec examples/falcon9/spec.ci.toml \
  --workers 1 \
  --out "$OUT_DIR"

echo "falcon9 monte-carlo CI passed (out_dir=$OUT_DIR)"
