#!/usr/bin/env bash
#
# Fast smoke-test elodin monte-carlo on the Apollo lander example for CI
# (Buildkite step :rocket: apollo monte-carlo).
# Runs one truncated deterministic campaign (infrastructure + SITL plumbing),
# not a full soft-landing validation. CI failure is enforced by the
# post_campaign ci_gate.py hook, not fail_on_run_errors.
#
# Usage (from repo root, inside nix develop):
#   scripts/test-apollo-monte-carlo.sh
#
# Optional:
#   APOLLO_MC_CI_OUT=/tmp/my-apollo-mc-out scripts/test-apollo-monte-carlo.sh
#   ELODIN_APOLLO_MAX_TICKS=1200 scripts/test-apollo-monte-carlo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${ELODIN_BIN:-}" ]]; then
  if command -v elodin >/dev/null 2>&1; then
    ELODIN_BIN=elodin
  elif [[ -x "$ROOT/target/release/elodin" ]]; then
    ELODIN_BIN="$ROOT/target/release/elodin"
  elif [[ -n "${CARGO_HOME:-}" && -x "${CARGO_HOME}/bin/elodin" ]]; then
    ELODIN_BIN="${CARGO_HOME}/bin/elodin"
  elif [[ -x "$HOME/.cargo/bin/elodin" ]]; then
    ELODIN_BIN="$HOME/.cargo/bin/elodin"
  else
    echo "error: elodin not found; run \`just install editor\` first" >&2
    exit 127
  fi
fi

OUT_DIR="${APOLLO_MC_CI_OUT:-$ROOT/target/apollo-mc-ci}"
rm -rf "$OUT_DIR"

# ~5 s of sim at 120 Hz: enough to exercise JAX, SITL, hooks, and reporting.
export ELODIN_APOLLO_MAX_TICKS="${ELODIN_APOLLO_MAX_TICKS:-600}"

# Serial run (one at a time) keeps CI light and deterministic.
"$ELODIN_BIN" monte-carlo run examples/apollo-lander/main.py \
  --campaign examples/apollo-lander/campaign.ci.toml \
  --spec examples/apollo-lander/spec.ci.toml \
  --workers 1 \
  --out "$OUT_DIR"

echo "apollo-lander monte-carlo CI passed (out_dir=$OUT_DIR)"
