#!/usr/bin/env bash
#
# Fast smoke-test elodin monte-carlo on the Apollo lander example for CI.
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

OUT_DIR="${APOLLO_MC_CI_OUT:-$ROOT/target/apollo-mc-ci}"
rm -rf "$OUT_DIR"

# ~5 s of sim at 120 Hz: enough to exercise JAX, SITL, hooks, and reporting.
export ELODIN_APOLLO_MAX_TICKS="${ELODIN_APOLLO_MAX_TICKS:-600}"

elodin monte-carlo run examples/apollo-lander/main.py \
  --campaign examples/apollo-lander/campaign.ci.toml \
  --spec examples/apollo-lander/spec.ci.toml \
  --out "$OUT_DIR" \
  --workers 1

echo "apollo-lander monte-carlo CI smoke passed (out_dir=$OUT_DIR)"
