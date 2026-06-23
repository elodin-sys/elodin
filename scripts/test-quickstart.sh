#!/usr/bin/env bash
#
# Smoke-test `elodin monte-carlo quickstart` end-to-end (Buildkite step
# :sparkles: monte-carlo quickstart).
#
# Scaffolds a campaign from the lightweight examples/monte-carlo sim (declares
# params, pure-Python fallback controller), then RUNS the generated artifacts
# unedited to prove the scaffold is actually runnable: param extraction, spec +
# campaign validity, and the generated post_run/post_campaign hooks resolving
# and passing.
#
# Usage (from repo root, inside nix develop):
#   scripts/test-quickstart.sh

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

SIM="examples/monte-carlo/main.py"
WORK="${QUICKSTART_CI_OUT:-$ROOT/target/quickstart-ci}"
CAMPAIGN_DIR="$WORK/campaign"
OUT_DIR="$WORK/out"
rm -rf "$WORK"
mkdir -p "$WORK"

# 1. Scaffold the campaign from the sim's declared params.
"$ELODIN_BIN" monte-carlo quickstart "$SIM" "$CAMPAIGN_DIR"

# 2. The expected skeleton must exist.
for f in spec.toml campaign.toml hooks/score.py hooks/gate.py; do
  if [[ ! -f "$CAMPAIGN_DIR/$f" ]]; then
    echo "error: quickstart did not generate $f" >&2
    exit 1
  fi
done

# 3. Keep CI fast: 2 samples, no external controller, tiny lookup table.
sed -i.bak 's/^n_samples = .*/n_samples = 2/' "$CAMPAIGN_DIR/spec.toml"
rm -f "$CAMPAIGN_DIR/spec.toml.bak"
export ELODIN_MONTE_CARLO_CONTROLLER=0
export ELODIN_MONTE_CARLO_GRID_SIZE="${ELODIN_MONTE_CARLO_GRID_SIZE:-1024}"
# Serial run (one in flight at a time); replaces the old --workers 1.
export S10_MAX_INFLIGHT=1

# 4. Run the generated artifacts unedited.
"$ELODIN_BIN" monte-carlo run "$SIM" \
  --campaign "$CAMPAIGN_DIR/campaign.toml" \
  --spec "$CAMPAIGN_DIR/spec.toml" \
  --out "$OUT_DIR"

# 5. Every run must have completed and passed the generated score hook,
#    which proves the hooks resolved and ran (not just that the sim ran).
python3 - "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads((Path(sys.argv[1]) / "summary.json").read_text())
total = int(summary.get("total_runs", 0))
passed = int(summary.get("passed", 0))
failed = int(summary.get("failed", 0))
if total < 1:
    raise SystemExit(f"quickstart smoke: no runs in summary {summary}")
if failed != 0:
    raise SystemExit(f"quickstart smoke: {failed}/{total} run(s) failed {summary}")
if passed != total:
    raise SystemExit(f"quickstart smoke: only {passed}/{total} passed the score hook {summary}")
print(f"quickstart smoke: {passed}/{total} runs passed")
PY

echo "monte-carlo quickstart CI passed (campaign=$CAMPAIGN_DIR out_dir=$OUT_DIR)"
