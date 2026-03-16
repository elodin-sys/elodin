#!/usr/bin/env bash
set -euo pipefail

tracy_out="$(pwd)/profile_output/tracy-sensor-camera-ci"
log_file="$(mktemp -t sensor-camera-perf.XXXXXX.log)"
trap 'rm -f "$log_file"' EXIT

mkdir -p "${tracy_out}"

# ── Tracy capture ────────────────────────────────────────────────────────────
tracy_pids=()

capture() {
  "$1" -a 127.0.0.1 -p "$2" -o "$3" -s 15 >/dev/null 2>&1 &
  tracy_pids+=("$!")
}

if command -v tracy-capture >/dev/null 2>&1; then
  capture tracy-capture 8087 "${tracy_out}/trace-run.tracy"
  capture tracy-capture 8088 "${tracy_out}/trace-render.tracy"
fi
if command -v iree-tracy-capture >/dev/null 2>&1; then
  capture iree-tracy-capture 8089 "${tracy_out}/trace-sim.tracy"
fi
[[ ${#tracy_pids[@]} -gt 0 ]] && sleep 1

# ── Run ──────────────────────────────────────────────────────────────────────
set +e
env ELODIN_SENSOR_CAMERA_MAX_TICKS=600 \
  timeout --signal=INT 60 elodin run examples/sensor-camera/main.py 2>&1 | tee "${log_file}"
rc="${PIPESTATUS[0]}"
set -e

[[ "${rc}" -eq 124 ]] || [[ "${rc}" -eq 0 ]] || { echo "error: exit ${rc}"; exit "${rc}"; }

# ── Export Tracy CSVs ────────────────────────────────────────────────────────
set +e
for pid in "${tracy_pids[@]}"; do wait "${pid}" >/dev/null 2>&1; done
set -e

if command -v tracy-csvexport >/dev/null 2>&1; then
  for f in "${tracy_out}"/*.tracy; do
    [[ -f "${f}" ]] && tracy-csvexport "${f}" > "${f%.tracy}.csv" 2>/dev/null || true
  done
fi

# ── Verdict ──────────────────────────────────────────────────────────────────
render_csv="${tracy_out}/trace-render.csv"

if [[ ! -f "${render_csv}" ]] || [[ ! -s "${render_csv}" ]]; then
  echo "FAIL: Tracy render CSV not found"
  exit 1
fi

exec python3 scripts/ci/sensor_camera_tracy_analysis.py "${render_csv}"
