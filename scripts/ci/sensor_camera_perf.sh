#!/usr/bin/env bash
set -euo pipefail

: "${ELODIN_SENSOR_CAMERA_MAX_TICKS:=1200}"
: "${ELODIN_SENSOR_CAMERA_TIMEOUT_S:=180}"
: "${ELODIN_SENSOR_CAMERA_MIN_RTF:=0.10}"
: "${ELODIN_SENSOR_CAMERA_MIN_RGB_FPS:=6.0}"
: "${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FPS:=3.0}"
: "${ELODIN_SENSOR_CAMERA_MIN_RGB_FRAMES:=500}"
: "${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FRAMES:=250}"
: "${ELODIN_SENSOR_CAMERA_SIM_TIME_STEP:=0.008333333333333333}"
: "${ELODIN_SENSOR_CAMERA_ENFORCE_THRESHOLDS:=0}"
: "${ELODIN_SENSOR_CAMERA_LOG_METRICS:=1}"
: "${ELODIN_SENSOR_CAMERA_CAPTURE_TRACY:=0}"
: "${ELODIN_SENSOR_CAMERA_TRACY_CAPTURE_SECONDS:=30}"
: "${ELODIN_SENSOR_CAMERA_TRACY_OUT_DIR:=}"

log_file="$(mktemp -t sensor-camera-perf.XXXXXX.log)"
trap 'rm -f "$log_file"' EXIT

echo "Running sensor-camera performance check"
echo "  max_ticks=${ELODIN_SENSOR_CAMERA_MAX_TICKS}"
echo "  timeout_s=${ELODIN_SENSOR_CAMERA_TIMEOUT_S}"
echo "  min_rtf=${ELODIN_SENSOR_CAMERA_MIN_RTF}"
echo "  min_rgb_fps=${ELODIN_SENSOR_CAMERA_MIN_RGB_FPS}"
echo "  min_thermal_fps=${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FPS}"
echo "  sim_time_step=${ELODIN_SENSOR_CAMERA_SIM_TIME_STEP}"
echo "  enforce_thresholds=${ELODIN_SENSOR_CAMERA_ENFORCE_THRESHOLDS}"
echo "  log_metrics_probes=${ELODIN_SENSOR_CAMERA_LOG_METRICS}"
echo "  capture_tracy=${ELODIN_SENSOR_CAMERA_CAPTURE_TRACY}"

tracy_capture_pids=()
tracy_out_dir=""

start_tracy_capture() {
  local bin="$1"
  local port="$2"
  local output="$3"
  "${bin}" -a 127.0.0.1 -p "${port}" -o "${output}" -s "${ELODIN_SENSOR_CAMERA_TRACY_CAPTURE_SECONDS}" \
    >/dev/null 2>&1 &
  tracy_capture_pids+=("$!")
}

if [[ "${ELODIN_SENSOR_CAMERA_CAPTURE_TRACY}" == "1" ]]; then
  if [[ -n "${ELODIN_SENSOR_CAMERA_TRACY_OUT_DIR}" ]]; then
    tracy_out_dir="${ELODIN_SENSOR_CAMERA_TRACY_OUT_DIR}"
  else
    tracy_out_dir="$(pwd)/profile_output/tracy-sensor-camera-ci"
  fi
  mkdir -p "${tracy_out_dir}"

  if command -v tracy-capture >/dev/null 2>&1; then
    start_tracy_capture "tracy-capture" 8087 "${tracy_out_dir}/trace-run.tracy"
    start_tracy_capture "tracy-capture" 8088 "${tracy_out_dir}/trace-render.tracy"
  else
    echo "warning: tracy-capture not found; skipping ports 8087/8088 capture"
  fi

  if command -v iree-tracy-capture >/dev/null 2>&1; then
    start_tracy_capture "iree-tracy-capture" 8089 "${tracy_out_dir}/trace-sim.tracy"
  else
    echo "warning: iree-tracy-capture not found; skipping port 8089 capture"
  fi

  if [[ "${#tracy_capture_pids[@]}" -gt 0 ]]; then
    # Start capture listeners before launching the run.
    sleep 1
    echo "Tracy capture enabled: ${tracy_out_dir}"
  fi
fi

set +e
if command -v timeout >/dev/null 2>&1; then
  ELODIN_SENSOR_CAMERA_MAX_TICKS="${ELODIN_SENSOR_CAMERA_MAX_TICKS}" \
  ELODIN_SENSOR_CAMERA_PROFILE=1 \
  ELODIN_SENSOR_CAMERA_LOG_METRICS="${ELODIN_SENSOR_CAMERA_LOG_METRICS}" \
  timeout --signal=INT "${ELODIN_SENSOR_CAMERA_TIMEOUT_S}" \
    elodin run examples/sensor-camera/main.py 2>&1 | tee "${log_file}"
  run_exit_code="${PIPESTATUS[0]}"
elif command -v gtimeout >/dev/null 2>&1; then
  ELODIN_SENSOR_CAMERA_MAX_TICKS="${ELODIN_SENSOR_CAMERA_MAX_TICKS}" \
  ELODIN_SENSOR_CAMERA_PROFILE=1 \
  ELODIN_SENSOR_CAMERA_LOG_METRICS="${ELODIN_SENSOR_CAMERA_LOG_METRICS}" \
  gtimeout --signal=INT "${ELODIN_SENSOR_CAMERA_TIMEOUT_S}" \
    elodin run examples/sensor-camera/main.py 2>&1 | tee "${log_file}"
  run_exit_code="${PIPESTATUS[0]}"
else
  echo "warning: timeout/gtimeout command not found, using Python timeout fallback"
  ELODIN_SENSOR_CAMERA_MAX_TICKS="${ELODIN_SENSOR_CAMERA_MAX_TICKS}" \
  ELODIN_SENSOR_CAMERA_PROFILE=1 \
  ELODIN_SENSOR_CAMERA_LOG_METRICS="${ELODIN_SENSOR_CAMERA_LOG_METRICS}" \
  python3 - "${ELODIN_SENSOR_CAMERA_TIMEOUT_S}" "${log_file}" <<'PY'
import os
import signal
import subprocess
import sys

timeout_s = float(sys.argv[1])
log_path = sys.argv[2]
cmd = ["elodin", "run", "examples/sensor-camera/main.py"]
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    start_new_session=True,
    env=os.environ.copy(),
)

timed_out = False
output = ""
try:
    output, _ = proc.communicate(timeout=timeout_s)
except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(proc.pid, signal.SIGINT)
    try:
        output, _ = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()

with open(log_path, "w", encoding="utf-8", errors="replace") as f:
    f.write(output)
sys.stdout.write(output)
sys.stdout.flush()

if timed_out:
    sys.exit(124)
sys.exit(proc.returncode if proc.returncode is not None else 1)
PY
  run_exit_code="$?"
fi
set -e

if [[ "${run_exit_code}" -eq 124 ]]; then
  echo "info: elodin run timed out after ${ELODIN_SENSOR_CAMERA_TIMEOUT_S}s (watch mode), continuing with collected metrics"
elif [[ "${run_exit_code}" -ne 0 ]]; then
  echo "error: elodin run failed with exit code ${run_exit_code}"
  exit "${run_exit_code}"
fi

if [[ "${#tracy_capture_pids[@]}" -gt 0 ]]; then
  set +e
  for pid in "${tracy_capture_pids[@]}"; do
    wait "${pid}" >/dev/null 2>&1
  done
  set -e

  if command -v tracy-csvexport >/dev/null 2>&1; then
    for trace_file in "${tracy_out_dir}"/*.tracy; do
      if [[ -f "${trace_file}" ]]; then
        tracy-csvexport "${trace_file}" > "${trace_file%.tracy}.csv" || true
      fi
    done
  fi
fi

perf_line="$(grep '^PERF sensor_camera ' "${log_file}" | tail -n 1 || true)"
if [[ -z "${perf_line}" ]]; then
  echo "error: PERF line not found in output"
  exit 1
fi

echo
python3 scripts/ci/sensor_camera_log_summary.py \
  "${log_file}" \
  --sim-time-step "${ELODIN_SENSOR_CAMERA_SIM_TIME_STEP}" || true
echo

extract_metric() {
  local key="$1"
  echo "${perf_line}" | tr ' ' '\n' | awk -F= -v key="${key}" '$1 == key {print $2}'
}

rtf="$(extract_metric rtf)"
rgb_fps="$(extract_metric rgb_fps)"
thermal_fps="$(extract_metric thermal_fps)"
rgb_frames="$(extract_metric rgb_frames)"
thermal_frames="$(extract_metric thermal_frames)"

echo "Parsed metrics"
echo "  rtf=${rtf}"
echo "  rgb_fps=${rgb_fps}"
echo "  thermal_fps=${thermal_fps}"
echo "  rgb_frames=${rgb_frames}"
echo "  thermal_frames=${thermal_frames}"

fail=0

if ! awk -v actual="${rtf}" -v min="${ELODIN_SENSOR_CAMERA_MIN_RTF}" 'BEGIN { exit !(actual >= min) }'; then
  echo "error: rtf=${rtf} is below threshold ${ELODIN_SENSOR_CAMERA_MIN_RTF}"
  fail=1
fi

if ! awk -v actual="${rgb_fps}" -v min="${ELODIN_SENSOR_CAMERA_MIN_RGB_FPS}" 'BEGIN { exit !(actual >= min) }'; then
  echo "error: rgb_fps=${rgb_fps} is below threshold ${ELODIN_SENSOR_CAMERA_MIN_RGB_FPS}"
  fail=1
fi

if ! awk -v actual="${thermal_fps}" -v min="${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FPS}" 'BEGIN { exit !(actual >= min) }'; then
  echo "error: thermal_fps=${thermal_fps} is below threshold ${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FPS}"
  fail=1
fi

if ! awk -v actual="${rgb_frames}" -v min="${ELODIN_SENSOR_CAMERA_MIN_RGB_FRAMES}" 'BEGIN { exit !(actual + 0 >= min + 0) }'; then
  echo "error: rgb_frames=${rgb_frames} is below threshold ${ELODIN_SENSOR_CAMERA_MIN_RGB_FRAMES}"
  fail=1
fi

if ! awk -v actual="${thermal_frames}" -v min="${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FRAMES}" 'BEGIN { exit !(actual + 0 >= min + 0) }'; then
  echo "error: thermal_frames=${thermal_frames} is below threshold ${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FRAMES}"
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  if [[ "${ELODIN_SENSOR_CAMERA_ENFORCE_THRESHOLDS}" == "1" ]]; then
    exit 1
  fi
  echo "warning: thresholds not met, but enforcement is disabled"
fi

if [[ -n "${tracy_out_dir}" ]]; then
  echo "Tracy artifacts: ${tracy_out_dir}"
fi
echo "sensor-camera performance check passed"
