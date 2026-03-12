#!/usr/bin/env bash
set -euo pipefail

: "${ELODIN_SENSOR_CAMERA_MAX_TICKS:=1200}"
: "${ELODIN_SENSOR_CAMERA_TIMEOUT_S:=180}"
: "${ELODIN_SENSOR_CAMERA_MIN_RTF:=0.10}"
: "${ELODIN_SENSOR_CAMERA_MIN_RGB_FPS:=6.0}"
: "${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FPS:=3.0}"
: "${ELODIN_SENSOR_CAMERA_MIN_RGB_FRAMES:=500}"
: "${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FRAMES:=250}"

log_file="$(mktemp -t sensor-camera-perf.XXXXXX.log)"
trap 'rm -f "$log_file"' EXIT

echo "Running sensor-camera performance check"
echo "  max_ticks=${ELODIN_SENSOR_CAMERA_MAX_TICKS}"
echo "  timeout_s=${ELODIN_SENSOR_CAMERA_TIMEOUT_S}"
echo "  min_rtf=${ELODIN_SENSOR_CAMERA_MIN_RTF}"
echo "  min_rgb_fps=${ELODIN_SENSOR_CAMERA_MIN_RGB_FPS}"
echo "  min_thermal_fps=${ELODIN_SENSOR_CAMERA_MIN_THERMAL_FPS}"

set +e
if command -v timeout >/dev/null 2>&1; then
  ELODIN_SENSOR_CAMERA_MAX_TICKS="${ELODIN_SENSOR_CAMERA_MAX_TICKS}" \
  ELODIN_SENSOR_CAMERA_PROFILE=1 \
  timeout --signal=INT "${ELODIN_SENSOR_CAMERA_TIMEOUT_S}" \
    elodin run examples/sensor-camera/main.py 2>&1 | tee "${log_file}"
  run_exit_code="${PIPESTATUS[0]}"
elif command -v gtimeout >/dev/null 2>&1; then
  ELODIN_SENSOR_CAMERA_MAX_TICKS="${ELODIN_SENSOR_CAMERA_MAX_TICKS}" \
  ELODIN_SENSOR_CAMERA_PROFILE=1 \
  gtimeout --signal=INT "${ELODIN_SENSOR_CAMERA_TIMEOUT_S}" \
    elodin run examples/sensor-camera/main.py 2>&1 | tee "${log_file}"
  run_exit_code="${PIPESTATUS[0]}"
else
  echo "warning: timeout/gtimeout command not found, running without timeout"
  ELODIN_SENSOR_CAMERA_MAX_TICKS="${ELODIN_SENSOR_CAMERA_MAX_TICKS}" \
  ELODIN_SENSOR_CAMERA_PROFILE=1 \
  elodin run examples/sensor-camera/main.py 2>&1 | tee "${log_file}"
  run_exit_code="${PIPESTATUS[0]}"
fi
set -e

if [[ "${run_exit_code}" -eq 124 ]]; then
  echo "info: elodin run timed out after ${ELODIN_SENSOR_CAMERA_TIMEOUT_S}s (watch mode), continuing with collected metrics"
elif [[ "${run_exit_code}" -ne 0 ]]; then
  echo "error: elodin run failed with exit code ${run_exit_code}"
  exit "${run_exit_code}"
fi

perf_line="$(grep '^PERF sensor_camera ' "${log_file}" | tail -n 1 || true)"
if [[ -z "${perf_line}" ]]; then
  echo "error: PERF line not found in output"
  exit 1
fi

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
  exit 1
fi

echo "sensor-camera performance check passed"
