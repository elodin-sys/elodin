#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Capture an Elodin example through headless Gamescope.

Usage:
  ./scripts/elodin_capture.sh [options] <simulation.py>

Options:
  -d, --duration SECONDS   Video duration (default: 10)
  -o, --output PATH        Output MP4 (default: /tmp/elodin-capture.mp4)
      --width PIXELS       Capture width (default: 1280)
      --height PIXELS      Capture height (default: 720)
      --refresh HZ         Gamescope refresh rate (default: 30)
      --bitrate KBIT       H.264 bitrate (default: 8000)
      --encoder NAME       auto, vaapi, nvenc, or x264 (default: auto)
      --editor PATH        Editor executable (default: target/release/elodin)
      --ready-regex REGEX  Editor-log readiness expression
      --ready-timeout SEC  Maximum readiness wait (default: 120)
      --warmup SECONDS     Delay after readiness (default: 2)
      --keep-log PATH      Copy the Gamescope/editor log to PATH
  -h, --help               Show this help

Environment:
  GPU selection comes from the Nix development shell. Set ELODIN_GPU before
  entering the shell to override automatic selection. An explicitly set
  GBM_BACKENDS_PATH is always preserved.
EOF
}

fail() {
  echo "elodin-capture: $*" >&2
  exit 1
}

is_positive_number() {
  awk -v value="$1" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'
}

duration=10
output=/tmp/elodin-capture.mp4
width=1280
height=720
refresh=30
bitrate=8000
encoder=auto
editor="${ELODIN_EDITOR_BIN:-$PWD/target/release/elodin}"
ready_regex='running server with cancellation'
ready_timeout=120
warmup=2
keep_log=
simulation=

while (($#)); do
  case "$1" in
    -d | --duration)
      (($# >= 2)) || fail "$1 requires a value"
      duration=$2
      shift 2
      ;;
    -o | --output)
      (($# >= 2)) || fail "$1 requires a value"
      output=$2
      shift 2
      ;;
    --width)
      (($# >= 2)) || fail "$1 requires a value"
      width=$2
      shift 2
      ;;
    --height)
      (($# >= 2)) || fail "$1 requires a value"
      height=$2
      shift 2
      ;;
    --refresh)
      (($# >= 2)) || fail "$1 requires a value"
      refresh=$2
      shift 2
      ;;
    --bitrate)
      (($# >= 2)) || fail "$1 requires a value"
      bitrate=$2
      shift 2
      ;;
    --encoder)
      (($# >= 2)) || fail "$1 requires a value"
      encoder=$2
      shift 2
      ;;
    --editor)
      (($# >= 2)) || fail "$1 requires a value"
      editor=$2
      shift 2
      ;;
    --ready-regex)
      (($# >= 2)) || fail "$1 requires a value"
      ready_regex=$2
      shift 2
      ;;
    --ready-timeout)
      (($# >= 2)) || fail "$1 requires a value"
      ready_timeout=$2
      shift 2
      ;;
    --warmup)
      (($# >= 2)) || fail "$1 requires a value"
      warmup=$2
      shift 2
      ;;
    --keep-log)
      (($# >= 2)) || fail "$1 requires a value"
      keep_log=$2
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      (($# == 1)) || fail "expected exactly one simulation after --"
      simulation=$1
      shift
      ;;
    -*)
      fail "unknown option: $1"
      ;;
    *)
      [[ -z "$simulation" ]] || fail "expected exactly one simulation"
      simulation=$1
      shift
      ;;
  esac
done

[[ "$(uname -s)" == Linux ]] || fail "headless capture is Linux-only"
[[ -n "$simulation" ]] || {
  usage >&2
  exit 2
}
[[ -f "$simulation" ]] || fail "simulation not found: $simulation"
[[ -x "$editor" ]] || fail "editor not found at $editor; run 'cargo build --release -p elodin' or pass --editor"
is_positive_number "$duration" || fail "duration must be positive"
is_positive_number "$warmup" || [[ "$warmup" == 0 ]] || fail "warmup must be non-negative"
for value_name in width height refresh bitrate ready_timeout; do
  value=${!value_name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$value_name must be a positive integer"
done
case "$encoder" in
  auto | vaapi | nvenc | x264) ;;
  *) fail "encoder must be auto, vaapi, nvenc, or x264" ;;
esac

for command in gamescope Xwayland pw-cli pw-dump jq gst-inspect-1.0 gst-launch-1.0 ffmpeg ffprobe timeout; do
  command -v "$command" >/dev/null || fail "missing required command: $command"
done
pw-cli info 0 >/dev/null 2>&1 || fail "PipeWire is unavailable; start the user PipeWire service"
for element in pipewiresrc h264parse mp4mux; do
  gst-inspect-1.0 "$element" >/dev/null 2>&1 || fail "missing GStreamer element: $element"
done

output=$(realpath -m "$output")
mkdir -p "$(dirname "$output")"
if [[ -n "$keep_log" ]]; then
  keep_log=$(realpath -m "$keep_log")
  mkdir -p "$(dirname "$keep_log")"
fi

tmp_dir=$(mktemp -d -t elodin-capture.XXXXXX)
gamescope_log=$tmp_dir/gamescope.log
gamescope_pid=

cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$gamescope_pid" ]] && kill -0 "$gamescope_pid" 2>/dev/null; then
    kill "$gamescope_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$gamescope_pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$gamescope_pid" 2>/dev/null || true
    wait "$gamescope_pid" 2>/dev/null || true
  fi
  if [[ -n "$keep_log" && -f "$gamescope_log" ]]; then
    cp "$gamescope_log" "$keep_log"
  fi
  if ((status != 0)) && [[ -f "$gamescope_log" ]]; then
    echo "--- recent Gamescope/editor log ---" >&2
    tail -80 "$gamescope_log" >&2
  fi
  rm -rf "$tmp_dir"
  exit "$status"
}
trap cleanup EXIT INT TERM

# The script intentionally relies on nix develop for a coherent graphics and
# GStreamer environment. In particular, Gamescope must not launch the host's
# Xwayland with Nix libraries injected into it.
[[ "$(command -v Xwayland)" == /nix/store/* ]] \
  || fail "Xwayland is not from Nix; enter a fresh 'nix develop' shell"
mesa_dri=${LIBGL_DRIVERS_PATH:-}
[[ "$mesa_dri" == /nix/store/*/lib/dri ]] \
  || fail "Mesa driver environment is missing; enter a fresh 'nix develop' shell"
mesa_prefix=${mesa_dri%/lib/dri}

selected_gpu=mesa
if [[ "${__GLX_VENDOR_LIBRARY_NAME:-mesa}" == nvidia ]]; then
  selected_gpu=nvidia
fi
case "${ELODIN_GPU:-auto}" in
  auto) ;;
  mesa) selected_gpu=mesa ;;
  nvidia)
    [[ "$selected_gpu" == nvidia ]] \
      || fail "NVIDIA is not configured; set ELODIN_GPU=nvidia before entering nix develop"
    ;;
  *) fail "ELODIN_GPU must be auto, mesa, or nvidia" ;;
esac

if [[ "$selected_gpu" == mesa ]]; then
  mesa_manifests=()
  for manifest in radeon intel; do
    path="$mesa_prefix/share/vulkan/icd.d/${manifest}_icd.$(uname -m).json"
    [[ -e "$path" ]] && mesa_manifests+=("$path")
  done
  ((${#mesa_manifests[@]})) || fail "no supported Mesa Vulkan manifests were found"
  mesa_vulkan_icds=$(IFS=:; echo "${mesa_manifests[*]}")
  export LIBGL_DRIVERS_PATH="$mesa_dri"
  export LIBVA_DRIVERS_PATH="$mesa_dri"
  export VK_ICD_FILENAMES="$mesa_vulkan_icds"
  export VK_DRIVER_FILES="$mesa_vulkan_icds"
  export __GLX_VENDOR_LIBRARY_NAME=mesa
fi

if [[ -z "${GBM_BACKENDS_PATH+x}" ]]; then
  if [[ "$selected_gpu" == mesa ]]; then
    export GBM_BACKENDS_PATH="$mesa_prefix/lib/gbm"
  else
    for candidate in \
      /run/opengl-driver/lib/gbm \
      /usr/lib/x86_64-linux-gnu/gbm \
      /usr/lib/aarch64-linux-gnu/gbm \
      /usr/lib64/gbm; do
      if compgen -G "$candidate/nvidia*_gbm.so*" >/dev/null; then
        export GBM_BACKENDS_PATH=$candidate
        break
      fi
    done
  fi
fi

echo "GPU path: $selected_gpu"
echo "Gamescope: $(command -v gamescope)"
echo "Xwayland: $(command -v Xwayland)"
echo "GBM backends: ${GBM_BACKENDS_PATH:-driver default}"

child_path=$PATH
if [[ -x "$PWD/.venv/bin/python" ]]; then
  child_path="$PWD/.venv/bin:$child_path"
fi

gamescope --backend headless \
  -w "$width" -h "$height" -W "$width" -H "$height" -r "$refresh" \
  -- env PATH="$child_path" "$editor" editor "$simulation" \
  >"$gamescope_log" 2>&1 &
gamescope_pid=$!

pipewire_node=
for _ in $(seq 1 "$ready_timeout"); do
  kill -0 "$gamescope_pid" 2>/dev/null || fail "Gamescope/editor exited during startup"
  pipewire_node=$(pw-dump 2>/dev/null | jq -r '
    [
      .[]
      | select(.type == "PipeWire:Interface:Node")
      | select(.info.props["media.name"] == "gamescope")
      | {id: .id, name: .info.props["node.name"]}
    ]
    | sort_by(.id)
    | last
    | .name // empty
  ')
  [[ -n "$pipewire_node" ]] && break
  sleep 1
done
[[ -n "$pipewire_node" ]] || fail "Gamescope did not publish a PipeWire source"

for _ in $(seq 1 "$ready_timeout"); do
  kill -0 "$gamescope_pid" 2>/dev/null || fail "Gamescope/editor exited before becoming ready"
  grep -Eq "$ready_regex" "$gamescope_log" && break
  sleep 1
done
grep -Eq "$ready_regex" "$gamescope_log" || fail "editor readiness timed out (regex: $ready_regex)"
sleep "$warmup"

run_pipeline() {
  local selected_encoder=$1
  local location=$2
  local requested_duration=$3
  local wall_duration
  local status
  wall_duration=$(awk -v duration="$requested_duration" 'BEGIN { printf "%.3f", duration + 0.30 }')

  set +e
  case "$selected_encoder" in
    vaapi)
      timeout --signal=INT --kill-after=8s "${wall_duration}s" gst-launch-1.0 -q -e \
        pipewiresrc target-object="$pipewire_node" do-timestamp=true \
        ! video/x-raw,format=BGRx \
        ! queue \
        ! videoconvert \
        ! video/x-raw,format=NV12 \
        ! vah264enc bitrate="$bitrate" \
        ! video/x-h264,profile=main \
        ! h264parse \
        ! mp4mux faststart=true \
        ! filesink location="$location"
      status=$?
      ;;
    nvenc)
      timeout --signal=INT --kill-after=8s "${wall_duration}s" gst-launch-1.0 -q -e \
        pipewiresrc target-object="$pipewire_node" do-timestamp=true \
        ! video/x-raw,format=BGRx \
        ! queue \
        ! videoconvert \
        ! video/x-raw,format=NV12 \
        ! nvh264enc bitrate="$bitrate" \
        ! video/x-h264,profile=main \
        ! h264parse \
        ! mp4mux faststart=true \
        ! filesink location="$location"
      status=$?
      ;;
    x264)
      timeout --signal=INT --kill-after=8s "${wall_duration}s" gst-launch-1.0 -q -e \
        pipewiresrc target-object="$pipewire_node" do-timestamp=true \
        ! video/x-raw,format=BGRx \
        ! queue \
        ! videoconvert \
        ! video/x-raw,format=I420 \
        ! x264enc bitrate="$bitrate" speed-preset=veryfast \
        ! video/x-h264,profile=main \
        ! h264parse \
        ! mp4mux faststart=true \
        ! filesink location="$location"
      status=$?
      ;;
  esac
  set -e
  [[ "$status" == 0 || "$status" == 124 ]]
}

video_is_valid() {
  local video=$1
  local probe_duration
  local midpoint
  local stats
  local ymax
  [[ -s "$video" ]] || return 1
  probe_duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video" 2>/dev/null) || return 1
  is_positive_number "$probe_duration" || return 1
  midpoint=$(awk -v duration="$probe_duration" 'BEGIN { printf "%.3f", duration / 2 }')
  stats=$(ffmpeg -v error -ss "$midpoint" -i "$video" -frames:v 1 \
    -vf signalstats,metadata=print:file=- -f null - 2>&1) || return 1
  ymax=$(awk -F= '/lavfi.signalstats.YMAX=/{value=$2} END{print value}' <<<"$stats")
  [[ -n "$ymax" ]] && awk -v value="$ymax" 'BEGIN { exit !(value > 20) }'
}

candidates=()
if [[ "$encoder" == auto ]]; then
  if [[ "$selected_gpu" == nvidia ]]; then
    candidates=(nvenc vaapi x264)
  else
    candidates=(vaapi nvenc x264)
  fi
else
  candidates=("$encoder")
fi

selected_encoder=
for candidate in "${candidates[@]}"; do
  case "$candidate" in
    vaapi) element=vah264enc ;;
    nvenc) element=nvh264enc ;;
    x264) element=x264enc ;;
  esac
  gst-inspect-1.0 "$element" >/dev/null 2>&1 || continue
  test_video=$tmp_dir/encoder-test-$candidate.mp4
  echo "Validating $candidate encoder..."
  if run_pipeline "$candidate" "$test_video" 1 && video_is_valid "$test_video"; then
    selected_encoder=$candidate
    break
  fi
  echo "$candidate encoder validation failed; trying the next candidate" >&2
done
[[ -n "$selected_encoder" ]] || fail "no working H.264 encoder was found"
echo "Encoder: $selected_encoder"

partial_output=$tmp_dir/capture.mp4
run_pipeline "$selected_encoder" "$partial_output" "$duration" || fail "recording pipeline failed"
video_is_valid "$partial_output" || fail "recording is invalid or blank"

actual_duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$partial_output")
minimum_duration=$(awk -v duration="$duration" 'BEGIN { printf "%.3f", duration * 0.95 }')
awk -v actual="$actual_duration" -v minimum="$minimum_duration" 'BEGIN { exit !(actual >= minimum) }' \
  || fail "recording is too short: ${actual_duration}s (requested ${duration}s)"

mv "$partial_output" "$output"
echo "Captured $output"
ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 "$output"
