---
name: elodin-headless-capture
description: Run the graphical Elodin Editor without a physical display and capture it. Use when launching Elodin in headless Gamescope, recording an editor session to MP4 with GStreamer and a private PipeWire session, taking a Gamescope screenshot, or troubleshooting headless GPU capture.
---

# Headless Elodin Editor Capture

Use Gamescope as a headless compositor and record the PipeWire video node that Gamescope publishes. This captures the complete graphical editor, including its 3D viewport and UI.

This is different from `elodin run`, which runs a simulation without the graphical editor and therefore has no editor window to record.

## Platform support

**This workflow is Linux-only.** Gamescope, PipeWire, and the Gamescope capture node are Linux technologies. GStreamer itself is cross-platform, but this source pipeline is not. On macOS, use the native screenshot harness documented in the `elodin-editor-dev` skill for still images and a macOS screen-recording tool for video.

The Linux Nix development shell supplies:

- `gamescope` and `gamescopectl`
- PipeWire, WirePlumber, `pw-cli`, `pw-dump`, and `wpctl`
- GStreamer core plus the base, good, bad, and ugly plugin sets
- `pipewiresrc`, `videoconvert`, `x264enc`, `h264parse`, and `mp4mux`

### Distribution portability

The capture userland comes from Nix and is not Ubuntu-specific. The workflow is expected to work on current x86-64 Debian, Ubuntu, Arch, Rocky Linux, and similar distributions, provided the host kernel exposes a working Vulkan GPU and the user can access its render device. It does not depend on a host PipeWire, WirePlumber, GStreamer, desktop, or login session.

The unavoidable distro-specific boundary is a proprietary NVIDIA driver: its kernel module and matching user-space ICD come from the host. The Nix hook recognizes the common Debian/Ubuntu multiarch directories, Arch `/usr/lib`, RPM `/usr/lib64`, NixOS `/run/opengl-driver`, and `nvidia_icd*.json` filename variants. Unusual layouts can set `ELODIN_NVIDIA_ICD` explicitly. Intel and AMD use the Mesa drivers supplied by Nix.

Portability still depends on sufficiently recent kernel/GPU drivers and Gamescope support; evaluate and smoke-test on each CI image rather than assuming every vendor driver release behaves identically.

## Private media session

Captures should create a private, mode-0700 session directory containing separate XDG and PipeWire runtime directories. Start one `pipewire` daemon and one `wireplumber` process in that environment. Gamescope and GStreamer inherit the variables and connect only to that private session.

A private `XDG_RUNTIME_DIR` gives Gamescope a valid place for its Wayland/control sockets on a server where no login session exists. Because the override is scoped to the capture process tree, it does not alter the host login session. It also prevents concurrent Gamescope captures from sharing control sockets. A private `PIPEWIRE_RUNTIME_DIR` similarly isolates PipeWire object registries.

Do not start these processes from the Nix `shellHook`: every `nix develop` terminal would start a competing daemon with unclear ownership. The shell provides the software and Nix configuration; each capture owns startup and cleanup.

WirePlumber is intentional. A bare PipeWire daemon works, but the `target.object` property then has no policy manager to create the media link. An agent would have to discover both ports and invoke `pw-link` itself. For an interactive capture, GStreamer would also need to run in the background before its input port could be linked, complicating Ctrl+C and MP4 finalization. Private WirePlumber lets a human run GStreamer in the foreground and stop it normally.

## NVIDIA compatibility

Launch Elodin with:

```bash
WGPU_SETTINGS_PRIO=webgpu
```

Bevy otherwise defaults to `functionality`, which requests every feature that the Vulkan adapter advertises. NVIDIA 550 drivers were observed to enumerate successfully but return `VK_ERROR_DEVICE_LOST` from device creation under that setting. `webgpu` requests the conservative WebGPU feature/limit baseline plus Elodin's explicitly required features. This setting was validated with Elodin running under the Nix-provided Gamescope on NVIDIA.

The capture commands set this only on the Elodin child. It is deliberately not a global Nix-shell default, because an unrelated workflow may intentionally use Bevy's full adapter feature set. The Nix NVIDIA hook locates host ICDs under `/usr/share`, `/etc`, or `/run/opengl-driver`, and exposes the matching host driver libraries. `ELODIN_NVIDIA_ICD=/path/to/nvidia_icd.json` can override discovery.

## Before a capture

From the repository root:

```bash
nix develop
just install                         # if the Python environment is not installed
cargo build --release -p elodin
source .venv/bin/activate            # when capturing a Python example
```

Check the capture tools and plugins:

```bash
command -v gamescope gamescopectl pipewire wireplumber pw-cli pw-dump wpctl gst-launch-1.0
test -f "$ELODIN_WIREPLUMBER_CONFIG"
for element in pipewiresrc videoconvert x264enc h264parse mp4mux; do
  gst-inspect-1.0 "$element" >/dev/null || exit 1
done
```

Do not run `pw-cli info 0` yet; the private daemon does not exist and the host session is irrelevant.

Before using a GPU on a shared machine, check whether it is busy. For NVIDIA GPUs, use `nvidia-smi`; for other GPUs, inspect `/dev/dri/renderD*` users. Prefer an idle GPU and the lowest resolution that keeps relevant UI readable. `1280x720` is a useful default; `960x540` consumes fewer resources.

## Interactive two-terminal workflow

### 1. Start an isolated session

In the first `nix develop` terminal:

```bash
export ELODIN_CAPTURE_SESSION="$(mktemp -d "${TMPDIR:-/tmp}/elodin-capture-session.XXXXXX")"
chmod 700 "$ELODIN_CAPTURE_SESSION"
export XDG_RUNTIME_DIR="$ELODIN_CAPTURE_SESSION/xdg"
export PIPEWIRE_RUNTIME_DIR="$ELODIN_CAPTURE_SESSION/pipewire"
mkdir -m 700 "$XDG_RUNTIME_DIR" "$PIPEWIRE_RUNTIME_DIR"

printf 'Run these in the capture terminal:\n'
printf 'export XDG_RUNTIME_DIR=%q\n' "$XDG_RUNTIME_DIR"
printf 'export PIPEWIRE_RUNTIME_DIR=%q\n' "$PIPEWIRE_RUNTIME_DIR"

pipewire >"$ELODIN_CAPTURE_SESSION/pipewire.log" 2>&1 &
PIPEWIRE_PID=$!
until pw-cli info 0 >/dev/null 2>&1; do
  kill -0 "$PIPEWIRE_PID" || { cat "$ELODIN_CAPTURE_SESSION/pipewire.log"; exit 1; }
  sleep 0.05
done

wireplumber -c "$ELODIN_WIREPLUMBER_CONFIG" \
  >"$ELODIN_CAPTURE_SESSION/wireplumber.log" 2>&1 &
WIREPLUMBER_PID=$!
until wpctl status >/dev/null 2>&1; do
  kill -0 "$WIREPLUMBER_PID" || { cat "$ELODIN_CAPTURE_SESSION/wireplumber.log"; exit 1; }
  sleep 0.05
done
```

### 2. Launch Elodin in headless Gamescope

Still in the first terminal:

```bash
gamescope \
  --backend headless \
  -W 1280 -H 720 \
  -w 1280 -h 720 \
  -r 30 \
  --force-windows-fullscreen \
  -- env WGPU_SETTINGS_PRIO=webgpu \
    ./target/release/elodin editor examples/three-body/main.py
```

Replace the example path as needed.

### 3. Record from a second terminal

Open a second `nix develop` terminal and run the two exports printed by the first terminal. Then discover Gamescope's PipeWire object serial:

```bash
GAMESCOPE_TARGET="$(
  pw-dump | jq -r '
    [.[]
      | select(.type == "PipeWire:Interface:Node")
      | select(.info.props["media.class"] == "Video/Source")
      | select((.info.props["node.name"] // "") | contains("gamescope"))
      | .info.props["object.serial"]
    ][0] // empty
  '
)"
test -n "$GAMESCOPE_TARGET"
printf 'Capturing PipeWire object serial %s\n' "$GAMESCOPE_TARGET"
```

Do not assume the node is literally named `gamescope`. The Nix binary wrapper currently yields `.gamescope-wrapped`, while an unwrapped system installation commonly yields `gamescope`. `object.serial` is unique within the private PipeWire registry and is the preferred `target-object` value.

Start GStreamer in the foreground:

```bash
gst-launch-1.0 -e \
  pipewiresrc target-object="$GAMESCOPE_TARGET" do-timestamp=true \
  ! queue \
  ! videoconvert \
  ! video/x-raw,format=I420 \
  ! x264enc bitrate=8000 speed-preset=veryfast \
  ! video/x-h264,profile=main \
  ! h264parse \
  ! mp4mux faststart=true \
  ! filesink location=elodin-capture.mp4
```

When the desired segment is complete:

1. Press **Ctrl+C in the GStreamer terminal first**.
2. Wait for GStreamer to emit end-of-stream and finalize the MP4.
3. Stop Gamescope/the editor.
4. In the first terminal, stop and remove the private session:

```bash
kill -TERM "$WIREPLUMBER_PID" "$PIPEWIRE_PID"
wait "$WIREPLUMBER_PID" "$PIPEWIRE_PID" 2>/dev/null || true
rm -rf "$ELODIN_CAPTURE_SESSION"
```

Validate the result:

```bash
ffprobe -v error \
  -show_entries format=duration:stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 \
  elodin-capture.mp4
```

The pipeline records video only. Do not assume the Gamescope video node contains audio.

## Automated bounded capture

Coding agents should prefer a single bounded Bash invocation that owns PipeWire, WirePlumber, Gamescope, and GStreamer. This recipe warms up for ten seconds, records thirty seconds, finalizes the MP4, and tears down the private session:

```bash
#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${OUTPUT:-/tmp/elodin-capture-$$.mp4}"
DURATION_SECONDS="${DURATION_SECONDS:-30}"
WARMUP_SECONDS="${WARMUP_SECONDS:-10}"
LOG_DIR="${LOG_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/elodin-capture-logs.XXXXXX")}"
SESSION_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/elodin-capture-session.XXXXXX")"
chmod 700 "$SESSION_RUNTIME_DIR"
export XDG_RUNTIME_DIR="$SESSION_RUNTIME_DIR/xdg"
export PIPEWIRE_RUNTIME_DIR="$SESSION_RUNTIME_DIR/pipewire"
mkdir -m 700 "$XDG_RUNTIME_DIR" "$PIPEWIRE_RUNTIME_DIR"

pipewire_pid=
wireplumber_pid=
gamescope_pid=
capture_pid=

stop_process() {
  local pid="$1"
  local signal="${2:-TERM}"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  kill -"$signal" "$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  # Finalize the muxer before removing its video source.
  stop_process "$capture_pid" INT
  stop_process "$gamescope_pid" TERM
  stop_process "$wireplumber_pid" TERM
  stop_process "$pipewire_pid" TERM
  rm -rf "$SESSION_RUNTIME_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

mkdir -p "$LOG_DIR"

pipewire >"$LOG_DIR/pipewire.log" 2>&1 &
pipewire_pid=$!
pipewire_ready=0
for _ in $(seq 1 200); do
  if pw-cli info 0 >/dev/null 2>&1; then
    pipewire_ready=1
    break
  fi
  kill -0 "$pipewire_pid" 2>/dev/null || break
  sleep 0.05
done
if [[ "$pipewire_ready" != 1 ]]; then
  echo "Private PipeWire failed to start" >&2
  cat "$LOG_DIR/pipewire.log" >&2
  exit 1
fi

wireplumber -c "$ELODIN_WIREPLUMBER_CONFIG" >"$LOG_DIR/wireplumber.log" 2>&1 &
wireplumber_pid=$!
wireplumber_ready=0
for _ in $(seq 1 200); do
  if wpctl status >/dev/null 2>&1; then
    wireplumber_ready=1
    break
  fi
  kill -0 "$wireplumber_pid" 2>/dev/null || break
  sleep 0.05
done
if [[ "$wireplumber_ready" != 1 ]]; then
  echo "Private WirePlumber failed to start" >&2
  cat "$LOG_DIR/wireplumber.log" >&2
  exit 1
fi

gamescope \
  --backend headless \
  -W 1280 -H 720 \
  -w 1280 -h 720 \
  -r 30 \
  --force-windows-fullscreen \
  -- env WGPU_SETTINGS_PRIO=webgpu \
    ./target/release/elodin editor examples/three-body/main.py \
  >"$LOG_DIR/gamescope.log" 2>&1 &
gamescope_pid=$!

gamescope_target=
for _ in $(seq 1 240); do
  gamescope_target="$(
    pw-dump 2>/dev/null | jq -r '
      [.[]
        | select(.type == "PipeWire:Interface:Node")
        | select(.info.props["media.class"] == "Video/Source")
        | select((.info.props["node.name"] // "") | contains("gamescope"))
        | .info.props["object.serial"]
      ][0] // empty
    '
  )"
  if [[ -n "$gamescope_target" ]]; then
    break
  fi
  kill -0 "$gamescope_pid" 2>/dev/null || break
  sleep 0.25
done
if [[ -z "$gamescope_target" ]]; then
  echo "Gamescope failed to publish its PipeWire node" >&2
  tail -100 "$LOG_DIR/gamescope.log" >&2
  exit 1
fi

sleep "$WARMUP_SECONDS"
if ! kill -0 "$gamescope_pid" 2>/dev/null; then
  echo "Gamescope/editor exited during warm-up" >&2
  tail -100 "$LOG_DIR/gamescope.log" >&2
  exit 1
fi

rm -f "$OUTPUT"
gst-launch-1.0 -e \
  pipewiresrc target-object="$gamescope_target" do-timestamp=true \
  ! queue \
  ! videoconvert \
  ! video/x-raw,format=I420 \
  ! x264enc bitrate=8000 speed-preset=veryfast \
  ! video/x-h264,profile=main \
  ! h264parse \
  ! mp4mux faststart=true \
  ! filesink location="$OUTPUT" \
  >"$LOG_DIR/gstreamer.log" 2>&1 &
capture_pid=$!

sleep "$DURATION_SECONDS"
if ! kill -0 "$capture_pid" 2>/dev/null; then
  echo "GStreamer exited before the requested duration" >&2
  cat "$LOG_DIR/gstreamer.log" >&2
  exit 1
fi
stop_process "$capture_pid" INT
capture_pid=

ffprobe -v error "$OUTPUT" >/dev/null
printf 'Wrote %s\nLogs: %s\n' "$OUTPUT" "$LOG_DIR"
```

Adjust the command, resolution, warm-up, and duration for the task. The session runtime is always removed. Logs are intentionally retained for diagnosis.

## Concurrent agents

The media side supports concurrent captures when every invocation creates its own XDG/PipeWire session directory and starts its own PipeWire and WirePlumber. PipeWire object names are scoped to a daemon registry, and Gamescope Wayland/control sockets are scoped to `XDG_RUNTIME_DIR`. The automated recipe also uses unique log and output paths.

Resources outside that session can still conflict:

- Two examples that each launch an Elodin DB normally contend for port `2240`. Run them sequentially or arrange distinct DB addresses.
- Use distinct example-specific DB/replay directories.
- GPU memory and CPU encoder load remain shared. Assign idle GPUs where possible and lower resolution for parallel work.

## Still screenshots

With the private `XDG_RUNTIME_DIR` exported, locate Gamescope's control display and capture it:

```bash
for socket in "$XDG_RUNTIME_DIR"/gamescope-[0-9]*; do
  [[ "$socket" == *-ei ]] && continue
  [[ -S "$socket" ]] || continue
  export GAMESCOPE_WAYLAND_DISPLAY="$(basename "$socket")"
  break
done
gamescopectl screenshot /tmp/elodin-gamescope.png
```

For repeatable editor regression screenshots, prefer the editor's native `ELODIN_SCREENSHOT` harness described in `.cursor/skills/elodin-editor-dev/SKILL.md`.

## Troubleshooting

### Private PipeWire or WirePlumber does not start

Inspect the private-session logs, not host services. Ensure the diagnostic terminal exports the same `PIPEWIRE_RUNTIME_DIR`:

```bash
cat "$LOG_DIR/pipewire.log"
cat "$LOG_DIR/wireplumber.log"
pw-cli info 0
wpctl status
```

If WirePlumber reports that `/usr/share/wireplumber/wireplumber.conf` is an old configuration, it was not launched with `-c "$ELODIN_WIREPLUMBER_CONFIG"` from the project Nix shell. Re-enter `nix develop` and use the explicit configuration argument; the shell deliberately does not modify global XDG data lookup paths.

### `pipewiresrc` is missing

```bash
echo "$GST_PLUGIN_PATH"
gst-inspect-1.0 pipewiresrc
```

The Nix shell adds PipeWire's `lib/gstreamer-1.0` directory to `GST_PLUGIN_PATH`. Do not mix host and Nix GStreamer plugin/core versions.

### No Gamescope node or GStreamer cannot find the target

Make sure every process has the same `PIPEWIRE_RUNTIME_DIR`, then inspect:

```bash
pw-dump | jq -r '
  .[]
  | select(.type == "PipeWire:Interface:Node")
  | [.info.props["object.serial"], .info.props["node.name"], .info.props["media.class"]]
  | @tsv
'
```

The Nix-wrapped node may be `.gamescope-wrapped`; discover its `object.serial` rather than hard-coding a name.

### `RequestDeviceError ... Device(Lost)` on NVIDIA

Verify that Elodin, not just Gamescope, receives:

```bash
WGPU_SETTINGS_PRIO=webgpu
```

The documented Gamescope command sets it on the child with `env`. If diagnosing outside Gamescope, this should keep the headless render server alive rather than failing Vulkan device creation:

```bash
WGPU_SETTINGS_PRIO=webgpu ./target/release/elodin render-server --addr 127.0.0.1:29999
```

### Gamescope cannot initialize Vulkan

- Check GPU occupancy before retrying.
- Run `DISPLAY= vulkaninfo --summary` from the same Nix shell.
- Use `ELODIN_GPU=nvidia` or `ELODIN_GPU=mesa` to override broad driver routing.
- Use `ELODIN_NVIDIA_ICD=/path/to/nvidia_icd.json` if NVIDIA ICD discovery is wrong.
- On a multi-GPU host, `--prefer-vk-device VENDOR:DEVICE` can select a vendor/device pair, though it cannot distinguish identical GPUs.

A system-installed Gamescope can occasionally need `sudo ldconfig` after installation. The Nix-provided Gamescope does not; do not run it routinely.

### MP4 is empty or corrupt

- Send GStreamer `SIGINT` while Gamescope, WirePlumber, and PipeWire are alive.
- Keep `gst-launch-1.0 -e` and wait for it to exit.
- Stop in this order: GStreamer, Gamescope, WirePlumber, PipeWire.
- Remove stale output before recording and validate with `ffprobe`.

### Recording is blank, stutters, or uses too many resources

- Allow a longer warm-up.
- Reduce Gamescope to `960x540` and/or lower `-r` from 30.
- Lower `x264enc bitrate` if encoding load or file size is excessive.
- Keep `queue` between PipeWire and conversion/encoding.
- Do not run captures in parallel on the same GPU unless required.
