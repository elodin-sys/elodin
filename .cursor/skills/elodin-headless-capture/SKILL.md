---
name: elodin-headless-capture
description: Run the Elodin Editor without a physical display in Gamescope, take screenshots, and record video through PipeWire and GStreamer. Use for visual testing or capture on a headless Linux GPU host.
---

# Headless Elodin capture

This workflow is **Linux-only**. Gamescope's headless compositor, its PipeWire
video source, and the host GPU-driver integration used here are Linux
facilities. Keep the related Nix dependencies guarded by `stdenv.isLinux`.

Run commands from the repository root inside `nix develop`. Do not use `sudo`:
Gamescope and GStreamer must use the same user's PipeWire socket under
`XDG_RUNTIME_DIR`.

## Prerequisites

1. Build the release editor:

   ```bash
   cargo build --release -p elodin
   ```

   Running a Python example also requires the project virtual environment. If
   it is not already installed and active, run `just install` and then
   `source .venv/bin/activate`.

2. Check that the host PipeWire service and portable software encoder are
   available:

   ```bash
   pw-cli info 0
   gst-inspect-1.0 pipewiresrc >/dev/null
   gst-inspect-1.0 x264enc >/dev/null
   ```

   On a systemd desktop, start a missing PipeWire service with
   `systemctl --user start pipewire`. The host must expose its GPU devices and
   graphics drivers for accelerated editor rendering; Nix supplies the
   user-space tools, not the kernel driver.

## Automated capture (preferred)

The repository provides `scripts/elodin_capture.sh`, which performs the
PipeWire preflight, selects a vendor-compatible graphics path, starts Gamescope
with Nix's Xwayland, validates a hardware encoder when available, falls back to
x264, records, decodes a frame, rejects blank output, and cleans up all child
processes:

```bash
./scripts/elodin_capture.sh --duration 10 --output /tmp/elodin.mp4 examples/cube-sat/main.py
```

Run it inside `nix develop`. The default readiness check waits for the
simulation database server and then allows a two-second warmup. Use
`--ready-regex` or `--warmup` for examples with unusual startup behavior. Use
`--encoder x264` to force the portable fallback, or `--encoder vaapi` /
`--encoder nvenc` when testing a specific hardware path. Set
`ELODIN_GPU=mesa` or `ELODIN_GPU=nvidia` before entering the development shell
to override automatic GPU selection. An explicitly set `GBM_BACKENDS_PATH` is
always preserved.

The manual workflow below remains useful for debugging capture infrastructure.

## Start the editor manually

Use the lowest practical resolution so the editor and encoder consume fewer GPU
resources. In terminal 1:

```bash
gamescope --backend headless \
  -w 1280 -h 720 -W 1280 -H 720 -r 30 \
  -- ./target/release/elodin editor examples/three-body/main.py
```

Gamescope starts a nested Xwayland display for the editor and publishes its
composited output with the PipeWire media name `gamescope`. Wait for the editor
to finish loading before capturing.

If startup fails on a non-NixOS host because the dynamic linker has not noticed
newly installed host driver libraries, run `sudo ldconfig` once outside the
capture workflow and retry as the normal user.

## Screenshot

While Gamescope is running, use terminal 2:

```bash
gamescopectl screenshot /tmp/elodin.png
```

Use an absolute output path. Read the image afterward to verify that the scene
and editor chrome rendered correctly.

For a one-shot editor screenshot where a composited video is not needed, prefer
the editor's `ELODIN_SCREENSHOT` mechanism documented in the
`elodin-editor-dev` skill.

## Record video

In terminal 2, start this after the editor has loaded. Resolve the newest
Gamescope node's PipeWire object serial instead of hard-coding its name. Current
PipeWire can publish multiple nodes with the same `.gamescope-wrapped` name, so
the serial uniquely identifies the live compositor.

```bash
GAMESCOPE_TARGET="$(pw-dump | jq -r '
  [
    .[]
    | select(.type == "PipeWire:Interface:Node")
    | select(.info.props["media.name"] == "gamescope")
    | select(.info.props["object.serial"] != null)
    | {id: .id, serial: (.info.props["object.serial"] | tostring)}
  ]
  | sort_by(.id)
  | last
  | .serial // empty
')"
test -n "$GAMESCOPE_TARGET"

gst-launch-1.0 -e \
  pipewiresrc target-object="$GAMESCOPE_TARGET" do-timestamp=true \
  ! video/x-raw,format=BGRx \
  ! queue \
  ! videoconvert \
  ! video/x-raw,format=I420 \
  ! x264enc bitrate=8000 speed-preset=veryfast \
  ! video/x-h264,profile=main \
  ! h264parse \
  ! mp4mux faststart=true \
  ! filesink location=/tmp/elodin.mp4
```

This software-encoding command is the reliable baseline and fallback across
NVIDIA, AMD, and Intel systems. The first caps filter is required: forcing
Gamescope to provide `BGRx` avoids capture paths that can produce an all-black
video. `videoconvert` then converts the valid BGRx frames to the I420 input used
by x264.

The x264 command above is the known-good fallback, but an agent **should use a
hardware encoder when one is detected and proven to work**. Inspect available
GStreamer elements for NVENC on NVIDIA or VA-API on AMD/Intel, then validate the
candidate before using it for the requested capture:

1. Confirm that the candidate encoder is registered and can initialize.
2. Keep the explicit BGRx filter immediately after `pipewiresrc` and convert to
   a format accepted by the selected encoder.
3. Make a short test recording, decode a frame from it, and verify that it is
   nonblank. Plugin discovery and a valid MP4 alone are not sufficient.
4. Verify hardware-engine activity with an appropriate vendor tool when
   practical.
5. Use the working hardware path for the full capture. Fall back to the x264
   command only if hardware encoding is unavailable or fails validation.

Prefer validated hardware encoding, but never skip output validation or retain
a broken hardware path merely to avoid the software fallback.

Stop recording with **Ctrl-C**. The `-e` option sends end-of-stream so `mp4mux`
can finalize the MP4. Do not kill GStreamer with `SIGKILL`, or the output may be
unplayable.

Confirm the result:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 /tmp/elodin.mp4
```

## Verify rendering and output

Gamescope and Elodin should create graphics contexts and increase GPU
utilization while the editor is rendering. Use the appropriate vendor tool if
available. Software x264 encoding is expected to use the CPU.

After every capture, check the stream metadata and decode a representative
frame. Confirm visually, or with image statistics, that the decoded frame is
not all black. This catches a valid-looking MP4 produced from invalid capture
buffers.

## Troubleshooting

- **`pipewiresrc` is missing:** enter a fresh `nix develop`; the shell adds the
  PipeWire GStreamer plugin to `GST_PLUGIN_PATH`.
- **A hardware encoder is missing or fails to initialize:** use the documented
  x264 pipeline. If hardware encoding is important, verify the host driver and
  device permissions, re-enter `nix develop`, and clear a stale plugin cache
  with `rm -f ~/.cache/gstreamer-1.0/registry.*.bin` before probing again.
- **No `gamescope` source:** make sure Gamescope is already running. Inspect
  video node names and media names with `pw-dump | jq '.[] | select(.type ==
  "PipeWire:Interface:Node") | .info.props | select(."media.class" ==
  "Video/Source") | {node_name: ."node.name", media_name: ."media.name"}'`.
- **PipeWire connection refused:** check `echo "$XDG_RUNTIME_DIR"` and
  `pw-cli info 0`; run both terminals as the same non-root user.
- **All-black recording:** ensure the `video/x-raw,format=BGRx` filter appears
  immediately after `pipewiresrc`. Do not let a downstream encoder negotiate
  the source format directly.
- **Partially loaded recording:** wait longer before starting GStreamer, or use
  a lighter example and lower resolution.
- **Stale editor process or port 2240 conflict:** stop the previous Gamescope
  child before starting another editor.
