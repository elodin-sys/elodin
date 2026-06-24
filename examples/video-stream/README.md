# Video Streaming Example

This example demonstrates streaming video into Elodin DB and displaying it in the Elodin Editor. It includes three video sources:

1. **Test Pattern** - A local GStreamer test pattern that starts automatically
2. **OBS Camera** - An SRT receiver that accepts a live stream from OBS Studio
3. **RTSP Camera** - An H.264 RTSP stream pulled from an IP camera or OBS, ingested **natively by Elodin DB** (or via a GStreamer sidecar)

## What It Does

- Runs a rolling ball simulation with rotating wind and wall bouncing
- Launches a GStreamer pipeline that streams an H.264 test pattern to Elodin DB
- Launches an SRT receiver that listens for OBS Studio connections on port 9000
- Displays the video streams (Test Pattern, OBS Camera, RTSP Camera) in tabs alongside the 3D viewport
- Stores all data (telemetry + video) to `./video-stream-db` for replay and export

## Simulation Features

- **Rolling ball**: A ball rolls around on a flat surface, visually spinning as it moves
- **Rotating wind**: Applies force that pushes the ball in a rotating direction
- **Wall bouncing**: Ball bounces off walls at the viewport edges (±4 units)
- **Friction**: Viscous damping for smooth, natural deceleration
- **Semi-implicit integrator**: Provides stable, game-like motion

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         S10 Process Group                                │
│                                                                          │
│  ┌────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │  Python Simulation │  │  GStreamer Pipeline  │  │  SRT Receiver    │  │
│  │  (main.py)         │  │  (stream-video.sh)   │  │  (receive-obs-   │  │
│  │                    │  │                      │  │   stream.sh)     │  │
│  │  - Rolling ball    │  │  videotestsrc ->     │  │                  │  │
│  │  - Wind/friction   │  │  x264enc ->          │  │  srtsrc :9000 -> │  │
│  │  - Wall bouncing   │  │  h264parse ->        │  │  tsdemux ->      │  │
│  │                    │  │  elodinsink          │  │  h264parse ->    │  │
│  │                    │  │  msg:"test-video"    │  │  elodinsink      │  │
│  │                    │  │                      │  │  msg:"obs-camera"│  │
│  └────────┬───────────┘  └──────────┬───────────┘  └────────┬─────────┘  │
│           │                         │                       │            │
└───────────┼─────────────────────────┼───────────────────────┼────────────┘
            │                         │                       │
            │        ┌────────────────┴───────────────────────┘
            │        │
            │   ┌────▼─────────┐         ┌──────────────┐
            └──►│  Elodin DB   │         │  OBS Studio  │
                │  :2240       │         │  (optional)  │
                └──────┬───────┘         └──────┬───────┘
                       │                        │
                ┌──────▼───────┐          SRT over network
                │Elodin Editor │          to :9000
                │ - 3D View    │                │
                │ - Test Video │                │
                │ - OBS Camera │◄───────────────┘
                └──────────────┘
```

## Requirements

- Nix development shell (`nix develop`) which provides GStreamer, SRT, and x264
- Rust toolchain (for building the elodinsink plugin)
- OBS Studio (optional, for the OBS Camera stream)

## Walkthrough

This walkthrough takes you through the full lifecycle: running the example, optionally connecting OBS Studio, replaying the recorded data, and exporting the video to an MP4 file. All commands are run from the repository root.

### Step 1: Enter the Nix Development Shell

```bash
nix develop
```

This provides GStreamer, x264, SRT, and all other dependencies needed by the example.

### Step 2: Build the Elodin Tools

```bash
just install
```

This builds the Elodin Editor, Elodin DB, and all supporting tools.

### Step 3: Run the Example

```bash
elodin editor examples/video-stream/main.py
```

The editor opens with a 3D viewport (showing a rolling ball) and three video stream tabs:
- **Test Pattern**: Shows a GStreamer test pattern immediately
- **OBS Camera**: Shows "Initializing..." until OBS connects
- **RTSP Camera**: Shows "No video at this time" until an RTSP source feeds the `rtsp-camera` message (see [RTSP Camera](#rtsp-camera))

The GStreamer plugin (`elodinsink`) is built automatically on first run.

### Step 4: Connect OBS Studio (Optional)

To stream live video from OBS Studio into the "OBS Camera" tab:

1. Open OBS Studio
2. Go to **Settings** -> **Stream**
3. Set **Service** to **Custom...**
4. Set **Server** to:
   ```
   srt://ELODIN_IP:9000?mode=caller
   ```
   Replace `ELODIN_IP` with the IP address of the machine running Elodin. If OBS and Elodin are on the same machine, use `127.0.0.1`.
5. Leave **Stream Key** empty
6. Click **OK**
7. Click **Start Streaming** in OBS

The video should appear in the "OBS Camera" tab within a few seconds.

**Recommended OBS encoder settings** (Settings -> Output -> Advanced -> Streaming):

| Setting | Value |
|---|---|
| Encoder | x264 (Software) or NVENC (Hardware) |
| Rate Control | CBR |
| Bitrate | 2500-6000 kbps |
| Keyframe Interval | 2 seconds |
| Profile | Baseline or Main (High also works) |
| Tune | `zerolatency` |

> **Important**: Use H.264, not H.265/HEVC. Elodin's video decoder only supports H.264.

Let the example run for about **60 seconds** so it records enough video and telemetry data, then close the editor window. All data is written to `./video-stream-db`.

### Step 5: Replay the Recorded Data

Now start Elodin DB pointing at the data that was just recorded:

```bash
elodin-db run [::]:2240 ./video-stream-db
```

In a **second terminal** (also in the nix shell), connect the editor to the running database:

```bash
nix develop --command elodin editor 127.0.0.1:2240
```

You can now scrub through the timeline to review the recorded simulation and video. When you're done, close the editor and stop the database with `Ctrl+C`.

### Step 6: Export the Video to MP4

Export the recorded video stream to an MP4 file:

```bash
elodin-db export-videos ./video-stream-db -o ./videos
```

This reads the H.264 frames stored in the database and muxes them into a standards-compliant MP4. The output file is written to `./videos/`.

### Step 7: Play the Exported Video

Open the MP4 to confirm the result:

```bash
# macOS
open ./videos/test-video.mp4

# Linux
xdg-open ./videos/test-video.mp4
```

You should see the GStreamer test pattern that was streamed during the simulation. If you also streamed from OBS, `obs-camera.mp4` will be present; if you fed the RTSP Camera, `rtsp-camera.mp4` will be too.

## RTSP Camera

The **RTSP Camera** tab shows an H.264 RTSP stream (an IP camera, or OBS via the
[OBS-RTSPServer](https://github.com/iamscottxu/obs-rtspserver) plugin). There are
two ways to get an RTSP stream into the `rtsp-camera` message:

### Quick manual test (one command, no camera)

For reviewers and quick local checks, a single script wires up the whole native
path end-to-end — a fake RTSP source, the RTSP-pulling DB, and the editor:

```bash
nix develop --command bash examples/video-stream/run-rtsp-demo.sh
```

It builds `elodin` + `elodin-db --features rtsp`, starts `mediamtx` (which
self-publishes an H.264 test pattern via `ffmpeg`), runs `elodin-db` pulling
`rtsp://127.0.0.1:8554/test` into the `rtsp-camera` message, and opens the editor
on the `rtsp_stream` panel. You should see the moving test pattern in the **RTSP
Camera** tab within a few seconds. Close the editor window to stop everything.

`mediamtx` is fetched automatically via `nix build nixpkgs#mediamtx` (needs nix +
network); runtime data/logs go to `examples/video-stream/.rtsp-demo/` (git-ignored).

The two manual routes below explain the moving parts the script automates.

### Option A — Native ingest (the feature)

Elodin DB can act as the RTSP **client** itself (via [`retina`](https://crates.io/crates/retina)):
it pulls the stream, reframes each access unit to Annex-B (repeating SPS/PPS ahead
of every keyframe), and stores it through the same `MsgLog` path as every other
video source. No inbound port is opened and no GStreamer sidecar is needed — once
the bytes land, playback/scrub/export/replication all work unchanged.

The ingest is gated behind the opt-in `rtsp` Cargo feature (off by default), so
run a standalone DB built with it (the `main.py` editor flow embeds a DB without
this feature, so use a standalone server for the native path):

```bash
cargo run -p elodin-db --features rtsp -- run \
    --rtsp-source rtsp-camera=rtsp://USER:PASS@CAMERA_IP:554/stream \
    "[::]:2240" ./video-stream-db
```

Then connect the editor with this example's schematic:

```bash
elodin editor 127.0.0.1:2240
```

`--rtsp-source NAME=URL` is repeatable (one per camera). The `NAME` must match the
panel — here `rtsp-camera`. Credentials may be embedded in the URL. The DB
reconnects automatically if the source drops. v1 expects H.264 **baseline/main**
with `bframes=0`.

### Option B — GStreamer sidecar (feeds the editor's embedded DB)

When running the full example via `elodin editor examples/video-stream/main.py`,
the DB is embedded in the simulation and has no `--rtsp-source` flag. Use the
sidecar helper to pull RTSP and push it into that DB:

```bash
bash examples/video-stream/receive-rtsp-stream.sh --rtsp-url rtsp://CAMERA_IP:554/stream
```

### Local test source (no camera needed)

Use [`mediamtx`](https://github.com/bluenviron/mediamtx) as a tiny RTSP server and
publish a test pattern to it with `ffmpeg`:

```bash
# Terminal 1: RTSP server
mediamtx

# Terminal 2: publish an H.264 baseline test pattern to it
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
    -c:v libx264 -profile:v baseline -tune zerolatency -pix_fmt yuv420p -g 30 \
    -f rtsp rtsp://127.0.0.1:8554/test
```

Then point either option above at `rtsp://127.0.0.1:8554/test`.

### Using OBS

Install the [OBS-RTSPServer](https://github.com/iamscottxu/obs-rtspserver) plugin,
start the RTSP server in OBS (Tools → RTSP Server), and point `--rtsp-source` /
`--rtsp-url` at the URL it advertises (typically `rtsp://OBS_IP:554/live`). Use an
H.264 encoder (not H.265/HEVC).

## Alternative: obs-gstreamer Direct Pipeline

If you install the [obs-gstreamer](https://github.com/fzwoch/obs-gstreamer) plugin on the OBS machine, you can pipe H.264 directly into `elodinsink` without the SRT receiver process.

### Setup

1. Install `obs-gstreamer` following its [installation instructions](https://github.com/fzwoch/obs-gstreamer#installation)
2. Build `elodinsink` on the OBS machine (requires the Elodin repo and nix shell)
3. In OBS, go to **Settings** -> **Output** -> **Output Mode: Advanced**
4. Under the **Streaming** tab, set **Encoder** to **GStreamer**
5. Set the pipeline to:
   ```
   video. ! h264parse config-interval=-1 ! elodinsink db-address=ELODIN_IP:2240 msg-name="obs-camera" audio. ! fakesink
   ```
   Replace `ELODIN_IP` with the Elodin server's IP address.
6. Click **Start Streaming**

This approach eliminates the SRT transport layer and sends H.264 NAL units directly to Elodin DB over TCP. It requires `elodinsink` to be installed on the OBS machine.

## SRT Receiver Script Options

The `receive-obs-stream.sh` script accepts the following options:

| Option | Default | Description |
|---|---|---|
| `--srt-port PORT` | `9000` | SRT listen port |
| `--db-address ADDR` | `127.0.0.1:2240` | Elodin DB address |
| `--msg-name NAME` | `obs-camera` | Video message name (must match schematic) |
| `--latency MS` | `125` | SRT latency in milliseconds |

Example with custom settings:

```bash
bash examples/video-stream/receive-obs-stream.sh --srt-port 9001 --msg-name "webcam" --latency 200
```

## RTSP Receiver Script Options

The `receive-rtsp-stream.sh` sidecar accepts the following options:

| Option | Default | Description |
|---|---|---|
| `--rtsp-url URL` | `rtsp://127.0.0.1:8554/test` | RTSP source URL (IP camera / OBS / mediamtx) |
| `--db-address ADDR` | `127.0.0.1:2240` | Elodin DB address |
| `--msg-name NAME` | `rtsp-camera` | Video message name (must match schematic) |

Example:

```bash
bash examples/video-stream/receive-rtsp-stream.sh --rtsp-url rtsp://192.168.1.50:554/stream --msg-name rtsp-camera
```

## Troubleshooting

### Video tile shows "Initializing..." or "Connecting..."

This is normal - the video stream tile defers connection for about 0.5 seconds during startup to ensure the system is fully initialized. The test pattern video will appear once the GStreamer pipeline starts streaming. The OBS Camera tile will show this until OBS connects.

### No video appears (Test Pattern)

- Verify the GStreamer pipeline started (check terminal output)
- Ensure the message name matches exactly: `test-video`
- Check that Elodin DB is running (it starts automatically with the editor)

### No video appears (OBS Camera)

- **Check the SRT URL**: Ensure OBS is pointing to the correct IP and port. If on the same machine, use `srt://127.0.0.1:9000?mode=caller`.
- **Check the encoder**: OBS must be using H.264 (not H.265/HEVC). Go to Settings -> Output -> Streaming and verify the encoder.
- **Check firewall**: Port 9000 (UDP) must be open on the Elodin machine. SRT uses UDP.
  ```bash
  # Linux - allow SRT port
  sudo ufw allow 9000/udp
  ```
- **Check GStreamer output**: Look at the terminal output for error messages from the GStreamer pipeline.

### High latency (OBS stream)

- Set OBS encoder tune to `zerolatency`
- Reduce SRT latency: `--latency 80` (minimum depends on network conditions)
- Use hardware encoding (NVENC/QSV) instead of x264 for lower encoding latency
- Reduce resolution and bitrate in OBS

### Plugin build fails

- Make sure you're in a nix develop shell
- Check that GStreamer development libraries are available: `pkg-config --libs gstreamer-1.0`

### "Loss of Signal" overlay

This appears when the video stream stops or timestamps become stale. Check if the GStreamer process is still running.

### "Stream disconnected. Reconnecting..."

The video stream tile automatically detects when frames stop arriving and will attempt to reconnect every 2 seconds. This handles cases where the video source temporarily drops.

### SRT connection refused

- Verify the receiver script is running (`receive-obs-stream.sh`)
- Ensure the SRT port matches between OBS (`?mode=caller`) and the receiver (`--srt-port`)
- Check that no other process is using the SRT port

### OBS shows "Failed to connect to server"

- Verify the Elodin machine's IP address is correct
- Check that the SRT receiver is running before starting OBS streaming
- Try with `srt://127.0.0.1:9000?mode=caller` if both are on the same machine

### export-videos produces no output

- Make sure the example ran for long enough to record at least one keyframe (a few seconds is sufficient)
- Check that `./video-stream-db/msgs/` contains data directories
- Try with `--pattern "test-*"` to filter by the message name used in this example

## Technical Details

- **Video codec**: H.264 baseline profile (test pattern), any H.264 profile (OBS)
- **Encoder settings**: `tune=zerolatency key-int-max=12` for low latency (test pattern)
- **h264parse config-interval=-1**: Ensures SPS/PPS sent with every keyframe (required for decoder)
- **SRT transport**: Secure Reliable Transport over UDP, listener on receiver, caller on OBS
- **Default SRT latency**: 125 ms (tunable for network conditions)
- **Message protocol**: Uses `MsgWithTimestamp` packets via TCP to Elodin DB port 2240
- **Database path**: `./video-stream-db` (relative to where you run the command)
