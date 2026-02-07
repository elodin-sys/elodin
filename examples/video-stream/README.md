# Video Streaming Example

This example demonstrates streaming video from GStreamer into Elodin DB and displaying it in the Elodin Editor.

## What It Does

- Runs a rolling ball simulation with random wind and wall bouncing
- Launches a GStreamer pipeline that streams an H.264 test pattern to Elodin DB
- Displays the video stream automatically alongside the 3D viewport
- Stores all data (telemetry + video) to `./video-stream-db` for replay and export

## Simulation Features

- **Rolling ball**: A ball rolls around on a flat surface, visually spinning as it moves
- **Random wind**: Applies drag force that pushes the ball in random directions
- **Wall bouncing**: Ball bounces off walls at the viewport edges (±4 units)
- **Friction**: Viscous damping for smooth, natural deceleration
- **Semi-implicit integrator**: Provides stable, game-like motion

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         S10 Process Group                            │
│                                                                      │
│  ┌────────────────────┐         ┌─────────────────────────────────┐  │
│  │  Python Simulation │         │     GStreamer Pipeline          │  │
│  │  (main.py)         │         │     (stream-video.sh)           │  │
│  │                    │         │                                 │  │
│  │  - Rolling ball    │         │  videotestsrc -> x264enc ->     │  │
│  │  - Wind/friction   │         │  h264parse -> elodinsink        │  │
│  │  - Wall bouncing   │         │                                 │  │
│  └─────────┬──────────┘         └───────────────┬─────────────────┘  │
│            │                                    │                    │
└────────────┼────────────────────────────────────┼────────────────────┘
             │                                    │
             │        ┌──────────────┐            │
             └───────►│  Elodin DB   │◄───────────┘
                      │  :2240       │
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │Elodin Editor │
                      │ - 3D View    │
                      │ - Video Tile │
                      └──────────────┘
```

## Requirements

- Nix development shell (`nix develop`) which provides GStreamer and x264
- Rust toolchain (for building the elodinsink plugin)

## Walkthrough

This walkthrough takes you through the full lifecycle: running the example, replaying the recorded data, and exporting the video to an MP4 file. All commands are run from the repository root.

### Step 1: Enter the Nix Development Shell

```bash
nix develop
```

This provides GStreamer, x264, and all other dependencies needed by the example.

### Step 2: Build the Elodin Tools

```bash
just install
```

This builds the Elodin Editor, Elodin DB, and all supporting tools.

### Step 3: Run the Example

```bash
elodin editor examples/video-stream/main.py
```

The editor opens with a 3D viewport (showing a rolling ball) and a video stream tile (showing a GStreamer test pattern). The GStreamer plugin (`elodinsink`) is built automatically on first run.

Let the example run for about **60 seconds** so it records enough video and telemetry data, then close the editor window. All data is written to `./video-stream-db`.

### Step 4: Replay the Recorded Data

Now start Elodin DB pointing at the data that was just recorded:

```bash
elodin-db run [::]:2240 ./video-stream-db
```

In a **second terminal** (also in the nix shell), connect the editor to the running database:

```bash
nix develop --command elodin editor 127.0.0.1:2240
```

You can now scrub through the timeline to review the recorded simulation and video. When you're done, close the editor and stop the database with `Ctrl+C`.

### Step 5: Export the Video to MP4

Export the recorded video stream to an MP4 file:

```bash
elodin-db export-videos ./video-stream-db -o ./videos
```

This reads the H.264 frames stored in the database and muxes them into a standards-compliant MP4. The output file is written to `./videos/`.

### Step 6: Play the Exported Video

Open the MP4 to confirm the result:

```bash
# macOS
open ./videos/test-video.mp4

# Linux
xdg-open ./videos/test-video.mp4
```

You should see the GStreamer test pattern that was streamed during the simulation.

## Troubleshooting

### Video tile shows "Initializing..." or "Connecting..."

This is normal - the video stream tile defers connection for about 0.5 seconds during startup to ensure the system is fully initialized. The video will appear once the GStreamer pipeline starts streaming.

### No video appears

- Verify the GStreamer pipeline started (check terminal output)
- Ensure the message name matches exactly: `test-video`
- Check that Elodin DB is running (it starts automatically with the editor)

### Plugin build fails

- Make sure you're in a nix develop shell
- Check that GStreamer development libraries are available: `pkg-config --libs gstreamer-1.0`

### "Loss of Signal" overlay

This appears when the video stream stops or timestamps become stale. Check if the GStreamer process is still running.

### "Stream disconnected. Reconnecting..."

The video stream tile automatically detects when frames stop arriving and will attempt to reconnect every 2 seconds. This handles cases where the video source temporarily drops.

### export-videos produces no output

- Make sure the example ran for long enough to record at least one keyframe (a few seconds is sufficient)
- Check that `./video-stream-db/msgs/` contains data directories
- Try with `--pattern "test-*"` to filter by the message name used in this example

## Technical Details

- **Video codec**: H.264 baseline profile
- **Encoder settings**: `tune=zerolatency key-int-max=12` for low latency
- **h264parse config-interval=-1**: Ensures SPS/PPS sent with every keyframe (required for decoder)
- **Message protocol**: Uses `MsgWithTimestamp` packets via TCP to Elodin DB port 2240
- **Database path**: `./video-stream-db` (relative to where you run the command)
