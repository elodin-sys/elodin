# Video Streaming Example

This example demonstrates streaming video from GStreamer into Elodin DB and displaying it in the Elodin Editor.

## What It Does

- Runs a rolling ball simulation with random wind and wall bouncing
- Launches a GStreamer pipeline that streams an H.264 test pattern to Elodin DB
- Displays the video stream automatically alongside the 3D viewport

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

## Running the Example

From the repository root:

```bash
elodin editor examples/video-stream/main.py
```

The GStreamer plugin (`elodinsink`) is built automatically on first run.

The video stream tile appears automatically - no manual setup required. The schematic includes a `video_stream` panel that connects to the `test-video` message.

## Requirements

- Nix development shell (`nix develop`) which provides GStreamer and x264
- Rust toolchain (for building the elodinsink plugin)

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

## Technical Details

- **Video codec**: H.264 baseline profile
- **Encoder settings**: `tune=zerolatency key-int-max=12` for low latency
- **h264parse config-interval=-1**: Ensures SPS/PPS sent with every keyframe (required for decoder)
- **Message protocol**: Uses `MsgWithTimestamp` packets via TCP to Elodin DB port 2240
