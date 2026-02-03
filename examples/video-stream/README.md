# Video Streaming Example

This example demonstrates streaming video from GStreamer into Elodin DB and displaying it in the Elodin Editor.

## What It Does

- Runs a minimal simulation with a rotating cube
- Launches a GStreamer pipeline that streams an H.264 test pattern to Elodin DB
- Displays the video stream automatically in the Elodin Editor

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         S10 Process Group                            │
│                                                                      │
│  ┌────────────────────┐         ┌─────────────────────────────────┐ │
│  │  Python Simulation │         │     GStreamer Pipeline          │ │
│  │  (main.py)         │         │     (stream-video.sh)           │ │
│  │                    │         │                                 │ │
│  │  - Rotating cube   │         │  videotestsrc -> x264enc ->     │ │
│  │  - Schematic with  │         │  h264parse -> elodinsink        │ │
│  │    video_stream    │         │                                 │ │
│  └─────────┬──────────┘         └───────────────┬─────────────────┘ │
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

### No video appears

- Verify the GStreamer pipeline started (check terminal output)
- Ensure the message name matches exactly: `test-video`
- Check that Elodin DB is running (it starts automatically with the editor)

### Plugin build fails

- Make sure you're in a nix develop shell
- Check that GStreamer development libraries are available: `pkg-config --libs gstreamer-1.0`

### "Loss of Signal" overlay

This appears when the video stream stops or timestamps become stale. Check if the GStreamer process is still running.

## Technical Details

- **Video codec**: H.264 baseline profile
- **Encoder settings**: `tune=zerolatency key-int-max=12` for low latency
- **h264parse config-interval=-1**: Ensures SPS/PPS sent with every keyframe (required for decoder)
- **Message protocol**: Uses `MsgWithTimestamp` packets via TCP to Elodin DB port 2240
