# Video Streaming Example

This example demonstrates two ways to stream video into Elodin DB and display it in the Elodin Editor:

1. **GStreamer + elodinsink** (direct to DB) - a GStreamer pipeline sends H.264 via the custom `elodinsink` plugin
2. **Native RTMP ingest** (OBS Studio) - OBS Studio streams via the DB's built-in RTMP server

## What It Does

- Runs a rolling ball simulation with random wind and wall bouncing
- Launches a GStreamer pipeline that streams an H.264 test pattern to Elodin DB (elodinsink)
- Provides an RTMP ingest endpoint for OBS Studio to stream into
- Displays video streams automatically alongside the 3D viewport
- Stores all data (telemetry + video) to `./video-stream-db` for replay and export

## Architecture

```
                                              ┌──────────────────────┐
                                              │   OBS Studio         │
                                              │   (customer laptop)  │
                                              └──────────┬───────────┘
                                                         │ RTMP
┌────────────────────────────────────────────────────┐   │
│                S10 Process Group                    │   │
│                                                    │   │
│  ┌──────────────────┐  ┌───────────────────────┐  │   │
│  │  Python Sim       │  │  GStreamer Pipeline   │  │   │
│  │  (main.py)        │  │  (stream-video.sh)    │  │   │
│  │                   │  │                       │  │   │
│  │  - Rolling ball   │  │  videotestsrc ->      │  │   │
│  │  - Wind/friction  │  │  x264enc ->           │  │   │
│  │  - Wall bouncing  │  │  h264parse ->         │  │   │
│  │                   │  │  elodinsink           │  │   │
│  └─────────┬─────────┘  └──────────┬────────────┘  │   │
│            │                       │                │   │
└────────────┼───────────────────────┼────────────────┘   │
             │                       │                    │
             │       ┌───────────────┴─────────┐          │
             └──────►│       Elodin DB         │◄─────────┘
                     │  :2240 (impeller2/TCP)   │
                     │  :2241 (RTMP ingest)     │
                     └───────────┬─────────────┘
                                 │
                     ┌───────────▼─────────────┐
                     │     Elodin Editor       │
                     │  - 3D Viewport          │
                     │  - Test Pattern (GST)   │
                     │  - OBS Studio Feed      │
                     └─────────────────────────┘
```

## Simulation Features

- **Rolling ball**: A ball rolls around on a flat surface, visually spinning as it moves
- **Random wind**: Applies drag force that pushes the ball in random directions
- **Wall bouncing**: Ball bounces off walls at the viewport edges (±4 units)
- **Friction**: Viscous damping for smooth, natural deceleration
- **Semi-implicit integrator**: Provides stable, game-like motion

## Requirements

- Nix development shell (`nix develop`) which provides GStreamer and x264
- Rust toolchain (for building the elodinsink plugin)

## Walkthrough

This walkthrough takes you through the full lifecycle: running the example, streaming from OBS Studio, replaying the recorded data, and exporting the video to an MP4 file. All commands are run from the repository root.

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

The editor opens with:
- A **3D viewport** showing a rolling ball pushed by wind
- A **Test Pattern** tile showing a GStreamer bouncing ball (streamed via `elodinsink`)
- An **OBS Studio Feed** tile (shows "Connecting..." until OBS is connected)
- A **Wind graph** showing the rotating wind force

The RTMP ingest server starts automatically on port **2241** (DB port + 1). The GStreamer plugin (`elodinsink`) is built automatically on first run.

### Step 4: Connect OBS Studio

While the example is running, open OBS Studio and configure it to stream to the Elodin DB:

Ubuntu install:
```sh
sudo add-apt-repository ppa:obsproject/obs-studio
sudo apt update
sudo apt install obs-studio
```
(or others [here](https://obsproject.com/download))

1. Go to **Settings -> Stream**
2. Set **Service** to **Custom**
3. Set **Server** to `rtmp://YOUR_ELODIN_IP:2241/live`
4. Set **Stream Key** to `rtmp-feed`
5. Click **OK** to save

Then click **Start Streaming** in OBS. The "OBS Studio Feed" tile in the Elodin Editor will show your OBS output.

> **Note**: The stream key (`rtmp-feed`) must match the `video_stream` panel name in the schematic. You can add multiple `video_stream` panels with different names and use different stream keys for multiple cameras.

#### Recommended OBS Encoder Settings

For best results with Elodin, use **Advanced** output mode (Settings -> Output -> Output Mode: Advanced):

| Setting | Value |
|---------|-------|
| **Encoder** | x264 (Software) or NVENC (Hardware) |
| **Rate Control** | CBR |
| **Bitrate** | 5000-15000 kbps (higher than usual -- see below) |
| **Profile** | Baseline |
| **Tune** | `zerolatency` |
| **x264 Options** | `keyint=1` |
| **Output format** | H.264 only (H.265/AV1 are not supported) |

> **Important**: The `keyint=1` option makes every frame a keyframe (IDR frame). This is required because the Elodin Editor decodes each video frame independently. Without this setting, only keyframes will display and the video will appear to update only every few seconds. The tradeoff is higher bandwidth, but for telemetry use cases over a LAN this is not a concern. Increase the bitrate accordingly (5000-15000 kbps) to maintain quality.

### Step 5: Replay the Recorded Data

Let the example run for about **60 seconds** so it records enough video and telemetry data, then close the editor window. All data is written to `./video-stream-db`.

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

You should see the GStreamer test pattern that was streamed during the simulation.

## Troubleshooting

### Video tile shows "Initializing..." or "Connecting..."

For the **Test Pattern** tile: this is normal during startup -- the video will appear once the GStreamer pipeline starts streaming (a few seconds).

For the **OBS Studio Feed** tile: this is expected until you connect OBS Studio to the RTMP endpoint (see Step 4 above).

### No video appears

- Verify the GStreamer pipeline started (check terminal output for `video-stream` messages)
- Ensure the message name matches exactly: `test-video` for the elodinsink stream
- Check that Elodin DB is running (it starts automatically with the editor)

### OBS Studio can't connect

- Make sure the example is running (the RTMP server starts on port 2241)
- Verify the server URL: `rtmp://HOST:2241/live` (not port 1935)
- Check the stream key matches a `video_stream` panel name in the schematic

### Plugin build fails

- Make sure you're in a nix develop shell
- Check that GStreamer development libraries are available: `pkg-config --libs gstreamer-1.0`

### "Loss of Signal" overlay

This appears when the video stream stops or timestamps become stale. Check if the GStreamer process or OBS Studio is still streaming.

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
- **elodinsink path**: Uses `MsgWithTimestamp` packets via TCP to Elodin DB port 2240
- **RTMP path**: Uses native RTMP ingest on port 2241 (DB port + 1), converting FLV/AVCC to Annex-B H.264 internally
- **Database path**: `./video-stream-db` (relative to where you run the command)
