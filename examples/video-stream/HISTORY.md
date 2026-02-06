# Video Streaming Architecture

> Technical documentation for the Elodin video streaming feature introduced in PR #67.  
> This document explains how to stream H.264 video from GStreamer pipelines into Elodin DB and play it back in real-time in the Elodin Editor.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Deep Dive](#component-deep-dive)
4. [Protocol Details](#protocol-details)
5. [Local Setup Instructions](#local-setup-instructions)
6. [Example Pipelines](#example-pipelines)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [File Reference](#file-reference)

---

## Executive Summary

The video streaming feature enables real-time video transmission from cameras and video files to the Elodin Editor through Elodin DB. This capability supports critical use cases including:

- **Live camera feeds** from Aleph flight computers during testing
- **Sensor visualization** from USB cameras and GigE Vision cameras
- **Mission monitoring** with low-latency video from drones and spacecraft
- **Hardware-in-the-loop testing** with visual feedback
- **Replay and analysis** of recorded video synchronized with telemetry data

### Key Design Decisions

1. **H.264 codec** - Chosen for broad platform support and hardware acceleration availability
2. **Annex-B NAL format** - Standard byte-stream format compatible with GStreamer
3. **Message-based storage** - Video frames stored as timestamped messages in Elodin DB
4. **Fixed-rate playback** - Synchronized frame delivery for smooth video playback
5. **Platform-specific decoding** - VideoToolbox on macOS, OpenH264 on Linux/Windows

---

## System Architecture

### High-Level Data Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              VIDEO SOURCES                                │
│  ┌──────────┐  ┌─────────────────┐  ┌────────────┐  ┌──────────────┐      │
│  │USB Camera│  │GigE Vision Cam  │  │Video File  │  │Test Pattern  │      │
│  └────┬─────┘  └───────┬─────────┘  └─────┬──────┘  └──────┬───────┘      │
└───────┼────────────────┼──────────────────┼────────────────┼──────────────┘
        │                │                  │                │
        └────────────────┴─────────┬────────┴────────────────┘
                                   ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          GSTREAMER PIPELINE                                │
│  ┌────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐│
│  │ Video Capture  │───►│ H.264 Encoder│───►│  h264parse   │───►│elodinsink││
│  └────────────────┘    └──────────────┘    └──────────────┘    └────┬─────┘│
└─────────────────────────────────────────────────────────────────────┼──────┘
                                                                      │
                                                          TCP :2240   │
                                                     MsgWithTimestamp │
                                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ELODIN DB                                      │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │ Message Log             │────────►│ Fixed Rate Stream State         │    │
│  │ (msgs/{msg_id}/)        │         │ (playback synchronization)      │    │
│  └─────────────────────────┘         └───────────────┬─────────────────┘    │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                                       FixedRateMsgStream Response
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ELODIN EDITOR                                    │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐  │
│  │ Video Stream Tile   │───►│ H.264 Decoder       │───►│ RGBA Display    │  │
│  │ (requests stream)   │    │ (VideoToolbox/      │    │ (rendered frame)│  │
│  │                     │    │  OpenH264)          │    │                 │  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Location | Purpose |
|-----------|----------|---------|
| **elodinsink** | `fsw/gstreamer/` | GStreamer plugin that sends H.264 NAL units to Elodin DB |
| **video-streamer** | `fsw/video-streamer/` | FFMPEG-based utility for streaming video files |
| **video-toolbox** | `libs/video-toolbox/` | H.264 decoder abstraction (VideoToolbox on macOS) |
| **impeller2** | `libs/impeller2/` | Protocol with `MsgWithTimestamp` packet type |
| **Elodin DB** | `libs/db/` | Message storage and fixed-rate streaming |
| **Elodin Editor** | `libs/elodin-editor/` | Video tile with decoder and display |

---

## Component Deep Dive

### A. GStreamer Plugin (`elodinsink`)

**Location:** `fsw/gstreamer/src/lib.rs`

The `elodinsink` element is a custom GStreamer sink that transmits H.264 NAL units to Elodin DB over TCP.

#### Capabilities

```
video/x-h264, stream-format=byte-stream, alignment=au
```

The plugin accepts H.264 video in Annex-B byte-stream format with access unit alignment. This is the standard output format from `h264parse`.

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `db-address` | string | `127.0.0.1:2240` | Elodin DB TCP address |
| `msg-name` | string | `video` | Message name (hashed to create msg_id) |

#### Implementation Details

```rust
// Key packet creation from fsw/gstreamer/src/lib.rs
fn send_packet(&self, data: &[u8]) -> Result<(), gst::ErrorMessage> {
    let msg_id = msg_id(&state.msg_name);
    
    // Creates packet with embedded timestamp
    let mut packet = LenPacket::msg_with_timestamp(msg_id, Timestamp::now(), data.len());
    packet.extend_from_slice(data);
    
    stream.write_all(&packet.inner)?;
}
```

The plugin:
1. Connects to Elodin DB via TCP on startup
2. Hashes the `msg-name` to create a 16-bit message ID
3. Wraps each H.264 NAL unit in a `MsgWithTimestamp` packet
4. Uses wall-clock timestamps (`Timestamp::now()`)
5. Automatically reconnects on connection loss

---

### B. Video Streamer (FFMPEG-based)

**Location:** `fsw/video-streamer/src/main.rs`

A command-line utility that reads video files and streams them to Elodin DB. Useful for testing and replaying recorded video.

#### Usage

```bash
video-streamer <input_file> <msg_name> [OPTIONS]

Options:
  -d, --db-addr <IP:PORT>      Elodin DB address [default: 127.0.0.1:2240]
  -b, --bitrate <KBPS>         Output bitrate [default: 1000]
  -k, --keyframe-interval <N>  Keyframe interval [default: 12]
  -e, --encoder <NAME>         H.264 encoder [default: libopenh264]
  -l, --live                   Use real-time timestamps
```

#### Implementation Details

The video streamer:
1. Decodes input video using FFMPEG
2. Re-encodes to H.264 with baseline profile
3. Sends encoded packets to Elodin DB with timestamps
4. In `--live` mode, uses wall-clock time; otherwise preserves original timestamps

```rust
// Timestamp handling from fsw/video-streamer/src/main.rs
let mut pkt = LenPacket::msg_with_timestamp(
    msg_id,
    if self.live {
        Timestamp::now()  // Real-time timestamps
    } else {
        pkt_timestamp     // Preserved video timestamps
    },
    data.len(),
);
```

---

### C. Video Toolbox Library

**Location:** `libs/video-toolbox/`

A cross-platform H.264 decoder abstraction that provides:
- **macOS/iOS:** Hardware-accelerated decoding via Apple's VideoToolbox API
- **Linux/Windows:** Software decoding via OpenH264 (handled in the editor)

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Aleph (Flight Computer)                │
│  ┌──────────┐                                           │
│  │USB Camera│──┐                                        │
│  └──────────┘  │     ┌─────────────────────┐            │
│                ├───► │    GStreamer        │            │
│  ┌──────────┐  │     │   ┌─────────────┐   │            │
│  │GigE Cam  │──┘     │   │  elodinsink │   │            │
│  └──────────┘        │   │   (plugin)  │   │            │
│                      │   └─────┬───────┘   │            │
│                      └─────────┼───────────┘            │
└────────────────────────────────┼────────────────────────┘
                                 │ H.264 NAL Units
                                 ▼
                      ┌──────────────────┐
                      │   Elodin DB      │
                      │  (Stores video   │
                      │   as messages)   │
                      └─────────┬────────┘
                                │ Impeller2 Protocol
                                ▼
                  ┌─────────────────────────┐
                  │    Elodin Editor        │
                  │  ┌─────────────────┐    │
                  │  │  video-toolbox  │    │
                  │  │   (Decoding)    │    │
                  │  └────────┬────────┘    │
                  │           │ RGBA Frames │
                  │           ▼             │
                  │    ┌────────────┐       │
                  │    │  Display   │       │
                  │    └────────────┘       │
                  └─────────────────────────┘
```

#### Key Types

```rust
// From libs/video-toolbox/src/lib.rs

/// Decoded video frame
pub struct DecodedFrame {
    pub rgba: Vec<u8>,    // RGBA pixel data
    pub width: usize,
    pub height: usize,
}

/// NAL unit types
pub enum NalType {
    Sps = 7,    // Sequence Parameter Set
    Pps = 8,    // Picture Parameter Set
    Idr = 5,    // Instantaneous Decoder Refresh (keyframe)
    Slice = 1,  // Non-IDR slice
    Other,
}
```

#### Decoding Process

1. Parse incoming Annex-B data to find NAL units
2. Extract SPS and PPS parameter sets
3. Create VideoToolbox decompression session
4. Convert Annex-B to AVCC format (length-prefixed)
5. Decode frame and convert to RGBA
6. Scale frame to desired output width

---

### D. Impeller2 Protocol Extensions

**Location:** `libs/impeller2/src/types.rs`

PR #67 added the `MsgWithTimestamp` packet type to support timestamped messages.

#### Packet Types

```rust
#[repr(u8)]
pub enum PacketTy {
    Msg = 0,              // Standard message
    Table = 1,            // Component table
    TimeSeries = 2,       // Time series data
    MsgWithTimestamp = 3, // Message with embedded timestamp (NEW)
}
```

#### Packet Format

```
┌──────────────────────────────────────────────────────────────┐
│                    MsgWithTimestamp Packet                   │
├──────────┬───────────┬──────────┬───────────┬───────────────┤
│ Length   │ PacketTy  │ Msg ID   │ Req ID    │ Timestamp     │
│ (4 bytes)│ (1 byte)  │ (2 bytes)│ (1 byte)  │ (8 bytes)     │
├──────────┴───────────┴──────────┴───────────┴───────────────┤
│                       Payload (H.264 NAL data)              │
└─────────────────────────────────────────────────────────────┘
```

#### Creating Timestamped Packets

```rust
// Create a packet with timestamp
let packet = LenPacket::msg_with_timestamp(
    msg_id,           // 2-byte message identifier
    Timestamp::now(), // Microseconds since Unix epoch
    data.len(),       // Payload capacity
);
packet.extend_from_slice(data);  // Add H.264 NAL data
```

---

### E. Elodin DB Message Handling

**Location:** `libs/db/src/lib.rs`

Elodin DB stores video frames as messages and provides fixed-rate playback.

#### Message Storage

Video messages are stored in the `msgs/` directory:
```
db_path/
  msgs/
    {msg_id}/       # Directory for each message type
      data          # Message payloads
      timestamps    # Message timestamps
```

#### Fixed-Rate Streaming

The `FixedRateMsgStream` request enables synchronized video playback:

```rust
// From libs/impeller2/wkt/src/msgs.rs
pub struct FixedRateMsgStream {
    pub msg_id: PacketId,
    pub fixed_rate: FixedRateOp,
}

pub struct FixedRateBehavior {
    pub initial_timestamp: InitialTimestamp,
    pub timestep: u64,    // Nanoseconds between ticks
    pub frequency: u64,   // Ticks per second
}
```

The DB handler (`handle_fixed_rate_msg_stream`) delivers frames at the configured rate:

```rust
// Simplified from libs/db/src/lib.rs
loop {
    let current_timestamp = stream_state.current_timestamp();
    let (msg_timestamp, msg) = msg_log.get_nearest(current_timestamp)?;
    
    // Send frame with its original timestamp
    pkt.extend_from_slice(msg_timestamp.as_bytes());
    pkt.extend_from_slice(msg);
    tx.send(pkt).await?;
    
    // Wait for next tick
    stream_state.wait_for_tick(elapsed, current_timestamp).await;
}
```

---

### F. Elodin Editor Video Tile

**Location:** `libs/elodin-editor/src/ui/video_stream.rs`

The video stream tile displays decoded video frames synchronized with playback.

#### Key Components

```rust
// Video stream state
pub struct VideoStream {
    pub msg_id: [u8; 2],           // Message identifier
    pub current_frame: Option<Image>,
    pub frame_timestamp: Option<Timestamp>,
    pub texture_handle: Option<TextureHandle>,
    pub size: Vec2,
    pub frame_count: usize,
    pub state: StreamState,
}

// Decoder runs in separate thread
pub struct VideoDecoderHandle {
    tx: flume::Sender<(Vec<u8>, Timestamp)>,   // Send packets to decoder
    rx: flume::Receiver<(Image, Timestamp)>,   // Receive decoded frames
    width: Arc<AtomicUsize>,                   // Desired output width
    _handle: std::thread::JoinHandle<()>,
}
```

#### Platform-Specific Decoding

```rust
// macOS: Hardware-accelerated
#[cfg(target_os = "macos")]
fn decode_video(...) {
    let mut video_toolbox = video_toolbox::VideoToolboxDecoder::new(frame_width).unwrap();
    while let Ok((packet, timestamp)) = packet_rx.recv() {
        if let Ok(Some(frame)) = video_toolbox.decode(&packet, 0) {
            // Convert to Bevy Image and send
        }
    }
}

// Linux/Windows: Software decoding
#[cfg(not(target_os = "macos"))]
fn decode_video(...) {
    let mut decoder = openh264::decoder::Decoder::new().unwrap();
    while let Ok((packet, timestamp)) = packet_rx.recv() {
        if let Ok(Some(yuv)) = decoder.decode(&packet) {
            // Convert YUV to RGBA and scale
        }
    }
}
```

#### Stream Initialization

When the video tile starts, it sends a `FixedRateMsgStream` request:

```rust
commands.send_req_reply_raw(
    FixedRateMsgStream {
        msg_id: stream.msg_id,
        fixed_rate: FixedRateOp {
            stream_id: state.stream_id.0,
            behavior: Default::default(),
        },
    },
    move |packet, mut decoders| {
        if let OwnedPacket::Msg(msg_buf) = packet {
            decoder.process_frame(timestamp, &msg_buf.buf);
        }
    },
);
```

#### Loss of Signal Detection

The tile shows an overlay when frames are stale (>500ms behind current time):

```rust
if (frame_timestamp.0 - state.current_time.0.0).abs() > 500000 {
    ui.painter().rect_filled(max_rect, 0, Color32::BLACK.opacity(0.75));
    ui.label("Loss of Signal - Frame out of date. Waiting for new keyframe");
}
```

---

## Protocol Details

### Message Flow Sequence

```
  GStreamer                    Elodin DB                    Elodin Editor
      │                            │                              │
      │  TCP Connect (:2240)       │                              │
      │───────────────────────────►│                              │
      │                            │                              │
      │  ┌─────────────────────────┼──────────────────────────────┤
      │  │ Each H.264 Frame        │                              │
      │  │                         │                              │
      │  │  MsgWithTimestamp       │                              │
      │  │  (msg_id, ts, NAL)      │                              │
      │──┼────────────────────────►│                              │
      │  │                         │                              │
      │  │                         │ Store in MsgLog              │
      │  │                         │─────────┐                    │
      │  │                         │         │                    │
      │  │                         │◄────────┘                    │
      │  └─────────────────────────┼──────────────────────────────┤
      │                            │                              │
      │                            │   FixedRateMsgStream         │
      │                            │   (msg_id, behavior)         │
      │                            │◄─────────────────────────────│
      │                            │                              │
      │                            │  ┌───────────────────────────┤
      │                            │  │ Fixed Rate Playback       │
      │                            │  │                           │
      │                            │  │  MsgWithTimestamp         │
      │                            │  │  (timestamp, NAL_data)    │
      │                            │──┼──────────────────────────►│
      │                            │  │                           │
      │                            │  │           Decode H.264    │
      │                            │  │                     ─────►│
      │                            │  │           Display Frame   │
      │                            │  │                     ─────►│
      │                            │  └───────────────────────────┤
      │                            │                              │
      │                            │   SetStreamState             │
      │                            │   (pause/seek)               │
      │                            │◄─────────────────────────────│
      │                            │                              │
      │                            │ Update stream state          │
      │                            │─────────┐                    │
      │                            │         │                    │
      │                            │◄────────┘                    │
      │                            │                              │
```

### Key Message Types

| Message | ID | Direction | Purpose |
|---------|-----|-----------|---------|
| `FixedRateMsgStream` | auto | Editor→DB | Request video stream |
| `SetStreamState` | `[224, 2]` | Editor→DB | Control playback |
| `MsgWithTimestamp` | varies | GS→DB, DB→Editor | Video frame with timestamp |

---

## Local Setup Instructions

### Prerequisites

1. **Nix development shell** (includes GStreamer, FFMPEG, and all dependencies):
   ```bash
   cd /home/dan/dev/elodin
   nix develop
   ```

2. **Build the Elodin tools**:
   ```bash
   just install
   ```

### Step 1: Start Elodin DB

```bash
# From repository root, in nix shell
elodin db --path ./test-db
```

This starts Elodin DB listening on `127.0.0.1:2240`.

### Step 2: Build the GStreamer Plugin

```bash
cd fsw/gstreamer
cargo build --release
```

The plugin will be at `target/release/libgstelodin.so` (Linux) or `libgstelodin.dylib` (macOS).

### Step 3: Set GStreamer Plugin Path

```bash
export GST_PLUGIN_PATH=$PWD/target/release
```

### Step 4: Start a Test Video Stream

**Option A: Synthetic Test Pattern**
```bash
gst-launch-1.0 videotestsrc ! \
    x264enc ! video/x-h264,profile=baseline ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="test"
```

**Option B: Webcam (macOS)**
```bash
gst-launch-1.0 avfvideosrc ! \
    vtenc_h264_hw max-keyframe-interval=12 realtime=true ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

**Option C: Webcam (Linux)**
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! x264enc tune=zerolatency ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

### Step 5: Launch Elodin Editor

```bash
elodin editor --db 127.0.0.1:2240
```

### Step 6: Add Video Tile

1. Open the command palette (Cmd/Ctrl + K)
2. Search for "Add Video Stream"
3. Enter the message name (e.g., "test" or "webcam")
4. The video should appear in the new tile

---

## Example Pipelines

### Basic Test Pattern

```bash
gst-launch-1.0 videotestsrc ! \
    x264enc ! video/x-h264,profile=baseline ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="test"
```

### MP4 File Streaming

```bash
gst-launch-1.0 filesrc location=video.mp4 ! \
    qtdemux ! h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="file"
```

### Webcam on macOS

```bash
gst-launch-1.0 avfvideosrc ! \
    vtenc_h264_hw max-keyframe-interval=12 realtime=true ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

### Webcam on Linux

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! x264enc tune=zerolatency ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

### JPEG Webcam on Aleph (Jetson)

```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    jpegdec ! nvvidconv ! nvv4l2h264enc ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

### GigE Vision Camera on Aleph (via Aravis)

```bash
gst-launch-1.0 aravissrc ! bayer2rgb ! videoconvert ! \
    'video/x-raw,format=NV12,width=1280,height=720' ! \
    nvvidconv ! nvv4l2h264enc ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="genicam"
```

### Using video-streamer for File Playback

```bash
# Basic usage
video-streamer video.mp4 myvideo

# With options
video-streamer video.mp4 myvideo \
    --db-addr 192.168.1.100:2240 \
    --bitrate 2000 \
    --keyframe-interval 30 \
    --live
```

---

## Troubleshooting Guide

### "No SPS or PPS NAL units found"

**Cause:** The H.264 stream doesn't include parameter sets with keyframes.

**Solution:** Use `h264parse config-interval=-1` in your GStreamer pipeline. This ensures SPS and PPS are sent with every IDR frame.

```bash
# Correct
h264parse config-interval=-1 ! elodinsink ...

# Wrong (missing config-interval)
h264parse ! elodinsink ...
```

### Connection Refused / Connection Failed

**Cause:** Elodin DB is not running or listening on the wrong address.

**Solutions:**
1. Verify Elodin DB is running: `elodin db --path ./test-db`
2. Check the address matches: default is `127.0.0.1:2240`
3. For remote connections, ensure firewall allows TCP 2240

### Corrupted or Glitchy Video

**Cause:** Incomplete NAL units or network packet loss.

**Solutions:**
1. Increase keyframe frequency: `max-keyframe-interval=12`
2. Check network stability between source and DB
3. Ensure the encoder outputs complete access units

### High Latency

**Cause:** Large GOP (Group of Pictures) or buffering.

**Solutions:**
1. Reduce keyframe interval: `max-keyframe-interval=12` or lower
2. Use `tune=zerolatency` with x264
3. Enable `realtime=true` for hardware encoders
4. Check network latency between components

### "Loss of Signal" in Editor

**Cause:** Frame timestamps are more than 500ms behind current playback time.

**Solutions:**
1. Ensure the video source is running
2. Check for network connectivity issues
3. Verify timestamps are being generated correctly
4. Try seeking to the current time in the playback controls

### Video Tile Shows Nothing

**Cause:** Message name mismatch or no video data.

**Solutions:**
1. Verify the `msg-name` in GStreamer matches the tile configuration
2. Check that the GStreamer pipeline is running without errors
3. Verify data is being stored: check `test-db/msgs/` directory
4. Ensure the codec is H.264 (not H.265/HEVC or AV1)

### OpenH264 Decoder Issues (Linux/Windows)

**Cause:** Missing or incompatible OpenH264 library.

**Solutions:**
1. The nix shell includes OpenH264; ensure you're in `nix develop`
2. On non-nix systems, install OpenH264 from your package manager
3. Check for architecture mismatches (x86 vs ARM)

---

## File Reference

### Core Implementation Files

| File | Description |
|------|-------------|
| [`fsw/gstreamer/src/lib.rs`](../fsw/gstreamer/src/lib.rs) | GStreamer `elodinsink` plugin implementation |
| [`fsw/gstreamer/Cargo.toml`](../fsw/gstreamer/Cargo.toml) | Plugin dependencies and build configuration |
| [`fsw/gstreamer/README.md`](../fsw/gstreamer/README.md) | GStreamer plugin usage documentation |
| [`fsw/video-streamer/src/main.rs`](../fsw/video-streamer/src/main.rs) | FFMPEG video streamer implementation |
| [`fsw/video-streamer/README.md`](../fsw/video-streamer/README.md) | Video streamer usage documentation |

### Video Decoding

| File | Description |
|------|-------------|
| [`libs/video-toolbox/src/lib.rs`](../libs/video-toolbox/src/lib.rs) | Decoder abstraction and NAL parsing |
| [`libs/video-toolbox/src/platform.rs`](../libs/video-toolbox/src/platform.rs) | macOS VideoToolbox implementation |
| [`libs/video-toolbox/README.md`](../libs/video-toolbox/README.md) | Video toolbox documentation with architecture |

### Protocol and Database

| File | Description |
|------|-------------|
| [`libs/impeller2/src/types.rs`](../libs/impeller2/src/types.rs) | `PacketTy::MsgWithTimestamp` and `LenPacket` |
| [`libs/impeller2/wkt/src/msgs.rs`](../libs/impeller2/wkt/src/msgs.rs) | `FixedRateMsgStream`, `SetStreamState` messages |
| [`libs/db/src/lib.rs`](../libs/db/src/lib.rs) | `handle_fixed_rate_msg_stream` implementation |
| [`libs/db/src/msg_log.rs`](../libs/db/src/msg_log.rs) | Message log storage |

### Editor Integration

| File | Description |
|------|-------------|
| [`libs/elodin-editor/src/ui/video_stream.rs`](../libs/elodin-editor/src/ui/video_stream.rs) | Video tile widget implementation |
| [`libs/elodin-editor/src/ui/tiles.rs`](../libs/elodin-editor/src/ui/tiles.rs) | Tile system integration |
| [`libs/elodin-editor/Cargo.toml`](../libs/elodin-editor/Cargo.toml) | Editor dependencies (video-toolbox, openh264) |

### Build Configuration

| File | Description |
|------|-------------|
| [`nix/shell.nix`](../nix/shell.nix) | Development shell with GStreamer dependencies |
| [`flake.nix`](../flake.nix) | Nix flake configuration |

---

## Summary

The video streaming feature provides a complete pipeline for transmitting H.264 video from cameras and files through Elodin DB to the Elodin Editor. Key points:

1. **Use GStreamer** with the `elodinsink` plugin to capture and encode video
2. **Always include** `h264parse config-interval=-1` for proper parameter set handling
3. **Elodin DB** stores video as timestamped messages for synchronized playback
4. **The Editor** automatically selects the best decoder for your platform
5. **Fixed-rate streaming** ensures smooth, synchronized video playback

For customer deployments, the same architecture applies whether streaming from Aleph flight computers, USB cameras, or recorded video files. The key is ensuring proper H.264 encoding with frequent keyframes and parameter sets.
