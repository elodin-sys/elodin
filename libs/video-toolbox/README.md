# video-toolbox

A high-level Rust wrapper for Apple's VideoToolbox framework, providing hardware-accelerated H.264 video decoding for the Elodin ecosystem.

## Overview

video-toolbox enables real-time video streaming from flight hardware to the Elodin Editor, supporting critical use cases like:
- **Live camera feeds** from Aleph flight computers during testing
- **Sensor visualization** from USB cameras and GigE Vision cameras
- **Mission monitoring** with low-latency video from drones and spacecraft
- **Hardware-in-the-loop** testing with visual feedback

The library provides a unified API that uses:
- **macOS/iOS**: Apple's VideoToolbox for hardware-accelerated H.264 decoding
- **Linux/Windows**: Graceful fallback to OpenH264 software decoding (handled by the editor)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Aleph (Flight Computer)            │
│  ┌──────────┐                                       │
│  │USB Camera│──┐                                    │
│  └──────────┘  │     ┌─────────────────────┐        │
│                ├───► │    GStreamer        │        │
│  ┌──────────┐  │     │   ┌─────────────┐   │        │
│  │GigE Cam  │──┘     │   │  elodinsink │   │        │
│  └──────────┘        │   │   (plugin)  │   │        │
│                      │   └─────┬───────┘   │        │
│                      └─────────┼───────────┘        │
└────────────────────────────────┼────────────────────┘
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

## Features

### Core Functionality
- **H.264 Annex-B NAL decoding** - Processes raw H.264 streams from cameras
- **Automatic SPS/PPS parsing** - Extracts and manages H.264 parameters
- **Frame scaling** - Adaptive resolution based on display requirements
- **Zero-copy where possible** - Minimal memory allocations during decode
- **Hardware acceleration** - Uses VideoToolbox on Apple platforms

### Platform Support
| Platform | Backend | Hardware Acceleration |
|----------|---------|----------------------|
| macOS    | VideoToolbox | ✅ Yes |
| iOS      | VideoToolbox | ✅ Yes |
| Linux    | Fallback* | ❌ No |
| Windows  | Fallback* | ❌ No |

*On non-Apple platforms, the Elodin Editor uses OpenH264 directly

## Usage

### Basic Decoding

```rust
use video_toolbox::{VideoToolboxDecoder, DecodedFrame};
use std::sync::{Arc, atomic::AtomicUsize};

// Create decoder with desired output width
let desired_width = Arc::new(AtomicUsize::new(1280));
let mut decoder = VideoToolboxDecoder::new(desired_width)?;

// Decode H.264 NAL units (Annex-B format)
let h264_data = get_h264_stream(); // Your H.264 source
match decoder.decode(&h264_data, pts)? {
    Some(frame) => {
        // frame.rgba contains RGBA pixel data
        // frame.width and frame.height are the output dimensions
        display_frame(frame);
    }
    None => {
        // No frame available yet (buffering SPS/PPS)
    }
}
```

### Integration with GStreamer

The video-toolbox library works seamlessly with the elodinsink GStreamer plugin for streaming from Aleph:

```bash
# Stream from USB camera on Aleph to Elodin DB
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    nvvidconv ! nvv4l2h264enc ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="camera"
```

The Elodin Editor will automatically decode these streams using video-toolbox on macOS.

## Aleph Camera Streaming

### Supported Camera Types

1. **USB Webcams** - Standard UVC cameras
2. **GigE Vision Cameras** - Industrial cameras via Aravis
3. **MIPI CSI Cameras** - Direct sensor interfaces

### GStreamer Pipeline Examples

#### Generic USB Webcam
```bash
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    jpegdec ! nvvidconv ! nvv4l2h264enc ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

#### GigE Vision Camera (via Aravis)
```bash
gst-launch-1.0 aravissrc ! bayer2rgb ! videoconvert ! \
    'video/x-raw, format=NV12, width=1280, height=720' ! \
    nvvidconv ! nvv4l2h264enc ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="genicam"
```

### Elodinsink Plugin

The elodinsink GStreamer plugin sends H.264 NAL units to Elodin DB:

**Key Properties:**
- `db-address`: Elodin DB address (default: `127.0.0.1:2240`)
- `msg-name`: Message identifier for the video stream
- `msg-id`: Alternative to msg-name using numeric ID

**Requirements:**
- Sends Annex-B NAL units with timestamps
- Includes SPS/PPS with every IDR frame (`config-interval=-1`)
- Supports automatic reconnection on network issues

## Technical Details

### H.264 Stream Requirements

1. **Format**: Annex-B byte stream (NAL units with start codes)
2. **Parameters**: SPS and PPS must precede IDR frames
3. **Keyframes**: Regular keyframes improve stream resilience (every 12-30 frames recommended)
4. **Profile**: Baseline or Main profile for maximum compatibility

### VideoToolbox Specifics

The decoder handles the complexity of VideoToolbox's requirements:
- Converts Annex-B to AVCC format internally (length-prefixed)
- Manages CMFormatDescription creation from SPS/PPS
- Handles decompression session lifecycle
- Provides frame scaling via VTPixelTransferSession

### Memory Management

- **Input buffers**: Minimal copying, reuses internal buffers
- **Output frames**: RGBA format for direct rendering
- **Frame scaling**: Hardware-accelerated on Apple platforms

## Development

### Building

```bash
# Build the library
cargo build --package video-toolbox

# Run tests (macOS only)
cargo test --package video-toolbox
```

### Testing with Real Hardware

1. **Set up Aleph** with a USB camera
2. **Install elodinsink** on Aleph (included in Aleph NixOS image)
3. **Start Elodin DB** on your network
4. **Run GStreamer pipeline** on Aleph
5. **Open Elodin Editor** and add a video stream tile

### Example Test Pipeline

```bash
# On Aleph:
gst-launch-1.0 videotestsrc ! \
    x264enc ! video/x-h264, profile=baseline ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=192.168.1.100:2240 msg-name="test"

# On macOS development machine:
cargo run --package elodin-editor
# Add video tile with msg-name "test"
```

## Performance

### Benchmarks (M1 MacBook Pro)

| Resolution  | Decode Time  | CPU Usage |
|-------------|--------------|-----------|
| 720p 30fps  | ~1ms         | 2-3%      |
| 1080p 30fps | ~2ms         | 4-5%      |
| 4K 30fps    | ~5ms         | 8-10%     |

Hardware acceleration provides 10-20x performance improvement over software decoding.

## Troubleshooting

### Common Issues

1. **"No SPS or PPS NAL units found"**
   - Ensure your H.264 stream includes parameter sets
   - Use `h264parse config-interval=-1` in GStreamer

2. **Corrupted frames**
   - Check that NAL units are complete (not fragmented)
   - Verify network stability between Aleph and Elodin DB

3. **High latency**
   - Reduce keyframe interval in encoder
   - Check network bandwidth and latency
   - Consider reducing resolution

## History

video-toolbox was introduced in [PR #67](https://github.com/elodin-sys/elodin/pull/67) to add video streaming capabilities to the Elodin ecosystem. The initial implementation focused on:

- Enabling live camera feeds from Aleph flight computers
- Hardware-accelerated decoding for smooth playback
- Integration with GStreamer for flexible camera support
- Support for various camera types used in aerospace applications

The library filled a critical gap in monitoring and debugging flight hardware, allowing engineers to see real-time visual feedback during testing and operations.

## License

See the repository's LICENSE file for details.
