# GStreamer Elodin-DB Plugin

A GStreamer plugin that loads H.264 video streams, send Annex B NAL units to elodin-db as msgs
## Requirements

- Rust
- GStreamer development libraries (1.18 or newer)
- elodin-db instance running and accessible

### Installing Dependencies

#### Ubuntu/Debian
```
sudo apt-get update
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

#### macOS (using Homebrew)
```
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-libav
```

## Installation

`elodinsink` is a flake package. `nix develop` and `nix develop .#run` put it on
`GST_PLUGIN_PATH` automatically (same pattern as `x264enc` / `srtsrc`).

```
nix build .#elodinsink
```

The plugin lands at `result/lib/gstreamer-1.0/libgstelodin.{so,dylib}`.

Do not add cargo `target/release` to `GST_PLUGIN_PATH`. That directory also
contains `libelodin`, and `gst-plugin-scanner` will try to dlopen every dylib
there — on GStreamer 1.26 a failed scan can hide the rest of the plugin set.

## Usage

Run pipelines from `nix develop` or `nix develop .#run`. Aleph sets the same
path via the `elodinsink` NixOS module.

The plugin provides a new GStreamer element called `elodinsink`. elodin-db expects the h264 stream to be made up of annex-b NAL units. For best results it is important to send frequency keyframes -- 12 seems to provide good results.
Further you will want to ensure that SPS and PPS are sent with every IDR frame. You can do that with `h264 parse config-interva=-1`

### Basic Pipeline Example

```bash
gst-launch-1.0 videotestsrc ! x264enc ! video/x-h264, profile=baseline ! h264parse config-interval=-1 ! elodinsink db-address=127.0.0.1:2240 msg-name="h264stream"
```

### Streaming from a File

```bash

gst-launch-1.0 filesrc location=video.mp4 ! qtdemux !  ! h264parse config-interval=-1 ! elodinsink db-address=192.168.1.100:2240 msg-name="file"
```

### Streaming from a Camera

#### Stream from a Genicam on Aleph
This example uses aravis to load the video from a USB
```bash
gst-launch-1.0 -v aravissrc ! bayer2rgb  ! videoconvert ! 'video/x-raw, format=NV12, width=1280,height=720' ! nvvidconv  ! nvv4l2h264enc ! video/x-h264, profile=baseline !  h264parse config-interval=-1 ! elodinsink db-address=127.0.0.1:2240 msg-name="cam"
```

#### Streaming from a JPG Webcam on Aleph
```bash
gst-launch-1.0 -v v4l2src device=/dev/video0 ! jpegdec ! nvvidconv ! nvv4l2h264enc  ! h264parse config-interval=-1 ! elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```


#### Streaming from a Webcam on Aleph
```bash
gst-launch-1.0 -v v4l2src device=/dev/video0 ! nvv4l2decoder! nvvidconv ! nvv4l2h264enc  ! h264parse config-interval=-1 ! elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

#### Streaming from a Webcam on macOS
```bash
gst-launch-1.0 avfvideosrc ! vtenc_h264_hw max-keyframe-interval=12 realtime=true ! h264parse config-interval=-1 ! elodinsink db-address=127.0.0.1:2240 msg-name="webcam"
```

## Properties

The `elodinsink` element supports the following properties:

- `db-address` (string): The address of the elodin-db instance in the format `IP:PORT`. Default: `127.0.0.1:2240`
- `msg-name` (string): The message name to use (will be hashed to get message ID). Optional.
- `msg-id` (uint): The message ID to use (ignored if msg-name is set). Default: 0
