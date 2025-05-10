# Video Streamer for Elodin

Video Streamer is a utility that loads video files from disk, re-encodes them to AV1, and sends the resulting Open Bitstream Units (OBUs) to an elodin-db instance as messages.

## Features

- Loads video files using FFmpeg
- Re-encodes video to AV1 format
- Streams OBUs to elodin-db using the impeller2_stellar client
- Configurable message IDs, bitrate, keyframe intervals, and encoding speed

## Requirements

- Rust (latest stable version recommended)
- FFmpeg libraries (libavcodec, libavformat, libavutil, etc.)
- AV1 encoder (libaom-av1 or libsvtav1)

### Installing FFmpeg Dependencies

#### Ubuntu/Debian
```
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libaom-dev libsvt-av1-dev
```

#### macOS (using Homebrew)
```
brew install ffmpeg aom svt-av1
```

## Installation

```
cargo build --release
```

The binary will be available at `target/release/video-streamer`.

## Usage

```
video-streamer --input <VIDEO_FILE> --message-id <MESSAGE_ID> [OPTIONS]
```

### Options

- `-i, --input <PATH>`: Path to the input video file (required)
- `-m, --message-id <ID>`: Message ID to use for the OBUs in elodin-db (required)
- `-d, --db-address <IP:PORT>`: Elodin DB address (default: 127.0.0.1:8080)
- `-b, --bitrate <KBPS>`: Output bitrate in kbps (default: 1000)
- `-k, --keyframe-interval <FRAMES>`: Keyframe interval in frames (default: 60)
- `-s, --speed <1-8>`: Speed preset (1 is highest quality, 8 is fastest) (default: 6)
- `-h, --help`: Print help information
- `-V, --version`: Print version information

### Examples

Basic usage:
```
video-streamer --input video.mp4 --message-id 12345
```

Advanced usage:
```
video-streamer --input video.mp4 --message-id 12345 --db-address 192.168.1.100:9090 --bitrate 2000 --keyframe-interval 30 --speed 4
```

## How It Works

1. The application connects to the specified elodin-db instance
2. It opens the input video file using FFmpeg
3. The video is decoded and then re-encoded to AV1 format with the specified parameters
4. Each encoded packet (OBU) is sent to elodin-db as a message with the specified message ID
5. Progress is logged during the encoding and streaming process

## Troubleshooting

- If you encounter "encoder not found" errors, ensure you have either libaom-av1 or libsvtav1 installed
- For connection issues, verify that elodin-db is running and accessible at the specified address
- Use the RUST_LOG environment variable to control logging level (e.g., `RUST_LOG=info`)

## License

See the LICENSE file in the project repository.