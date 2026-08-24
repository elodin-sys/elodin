#!/bin/bash
# Video streaming script for Elodin
#
# This script:
# 1. Waits for Elodin DB to be ready
# 2. Runs a GStreamer pipeline to stream H.264 video to Elodin DB
#
# Requires `nix develop` or `nix develop .#run` (elodinsink, x264enc, gst-launch).

set -e

if ! command -v gst-inspect-1.0 >/dev/null 2>&1 || ! gst-inspect-1.0 elodinsink >/dev/null 2>&1; then
    echo "ERROR: GStreamer element elodinsink is not available." >&2
    echo "Run from \`nix develop\` or \`nix develop .#run\` — those shells set GST_PLUGIN_PATH." >&2
    exit 1
fi

echo "Waiting for elodin-db on 127.0.0.1:2240..."

# Wait for Elodin DB to be ready (retry up to 30 times, 1 second apart)
MAX_RETRIES=30
RETRY_COUNT=0
while ! nc -z 127.0.0.1 2240 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Elodin DB not available after ${MAX_RETRIES} seconds"
        exit 1
    fi
    sleep 1
done

echo "Elodin DB is ready!"
echo "Starting video stream..."
echo "Message name: test-video"
echo ""

# Run GStreamer pipeline with test pattern
# - videotestsrc pattern=ball: animated bouncing ball test pattern
# - video/x-raw,framerate=60/1: explicit 60 fps, matching default editor playback rate
# - videoconvert: converts to format suitable for encoder
# - x264enc: H.264 encoder with low-latency settings
# - h264parse config-interval=-1: ensures SPS/PPS sent with every keyframe
# - elodinsink: custom plugin that sends NAL units to Elodin DB
gst-launch-1.0 \
    videotestsrc pattern=ball !  \
    clockoverlay time-format="%H:%M:%S" font-desc="Sans, 48" color=-16776961 ! \
    video/x-raw,width=640,height=512,framerate=60/1 ! \
    videoconvert ! \
    x264enc tune=zerolatency key-int-max=12 ! \
    video/x-h264,profile=baseline ! \
    h264parse config-interval=-1 ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="test-video"
