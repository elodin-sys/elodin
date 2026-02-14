#!/bin/bash
# OBS Studio SRT Receiver for Elodin
#
# This script:
# 1. Builds the elodinsink GStreamer plugin from source
# 2. Waits for Elodin DB to be ready
# 3. Listens for an SRT stream (e.g. from OBS Studio) and forwards
#    the H.264 video into Elodin DB via elodinsink
#
# The plugin is built automatically - no manual prerequisite steps required.
#
# Usage:
#   ./receive-obs-stream.sh [OPTIONS]
#
# Options:
#   --srt-port PORT       SRT listen port (default: 9000)
#   --db-address ADDR     Elodin DB address (default: 127.0.0.1:2240)
#   --msg-name NAME       Video message name (default: obs-camera)
#   --latency MS          SRT latency in milliseconds (default: 125)
#   --help                Show this help message

set -e

# =============================================================================
# Default Configuration
# =============================================================================

SRT_PORT=9000
DB_ADDRESS="127.0.0.1:2240"
MSG_NAME="obs-camera"
SRT_LATENCY=125

# =============================================================================
# Parse CLI Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --srt-port)
            SRT_PORT="$2"
            shift 2
            ;;
        --db-address)
            DB_ADDRESS="$2"
            shift 2
            ;;
        --msg-name)
            MSG_NAME="$2"
            shift 2
            ;;
        --latency)
            SRT_LATENCY="$2"
            shift 2
            ;;
        --help)
            echo "OBS Studio SRT Receiver for Elodin"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --srt-port PORT       SRT listen port (default: 9000)"
            echo "  --db-address ADDR     Elodin DB address (default: 127.0.0.1:2240)"
            echo "  --msg-name NAME       Video message name (default: obs-camera)"
            echo "  --latency MS          SRT latency in milliseconds (default: 125)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Build elodinsink Plugin
# =============================================================================

# Get the repository root (this script is in examples/video-stream/)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Build the GStreamer plugin (cargo will skip if already up-to-date)
echo "Building elodinsink GStreamer plugin..."
cargo build --release --manifest-path="$REPO_ROOT/fsw/gstreamer/Cargo.toml"

# Set plugin path to include our built plugin (prepend so local build overrides nix)
export GST_PLUGIN_PATH="$REPO_ROOT/target/release:${GST_PLUGIN_PATH}"

# =============================================================================
# Wait for Elodin DB
# =============================================================================

# Extract host and port from DB_ADDRESS for the connectivity check
DB_HOST="${DB_ADDRESS%:*}"
DB_PORT="${DB_ADDRESS##*:}"

echo "Waiting for elodin-db on ${DB_ADDRESS}..."

MAX_RETRIES=30
RETRY_COUNT=0
while ! nc -z "$DB_HOST" "$DB_PORT" 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Elodin DB not available after ${MAX_RETRIES} seconds"
        exit 1
    fi
    sleep 1
done

echo "Elodin DB is ready!"

# =============================================================================
# Launch SRT Receiver Pipeline
# =============================================================================

echo ""
echo "Starting OBS SRT receiver..."
echo "  SRT listener:  srt://0.0.0.0:${SRT_PORT} (mode=listener)"
echo "  SRT latency:   ${SRT_LATENCY} ms"
echo "  DB address:    ${DB_ADDRESS}"
echo "  Message name:  ${MSG_NAME}"
echo ""
echo "Configure OBS Studio to stream to:"
echo "  srt://<THIS_MACHINE_IP>:${SRT_PORT}?mode=caller"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Graceful shutdown on Ctrl+C
cleanup() {
    echo ""
    echo "Shutting down SRT receiver..."
    kill "$GST_PID" 2>/dev/null || true
    wait "$GST_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

# GStreamer SRT receiver pipeline:
# - srtsrc: listens for incoming SRT connections (OBS connects as caller)
# - tsdemux: demuxes MPEG-TS container that OBS wraps around the H.264 stream
# - h264parse config-interval=-1: ensures SPS/PPS sent with every keyframe
# - queue: decouples the live source from the sink for latency negotiation
# - elodinsink: sends H.264 NAL units to Elodin DB
gst-launch-1.0 \
    srtsrc uri="srt://0.0.0.0:${SRT_PORT}?mode=listener" latency="${SRT_LATENCY}" ! \
    tsdemux ! \
    h264parse config-interval=-1 ! \
    queue max-size-buffers=10 ! \
    elodinsink sync=false db-address="${DB_ADDRESS}" msg-name="${MSG_NAME}" &

GST_PID=$!
wait "$GST_PID"
