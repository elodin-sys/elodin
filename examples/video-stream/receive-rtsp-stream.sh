#!/bin/bash
# RTSP Receiver for Elodin (interim GStreamer sidecar)
#
# Pulls an H.264 RTSP stream (IP camera, or OBS via the OBS-RTSPServer plugin)
# and forwards the video into the running example's Elodin DB via elodinsink.
#
# This is the GStreamer-sidecar path that feeds the embedded DB created by
# `elodin editor examples/video-stream/main.py`. The DB itself can also pull
# RTSP natively (no sidecar) when started as a standalone server built with the
# `rtsp` feature — see the README's "RTSP Camera" section.
#
# This script:
# 1. Builds the elodinsink GStreamer plugin from source
# 2. Waits for Elodin DB to be ready
# 3. Pulls the RTSP stream and pushes its H.264 NAL units into Elodin DB
#
# Usage:
#   ./receive-rtsp-stream.sh [OPTIONS]
#
# Options:
#   --rtsp-url URL        RTSP source URL (default: rtsp://127.0.0.1:8554/test)
#   --db-address ADDR     Elodin DB address (default: 127.0.0.1:2240)
#   --msg-name NAME       Video message name (default: rtsp-camera)
#   --help                Show this help message

set -e

# =============================================================================
# Default Configuration
# =============================================================================

RTSP_URL="rtsp://127.0.0.1:8554/test"
DB_ADDRESS="127.0.0.1:2240"
MSG_NAME="rtsp-camera"

# =============================================================================
# Parse CLI Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --rtsp-url)
            RTSP_URL="$2"
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
        --help)
            echo "RTSP Receiver for Elodin"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rtsp-url URL        RTSP source URL (default: rtsp://127.0.0.1:8554/test)"
            echo "  --db-address ADDR     Elodin DB address (default: 127.0.0.1:2240)"
            echo "  --msg-name NAME       Video message name (default: rtsp-camera)"
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
# Launch RTSP Receiver Pipeline
# =============================================================================

echo ""
echo "Starting RTSP receiver..."
echo "  RTSP source:   ${RTSP_URL}"
echo "  DB address:    ${DB_ADDRESS}"
echo "  Message name:  ${MSG_NAME}"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Graceful shutdown on Ctrl+C
SHUTDOWN=false
cleanup() {
    SHUTDOWN=true
    echo ""
    echo "Shutting down RTSP receiver..."
    kill "$GST_PID" 2>/dev/null || true
    wait "$GST_PID" 2>/dev/null || true
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# GStreamer RTSP receiver pipeline:
# - rtspsrc: RTSP client; protocols=tcp for interleaved (firewall-friendly) RTP
# - rtph264depay: depacketizes RTP into H.264 NAL units
# - h264parse config-interval=-1: repeats SPS/PPS ahead of every keyframe
# - queue: decouples the live source from the sink
# - elodinsink: sends H.264 NAL units to Elodin DB
#
# Wrapped in a restart loop so a dropped source (camera reboot, OBS stop) is
# retried automatically, mirroring the DB's native reconnect behavior.
while [ "$SHUTDOWN" = false ]; do
    gst-launch-1.0 \
        rtspsrc location="${RTSP_URL}" protocols=tcp latency=0 ! \
        rtph264depay ! \
        h264parse config-interval=-1 ! \
        queue max-size-buffers=10 ! \
        elodinsink sync=false db-address="${DB_ADDRESS}" msg-name="${MSG_NAME}" &

    GST_PID=$!
    wait "$GST_PID" || true

    if [ "$SHUTDOWN" = true ]; then
        break
    fi

    echo ""
    echo "RTSP source ended/unreachable. Retrying in 2 seconds..."
    sleep 2
done
