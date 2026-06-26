#!/bin/bash
# RTSP Receiver for Elodin
#
# Pulls an H.264 RTSP stream (an IP camera, or OBS with the OBS-RTSPServer
# plugin) and streams it into Elodin DB via the `rtsp-streamer` producer.
# Elodin DB stays passive — this producer connects to it, exactly like the
# OBS/SRT and GStreamer paths.
#
# Set the source URL via the RTSP_URL environment variable; if it is unset this
# script does nothing (so the example still runs without an RTSP source).
#
# Usage:
#   RTSP_URL="rtsp://user:pass@camera/Streaming/Channels/101" ./receive-rtsp-stream.sh
#
# Options (env vars):
#   RTSP_URL      RTSP source URL to pull from (required to do anything)
#   DB_ADDRESS    Elodin DB address (default: 127.0.0.1:2240)
#   MSG_NAME      Video message name (default: rtsp-camera)

set -e

DB_ADDRESS="${DB_ADDRESS:-127.0.0.1:2240}"
MSG_NAME="${MSG_NAME:-rtsp-camera}"

if [ -z "${RTSP_URL:-}" ]; then
    echo "RTSP_URL not set — skipping RTSP receiver."
    echo "To enable: RTSP_URL=\"rtsp://<host>/<path>\" (re)run the example."
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Building rtsp-streamer..."
cargo build --release --manifest-path="$REPO_ROOT/fsw/rtsp-streamer/Cargo.toml"
RTSP_STREAMER="$REPO_ROOT/target/release/rtsp-streamer"

# Wait for Elodin DB to be ready.
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

# Strip any user:pass@ userinfo so credentials never reach the terminal/CI logs
# (the rtsp-streamer binary redacts URLs in its own traces too).
REDACTED_URL="$(printf '%s' "$RTSP_URL" | sed -E 's#://[^/@]*@#://#')"

echo ""
echo "Starting RTSP receiver..."
echo "  RTSP source:  ${REDACTED_URL}"
echo "  DB address:   ${DB_ADDRESS}"
echo "  Message name: ${MSG_NAME}"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# rtsp-streamer auto-reconnects internally, so a single invocation suffices.
exec "$RTSP_STREAMER" "$RTSP_URL" "$MSG_NAME" --db-addr "$DB_ADDRESS"
