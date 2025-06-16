#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# Parse arguments
USE_DEFMT=false
ELF=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --defmt)
            USE_DEFMT=true
            shift
            ;;
        -*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            if [[ -z "$ELF" ]]; then
                ELF="$1"
            else
                echo "Multiple ELF files specified"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$ELF" ]]; then
    echo "Usage: $0 [--defmt] <firmware.elf>"
    echo "  --defmt    Use defmt-print for structured logging (default: plain RTT)"
    exit 1
fi

ADDR=$(nm -n "$ELF" | grep _SEGGER_RTT | awk '{print $1}')
ADDR_HEX="0x$ADDR"

LOG_FILE=$(mktemp)
OPENOCD_PID=""

cleanup() {
    if [[ -n "$OPENOCD_PID" ]]; then
        kill -INT "$OPENOCD_PID" 2>/dev/null || true
        wait "$OPENOCD_PID" 2>/dev/null || true
    fi
    rm -f "$LOG_FILE"
}
trap cleanup EXIT

# Find the directory of this script:
SCRIPT_DIR=$(dirname "$0")

# Launch OpenOCD in background, tee only the initial logs (exclude RTT spam)
openocd -f "$SCRIPT_DIR/aleph.cfg" \
  -c "program $ELF reset" \
  -c "rtt setup $ADDR_HEX 4096 \"SEGGER RTT\"" \
  -c "rtt start" \
  -c "rtt server start 19021 0" \
  2>&1 | tee "$LOG_FILE" &
OPENOCD_PID=$!

# Wait for RTT TCP server to become available
for _ in {1..100}; do
    if nc -z localhost 19021 2>/dev/null; then
        break
    fi
    sleep 0.05
done

if ! nc -z localhost 19021 2>/dev/null; then
    echo "ERROR: RTT TCP server did not start in time." >&2
    exit 1
fi

# Use defmt or plain RTT based on flag
if [[ "$USE_DEFMT" == "true" ]]; then
    echo "Using defmt-print for structured logging..."
    defmt-print -e "$ELF" tcp
else
    echo "Using plain SEGGER RTT..."
    echo "RTT output (Ctrl+C to exit):"
    echo "----------------------------------------"
    # Stream raw RTT data from TCP server
    nc localhost 19021
fi
