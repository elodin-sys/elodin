#!/usr/bin/env bash

# Script to scan for Aleph devices on the network using mDNS

set -e

show_help() {
  echo "Usage: $0 [OPTIONS] [DURATION]"
  echo "  DURATION: Scan duration in seconds (default: 0.5)"
  echo ""
  echo "Options:"
  echo "  -f, --first     Exit after finding the first device"
  echo "  -h, --help      Show this help message"
  exit 0
}

# Parse arguments
FIRST_ONLY=false
SCAN_DURATION=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--first) FIRST_ONLY=true; shift ;;
    -h|--help) show_help ;;
    *)
      if [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        SCAN_DURATION="$1"
      else
        echo "Unknown option: $1"; show_help
      fi
      shift ;;
  esac
done

# Detect operating system
OS="$(uname -s)"

# Function to clean up hostname - extracts base hostname and handles null bytes
clean_hostname() {
  local raw_host="$1"
  # Remove null bytes first to avoid warnings
  raw_host=$(tr -d '\0' <<< "$raw_host")
  # Process escape sequences and extract base hostname
  printf "%b" "$raw_host" | sed -E 's/^(aleph-[0-9a-zA-Z]+).*/\1/'
}

# Scan using avahi-browse on Linux
scan_with_avahi() {
  if [ "$FIRST_ONLY" = true ]; then
    avahi-browse -t -a --resolve --parsable 2>/dev/null |
    while read -r line; do
      echo "$line" | grep -qi "aleph" || continue
      echo "$line" | grep -q "=" || continue
      hostname=$(echo "$line" | tr -d '\0' | awk -F';' '{print $4}' | grep "^aleph" | head -1)
      if [ -n "$hostname" ]; then
        echo "$(clean_hostname "$hostname").local"
        exit 0
      fi
    done
  else
    TMP_FILE=$(mktemp)
    trap 'rm -f "$TMP_FILE"' EXIT
    timeout "$SCAN_DURATION" avahi-browse -t -a --resolve --parsable 2>/dev/null > "$TMP_FILE" || true
    grep -i "aleph" "$TMP_FILE" | grep "=" | tr -d '\0' | awk -F';' '{print $4}' | grep "^aleph" | sort -u |
    while read -r host; do
      echo "$(clean_hostname "$host").local"
    done
  fi
}

# Scan using dns-sd on macOS
scan_with_dns_sd() {
  TMP_FILE=$(mktemp)
  trap 'rm -f "$TMP_FILE"' EXIT

  {
    dns-sd -B _workstation._tcp local > "$TMP_FILE" &
    DNS_PID=$!

    if [ "$FIRST_ONLY" = true ]; then
      # First-only mode: check frequently and exit as soon as we find something
      while [ "$(date +%s)" -lt "$END_TIME" ]; do
        if grep -q "aleph" "$TMP_FILE" 2>/dev/null; then
          hostname=$(grep "aleph" "$TMP_FILE" 2>/dev/null | awk '{print $7}' | grep "^aleph" | head -1)
          if [ -n "$hostname" ]; then
            echo "${hostname}.local"
            break
          fi
        fi
        sleep 0.1
      done
    else
      sleep "$SCAN_DURATION"
    fi

    kill $DNS_PID 2>/dev/null || true
    wait $DNS_PID 2>/dev/null || true

    # For regular mode, output all devices
    if [ "$FIRST_ONLY" = false ]; then
      grep -i "aleph" "$TMP_FILE" 2>/dev/null | awk '{print $7}' | grep "^aleph" | sort -u |
      while read -r host; do
        echo "${host}.local"
      done
    fi
  } 2>/dev/null
}

# Start time and end time calculation
START_TIME=$(date +%s)
INT_DURATION=$(printf "%.0f" "$(echo "$SCAN_DURATION" | awk '{print int($1+0.5)}')")
END_TIME=$((START_TIME + INT_DURATION))

# Run the appropriate scanner
if [ "$OS" = "Linux" ]; then
  if command -v avahi-browse >/dev/null 2>&1; then
    scan_with_avahi
  else
    >&2 echo "Avahi tools not found. Please install avahi-utils package."
    exit 1
  fi
elif [ "$OS" = "Darwin" ]; then
  scan_with_dns_sd
else
  >&2 echo "Unsupported operating system."
  exit 1
fi
