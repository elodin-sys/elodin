#!/usr/bin/env bash
# Upload an Elodin MCAP export + Foxglove layout to the Foxglove Data Platform
# and print a browser view URL.
#
# Usage:
#   scripts/foxglove-upload.sh \
#       --mcap path/to/db.mcap \
#       --layout path/to/db.foxglove-layout.json \
#       --device elodin-video-stream \
#       --key elodin-video-stream-v1 \
#       --layout-name "Elodin Video Stream"
#
# Environment:
#   FOXGLOVE_API_KEY   Bearer token (required). If unset, reads
#                      ai-context/foxglove-dev.md relative to the repo root.
#   FOXGLOVE_ORG_SLUG  Default: singularity-defense-corporation
#   FOXGLOVE_PROJECT   Default: prj_0eP167LtnwhrIxZD
#   FOXGLOVE_FOLDER    Layout folder. Default: Elodin
#
# Flow: POST /v1/data/upload → PUT file → poll import → POST /v1/layouts → print URL.
set -euo pipefail

API="${FOXGLOVE_API:-https://api.foxglove.dev}"
ORG_SLUG="${FOXGLOVE_ORG_SLUG:-singularity-defense-corporation}"
PROJECT="${FOXGLOVE_PROJECT:-prj_0eP167LtnwhrIxZD}"
FOLDER="${FOXGLOVE_FOLDER:-Elodin}"

MCAP=""
LAYOUT=""
DEVICE=""
KEY=""
LAYOUT_NAME=""
POLL_SECS="${POLL_SECS:-180}"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mcap) MCAP="$2"; shift 2 ;;
    --layout) LAYOUT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --layout-name) LAYOUT_NAME="$2"; shift 2 ;;
    --folder) FOLDER="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --org) ORG_SLUG="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

if [[ -z "$MCAP" || -z "$DEVICE" || -z "$KEY" ]]; then
  echo "error: --mcap, --device, and --key are required" >&2
  usage
fi
if [[ ! -f "$MCAP" ]]; then
  echo "error: mcap not found: $MCAP" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${FOXGLOVE_API_KEY:-}" ]]; then
  KEY_FILE="$REPO_ROOT/ai-context/foxglove-dev.md"
  if [[ -f "$KEY_FILE" ]]; then
    FOXGLOVE_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
  else
    echo "error: FOXGLOVE_API_KEY unset and $KEY_FILE missing" >&2
    exit 1
  fi
fi

AUTH=(-H "Authorization: Bearer ${FOXGLOVE_API_KEY}")
FILENAME="$(basename "$MCAP")"

echo "==> Requesting upload link for device=${DEVICE} key=${KEY}"
UPLOAD_RESP=$(curl -sf -X POST "${API}/v1/data/upload" \
  "${AUTH[@]}" \
  -H "content-type: application/json" \
  -d "$(jq -n \
        --arg fn "$FILENAME" \
        --arg device "$DEVICE" \
        --arg key "$KEY" \
        '{filename: $fn, deviceName: $device, key: $key}')")
LINK=$(echo "$UPLOAD_RESP" | jq -r '.link // empty')
REQUEST_ID=$(echo "$UPLOAD_RESP" | jq -r '.requestId // .id // empty')
if [[ -z "$LINK" ]]; then
  echo "error: upload endpoint did not return a link:" >&2
  echo "$UPLOAD_RESP" | jq . >&2 || echo "$UPLOAD_RESP" >&2
  exit 1
fi
echo "    requestId=${REQUEST_ID:-unknown}"

echo "==> Uploading ${FILENAME} ($(du -h "$MCAP" | cut -f1))"
HTTP=$(curl -s -o /tmp/foxglove-put-body.txt -w "%{http_code}" \
  -X PUT -H "content-type: application/octet-stream" \
  --data-binary @"$MCAP" "$LINK")
if [[ "$HTTP" != "200" && "$HTTP" != "201" && "$HTTP" != "204" ]]; then
  echo "error: PUT failed with HTTP $HTTP" >&2
  cat /tmp/foxglove-put-body.txt >&2 || true
  exit 1
fi
echo "    PUT ok (HTTP $HTTP)"

echo "==> Waiting for import (up to ${POLL_SECS}s)"
DEADLINE=$((SECONDS + POLL_SECS))
REC_ID=""
IMPORT_STATUS=""
while (( SECONDS < DEADLINE )); do
  # Prefer key lookup; fall back to pending-imports.
  REC_JSON=$(curl -sf "${AUTH[@]}" \
    "${API}/v1/recordings?deviceName=$(jq -nr --arg d "$DEVICE" '$d|@uri')&limit=50" \
    || true)
  REC_ID=$(echo "$REC_JSON" | jq -r --arg k "$KEY" \
    '.[] | select(.key == $k) | .id' | head -1)
  IMPORT_STATUS=$(echo "$REC_JSON" | jq -r --arg k "$KEY" \
    '.[] | select(.key == $k) | .importStatus' | head -1)
  if [[ -n "$REC_ID" && "$IMPORT_STATUS" == "complete" ]]; then
    break
  fi
  if [[ -n "$REC_ID" && "$IMPORT_STATUS" == "error" ]]; then
    echo "error: import failed for key=$KEY" >&2
    echo "$REC_JSON" | jq --arg k "$KEY" '.[] | select(.key == $k)' >&2
    exit 1
  fi
  if [[ -n "$REQUEST_ID" ]]; then
    PENDING=$(curl -sf "${AUTH[@]}" "${API}/v1/data/pending-imports" || true)
    PSTATUS=$(echo "$PENDING" | jq -r --arg id "$REQUEST_ID" \
      '.[] | select(.requestId == $id) | .importStatus' | head -1)
    PERR=$(echo "$PENDING" | jq -r --arg id "$REQUEST_ID" \
      '.[] | select(.requestId == $id) | .errorMessage // empty' | head -1)
    if [[ "$PSTATUS" == "error" ]]; then
      echo "error: pending import failed: ${PERR:-unknown}" >&2
      exit 1
    fi
    echo "    pending=${PSTATUS:-waiting} recording=${IMPORT_STATUS:-none}"
  else
    echo "    recording=${IMPORT_STATUS:-none}"
  fi
  sleep 5
done

if [[ -z "$REC_ID" || "$IMPORT_STATUS" != "complete" ]]; then
  echo "error: timed out waiting for import (status=${IMPORT_STATUS:-none})" >&2
  exit 1
fi
echo "    recording=${REC_ID} status=${IMPORT_STATUS}"

LAYOUT_ID=""
if [[ -n "$LAYOUT" ]]; then
  if [[ ! -f "$LAYOUT" ]]; then
    echo "error: layout not found: $LAYOUT" >&2
    exit 1
  fi
  if [[ -z "$LAYOUT_NAME" ]]; then
    LAYOUT_NAME="Elodin $(basename "$LAYOUT" .foxglove-layout.json)"
  fi
  echo "==> Creating layout '${LAYOUT_NAME}' in folder '${FOLDER}'"
  # Always create a fresh layout ID so Foxglove clients pick up the new config
  # (PATCH of an existing layout can leave a stale local working copy).
  LAYOUT_RESP=$(jq -n \
      --slurpfile data "$LAYOUT" \
      --arg name "$LAYOUT_NAME" \
      --arg folder "$FOLDER" \
      '{name: $name, permission: "ORG_WRITE", folderName: $folder, data: $data[0]}' \
    | curl -sf -X POST "${API}/v1/layouts" \
        "${AUTH[@]}" \
        -H "content-type: application/json" \
        --data-binary @-)
  LAYOUT_ID=$(echo "$LAYOUT_RESP" | jq -r '.id // empty')
  if [[ -z "$LAYOUT_ID" ]]; then
    echo "error: layout create failed:" >&2
    echo "$LAYOUT_RESP" | jq . >&2 || echo "$LAYOUT_RESP" >&2
    exit 1
  fi
  echo "    layout=${LAYOUT_ID}"
fi

VIEW_URL="https://app.foxglove.dev/${ORG_SLUG}/p/${PROJECT}/view?ds=foxglove-stream&ds.recordingId=${REC_ID}"
if [[ -n "$LAYOUT_ID" ]]; then
  VIEW_URL="${VIEW_URL}&layoutId=${LAYOUT_ID}"
fi

echo
echo "Recording: ${REC_ID}"
echo "Layout:    ${LAYOUT_ID:-none}"
echo "View URL:  ${VIEW_URL}"

# Machine-readable summary for callers.
cat <<EOF
---
recording_id=${REC_ID}
layout_id=${LAYOUT_ID}
device=${DEVICE}
key=${KEY}
view_url=${VIEW_URL}
EOF
