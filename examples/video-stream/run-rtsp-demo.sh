#!/usr/bin/env bash
# One-command manual test for native RTSP ingest (no physical camera needed).
#
#   mediamtx (RTSP server, self-publishes an H.264 test pattern via ffmpeg)
#         |  retina pull  (elodin-db --features rtsp, --rtsp-source)
#         v
#   elodin-db  -->  MsgLog "rtsp-camera"  -->  elodin editor (rtsp_stream panel)
#
# Run from the repo root, inside the nix shell:
#   nix develop --command bash examples/video-stream/run-rtsp-demo.sh
#
# Close the editor window to stop every service.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RTSP_URL="rtsp://127.0.0.1:8554/test"
DB_ADDR="127.0.0.1:2240"
DEMO_DIR="$REPO_ROOT/examples/video-stream/.rtsp-demo"
DB_DIR="$DEMO_DIR/db"
LOG_DIR="$DEMO_DIR/logs"
SCHEMATIC="$REPO_ROOT/examples/video-stream/rtsp-demo.kdl"
MEDIAMTX_CFG="$REPO_ROOT/examples/video-stream/mediamtx.yml"
mkdir -p "$LOG_DIR"

TARGET="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
ELODIN="$TARGET/debug/elodin"
ELODIN_DB="$TARGET/debug/elodin-db"

# Resolve mediamtx (PATH, else fetch from nixpkgs).
MEDIAMTX="$(command -v mediamtx || true)"
if [ -z "$MEDIAMTX" ]; then
  echo "[setup] fetching mediamtx from nixpkgs..."
  MEDIAMTX="$(nix build nixpkgs#mediamtx --no-link --print-out-paths)/bin/mediamtx"
fi

pids=()
cleanup() {
  echo
  echo "[cleanup] stopping services..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_port() { # host port name
  for _ in $(seq 1 150); do
    if (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null; then echo "[ok] $3 up on $1:$2"; return 0; fi
    sleep 0.2
  done
  echo "[fail] $3 never came up on $1:$2"; return 1
}

echo "=== 0/3  build (elodin + elodin-db with rtsp feature) ==="
cargo build -p elodin -p elodin-db --features elodin-db/rtsp

echo "=== 1/3  RTSP source (mediamtx self-publishes a test pattern) ==="
"$MEDIAMTX" "$MEDIAMTX_CFG" >"$LOG_DIR/mediamtx.log" 2>&1 &
pids+=($!)
wait_for_port 127.0.0.1 8554 "mediamtx (RTSP)"
sleep 3   # let mediamtx's runOnInit ffmpeg connect and start publishing

echo "=== 2/3  elodin-db pulling RTSP (feature rtsp) -> MsgLog 'rtsp-camera' ==="
rm -rf "$DB_DIR"   # absent path => DB::create makes a fresh db
"$ELODIN_DB" run \
  --rtsp-source "rtsp-camera=$RTSP_URL" \
  "$DB_ADDR" "$DB_DIR" >"$LOG_DIR/elodin-db.log" 2>&1 &
pids+=($!)
if ! wait_for_port 127.0.0.1 2240 "elodin-db"; then
  echo "----- elodin-db.log -----"; cat "$LOG_DIR/elodin-db.log" || true; exit 1
fi
sleep 2
echo "[db] $(grep -m1 'RTSP connected' "$LOG_DIR/elodin-db.log" 2>/dev/null || echo 'waiting for first keyframe...')"

echo "=== 3/3  editor (rtsp_stream panel) -> $DB_ADDR ==="
echo "Logs: $LOG_DIR   |   Close the editor window to stop all services."
"$ELODIN" editor "$DB_ADDR" --kdl "$SCHEMATIC"
