#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/elodin-db-grpc-full.XXXXXX")"
server_pid=""
sim_pid=""

cleanup() {
  local status=$?
  if [[ -n "${sim_pid}" ]] && kill -0 "${sim_pid}" 2>/dev/null; then
    kill -TERM -- "-${sim_pid}" 2>/dev/null || true
    wait "${sim_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  if [[ "${status}" -ne 0 ]]; then
    for log in "${work}/rc-jet.log" "${work}/server.log"; do
      if [[ -s "${log}" ]]; then
        printf '==== %s (tail) ====\n' "${log}" >&2
        tail -n 120 "${log}" >&2 || true
      fi
    done
  fi
  if [[ "${ELODIN_GRPC_DEMO_KEEP:-0}" != "1" ]]; then
    rm -rf "${work}"
  else
    printf 'kept gRPC demo artifacts: %s\n' "${work}"
  fi
}
trap cleanup EXIT

cd "${root}"
python3 -c 'import elodin, grpc, grpc_tools, pyarrow'
db_port=$((30000 + $$ % 10000))
grpc_port=$((db_port + 2))

generated="${work}/python"
mkdir -p "${generated}"
python3 -m grpc_tools.protoc \
  -I libs/db/proto \
  --python_out="${generated}" \
  --grpc_python_out="${generated}" \
  libs/db/proto/elodin/db/v1/common.proto \
  libs/db/proto/elodin/db/v1/ingest.proto \
  libs/db/proto/elodin/db/v1/query.proto \
  libs/db/proto/elodin/db/v1/stream.proto \
  libs/db/proto/elodin/db/v1/msg.proto \
  libs/db/proto/elodin/db/v1/admin.proto
touch "${generated}/elodin/__init__.py"
touch "${generated}/elodin/db/__init__.py"
touch "${generated}/elodin/db/v1/__init__.py"

wait_for_port() {
  local port="$1"
  for _ in $(seq 1 100); do
    if python3 - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.1)
    sys.exit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
    then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

if [[ -n "${ELODIN_GRPC_DEMO_DB:-}" ]]; then
  cp -a "${ELODIN_GRPC_DEMO_DB}" "${work}/db"
else
  sim_log="${work}/rc-jet.log"
  touch "${sim_log}"
  controller_bin="${ELODIN_RC_JET_CONTROLLER_BIN:-}"
  if [[ -z "${controller_bin}" ]] && command -v cargo >/dev/null; then
    cargo build --release -p rc-jet-controller
    controller_bin="${root}/target/release/rc-jet-controller"
  fi
  setsid env \
    ELODIN_DB_PATH="${work}/db" \
    ELODIN_MAX_TICKS="${ELODIN_GRPC_DEMO_TICKS:-3000}" \
    ELODIN_NON_INTERACTIVE=1 \
    ELODIN_RC_JET_CONTROLLER_HOST="127.0.0.1:${db_port}" \
    ELODIN_RC_JET_CONTROLLER_BIN="${controller_bin}" \
    elodin run examples/rc-jet/main.py --addr "127.0.0.1:${db_port}" \
    >"${sim_log}" 2>&1 &
  sim_pid=$!
  if ! wait_for_port "${grpc_port}"; then
    rg -n '^' "${sim_log}" >&2 || true
    printf 'embedded RC-jet gRPC server did not become ready\n' >&2
    exit 1
  fi
  PYTHONPATH="${generated}" python3 libs/db/examples/grpc_full_api_demo.py \
    --live --target "127.0.0.1:${grpc_port}"
  complete=0
  for _ in $(seq 1 720); do
    if rg -q 'elodin simulation summary' "${sim_log}"; then
      complete=1
      break
    fi
    if ! kill -0 "${sim_pid}" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if [[ "${complete}" -ne 1 ]]; then
    rg -n '^' "${sim_log}" >&2 || true
    printf 'RC-jet simulation did not complete\n' >&2
    exit 1
  fi
  if ! rg -q 'controller.*Connected to' "${sim_log}"; then
    rg -n '^' "${sim_log}" >&2 || true
    printf 'RC controller did not connect\n' >&2
    exit 1
  fi
  if ! rg -q 'Sensor cameras (spawned and primed|late-spawned)' "${sim_log}"; then
    rg -n '^' "${sim_log}" >&2 || true
    printf 'headless renderer did not initialize the sensor camera\n' >&2
    exit 1
  fi
  sleep 1
  kill -TERM -- "-${sim_pid}" 2>/dev/null || true
  for _ in $(seq 1 40); do
    kill -0 "${sim_pid}" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "${sim_pid}" 2>/dev/null; then
    kill -KILL -- "-${sim_pid}" 2>/dev/null || true
  fi
  wait "${sim_pid}" 2>/dev/null || true
  sim_pid=""
fi

if [[ -n "${ELODIN_DB_BIN:-}" ]]; then
  server_bin="${ELODIN_DB_BIN}"
elif command -v elodin-db >/dev/null; then
  server_bin="$(type -P elodin-db)"
elif command -v cargo >/dev/null; then
  cargo build -p elodin-db --bin elodin-db --features grpc
  server_bin="${root}/target/debug/elodin-db"
else
  printf 'elodin-db or cargo is required\n' >&2
  exit 1
fi
if ! "${server_bin}" run --help | rg -q -- '--grpc-auth-token'; then
  printf 'elodin-db was built without the grpc feature\n' >&2
  exit 1
fi

start_server() {
  local token="${1:-}"
  local auth=()
  if [[ -n "${token}" ]]; then
    auth=(--grpc-auth-token "${token}")
  fi
  (
    trap - INT
    exec env RUST_LOG=warn "${server_bin}" run "127.0.0.1:${db_port}" "${work}/db" \
      "${auth[@]}"
  ) >"${work}/server.log" 2>&1 &
  server_pid=$!
  wait_for_port "${grpc_port}" && return
  rg -n '^' "${work}/server.log" >&2 || true
  printf 'elodin-db did not become ready\n' >&2
  exit 1
}

stop_server() {
  kill -INT "${server_pid}"
  set +e
  wait "${server_pid}"
  local status=$?
  set -e
  server_pid=""
  if [[ "${status}" -ne 0 ]] && [[ "${status}" -ne 130 ]]; then
    exit "${status}"
  fi
}

start_server
PYTHONPATH="${generated}" python3 libs/db/examples/grpc_full_api_demo.py \
  --target "127.0.0.1:${grpc_port}"
stop_server

start_server "demo-token"
PYTHONPATH="${generated}" python3 libs/db/examples/grpc_full_api_demo.py \
  --target "127.0.0.1:${grpc_port}" --token "demo-token"
stop_server

gse_state="${work}/gse-state"
start_server
PYTHONPATH="${generated}" python3 libs/db/examples/grpc_gse_client.py \
  --phase write1 --target "127.0.0.1:${grpc_port}" --state-dir "${gse_state}"
stop_server

start_server
PYTHONPATH="${generated}" python3 libs/db/examples/grpc_gse_client.py \
  --phase write2 --target "127.0.0.1:${grpc_port}" --state-dir "${gse_state}"
PYTHONPATH="${generated}" python3 libs/db/examples/grpc_gse_client.py \
  --phase verify --target "127.0.0.1:${grpc_port}" --state-dir "${gse_state}"
stop_server
