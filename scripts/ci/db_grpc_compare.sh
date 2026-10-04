#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

duration="${DB_BENCH_DURATION:-5}"
components="${DB_BENCH_COMPONENTS:-40}"
frequency="${DB_BENCH_FREQUENCY:-250}"
db_nice="${DB_BENCH_DB_NICE:-5}"
capture="${DB_BENCH_CAPTURE:-0}"
output="${DB_BENCH_OUTPUT:-}"
if [[ -z "${output}" ]]; then
  mkdir -p profile_output
  output="$(mktemp -d "${root}/profile_output/db-grpc-comparison.XXXXXX")"
else
  mkdir -p "${output}"
  output="$(cd "${output}" && pwd)"
fi

bench_bin="${ELODIN_DB_BENCH_BIN:-${root}/target/release/elodin-db-bench}"
server_bin="${ELODIN_DB_BIN:-${root}/target/release/elodin-db}"
if [[ "${DB_BENCH_SKIP_BUILD:-0}" != "1" ]] || [[ ! -x "${bench_bin}" ]] || [[ ! -x "${server_bin}" ]]; then
  cargo build --release -p elodin-db --bin elodin-db --bin elodin-db-bench --features grpc
fi

time_bin="$(type -P time || true)"
if [[ -z "${time_bin}" ]]; then
  printf 'GNU time is required; run this script from nix develop\n' >&2
  exit 1
fi

server_pgid=""
server_process=""
capture_pid=""
work=""
cleanup_processes() {
  if [[ -n "${server_process}" ]] && kill -0 "${server_process}" 2>/dev/null; then
    kill -TERM "${server_process}" 2>/dev/null || true
  elif [[ -n "${server_pgid}" ]] && kill -0 "${server_pgid}" 2>/dev/null; then
    kill -TERM -- "-${server_pgid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pgid}" ]]; then
    wait "${server_pgid}" 2>/dev/null || true
  fi
  server_pgid=""
  server_process=""
  if [[ -n "${capture_pid}" ]] && kill -0 "${capture_pid}" 2>/dev/null; then
    kill -INT "${capture_pid}" 2>/dev/null || true
    wait "${capture_pid}" 2>/dev/null || true
  fi
  capture_pid=""
  if [[ -n "${work}" ]]; then
    rm -rf "${work}"
  fi
  work=""
}
trap cleanup_processes EXIT

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
    if [[ -n "${server_pgid}" ]] && ! kill -0 "${server_pgid}" 2>/dev/null; then
      return 1
    fi
    sleep 0.1
  done
  return 1
}

run_mode() {
  local mode="$1"
  local offset="$2"
  local base=$((41000 + $$ % 1000 + offset))
  local db_addr="127.0.0.1:${base}"
  local grpc_addr="127.0.0.1:$((base + 2))"
  local result="${output}/${mode}.json"
  local server_time="${output}/${mode}.server-time.json"
  local client_time="${output}/${mode}.client-time.json"

  work="$(mktemp -d "${TMPDIR:-/tmp}/elodin-db-compare.XXXXXX")"
  if [[ "${capture}" == "1" ]] && [[ "${mode}" == "grpc-packed" ]]; then
    tcpdump -i lo -nn -s 0 -w "${output}/${mode}.pcap" "tcp port $((base + 2))" \
      >"${output}/${mode}.tcpdump.log" 2>&1 &
    capture_pid=$!
    sleep 0.2
    if ! kill -0 "${capture_pid}" 2>/dev/null; then
      wait "${capture_pid}" 2>/dev/null || true
      capture_pid=""
      printf 'warning: tcpdump capture unavailable\n' >&2
    fi
  fi

  setsid bash -c 'trap - INT; exec "$@"' db-time \
    "${time_bin}" -q -f '{"wall_seconds":%e,"user_seconds":%U,"sys_seconds":%S}' \
    -o "${server_time}" bash -c \
    'pid_file=$1; shift; printf "%s\n" "$$" >"${pid_file}"; exec "$@"' server-child \
    "${work}/server.pid" nice -n "${db_nice}" "${server_bin}" run "${db_addr}" "${work}/db" \
    >"${output}/${mode}.server.log" 2>&1 &
  server_pgid=$!
  if ! wait_for_port "${base}" || ! wait_for_port "$((base + 2))"; then
    printf 'server failed to start for %s\n' "${mode}" >&2
    rg -n '^' "${output}/${mode}.server.log" >&2 || true
    exit 1
  fi
  read -r server_process <"${work}/server.pid"
  if [[ -z "${server_process}" ]]; then
    printf 'failed to identify timed elodin-db process\n' >&2
    exit 1
  fi

  local args=(
    --components "${components}"
    --frequency "${frequency}"
    --duration "${duration}"
    --clients 1
    --mode "${mode}"
    --db-addr "${db_addr}"
    --json
  )
  if [[ "${mode}" == grpc-* ]]; then
    args+=(--grpc-addr "${grpc_addr}")
  fi
  RUST_LOG=warn "${time_bin}" -f '{"wall_seconds":%e,"user_seconds":%U,"sys_seconds":%S}' \
    -o "${client_time}" "${bench_bin}" "${args[@]}" >"${result}" \
    2>"${output}/${mode}.client.log"

  kill -TERM "${server_process}" 2>/dev/null || true
  wait "${server_pgid}" 2>/dev/null || true
  server_pgid=""
  server_process=""
  if [[ -n "${capture_pid}" ]]; then
    kill -INT "${capture_pid}" 2>/dev/null || true
    wait "${capture_pid}" 2>/dev/null || true
    capture_pid=""
    python3 scripts/ci/db_grpc_packet_shape.py "${output}/${mode}.pcap" \
      --port "$((base + 2))" --json >"${output}/${mode}.packet-shape.json"
  fi
  rm -rf "${work}"
  work=""
}

run_mode batch 0
run_mode grpc-packed 10

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'kernel=%s\n' "$(uname -srmo)"
  printf 'cpu=%s\n' "$(lscpu | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
  printf 'rustc=%s\n' "$(rustc --version)"
  printf 'components=%s\nfrequency_hz=%s\ntarget_component_writes_per_s=%s\n' \
    "${components}" "${frequency}" "$((components * frequency))"
  printf 'duration_s=%s\ndb_nice=%s\n' "${duration}" "${db_nice}"
} >"${output}/environment.txt"

jq -n \
  --slurpfile batch "${output}/batch.json" \
  --slurpfile grpc "${output}/grpc-packed.json" \
  --slurpfile batch_client "${output}/batch.client-time.json" \
  --slurpfile batch_server "${output}/batch.server-time.json" \
  --slurpfile grpc_client "${output}/grpc-packed.client-time.json" \
  --slurpfile grpc_server "${output}/grpc-packed.server-time.json" \
  '{
    impeller_batch: {
      benchmark: $batch[0],
      client_process: $batch_client[0],
      db_process: $batch_server[0]
    },
    grpc_packed: {
      benchmark: $grpc[0],
      client_process: $grpc_client[0],
      db_process: $grpc_server[0]
    }
  }' >"${output}/summary.json"

printf 'results: %s\n' "${output}"
jq . "${output}/summary.json"
