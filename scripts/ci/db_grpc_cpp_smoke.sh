#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/elodin-db-grpc-cpp.XXXXXX")"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -rf "${work}"
}
trap cleanup EXIT

cd "${root}"
cmake -S libs/db/proto -B "${work}/proto" -DCMAKE_INSTALL_PREFIX="${work}/install"
cmake --build "${work}/proto" --parallel
cmake --install "${work}/proto"
cmake -S libs/db/examples -B "${work}/examples" -DCMAKE_PREFIX_PATH="${work}/install"
cmake --build "${work}/examples" --target elodin-db-grpc-client-batched --parallel

if [[ -z "${ELODIN_DB_BIN:-}" ]]; then
  cargo build -p elodin-db --bin elodin-db --features grpc
  server_bin="${root}/target/debug/elodin-db"
else
  server_bin="${ELODIN_DB_BIN}"
fi
if [[ ! -x "${server_bin}" ]]; then
  printf 'elodin-db binary is not executable: %s\n' "${server_bin}" >&2
  exit 1
fi

db_port=$((30000 + $$ % 10000))
grpc_port=$((db_port + 2))
db_dir="${work}/db"
(
  trap - INT
  exec env RUST_LOG=warn "${server_bin}" run "127.0.0.1:${db_port}" "${db_dir}"
) >"${work}/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 100); do
  if python3 - "${grpc_port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.1)
    sys.exit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
  then
    ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

if [[ "${ready}" -ne 1 ]]; then
  printf 'elodin-db did not become ready\n' >&2
  rg -n '^' "${work}/server.log" >&2 || true
  exit 1
fi

ticks="${ELODIN_GRPC_CPP_TICKS:-8}"
"${work}/examples/elodin-db-grpc-client-batched" \
  --address "127.0.0.1:${grpc_port}" \
  --ticks "${ticks}" \
  --frequency 200

kill -INT "${server_pid}"
set +e
wait "${server_pid}"
server_rc=$?
set -e
server_pid=""
if [[ "${server_rc}" -ne 0 ]] && [[ "${server_rc}" -ne 130 ]]; then
  printf 'elodin-db exited with status %s\n' "${server_rc}" >&2
  exit "${server_rc}"
fi

components="$("${server_bin}" list-components -l "${db_dir}")"
line="$(printf '%s\n' "${components}" | rg '^grpc_cpp\.reference\.signal_0[[:space:]]')"
if [[ "${line##* }" != "${ticks}" ]]; then
  printf 'expected %s rows for signal_0, got:\n%s\n' "${ticks}" "${components}" >&2
  exit 1
fi
printf 'verified %s rows in each reference component\n' "${ticks}"
