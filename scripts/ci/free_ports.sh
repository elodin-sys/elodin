# shellcheck shell=bash
# Sourced by CI scripts: CI agents share a host, so fixed ports collide across jobs.

port_listening() {
  python3 - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(0.1)
    sys.exit(sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) != 0)
PY
}

# Sets db_port and grpc_port to a free elodin-db triple (db, assets, gRPC).
pick_ports() {
  local base
  base=$((20000 + $$ % 10000))
  (( base > 29997 )) && base=20000
  for _ in $(seq 1 64); do
    if ! port_listening "${base}" && ! port_listening "$((base + 1))" && ! port_listening "$((base + 2))"; then
      db_port="${base}"
      grpc_port="$((base + 2))"
      return 0
    fi
    base=$((base + 3))
    if (( base > 29997 )); then
      base=20000
    fi
  done
  printf 'no free elodin-db port triple in 20000-29999\n' >&2
  exit 1
}
