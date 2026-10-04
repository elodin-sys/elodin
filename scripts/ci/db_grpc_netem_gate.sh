#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root}"

if [[ "$(uname -s)" != "Linux" ]] || ! command -v tc >/dev/null || ! command -v tcpdump >/dev/null; then
  printf 'SKIP: Linux tc and tcpdump are required\n'
  exit 0
fi
if [[ "${ELODIN_RUN_NETEM:-0}" != "1" ]]; then
  printf 'SKIP: set ELODIN_RUN_NETEM=1 to authorize loopback qdisc changes\n'
  exit 0
fi
if [[ "${EUID}" -ne 0 ]]; then
  printf 'SKIP: run as root or in a CAP_NET_ADMIN/CAP_NET_RAW test environment\n'
  exit 0
fi

existing="$(tc qdisc show dev lo)"
if printf '%s\n' "${existing}" | rg -q 'qdisc (netem|fq_codel|tbf|htb|cake)'; then
  printf 'SKIP: loopback already has a managed qdisc:\n%s\n' "${existing}"
  exit 0
fi
if ! printf '%s\n' "${existing}" | rg -q '^qdisc noqueue .* root'; then
  printf 'SKIP: refusing to replace an unknown loopback qdisc:\n%s\n' "${existing}"
  exit 0
fi

installed=0
cleanup() {
  if [[ "${installed}" -eq 1 ]]; then
    tc qdisc del dev lo root 2>/dev/null || true
    installed=0
  fi
}
trap cleanup EXIT INT TERM

tc qdisc add dev lo root netem \
  delay "${ELODIN_NETEM_DELAY:-20ms}" "${ELODIN_NETEM_JITTER:-5ms}" \
  loss "${ELODIN_NETEM_LOSS:-0.1%}"
installed=1

mkdir -p profile_output
output="$(mktemp -d "${root}/profile_output/db-grpc-netem.XXXXXX")"
DB_BENCH_CAPTURE=1 \
DB_BENCH_DURATION="${DB_BENCH_DURATION:-3}" \
DB_BENCH_OUTPUT="${output}" \
scripts/ci/db_grpc_compare.sh

cleanup
if tc qdisc show dev lo | rg -q 'qdisc netem'; then
  printf 'error: loopback netem qdisc remained after cleanup\n' >&2
  exit 1
fi
printf 'netem gate results: %s\n' "${output}"
