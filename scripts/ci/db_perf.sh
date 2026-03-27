#!/usr/bin/env bash
set -euo pipefail

tracy_out="$(pwd)/profile_output/tracy-db-perf"
mkdir -p "${tracy_out}"

# Binary is pre-built by the pipeline's pre_command step (default nix shell with cargo).
bench_bin="./target/release/elodin-db-bench"

if [[ ! -x "${bench_bin}" ]]; then
  echo "error: ${bench_bin} not found -- build with: cargo build --release -p elodin-db --bin elodin-db-bench --features tracy"
  exit 1
fi

run_bench() {
  local label="$1"
  shift
  echo "--- :racehorse: ${label}"

  trace_file="${tracy_out}/trace-db-${label}.tracy"

  # Start Tracy capture on port 8090 (elodin-db) before the bench
  tracy-capture -a 127.0.0.1 -p 8090 -o "${trace_file}" -s 30 >/dev/null 2>&1 &
  tracy_pid=$!
  sleep 0.5

  # Run the bench -- TRACY_PORT is read by TracyClient's static constructor.
  # Exit 141 (SIGPIPE) is expected when tracy-capture disconnects; treat as success.
  set +e
  TRACY_PORT=8090 "${bench_bin}" "$@"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]] && [[ "${rc}" -ne 141 ]]; then
    echo "error: bench exited with status ${rc}"
    exit "${rc}"
  fi

  # Wait for tracy-capture to finish
  set +e
  wait "${tracy_pid}" >/dev/null 2>&1
  set -e

  # Export to CSV and print zone stats
  csv_file="${trace_file%.tracy}.csv"
  if [[ -f "${trace_file}" ]]; then
    tracy-csvexport "${trace_file}" > "${csv_file}" 2>/dev/null || true

    if [[ -s "${csv_file}" ]]; then
      echo ""
      echo "Tracy zone statistics (${label}):"
      echo "──────────────────────────────────────────────────"
      head -1 "${csv_file}"
      # sort|head triggers SIGPIPE when head closes early; harmless, suppress it
      tail -n +2 "${csv_file}" | sort -t',' -k4 -rn | head -20 || true
      echo "──────────────────────────────────────────────────"
    fi
  fi

  echo ""
}

# Customer scenario defaults to per-component; also run in batch for comparison
run_bench "customer"       --scenario customer
run_bench "customer-batch" --scenario customer --mode batch
run_bench "high-freq"      --scenario high-freq
run_bench "high-fanout"    --scenario high-fanout
run_bench "stress"         --scenario stress
