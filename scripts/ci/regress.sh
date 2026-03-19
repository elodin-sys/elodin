#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <example-name> <example-entrypoint>"
  echo
  echo "Example:"
  echo "  $0 ball examples/ball/main.py"
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

example_name="$1"
example_entrypoint="$2"
baseline_root="${BASELINE_ROOT:-scripts/ci/baseline}"
tolerances_file="${TOLERANCES_FILE:-${baseline_root}/tolerances.json}"
ticks="${REGRESSION_TICKS:-100}"
enable_profile="${REGRESSION_ENABLE_PROFILE:-1}"
file_prefix=""

if [[ ! -f "${example_entrypoint}" ]]; then
  echo "FAIL: example entrypoint not found: ${example_entrypoint}"
  exit 1
fi
if [[ ! -d "${baseline_root}" ]]; then
  echo "FAIL: baseline root not found: ${baseline_root}"
  exit 1
fi
if [[ ! -f "${tolerances_file}" ]]; then
  echo "FAIL: tolerance config not found: ${tolerances_file}"
  exit 1
fi

resolve_baseline_dir() {
  local root="$1"
  local example="$2"
  local dir
  local -a explicit_dirs=(
    "${root}/${example}"
    "${root}/${example}-csv"
    "${root}/${example}-csv-100"
    "${root}/${example}-iree-csv-100"
    "${root}/${example}-jax-csv-100"
    "${root}/${example}-xla-csv-100"
  )
  for dir in "${explicit_dirs[@]}"; do
    if [[ -d "${dir}" ]]; then
      printf "%s\n" "${dir}"
      return 0
    fi
  done
  for dir in "${root}/${example}"*; do
    if [[ -d "${dir}" ]]; then
      printf "%s\n" "${dir}"
      return 0
    fi
  done
  return 1
}

baseline_dir="$(resolve_baseline_dir "${baseline_root}" "${example_name}" || true)"
if [[ -z "${baseline_dir}" ]]; then
  if compgen -G "${baseline_root}/${example_name}"'*.csv' > /dev/null; then
    baseline_dir="${baseline_root}"
    file_prefix="${example_name}."
  else
    echo "FAIL: could not locate baseline directory for example '${example_name}' under ${baseline_root}"
    exit 1
  fi
fi

scratch_dir="$(mktemp -d -t "elodin-regress-${example_name}.XXXXXX")"
trap 'rm -rf "${scratch_dir}"' EXIT

db_path="${scratch_dir}/db"
export_dir="${scratch_dir}/csv"
mkdir -p "${db_path}" "${export_dir}"

bench_args=(bench --ticks "${ticks}")
if [[ "${enable_profile}" == "1" ]]; then
  bench_args+=(--profile)
fi

echo "==> [${example_name}] running benchmark (${example_entrypoint})"
run_log="${scratch_dir}/run.log"
ELODIN_DB_PATH="${db_path}" uv run "${example_entrypoint}" "${bench_args[@]}" 2>&1 | tee "${run_log}"

# In profile mode, the underlying bench path may ignore db_path and print the
# actual created path instead. If that happens, discover and use it.
if [[ ! -f "${db_path}/db_state" ]]; then
  discovered_db_path="$(
    python3 - "${run_log}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r'created db path="([^"]+)"', text)
print(matches[-1] if matches else "")
PY
  )"

  if [[ -n "${discovered_db_path}" ]] && [[ -f "${discovered_db_path}/db_state" ]]; then
    echo "==> [${example_name}] using discovered db path (${discovered_db_path})"
    db_path="${discovered_db_path}"
  fi
fi

if [[ ! -f "${db_path}/db_state" ]]; then
  echo "FAIL: benchmark completed but no db_state found at ${db_path}"
  exit 1
fi

echo "==> [${example_name}] exporting telemetry to CSV"
elodin-db export --format csv --flatten --output "${export_dir}" "${db_path}"

echo "==> [${example_name}] comparing against baseline (${baseline_dir})"
compare_args=(
  --example "${example_name}"
  --baseline-dir "${baseline_dir}"
  --candidate-dir "${export_dir}"
  --tolerances "${tolerances_file}"
)
if [[ -n "${file_prefix}" ]]; then
  compare_args+=(--file-prefix "${file_prefix}")
fi

uv run python3 scripts/ci/compare_baseline_csv.py "${compare_args[@]}"

