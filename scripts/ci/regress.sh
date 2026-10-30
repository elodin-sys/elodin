#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:"
  echo "  $0 [--update] <example-name> <example-entrypoint>"
  echo "  $0 --all [--update]"
  echo
  echo "Examples:"
  echo "  $0 ball examples/ball/main.py"
  echo "  $0 --update ball examples/ball/main.py"
  echo "  $0 --all"
  echo "  $0 --all --update"
}

array_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

normalize_example_name() {
  local name="$1"
  local suffix
  local -a suffixes=(
    "-jax-csv-100"
    "-xla-csv-100"
    "-csv-100"
    "-csv"
  )

  for suffix in "${suffixes[@]}"; do
    if [[ "${name}" == *"${suffix}" ]]; then
      printf "%s\n" "${name%"${suffix}"}"
      return 0
    fi
  done

  printf "%s\n" "${name}"
}

resolve_baseline_dir() {
  local root="$1"
  local example="$2"
  local dir
  local -a explicit_dirs=(
    "${root}/${example}"
    "${root}/${example}-csv"
    "${root}/${example}-csv-100"
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

discover_all_examples() {
  local baseline_path
  local baseline_name
  local example_name
  local -a discovered=()

  shopt -s nullglob
  for baseline_path in "${baseline_root}"/*/; do
    baseline_name="$(basename "${baseline_path}")"
    example_name="$(normalize_example_name "${baseline_name}")"

    if [[ -n "${example_name}" ]] && ! array_contains "${example_name}" "${discovered[@]}"; then
      discovered+=("${example_name}")
    fi
  done
  shopt -u nullglob

  if [[ "${#discovered[@]}" -eq 0 ]]; then
    echo "FAIL: could not locate any baseline example directories under ${baseline_root}"
    return 1
  fi

  printf "%s\n" "${discovered[@]}"
}

run_example() (
  local example_name="$1"
  local example_entrypoint="$2"
  local baseline_dir=""
  local file_prefix=""
  local scratch_dir=""
  local db_path=""
  local export_dir=""
  local run_log=""
  local discovered_db_path=""
  local csv_status=0
  local perf_status=0
  local i
  local -a bench_args=(bench --ticks "${ticks}" --profile --detail)
  local -a compare_args=()
  local -a candidate_metrics=()
  local -a base_metrics=()

  if [[ ! -f "${example_entrypoint}" ]]; then
    echo "FAIL: example entrypoint not found: ${example_entrypoint}"
    return 1
  fi

  baseline_dir="$(resolve_baseline_dir "${baseline_root}" "${example_name}" || true)"
  if [[ -z "${baseline_dir}" ]]; then
    if compgen -G "${baseline_root}/${example_name}"'*.csv' > /dev/null; then
      baseline_dir="${baseline_root}"
      file_prefix="${example_name}."
    elif [[ "${update_baseline}" == "1" ]]; then
      baseline_dir="${baseline_root}/${example_name}"
    else
      echo "FAIL: could not locate baseline directory for example '${example_name}' under ${baseline_root}"
      return 1
    fi
  fi

  scratch_dir="$(mktemp -d -t "elodin-regress-${example_name}.XXXXXX")"
  trap 'rm -rf "${scratch_dir}"' EXIT

  db_path="${scratch_dir}/db"
  export_dir="${scratch_dir}/csv"
  mkdir -p "${db_path}" "${export_dir}"

  extract_metrics() {
    "${python_bin}" scripts/ci/extract_profile_metrics.py \
      --run-log "$1.log" \
      --ticks "${ticks}" \
      --output "$1.json"
  }

  # Usage: bench_run <scratch path prefix> [VAR=value...]
  bench_run() {
    local prefix="$1"
    shift
    mkdir -p "${prefix}.db"
    env ELODIN_DB_PATH="${prefix}.db" "$@" "${python_bin}" "${example_entrypoint}" "${bench_args[@]}" \
      > "${prefix}.log" 2>&1 || tail -n 20 "${prefix}.log"
    extract_metrics "${prefix}"
  }

  echo "==> [${example_name}] running benchmark (${example_entrypoint})"
  run_log="${scratch_dir}/candidate-1.log"
  ELODIN_DB_PATH="${db_path}" "${python_bin}" "${example_entrypoint}" "${bench_args[@]}" 2>&1 | tee "${run_log}"
  extract_metrics "${scratch_dir}/candidate-1"
  candidate_metrics+=("${scratch_dir}/candidate-1.json")

  # If the runtime reports a different db path in logs, discover and use it.
  if [[ ! -f "${db_path}/db_state" ]]; then
    discovered_db_path="$(
      "${python_bin}" - "${run_log}" <<'PY'
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
    return 1
  fi

  echo "==> [${example_name}] exporting telemetry to CSV"
  elodin-db export --format csv --flatten --include-private --output "${export_dir}" "${db_path}"

  if [[ "${update_baseline}" == "1" ]]; then
    if [[ "${baseline_dir}" == "${baseline_root}" ]] && [[ -n "${file_prefix}" ]]; then
      baseline_dir="${baseline_root}/${example_name}"
      file_prefix=""
    fi

    echo "==> [${example_name}] updating baseline (${baseline_dir})"
    rm -rf "${baseline_dir}"
    mkdir -p "${baseline_dir}"
    cp -r "${export_dir}/." "${baseline_dir}/"
    "${python_bin}" scripts/ci/windows_paths.py sanitize-dir "${baseline_dir}"
    return 0
  fi

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

  if "${python_bin}" scripts/ci/compare_baseline_csv.py "${compare_args[@]}"; then
    :
  else
    csv_status=$?
  fi

  if [[ -n "${base_site}" ]]; then
    echo "==> [${example_name}] perf A/B: ${runs} interleaved run(s) each of candidate and base"
    # Interleaved so slow phases of the shared host hit both sides alike.
    for ((i = 1; i <= runs; i++)); do
      if [[ "${i}" -gt 1 ]]; then
        bench_run "${scratch_dir}/candidate-${i}"
        candidate_metrics+=("${scratch_dir}/candidate-${i}.json")
      fi
      bench_run "${scratch_dir}/base-${i}" PYTHONPATH="${base_site}${PYTHONPATH:+:${PYTHONPATH}}"
      base_metrics+=("${scratch_dir}/base-${i}.json")
    done

    if "${python_bin}" scripts/ci/compare_profile_metrics.py \
      --example "${example_name}" \
      --baseline "${base_metrics[@]}" \
      --candidate "${candidate_metrics[@]}" \
      --tolerances "${tolerances_file}"; then
      :
    else
      perf_status=$?
    fi
  fi

  if [[ "${csv_status}" -ne 0 || "${perf_status}" -ne 0 ]]; then
    return 1
  fi
)

run_all_examples() {
  local discovered_output=""
  local example_name
  local example_entrypoint
  local failures=0
  local joined_examples=""
  local -a examples=()
  local -a failed_examples=()
  local -a rtf_lines=()
  local run_output=""

  if ! discovered_output="$(discover_all_examples)"; then
    return 1
  fi

  while IFS= read -r example_name; do
    if [[ -n "${example_name}" ]]; then
      examples+=("${example_name}")
    fi
  done <<< "${discovered_output}"

  local IFS=", "
  joined_examples="${examples[*]}"
  if [[ "${update_baseline}" == "1" ]]; then
    echo "==> refreshing ${#examples[@]} baselined regression example(s): ${joined_examples}"
  else
    echo "==> checking ${#examples[@]} baselined regression example(s): ${joined_examples}"
  fi

  for example_name in "${examples[@]}"; do
    example_entrypoint="examples/${example_name}/main.py"
    run_output="$(run_example "${example_name}" "${example_entrypoint}" 2>&1)" || true
    echo "${run_output}" | grep -v "^RTF_DELTA: "

    if echo "${run_output}" | grep -q "^FAIL:"; then
      failures=$((failures + 1))
      failed_examples+=("${example_name}")
    fi

    local rtf_line
    rtf_line="$(echo "${run_output}" | grep "^RTF_DELTA: " | head -1)" || true
    if [[ -n "${rtf_line}" ]]; then
      rtf_lines+=("${rtf_line}")
    fi
  done

  if [[ "${failures}" -ne 0 ]]; then
    echo
    echo "FAIL: ${failures}/${#examples[@]} regression example(s) failed: ${failed_examples[*]}"
    _print_rtf_summary "${rtf_lines[@]}"
    return 1
  fi

  echo
  if [[ "${update_baseline}" == "1" ]]; then
    echo "PASS: refreshed ${#examples[@]} regression baseline(s)"
  else
    echo "PASS: checked ${#examples[@]} regression example(s)"
  fi
  _print_rtf_summary "${rtf_lines[@]}"
}

_print_rtf_summary() {
  local -a lines=("$@")
  if [[ "${#lines[@]}" -eq 0 ]]; then
    return
  fi
  echo
  echo "=== Performance Summary (real_time_factor_median vs base ${base_rev:0:7}) ==="
  local line name baseline candidate pct sign
  for line in "${lines[@]}"; do
    # Format: RTF_DELTA: <name> <baseline> <candidate> <pct>
    read -r _ name baseline candidate pct <<< "${line}"
    if [[ -z "${name}" ]]; then
      continue
    fi
    sign=""
    if [[ "${pct}" != -* ]]; then
      sign="+"
    fi
    printf "  %-14s %6.1fx  (base %6.1fx, %s%s%%)\n" \
      "${name}:" "${candidate}" "${baseline}" "${sign}" "${pct}"
  done
}

update_baseline=0
run_all=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      run_all=1
      shift
      ;;
    --update)
      update_baseline=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "FAIL: unknown option: $1"
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

baseline_root="${BASELINE_ROOT:-scripts/ci/baseline}"
tolerances_file="${TOLERANCES_FILE:-${baseline_root}/tolerances.json}"
ticks="${REGRESSION_TICKS:-100}"
runs="${REGRESSION_RUNS:-3}"
python_bin="${PYTHON:-python3}"

if [[ ! -d "${baseline_root}" ]]; then
  echo "FAIL: baseline root not found: ${baseline_root}"
  exit 1
fi
if [[ ! -f "${tolerances_file}" ]]; then
  echo "FAIL: tolerance config not found: ${tolerances_file}"
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "FAIL: python interpreter not found: ${python_bin}"
  exit 1
fi

resolve_base_rev() {
  local base
  if [[ -n "${REGRESSION_BASE:-}" ]]; then
    git rev-parse --verify "${REGRESSION_BASE}^{commit}"
    return
  fi
  base="$(git merge-base HEAD origin/main)" || return 1
  if [[ "${base}" == "$(git rev-parse HEAD)" ]]; then
    git rev-parse HEAD^
  else
    printf "%s\n" "${base}"
  fi
}

base_rev=""
base_site=""
# Cache-only: the base commit's own CI run built and pushed its wheel; never rebuild it here.
if [[ "${update_baseline}" != "1" ]]; then
  if ! base_rev="$(resolve_base_rev)"; then
    echo "WARN: could not resolve a base commit; skipping perf A/B"
  elif base_py="$(nix build --no-link --print-out-paths --max-jobs 0 --builders '' \
    "git+file://${PWD}?rev=${base_rev}&allRefs=1#elodin-py" 2>/dev/null)"; then
    base_site="$(echo "${base_py}"/lib/python3.*/site-packages)"
    echo "==> perf A/B against base ${base_rev:0:7} (${base_py})"
  else
    echo "WARN: base ${base_rev:0:7} elodin-py not in cache; skipping perf A/B"
  fi
fi

if [[ "${run_all}" == "1" ]]; then
  if [[ $# -ne 0 ]]; then
    usage
    exit 1
  fi
  run_all_examples
  exit $?
fi

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

run_example "$1" "$2"

