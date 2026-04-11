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
    "-iree-csv-100"
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

discover_all_examples() {
  local metrics_path
  local baseline_name
  local example_name
  local -a discovered=()

  shopt -s nullglob
  for metrics_path in "${baseline_root}"/*/profile-metrics.json; do
    baseline_name="$(basename "$(dirname "${metrics_path}")")"
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
  local metrics_path=""
  local run_log=""
  local discovered_db_path=""
  local csv_status=0
  local perf_status=0
  local -a bench_args=(bench --ticks "${ticks}" --profile --detail)
  local -a compare_args=()

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
  metrics_path="${scratch_dir}/profile-metrics.json"
  mkdir -p "${db_path}" "${export_dir}"

  echo "==> [${example_name}] running benchmark (${example_entrypoint})"
  run_log="${scratch_dir}/run.log"
  ELODIN_DB_PATH="${db_path}" "${python_bin}" "${example_entrypoint}" "${bench_args[@]}" 2>&1 | tee "${run_log}"

  "${python_bin}" scripts/ci/extract_profile_metrics.py \
    --run-log "${run_log}" \
    --ticks "${ticks}" \
    --output "${metrics_path}"

  # Optional: compile any .stablehlo.mlir fixtures in the example directory.
  # These are gitignored customer artifacts used for exact-reproduction testing.
  local example_dir
  example_dir="$(dirname "${example_entrypoint}")"
  local mlir_fixture
  local mlir_status=0
  local iree_compile_bin=""
  iree_compile_bin="$("${python_bin}" -c "
import importlib.resources, os
try:
    ref = importlib.resources.files('elodin').joinpath('_iree_compiler_dir')
    if ref.is_file():
        d = ref.read_text().strip()
        c = os.path.join(d, 'bin', 'iree-compile')
        if os.path.isfile(c): print(c); raise SystemExit
except SystemExit: raise
except Exception: pass
d = os.environ.get('IREE_COMPILER_DIR', '')
if d:
    c = os.path.join(d, 'bin', 'iree-compile')
    if os.path.isfile(c): print(c); raise SystemExit
import shutil
c = shutil.which('iree-compile')
if c: print(c)
" 2>/dev/null || true)"
  for mlir_fixture in "${example_dir}"/*.stablehlo.mlir; do
    [[ -f "${mlir_fixture}" ]] || continue
    if [[ -z "${iree_compile_bin}" ]]; then
      echo "SKIP: no iree-compile found, skipping MLIR fixture"
      break
    fi
    echo "==> [${example_name}] compiling MLIR fixture: $(basename "${mlir_fixture}")"
    if "${iree_compile_bin}" - -o /dev/null \
        --iree-hal-target-backends=llvm-cpu \
        --iree-vm-target-extension-f64 \
        --iree-input-demote-f64-to-f32=false \
        --iree-input-type=stablehlo \
        --iree-opt-level=O2 \
        --iree-stream-partitioning-favor=max-concurrency \
        --iree-dispatch-creation-enable-aggressive-fusion=true \
        --iree-llvmcpu-enable-ukernels=all \
        --iree-flow-inline-constants-max-byte-length=0 \
        --iree-llvmcpu-target-triple="${iree_triple}" \
        --iree-llvmcpu-target-cpu=host \
        < "${mlir_fixture}" 2>"${scratch_dir}/mlir_compile.log"; then
      echo "PASS: MLIR fixture $(basename "${mlir_fixture}") compiled successfully"
    else
      echo "FAIL: MLIR fixture $(basename "${mlir_fixture}") compilation failed"
      head -5 "${scratch_dir}/mlir_compile.log"
      mlir_status=1
    fi
  done

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
  elodin-db export --format csv --flatten --output "${export_dir}" "${db_path}"

  if [[ "${update_baseline}" == "1" ]]; then
    if [[ "${baseline_dir}" == "${baseline_root}" ]] && [[ -n "${file_prefix}" ]]; then
      baseline_dir="${baseline_root}/${example_name}"
      file_prefix=""
    fi

    echo "==> [${example_name}] updating baseline (${baseline_dir})"
    rm -rf "${baseline_dir}"
    mkdir -p "${baseline_dir}"
    cp -r "${export_dir}/." "${baseline_dir}/"
    cp "${metrics_path}" "${baseline_dir}/profile-metrics.json"
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

  if "${python_bin}" scripts/ci/compare_profile_metrics.py \
    --example "${example_name}" \
    --baseline "${baseline_dir}/profile-metrics.json" \
    --candidate "${metrics_path}" \
    --tolerances "${tolerances_file}"; then
    :
  else
    perf_status=$?
  fi

  # Structural check: the StableHLO MLIR must not contain _where functions
  # that convert integer types to float64.  These are the smoking gun of
  # uint64 type promotion: when jnp.uint64 mixes with int64 in the trace,
  # JAX creates _where variants like (i1, i64, i64) -> f64 with internal
  # stablehlo.convert i64 -> f64.  These produce structurally different IREE
  # kernels that cause numerical divergence in long-running simulations.
  #
  # A correctly sanitized trace has _where(i1, i64, i64) -> i64 (direct
  # select, no float promotion).
  local struct_status=0
  if [[ "${example_name}" == "linalg-iree" ]]; then
    local dump_dir="${scratch_dir}/iree_dump"
    mkdir -p "${dump_dir}"
    local stablehlo_mlir="${dump_dir}/stablehlo.mlir"
    # Re-compile with ELODIN_IREE_DUMP_DIR to capture the StableHLO MLIR.
    ELODIN_IREE_DUMP_DIR="${dump_dir}" \
      ELODIN_DB_PATH="${scratch_dir}/db_struct" \
      "${python_bin}" "${example_entrypoint}" bench --ticks 5 > /dev/null 2>&1 || true
    # Find the stablehlo.mlir in the dump (it may be in a timestamped subdir).
    stablehlo_mlir="$(find "${dump_dir}" -name 'stablehlo.mlir' -type f 2>/dev/null | head -1)"
    if [[ -n "${stablehlo_mlir}" ]]; then
      # Count _where functions that convert i64 -> f64 (the type-promotion smoking gun).
      # Pattern: func @_where_<N>(...i64..., ...i64...) -> ...f64 followed by convert i64 -> f64
      local bad_wheres
      bad_wheres="$(grep -c 'func.*@_where.*i64.*i64.*-> tensor<f64>' "${stablehlo_mlir}" 2>/dev/null)" || bad_wheres=0
      if [[ "${bad_wheres}" -gt 0 ]]; then
        echo "FAIL: uint64 type promotion detected in StableHLO (${bad_wheres} _where functions convert i64->f64)"
        echo "  These _where variants cause structurally different IREE kernels,"
        echo "  producing numerical divergence in customer simulations."
        grep 'func.*@_where.*i64.*i64.*-> tensor<f64>' "${stablehlo_mlir}" | head -3
        struct_status=1
      else
        echo "PASS: no uint64 type promotion in StableHLO _where functions"
      fi
    else
      echo "SKIP: no stablehlo.mlir found in dump (ELODIN_IREE_DUMP_DIR may not be supported)"
    fi
  fi

  if [[ "${csv_status}" -ne 0 || "${perf_status}" -ne 0 || "${mlir_status}" -ne 0 || "${struct_status}" -ne 0 ]]; then
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
  echo "=== Performance Summary (real_time_factor) ==="
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
    printf "  %-14s %6.1fx  (baseline %6.1fx, %s%s%%)\n" \
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
python_bin="${PYTHON:-python3}"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  iree_triple="arm64-apple-darwin" ;;
  Darwin-x86_64) iree_triple="x86_64-apple-darwin" ;;
  Linux-aarch64) iree_triple="aarch64-unknown-linux-gnu" ;;
  *)             iree_triple="x86_64-unknown-linux-gnu" ;;
esac

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

