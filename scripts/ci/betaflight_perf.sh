#!/usr/bin/env bash
# Whole-job Buildkite concurrency protects Betaflight's fixed UDP/CLI ports.
set -euo pipefail

repo="$(pwd)"
example="examples/betaflight-sitl"
python_bin="${PYTHON:-python3}"
output="${BETAFLIGHT_PERF_OUTPUT:-}"
if [[ -z "${output}" ]]; then
  mkdir -p "${repo}/profile_output/betaflight-perf-ci"
  output="$(mktemp -d "${repo}/profile_output/betaflight-perf-ci/run.XXXXXX")"
fi
mkdir -p "${output}"
output="$(cd "${output}" && pwd)"
# Never reuse old summaries if a new process fails before writing its own.
if [[ -n "$(ls -A "${output}")" ]]; then
  echo "FAIL: output directory must be empty: ${output}" >&2
  exit 1
fi
mkdir -p "${HOME}/.cache"
work="$(mktemp -d "${HOME}/.cache/betaflight-perf.XXXXXX")"

cleanup() {
  local status=$?
  trap - EXIT
  # All measured processes have private source paths below work, including s10's
  # firmware child. Cover abnormal job cancellation as well as normal shutdown.
  pkill -KILL -f "${work}/" 2>/dev/null || true
  rm -rf "${work}"
  if [[ -n "${BUILDKITE:-}" ]]; then
    buildkite-agent artifact upload "${output}/*" || { [[ "${status}" != 0 ]] || status=1; }
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
    printf '%s\n' "${base}"
  fi
}

main() {
  local base_rev="" base_site="${BETAFLIGHT_PERF_BASE_SITE:-}" base_py=""
  local candidate_source="${work}/candidate/${example}"
  local base_source="${work}/base/${example}"
  local file source
  local -a base_args=()

  if ! base_rev="$(resolve_base_rev)"; then
    if [[ -n "${base_site}" ]]; then
      echo "FAIL: explicit base site requires a resolvable REGRESSION_BASE" >&2
      return 1
    fi
    echo "WARN: cannot resolve baseline; performance A/B will be skipped"
  elif [[ -n "${base_site}" ]]; then
    base_site="$(cd "${base_site}" && pwd)"
  elif base_py="$(nix build --no-link --print-out-paths --max-jobs 0 --builders '' \
    "git+file://${repo}?rev=${base_rev}&allRefs=1#elodin-py")"; then
    local -a sites=("${base_py}"/lib/python3.*/site-packages)
    [[ "${#sites[@]}" == 1 && -d "${sites[0]}" ]] || return 1
    base_site="${sites[0]}"
  else
    echo "WARN: base ${base_rev:0:7} elodin-py unavailable in cache; performance A/B will be skipped"
  fi

  if [[ -n "${base_site}" && ! -f "${base_site}/elodin/__init__.py" ]]; then
    echo "FAIL: base site does not contain elodin: ${base_site}" >&2
    return 1
  fi

  # Snapshot tracked source only: never import a developer's custom schematic or
  # stale EEPROM. Copy working-tree candidate bytes, not just the HEAD version.
  mkdir -p "${work}/candidate"
  while IFS= read -r -d '' file; do
    if [[ -f "${file}" ]]; then
      cp --parents "${file}" "${work}/candidate/"
    fi
  done < <(git ls-files -z "${example}")
  if [[ -n "${base_site}" ]]; then
    mkdir -p "${work}/base"
    git archive "${base_rev}" "${example}" | tar -x -C "${work}/base"
    [[ -f "${base_source}/main.py" ]] || return 1
    base_args=(--base-source "${base_source}" --base-site "${base_site}")
  fi
  printf 'candidate=%s\nbase=%s\nbase_site=%s\n' \
    "$(git rev-parse HEAD) (working-tree source)" "${base_rev}" "${base_site}" > "${output}/revisions.txt"

  if [[ "${BETAFLIGHT_PERF_SKIP_BUILD:-0}" != 1 ]]; then
    git submodule update --init --depth 1 "${example}/betaflight"
    SITL_RATE_HZ=8000 RUN_LOOP_DELAY_US=0 bash "${example}/build.sh"
  fi
  local firmware="${repo}/${example}/betaflight/obj/main/betaflight_SITL.elf"
  [[ -x "${firmware}" ]] || { echo "FAIL: missing firmware ${firmware}"; return 1; }
  sha256sum "${firmware}" > "${output}/firmware.sha256"
  # The exact same executable and initial EEPROM are held fixed for A/B. This
  # isolates Elodin/its Python callback, not changes to firmware or its patch.
  for source in "${candidate_source}" ${base_site:+"${base_source}"}; do
    mkdir -p "${source}/betaflight/obj/main"
    ln -s "${firmware}" "${source}/betaflight/obj/main/betaflight_SITL.elf"
  done
  timeout --signal=TERM --kill-after=5s 90s "${python_bin}" "${candidate_source}/init_eeprom.py"
  cp "${candidate_source}/eeprom.bin" "${work}/verified-eeprom.bin"
  sha256sum "${work}/verified-eeprom.bin" > "${output}/eeprom.sha256"

  source scripts/ci/free_ports.sh
  pick_ports
  # Assets are shared, but each side retains its own schematic/example source.
  # The model is ingested into the recorded DB; no render server is started.
  ELODIN_ASSETS="${repo}/assets" "${python_bin}" scripts/ci/betaflight_perf.py \
    --candidate-source "${candidate_source}" "${base_args[@]}" \
    --eeprom "${work}/verified-eeprom.bin" --work "${work}" --output "${output}" \
    --addr "127.0.0.1:${db_port}" \
    --runs "${BETAFLIGHT_PERF_RUNS:-5}" \
    --threshold "${BETAFLIGHT_PERF_THRESHOLD:-0.10}" \
    --timeout "${BETAFLIGHT_PERF_RUN_TIMEOUT:-120}"
}

main 2>&1 | tee "${output}/job.log"
