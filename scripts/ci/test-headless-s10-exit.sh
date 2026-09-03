#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
elodin_bin="${ELODIN_BIN:-${root}/target/release/elodin}"
python_bin="${PYTHON:-$(command -v python3)}"
work="$(mktemp -d "${TMPDIR:-/tmp}/elodin-headless-s10.XXXXXX")"

cleanup() {
    local pid_file pid
    while IFS= read -r pid_file; do
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            for _ in {1..50}; do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 0.02
            done
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done < <(find "${work}" -name sidecar.pid -type f 2>/dev/null)
    rm -rf "${work}"
}
trap cleanup EXIT INT TERM

if [[ ! -x "${elodin_bin}" ]]; then
    echo "FAIL: elodin binary is not executable: ${elodin_bin}" >&2
    exit 1
fi

sidecar="${work}/sidecar.sh"
cat >"${sidecar}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$$" >"${SIDECAR_PID_FILE:?}"
exec sleep 60
SH
chmod +x "${sidecar}"

sim="${work}/sim.py"
cat >"${sim}" <<'PY'
import os
import sys
import time
from pathlib import Path

pid_file = Path(os.environ["SIDECAR_PID_FILE"])
deadline = time.monotonic() + 5.0
while not pid_file.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not pid_file.exists():
    print("SIM_FIXTURE_ERROR=sidecar-pid-timeout", flush=True)
    sys.exit(99)

exit_code = int(os.environ["SIM_EXIT_CODE"])
print(f"SIM_FIXTURE_EXIT={exit_code}", flush=True)
time.sleep(0.1)
sys.exit(exit_code)
PY

fail_case() {
    local name="$1"
    local message="$2"
    local log="${work}/${name}/elodin.log"
    echo "FAIL [${name}]: ${message}" >&2
    if [[ -f "${log}" ]]; then
        echo "--- ${name} log ---" >&2
        tail -100 "${log}" >&2
    fi
    return 1
}

run_case() {
    local name="$1"
    local sim_exit="$2"
    local expected="$3"
    local case_dir="${work}/${name}"
    local pid_file="${case_dir}/sidecar.pid"
    local plan="${case_dir}/s10.toml"
    local log="${case_dir}/elodin.log"
    local status pid

    mkdir -p "${case_dir}"
    cat >"${plan}" <<TOML
type = "group"
refs = []

[recipes.sidecar]
type = "process"
cmd = "${sidecar}"
args = []
restart_policy = "never"
fail_on_error = false
silence = false
depends_on = []
own_process_group = false
no_watch = true

[recipes.sidecar.env]
SIDECAR_PID_FILE = "${pid_file}"

[recipes.sim]
type = "sim"
path = "${sim}"
addr = "127.0.0.1:0"
optimize = false
depends-on = []
own-process-group = false

[recipes.sim.env]
SIDECAR_PID_FILE = "${pid_file}"
SIM_EXIT_CODE = "${sim_exit}"
TOML

    set +e
    ELODIN_PYTHON="${python_bin}" timeout --signal=TERM --kill-after=2s 15s \
        "${elodin_bin}" run "${plan}" >"${log}" 2>&1
    status=$?
    set -e

    if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
        fail_case "${name}" "elodin run timed out with status ${status}"
        return
    fi
    if [[ "${expected}" == "success" && "${status}" -ne 0 ]]; then
        fail_case "${name}" "expected status 0, got ${status}"
        return
    fi
    if [[ "${expected}" == "failure" && "${status}" -eq 0 ]]; then
        fail_case "${name}" "expected a nonzero status"
        return
    fi
    if ! grep -q "SIM_FIXTURE_EXIT=${sim_exit}" "${log}"; then
        fail_case "${name}" "simulation fixture did not reach its intended exit"
        return
    fi
    if [[ ! -s "${pid_file}" ]]; then
        fail_case "${name}" "sidecar did not write its PID file"
        return
    fi

    pid="$(cat "${pid_file}")"
    for _ in {1..50}; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.02
    done
    if kill -0 "${pid}" 2>/dev/null; then
        fail_case "${name}" "sidecar PID ${pid} survived elodin exit"
        return
    fi

    rm -f "${pid_file}"
    echo "PASS [${name}]: elodin status=${status}, sidecar ${pid} reaped"
}

run_case success 0 success
run_case failure 7 failure
