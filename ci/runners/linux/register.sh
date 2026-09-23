#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

lane="${1:-}"
require_lane "${lane}"
[[ -n "${RUNNER_TOKEN:-}" ]] || die 'set RUNNER_TOKEN (ci/runners/bin/mint-token.sh)'

if runner_exec "${lane}" test -s /opt/actions-runner/.runner; then
    die "${lane} is already registered"
fi

# Token is passed as an env var inside the guest; do not print it.
runner_exec "${lane}" bash -c '
    set -euo pipefail
    cd /opt/actions-runner
    ./config.sh \
        --unattended \
        --url "'"${REPO_URL}"'" \
        --name "'"${lane}"'" \
        --labels "ci,'"${lane}"'" \
        --work _work \
        --token "${RUNNER_TOKEN}"
    touch .ci-ready
'

echo "Registered ${lane}. Confirm it is online in GitHub Settings > Actions > Runners."
