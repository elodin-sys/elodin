#!/usr/bin/env bash
# Print a one-hour GitHub Actions runner registration token. Capture stdout only:
#   RUNNER_TOKEN=$(ci/runners/bin/mint-token.sh)
set -Eeuo pipefail
gh api -X POST repos/elodin-sys/elodin/actions/runners/registration-token --jq .token
