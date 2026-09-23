#!/usr/bin/env bash
set -Eeuo pipefail
cd /opt/actions-runner
if [[ ! -f .ci-ready ]]; then
    printf '%s\n' 'Waiting for an administrator to register this runner and create .ci-ready.'
    while [[ ! -f .ci-ready ]]; do
        sleep 2
    done
fi
if [[ ! -s .runner ]]; then
    printf '%s\n' 'ERROR: .ci-ready exists, but runner registration is missing.' >&2
    exit 1
fi
exec ./run.sh
