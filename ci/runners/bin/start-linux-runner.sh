#!/bin/bash
# Run as the dedicated macOS CI account, not root.
# Starts Apple Container and the existing ci-linux-arm64 container.
set -Eeuo pipefail
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
if [[ -e "${HOME}/ci/MAINTENANCE" ]]; then
    echo 'CI maintenance flag present; leaving the Linux runner stopped.'
    exit 0
fi
command -v container >/dev/null || {
    echo 'container is not installed' >&2
    exit 1
}
# Kernel installation must already have been completed interactively.
container system start --disable-kernel-install
if container exec ci-linux-arm64 /bin/true >/dev/null 2>&1; then
    echo 'Linux container is already running.'
else
    container start ci-linux-arm64
fi
