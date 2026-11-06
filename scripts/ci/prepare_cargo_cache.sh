#!/usr/bin/env bash
set -euo pipefail

# Each Buildkite agent has its own checkout and runs one job at a time.
# Never clean a developer's target directory or another agent's cache.
[[ "${BUILDKITE:-}" == true ]] || exit 0
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
checkout="$(cd "${BUILDKITE_BUILD_CHECKOUT_PATH:?}" && pwd -P)"
if [[ "$(pwd -P)" != "$repo_root" || "$checkout" != "$repo_root" ]]; then
    echo 'Cargo cache preparation must run at the Buildkite checkout root' >&2
    exit 1
fi

available_kb() { df -Pk . | awk 'END {print $4}'; }
minimum_kb=$((15 * 1024 * 1024))
echo '--- Disk space before build'
df -Ph .
if (( $(available_kb) >= minimum_kb )); then
    exit 0
fi

for path in target fsw fsw/sensor-fw fsw/sensor-fw/target; do
    if [[ -L "$path" ]]; then
        echo "Refusing to clean Cargo caches through symlink: $path" >&2
        exit 1
    fi
done
echo 'Less than 15 GiB free; removing this checkout’s Cargo build artifacts'
rm -rf -- target/debug target/release fsw/sensor-fw/target/debug fsw/sensor-fw/target/release
df -Ph .
if (( $(available_kb) < minimum_kb )); then
    echo 'Still less than 15 GiB free. The CI host needs disk maintenance before building.' >&2
    exit 1
fi
