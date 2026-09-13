#!/usr/bin/env bash
# Run as an administrator (stdio), not as ci. Prompts for sudo.
set -Eeuo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../versions.env"

if [[ "$(uname -s)" != Darwin ]]; then
    echo 'error: provision-host.sh is for macOS' >&2
    exit 1
fi
if [[ "$(id -un)" == ci ]]; then
    echo 'error: run this as the administrator account, not ci' >&2
    exit 1
fi

command -v brew >/dev/null || {
    echo 'error: Homebrew is required' >&2
    exit 1
}

if ! dscl . -read /Users/ci >/dev/null 2>&1; then
    if [[ -z "${CI_PASSWORD:-}" ]]; then
        read -r -s -p 'Password for new ci user: ' CI_PASSWORD
        echo
        [[ -n "${CI_PASSWORD}" ]] || {
            echo 'error: empty password' >&2
            exit 1
        }
    fi
    sudo sysadminctl -addUser ci -fullName 'Elodin CI' -password "${CI_PASSWORD}"
    echo 'Created user ci (standard).'
else
    echo 'User ci already exists.'
fi

echo 'Installing Homebrew packages (git-lfs, protobuf, ffmpeg@8)...'
brew install git-lfs protobuf ffmpeg@8

echo 'Accepting the Xcode license...'
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
sudo xcodebuild -runFirstLaunch

pkg_url="https://github.com/apple/container/releases/download/${APPLE_CONTAINER_VERSION}/container-${APPLE_CONTAINER_VERSION}-installer-signed.pkg"
pkg_path="${TMPDIR:-/tmp}/container-${APPLE_CONTAINER_VERSION}.pkg"
if command -v container >/dev/null; then
    echo "Apple Container already installed: $(container system version 2>/dev/null || echo present)"
else
    echo "Downloading Apple Container ${APPLE_CONTAINER_VERSION}..."
    curl --fail --location --retry 3 -o "${pkg_path}" "${pkg_url}"
    sudo installer -pkg "${pkg_path}" -target /
    rm -f "${pkg_path}"
fi

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/seed-python.sh"

cat <<'EOF'

Host provisioning finished. Next, as the administrator:

  sudo fdesetup add -usertoadd ci

Then log in as ci (Fast User Switching is fine) and run:

  container system start
  container run --rm --arch arm64 ubuntu:24.04 uname -m

Then register the macOS runner:

  RUNNER_TOKEN=$(/path/to/elodin/ci/runners/bin/mint-token.sh) \
    /path/to/elodin/ci/runners/mac/provision-runner.sh

EOF
