#!/usr/bin/env bash
# Run as the dedicated ci account after provision-host.sh.
set -Eeuo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../versions.env"

if [[ "$(id -un)" != ci ]]; then
    echo 'error: run provision-runner.sh as the ci user' >&2
    exit 1
fi
[[ "$(uname -m)" == arm64 ]] || {
    echo 'error: expected Apple silicon' >&2
    exit 1
}
[[ -n "${RUNNER_TOKEN:-}" ]] || {
    echo 'error: set RUNNER_TOKEN (ci/runners/bin/mint-token.sh)' >&2
    exit 1
}

export PATH="/opt/homebrew/bin:/usr/bin:/bin:${PATH}"
runner_dir="${HOME}/actions-runner"
mkdir -p "${runner_dir}"
cd "${runner_dir}"

if [[ ! -x ./config.sh ]]; then
    tarball="actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz"
    curl --fail --location --retry 3 \
        -o "${tarball}" \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${tarball}"
    printf '%s  %s\n' "${RUNNER_SHA256_OSX_ARM64}" "${tarball}" | shasum -a 256 -c -
    tar xzf "${tarball}"
    rm -f "${tarball}"
fi

if [[ ! -x "${HOME}/.cargo/bin/rustup" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}" --profile default
fi
# shellcheck disable=SC1091
source "${HOME}/.cargo/env"
rustup default "${RUST_TOOLCHAIN}"

if ! command -v dist >/dev/null; then
    curl --proto '=https' --tlsv1.2 -LsSf \
        "https://github.com/axodotdev/cargo-dist/releases/download/v${DIST_VERSION}/cargo-dist-installer.sh" \
        | sh
fi

command -v git-lfs >/dev/null && git lfs install --skip-repo

cat >"${runner_dir}/.env" <<EOF
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
HOMEBREW_NO_AUTO_UPDATE=1
CARGO_BUILD_JOBS=10
EOF

cat >"${runner_dir}/.path" <<EOF
/opt/homebrew/bin
${HOME}/.cargo/bin
EOF

if [[ ! -s "${runner_dir}/.runner" ]]; then
    ./config.sh \
        --unattended \
        --url "${REPO_URL}" \
        --name ci-macos-arm64 \
        --labels ci,ci-macos-arm64 \
        --work _work \
        --token "${RUNNER_TOKEN}"
fi

./svc.sh install
./svc.sh start
./svc.sh status

echo 'macOS runner registered. Confirm it is online in GitHub Settings > Actions > Runners.'
