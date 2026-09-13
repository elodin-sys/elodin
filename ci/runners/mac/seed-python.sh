#!/usr/bin/env bash
# Pre-seeds actions/setup-python's tool cache so jobs never need sudo.
# setup-python hardcodes /Users/runner/hostedtoolcache on macOS and its
# installer runs `sudo installer`; the ci account has neither.
# Run as an administrator (stdio). Prompts for sudo.
# Re-run whenever PYTHON_SERIES changes (python-version in release.yml).
set -Eeuo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../versions.env"

toolcache=/Users/runner/hostedtoolcache
manifest=https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json

[[ "$(uname -s)" == Darwin ]] || {
    echo 'error: seed-python.sh is for macOS' >&2
    exit 1
}
[[ "$(id -un)" != ci ]] || {
    echo 'error: run this as the administrator account, not ci' >&2
    exit 1
}
dscl . -read /Users/ci >/dev/null 2>&1 || {
    echo 'error: user ci does not exist; run provision-host.sh first' >&2
    exit 1
}

read -r version url < <(python3 - "${PYTHON_SERIES}" "${manifest}" <<'PY'
import json, sys, urllib.request

series, manifest_url = sys.argv[1:3]
with urllib.request.urlopen(manifest_url) as resp:
    manifest = json.load(resp)
for release in manifest:
    if release["stable"] and release["version"].startswith(series + "."):
        for f in release["files"]:
            if f["platform"] == "darwin" and f["arch"] == "arm64":
                print(release["version"], f["download_url"])
                sys.exit(0)
sys.exit(f"no darwin/arm64 build for Python {series}")
PY
)

if [[ -f "${toolcache}/Python/${version}/arm64.complete" ]]; then
    echo "Python ${version} already seeded in ${toolcache}."
else
    echo "Seeding Python ${version} into ${toolcache}..."
    sudo mkdir -p "${toolcache}"
    sudo chown "$(id -un)" "${toolcache}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    curl --fail --location --retry 3 -o "${tmp}/python.tar.gz" "${url}"
    tar -xzf "${tmp}/python.tar.gz" -C "${tmp}"
    (cd "${tmp}" && AGENT_TOOLSDIRECTORY="${toolcache}" bash ./setup.sh)
fi

sudo chown -R ci:staff "${toolcache}"
echo "setup-python cache ready: ${toolcache}/Python/${version}/arm64"
