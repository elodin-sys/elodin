#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${IN_NIX_SHELL:-}" ]]; then
    echo "Run this launcher from the repository root inside nix develop." >&2
    exit 2
fi
if [[ ! -f examples/rc-jet/main.py ]]; then
    echo "Run this launcher from the Elodin repository root." >&2
    exit 2
fi
if [[ -z "${CESIUM_ION_TOKEN:-}" && -n "${CESIUM_ION_TOKEN_FILE:-}" ]]; then
    if [[ ! -r "$CESIUM_ION_TOKEN_FILE" ]]; then
        echo "CESIUM_ION_TOKEN_FILE is not readable." >&2
        exit 2
    fi
    CESIUM_ION_TOKEN="$(< "$CESIUM_ION_TOKEN_FILE")"
fi
if [[ -z "${CESIUM_ION_TOKEN:-}" ]]; then
    echo "Cesium ion token missing. Set CESIUM_ION_TOKEN_FILE to a file containing the token, or set CESIUM_ION_TOKEN." >&2
    exit 2
fi
export CESIUM_ION_TOKEN
export CESIUM_ION_ASSET_ID="${CESIUM_ION_ASSET_ID:-1}"
export CESIUM_ION_IMAGERY_ASSET_ID="${CESIUM_ION_IMAGERY_ASSET_ID:-2}"
export ELODIN_ASSETS="${ELODIN_ASSETS:-$PWD/assets}"
export ELODIN_RC_JET_TERRAIN=cesium
# Fly toward the valley; the original northwestern route enters the mountains.
# Ground contact still uses the reference field elevation, not Cesium heights.
export ELODIN_RC_JET_HEADING_DEG="${ELODIN_RC_JET_HEADING_DEG:-45}"
export CESIUM_LOD_RETENTION_METERS="${CESIUM_LOD_RETENTION_METERS:-500}"
export CESIUM_MAXIMUM_SCREEN_ERROR="${CESIUM_MAXIMUM_SCREEN_ERROR:-16}"
unset CESIUM_TILESET_URL
exec "${ELODIN_EDITOR_BIN:-elodin}" editor examples/rc-jet/main.py "$@"
