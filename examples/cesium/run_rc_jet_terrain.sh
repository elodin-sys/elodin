#!/usr/bin/env bash
set -euo pipefail

photo3d=on
editor_args=()
while (($#)); do
    case "$1" in
        --photo3d)
            if (($# < 2)) || [[ "$2" != on && "$2" != off ]]; then
                echo "--photo3d requires 'on' or 'off'." >&2
                exit 2
            fi
            photo3d=$2
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash examples/cesium/run_rc_jet_terrain.sh [--photo3d on|off] [editor arguments]"
            echo "Photo 3D defaults to on. Physics heights are configured independently."
            exit 0
            ;;
        --)
            shift
            editor_args+=("$@")
            break
            ;;
        *)
            editor_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${IN_NIX_SHELL:-}" ]]; then
    echo "Run this launcher from the repository root inside nix develop." >&2
    exit 2
fi
if [[ ! -f examples/rc-jet/main.py ]]; then
    echo "Run this launcher from the repository root." >&2
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
case "$photo3d" in
    on)
        # Photogrammetry already contains imagery; it is not the physics surface.
        export CESIUM_ION_ASSET_ID=2275207
        export ELODIN_RC_JET_HEIGHT_ASSET_ID="${ELODIN_RC_JET_HEIGHT_ASSET_ID:-1}"
        export CESIUM_CREDIT="Google Maps"
        unset CESIUM_ION_IMAGERY_ASSET_ID
        ;;
    off)
        export CESIUM_ION_ASSET_ID="${CESIUM_ION_ASSET_ID:-1}"
        export ELODIN_RC_JET_HEIGHT_ASSET_ID="${ELODIN_RC_JET_HEIGHT_ASSET_ID:-$CESIUM_ION_ASSET_ID}"
        export CESIUM_ION_IMAGERY_ASSET_ID="${CESIUM_ION_IMAGERY_ASSET_ID:-2}"
        ;;
esac
export ELODIN_ASSETS="${ELODIN_ASSETS:-$PWD/assets}"
export ELODIN_RC_JET_TERRAIN=cesium
export CESIUM_LOD_RETENTION_METERS="${CESIUM_LOD_RETENTION_METERS:-500}"
export CESIUM_MAXIMUM_SCREEN_ERROR="${CESIUM_MAXIMUM_SCREEN_ERROR:-16}"
unset CESIUM_TILESET_URL
exec "${ELODIN_EDITOR_BIN:-elodin}" editor examples/rc-jet/main.py "${editor_args[@]}"
