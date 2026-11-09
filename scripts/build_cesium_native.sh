#!/usr/bin/env bash
# Native SDK build logic belongs to the pinned bevy-cesium dependency.
set -euo pipefail
plugin_dir="${BEVY_CESIUM_DIR:-}"
if [[ -z "$plugin_dir" ]]; then
    plugin_manifest="$(cargo metadata --locked --format-version 1 | uv run --no-project python -c 'import json,sys; print(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "bevy-cesium"))')"
    plugin_dir="$(dirname "$plugin_manifest")"
fi
export CESIUM_BUILD_ROOT="${CESIUM_BUILD_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/elodin/cesium-0.65.0}"
exec bash "$plugin_dir/scripts/build_cesium_native.sh"
