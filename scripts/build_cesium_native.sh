#!/usr/bin/env bash
# Run from the repository root, inside nix develop.
set -euo pipefail

root="${CESIUM_BUILD_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/elodin/cesium-0.65.0}"
source_dir="$root/source"
install_dir="$root/install"
vcpkg_commit=56bb2411609227288b70117ead2c47585ba07713
export EZVCPKG_BASEDIR="$root/vcpkg"
mkdir -p "$root"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 --branch v0.65.0 --recurse-submodules --shallow-submodules \
        https://github.com/CesiumGS/cesium-native.git "$source_dir"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == 1bca8021711294cc83a5a032599ac72d9b3166c9 ]] || {
    echo "Unexpected Cesium Native revision" >&2; exit 1;
}

# Draco 1.5.7 mistakes upstream Clang on Darwin for an ELF linker.
# Use an overlay, leaving both upstream checkouts unchanged.
vcpkg_dir="$EZVCPKG_BASEDIR/$vcpkg_commit"
if [[ ! -d "$vcpkg_dir/.git" ]]; then
    git clone --filter=blob:none https://github.com/microsoft/vcpkg.git "$vcpkg_dir"
    git -C "$vcpkg_dir" checkout "$vcpkg_commit"
fi
mkdir -p "$root/overlay"
cp -R "$vcpkg_dir/ports/draco" "$root/overlay/"
uv run --no-project python - "$root/overlay/draco/portfile.cmake" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
patch = '''file(READ "${SOURCE_PATH}/cmake/draco_targets.cmake" DRACO_TARGETS)
string(REPLACE "if(CMAKE_CXX_COMPILER_ID MATCHES \\"^Clang|^GNU\\")" "if(NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES \\"^Clang|^GNU\\")" DRACO_TARGETS "${DRACO_TARGETS}")
file(WRITE "${SOURCE_PATH}/cmake/draco_targets.cmake" "${DRACO_TARGETS}")

'''
path.write_text(text.replace('vcpkg_cmake_configure(', patch + 'vcpkg_cmake_configure('))
PY
cmake -S "$source_dir" -B "$root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCESIUM_TESTS_ENABLED=OFF \
    -DCESIUM_ENABLE_CLANG_TIDY=OFF -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DVCPKG_OVERLAY_PORTS="$root/overlay"
cmake --build "$root/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
cmake --install "$root/build"
printf 'Build ready. Set CESIUM_NATIVE_DIR=%s\n' "$install_dir"
