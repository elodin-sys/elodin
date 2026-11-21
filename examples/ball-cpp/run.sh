#!/usr/bin/env bash
# The whole example, from the repository root: export the ball, package it as an AOT and a JIT
# library, build and install the C++ host against each, run both, and verify them against Elodin.
set -euo pipefail
cd "$(dirname "$0")/../.."

build=examples/ball-cpp/build
python3 examples/ball-cpp/export.py "$build/model"
for backend in aot jit; do
  python3 libs/elodin-model/package.py "$build/model" "$build/ElodinModel-$backend" --backend "$backend"
  cmake -S examples/ball-cpp/cpp -B "$build/cpp-$backend" -DCMAKE_BUILD_TYPE=Release \
    -DElodinModel_DIR="$PWD/$build/ElodinModel-$backend"
  cmake --build "$build/cpp-$backend"
  rm -rf "$build/install-$backend"
  cmake --install "$build/cpp-$backend" --prefix "$build/install-$backend"
  installed="$build/install-$backend/bin/ball_cpp"
  if [[ "$(uname)" == Darwin ]]; then links="$(otool -L "$installed")"; else links="$(readelf -d "$installed")"; fi
  if grep -qF "$PWD" <<<"$links"; then
    printf 'the installed ball_cpp depends on the build tree, so it cannot move:\n%s\n' "$links" >&2
    exit 1
  fi
  "$installed" "$build/ball_cpp_$backend.csv"
done
python3 examples/ball-cpp/verify.py "$build/ball_cpp_aot.csv" "$build/ball_cpp_jit.csv"
