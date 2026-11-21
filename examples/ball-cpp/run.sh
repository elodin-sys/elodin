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
  # The installed tree must find the library through its runpath, so it runs after a move.
  if [[ "$(uname)" == Darwin ]]; then
    links="$(otool -L "$build/install-$backend/bin/ball_cpp")" library="@rpath/libelodin_model.dylib"
  else
    links="$(readelf -d "$build/install-$backend/bin/ball_cpp")" library="[libelodin_model.so]"
  fi
  if ! grep -qF "$library" <<<"$links"; then
    printf 'ball_cpp does not load %s through its runpath:\n%s\n' "$library" "$links" >&2
    exit 1
  fi
  # Run it moved, with the package out of reach too, so no runpath into either can load it.
  rm -rf "$build/moved-$backend" "$build/away"
  mv "$build/install-$backend" "$build/moved-$backend"
  mv "$build/ElodinModel-$backend" "$build/away"
  "$build/moved-$backend/bin/ball_cpp" "$build/ball_cpp_$backend.csv"
  mv "$build/away" "$build/ElodinModel-$backend"
done
python3 examples/ball-cpp/verify.py "$build/ball_cpp_aot.csv" "$build/ball_cpp_jit.csv"
