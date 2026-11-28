#!/usr/bin/env bash
# The whole example, from the repository root: export the ball, package it three ways, build and
# install the C++ host against each, run them all, and verify them against Elodin.
set -euo pipefail
cd "$(dirname "$0")/../.."

build=examples/ball-cpp/build
python3 examples/ball-cpp/export.py "$build/model"
# The two backends, and the precompiled step built again for any CPU of this architecture, as
# for delivery to another machine.
for variant in aot jit generic; do
  case "$variant" in
    generic) options=(--cpu generic) ;;
    *) options=(--backend "$variant") ;;
  esac
  python3 libs/elodin-model/package.py "$build/model" "$build/ElodinModel-$variant" "${options[@]}"
  cmake -S examples/ball-cpp/cpp -B "$build/cpp-$variant" -DCMAKE_BUILD_TYPE=Release \
    -DElodinModel_DIR="$PWD/$build/ElodinModel-$variant"
  cmake --build "$build/cpp-$variant"
  rm -rf "$build/install-$variant"
  cmake --install "$build/cpp-$variant" --prefix "$build/install-$variant"
  # The installed tree must find the library through its runpath, so it runs after a move.
  if [[ "$(uname)" == Darwin ]]; then
    links="$(otool -L "$build/install-$variant/bin/ball_cpp")" library="@rpath/libelodin_model.dylib"
  else
    links="$(readelf -d "$build/install-$variant/bin/ball_cpp")" library="[libelodin_model.so]"
  fi
  if ! grep -qF "$library" <<<"$links"; then
    printf 'ball_cpp does not load %s through its runpath:\n%s\n' "$library" "$links" >&2
    exit 1
  fi
  # Run it moved, with the package out of reach too, so no runpath into either can load it.
  rm -rf "$build/moved-$variant" "$build/away"
  mv "$build/install-$variant" "$build/moved-$variant"
  mv "$build/ElodinModel-$variant" "$build/away"
  "$build/moved-$variant/bin/ball_cpp" "$build/ball_cpp_$variant.csv"
  mv "$build/away" "$build/ElodinModel-$variant"
done
python3 examples/ball-cpp/verify.py \
  "$build/ball_cpp_aot.csv" "$build/ball_cpp_jit.csv" "$build/ball_cpp_generic.csv"
