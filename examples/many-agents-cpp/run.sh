#!/usr/bin/env bash
# The whole example, from the repository root: build the original's controller and hook, export
# and package the simulation, build and install the C++ programs, and verify them against Elodin.
#
#     bash examples/many-agents-cpp/run.sh [AGENTS] [TICKS]
set -euo pipefail
cd "$(dirname "$0")/../.."

agents=${1:-4}
ticks=${2:-600}
build=examples/many-agents-cpp/build
python3 examples/many-agents/controller/build.py
python3 examples/many-agents/hook/build.py
python3 examples/many-agents-cpp/export.py "$build/model" --agents "$agents" --workers 8
python3 libs/elodin-model/package.py "$build/model" "$build/ElodinModel"
cmake -S examples/many-agents-cpp/cpp -B "$build/cpp" -DCMAKE_BUILD_TYPE=Release \
  -DElodinModel_DIR="$PWD/$build/ElodinModel"
cmake --build "$build/cpp"
rm -rf "$build/install"
cmake --install "$build/cpp" --prefix "$build/install"
python3 examples/many-agents-cpp/verify.py --bin "$build/install/bin" --agents "$agents" --ticks "$ticks"
