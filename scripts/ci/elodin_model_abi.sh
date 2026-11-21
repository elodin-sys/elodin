#!/usr/bin/env bash
# Package the checked-in rooms model and run the C and C++ consumer tests against the package.
#
#     bash scripts/ci/elodin_model_abi.sh [package.py options]
#
# Without options the step is compiled for this CPU's features; `--cpu generic` and
# `--backend jit` test the other packages. Needs Rust, CMake, a C++17 compiler and Python 3.11:
# neither the Elodin Python SDK nor Nix.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ $# -gt 0 ] || set -- --cpu native
mkdir -p "${ELODIN_MODEL_ABI_BUILD:-target/elodin-model-abi}"
build=$(cd "${ELODIN_MODEL_ABI_BUILD:-target/elodin-model-abi}" && pwd)
rm -rf "$build/ElodinModel" "$build/cpp"

"${PYTHON:-python3}" libs/elodin-model/package.py libs/elodin-model/tests/fixtures/rooms \
  "$build/ElodinModel" "$@"
cmake -S libs/elodin-model/tests/cpp -B "$build/cpp" -DElodinModel_DIR="$build/ElodinModel"
cmake --build "$build/cpp"
ctest --test-dir "$build/cpp" --output-on-failure
