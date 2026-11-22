#!/usr/bin/env bash
# Build the passive model library on RHEL 8 userland (glibc 2.28, GCC 8) from the checked-in rooms
# model, run its C and C++ consumer tests there, and check the library can be delivered.
#
#     bash scripts/ci/elodin_model_rhel8.sh [OUTPUT_DIR]
#
# Needs only Docker. The Rust toolchain, Cargo's downloads and the build stay in OUTPUT_DIR
# (default target/elodin-model-rhel8), so a second run reuses them.
set -euo pipefail
cd "$(dirname "$0")/../.."

output=${1:-target/elodin-model-rhel8}
mkdir -p "$output"
toolchain=$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)

docker run --rm \
  --platform "${ELODIN_RHEL8_PLATFORM:-linux/amd64}" \
  -v "$PWD:/src:ro" \
  -v "$(cd "$output" && pwd):/out" \
  -e RUSTUP_TOOLCHAIN="$toolchain" \
  -e RUSTUP_HOME=/out/rustup \
  -e CARGO_HOME=/out/cargo \
  -e CARGO_TARGET_DIR=/out/target \
  -e ELODIN_MODEL_ABI_BUILD=/out/abi \
  -e PYTHON=python3.11 \
  "${ELODIN_RHEL8_IMAGE:-rockylinux:8}" bash -euo pipefail -c '
    # A mirror is sometimes mid-sync, so retry with fresh metadata.
    for attempt in 1 2 3; do
      dnf install -y gcc gcc-c++ make cmake binutils python3.11 >/dev/null && break
      dnf clean all >/dev/null
    done
    rpm -q gcc-c++ cmake python3.11 >/dev/null
    curl -sSf https://sh.rustup.rs |
      sh -s -- -y --profile minimal --default-toolchain "$RUSTUP_TOOLCHAIN" --no-modify-path
    export PATH=$CARGO_HOME/bin:$PATH
    cd /src
    bash scripts/ci/elodin_model_abi.sh --cpu generic
    python3.11 scripts/ci/check_elodin_model_elf.py /out/abi/ElodinModel/lib/libelodin_model.so
  '
