#!/usr/bin/env bash
# Compile all .mlir test fixtures to .vmfb for the selected architecture.
# Requires iree-compile on PATH (for example:
#   uv run --directory /path/to/elodin/libs/nox-py iree-compile --version
# )
# Version must match nix/pkgs/iree-runtime.nix. Run from this directory.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v iree-compile &>/dev/null; then
  echo "iree-compile not found. Install with: uv pip install iree-base-compiler==3.11.0" >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64) HOST_ARCH=x86_64 ;;
  arm64|aarch64) HOST_ARCH=aarch64 ;;
  *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

ARCH="${IREE_FIXTURE_ARCH:-$HOST_ARCH}"
TARGET_TRIPLE="${IREE_TARGET_TRIPLE:-}"

if [[ -z "${TARGET_TRIPLE}" && "${ARCH}" != "${HOST_ARCH}" ]]; then
  case "${ARCH}" in
    x86_64) TARGET_TRIPLE=x86_64-unknown-linux-gnu ;;
    aarch64) TARGET_TRIPLE=aarch64-unknown-linux-gnu ;;
    *) echo "Unsupported fixture arch: ${ARCH}" >&2; exit 1 ;;
  esac
fi

mkdir -p "$ARCH"

compile_args=(--iree-hal-target-backends=llvm-cpu)
if [[ -n "${TARGET_TRIPLE}" ]]; then
  compile_args+=(--iree-llvmcpu-target-triple="${TARGET_TRIPLE}")
fi

iree-compile "${compile_args[@]}" simple_mul.mlir -o "$ARCH/simple_mul.vmfb"
iree-compile "${compile_args[@]}" matmul.mlir -o "$ARCH/matmul.vmfb"
iree-compile "${compile_args[@]}" multi_output.mlir -o "$ARCH/multi_output.vmfb"
iree-compile "${compile_args[@]}" identity_i64.mlir -o "$ARCH/identity_i64.vmfb"

# f64 support (required for physics simulations)
iree-compile "${compile_args[@]}" \
  --iree-vm-target-extension-f64 \
  --iree-input-demote-f64-to-f32=false \
  simple_add_f64.mlir -o "$ARCH/simple_add_f64.vmfb"

if [[ -n "${TARGET_TRIPLE}" ]]; then
  echo "Compiled fixtures into $ARCH/ using target triple ${TARGET_TRIPLE}"
else
  echo "Compiled fixtures into $ARCH/"
fi
