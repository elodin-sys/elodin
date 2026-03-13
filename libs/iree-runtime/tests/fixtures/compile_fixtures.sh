#!/usr/bin/env bash
# Compile all .mlir test fixtures to .vmfb for the current host architecture.
# Requires iree-compile on PATH (e.g. from pip: uv pip install iree-base-compiler==3.10.0).
# Version must match nix/pkgs/iree-runtime.nix. Run from this directory.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v iree-compile &>/dev/null; then
  echo "iree-compile not found. Install with: uv pip install iree-base-compiler==3.10.0" >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64) ARCH=x86_64 ;;
  arm64|aarch64) ARCH=aarch64 ;;
  *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$ARCH"

# Standard llvm-cpu (omitting target triple compiles for host)
iree-compile --iree-hal-target-backends=llvm-cpu simple_mul.mlir -o "$ARCH/simple_mul.vmfb"
iree-compile --iree-hal-target-backends=llvm-cpu matmul.mlir -o "$ARCH/matmul.vmfb"
iree-compile --iree-hal-target-backends=llvm-cpu multi_output.mlir -o "$ARCH/multi_output.vmfb"
iree-compile --iree-hal-target-backends=llvm-cpu identity_i64.mlir -o "$ARCH/identity_i64.vmfb"

# f64 support (required for physics simulations)
iree-compile --iree-hal-target-backends=llvm-cpu \
  --iree-vm-target-extension-f64 \
  --iree-input-demote-f64-to-f32=false \
  simple_add_f64.mlir -o "$ARCH/simple_add_f64.vmfb"

echo "Compiled fixtures into $ARCH/"
