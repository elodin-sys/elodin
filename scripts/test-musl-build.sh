#!/usr/bin/env bash
#
# Reproduce locally the "build-local-artifacts" musl matrix entries from
# .github/workflows/release.yml, then run `cargo build -p elodin-db` for
# each. Use this to validate changes to the release workflow without
# burning a CI cycle.
#
# By default runs BOTH musl matrix targets sequentially (same as the CI
# matrix). Pass a specific target to run only that one (for debugging).
#
# Usage:
#   scripts/test-musl-build.sh                                 # both targets
#   scripts/test-musl-build.sh x86_64-unknown-linux-musl       # only x86_64
#   scripts/test-musl-build.sh aarch64-unknown-linux-musl      # only aarch64
#
# Requirements:
#   - zig in PATH (try: nix shell nixpkgs#zig, or brew install zig)
#   - rustup (targets will be added automatically if missing)
#
# The script appends a [target.<triple>] section to .cargo/config.toml
# for each target and restores the file on exit (Ctrl+C safe).

set -o pipefail
# Note: nounset (-u) intentionally NOT set — macOS ships bash 3.2 which
# treats empty arrays as unset, breaking ${#failed[@]} access at the end.

ALL_TARGETS=(x86_64-unknown-linux-musl aarch64-unknown-linux-musl)

if [ "$#" -eq 0 ]; then
  TARGETS=("${ALL_TARGETS[@]}")
else
  TARGETS=("$@")
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root" || { echo "error: cannot cd to repo root: $repo_root" >&2; exit 1; }

if ! command -v zig >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: zig not in PATH

install with one of:
  nix shell nixpkgs#zig
  brew install zig
  https://ziglang.org/download/
EOF
  exit 1
fi

if ! command -v rustup >/dev/null 2>&1; then
  echo "error: rustup not in PATH" >&2
  exit 1
fi

zig_bin="$(command -v zig)"
echo "==> using zig:    $zig_bin ($(zig version))"
echo "==> using cargo:  $(cargo --version)"
echo "==> using rustc:  $(rustc --version)"
echo "==> using rustup: $(rustup --version | head -1)"
echo "==> targets:      ${TARGETS[*]}"

zig_major_minor="$(zig version | awk -F. '{ print $1"."$2 }')"
if [ "$zig_major_minor" != "0.16" ]; then
  echo "warning: CI uses Zig 0.16.x; local Zig $zig_major_minor may produce different results" >&2
fi

cargo_config=".cargo/config.toml"
session_dir="$(mktemp -d "${TMPDIR:-/tmp}/musl-test.XXXXXX")"
config_backup="${session_dir}/.config.toml.bak"
had_config=0
if [ -f "$cargo_config" ]; then
  cp "$cargo_config" "$config_backup"
  had_config=1
fi

cleanup() {
  if [ "$had_config" = "1" ] && [ -f "$config_backup" ]; then
    mv "$config_backup" "$cargo_config"
  else
    rm -f "$cargo_config"
  fi
  rm -rf "$session_dir"
  echo "==> cleanup done (.cargo/config.toml restored)"
}
trap cleanup EXIT INT TERM

build_one_target() {
  local target="$1"
  local zig_target env_target

  case "$target" in
    x86_64-unknown-linux-musl)
      zig_target="x86_64-linux-musl"
      env_target="x86_64_unknown_linux_musl"
      ;;
    aarch64-unknown-linux-musl)
      zig_target="aarch64-linux-musl"
      env_target="aarch64_unknown_linux_musl"
      ;;
    *)
      echo "error: unsupported target: $target" >&2
      return 2
      ;;
  esac

  if ! rustup target list --installed 2>/dev/null | grep -q "^${target}$"; then
    echo "==> installing rustup target ${target}"
    rustup target add "$target"
  fi

  local wrap_dir="${session_dir}/zig-cc-${env_target}"
  mkdir -p "$wrap_dir"

  local wrapper_src="${repo_root}/scripts/zig-musl-wrapper.sh"
  if [ ! -f "$wrapper_src" ]; then
    echo "error: missing $wrapper_src" >&2
    return 2
  fi
  write_zig_trampoline() {
    local dest="$1" cmd="$2"
    printf '%s\n' \
      '#!/bin/bash' \
      "export ELODIN_ZIG_BIN=\"${zig_bin}\"" \
      "export ELODIN_ZIG_TARGET=\"${zig_target}\"" \
      "export ELODIN_ZIG_CMD=\"${cmd}\"" \
      "exec /bin/bash \"${wrapper_src}\" \"\$@\"" \
      >"$dest"
  }
  write_zig_trampoline "${wrap_dir}/cc" cc
  write_zig_trampoline "${wrap_dir}/c++" c++
  printf '%s\n' \
    '#!/bin/sh' \
    "exec \"${zig_bin}\" ar \"\$@\"" \
    >"${wrap_dir}/ar"
  chmod +x "${wrap_dir}/cc" "${wrap_dir}/c++" "${wrap_dir}/ar"
  echo "--- generated cc trampoline ---"
  cat "${wrap_dir}/cc"

  # Restore original .cargo/config.toml then append fresh target section
  if [ "$had_config" = "1" ]; then
    cp "$config_backup" "$cargo_config"
  else
    rm -f "$cargo_config"
  fi
  mkdir -p .cargo
  touch "$cargo_config"
  # aarch64: rustc CRT for zig ld.lld. x86_64: zig cc provides CRT.
  if [ "$target" = "aarch64-unknown-linux-musl" ]; then
    printf '\n[target.%s]\nlinker = "%s/cc"\n' \
      "$target" "$wrap_dir" >> "$cargo_config"
  else
    printf '\n[target.%s]\nlinker = "%s/cc"\nrustflags = ["-C", "link-self-contained=no"]\n' \
      "$target" "$wrap_dir" >> "$cargo_config"
  fi

  export "CC_${env_target}=${wrap_dir}/cc"
  export "CXX_${env_target}=${wrap_dir}/c++"
  export "AR_${env_target}=${wrap_dir}/ar"

  echo
  echo "============================================================"
  echo "==> [${target}] cargo build --release -p elodin-db"
  echo "============================================================"
  echo
  cargo build --release -p elodin-db --target "$target"
}

failed=()
passed=()
for target in "${TARGETS[@]}"; do
  if build_one_target "$target"; then
    passed+=("$target")
    echo "==> [${target}] PASS"
  else
    failed+=("$target")
    echo "==> [${target}] FAIL" >&2
  fi
done

echo
echo "============================================================"
echo "==> SUMMARY"
echo "============================================================"
for t in "${passed[@]}"; do echo "  PASS  $t"; done
for t in "${failed[@]}"; do echo "  FAIL  $t"; done

if [ "${#failed[@]}" -gt 0 ]; then
  exit 1
fi
