#!/usr/bin/env bash
set -Eeuo pipefail

[[ "$(uname -s)" == Linux ]] || {
    echo 'Expected Linux' >&2
    exit 1
}
[[ "$(id -u)" -ne 0 ]] || {
    echo 'Runner must not execute as root' >&2
    exit 1
}

host_arch="$(uname -m)"
case "$host_arch" in
    x86_64) cc_guard='__x86_64__' ;;
    aarch64) cc_guard='__aarch64__' ;;
    *)
        echo "Unexpected Linux arch: ${host_arch}" >&2
        exit 1
        ;;
esac

uname -a
id
cat /etc/os-release
ldd --version
command -v cargo
cargo --version
command -v rustc
rustc --version
command -v dist
dist --version
command -v cargo-deb
cargo-deb --version
command -v git-lfs
git lfs version
command -v python3
python3 --version

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat >"${work}/smoke.c" <<C
#include <stdio.h>
#ifndef ${cc_guard}
#error This smoke test must produce a native executable for ${host_arch}.
#endif
int main(void) { puts("Native Linux compile-and-execute: PASS"); return 0; }
C
cc -O2 -o "${work}/smoke" "${work}/smoke.c"
file "${work}/smoke"
"${work}/smoke"
echo "Linux runner environment: PASS (${host_arch})"
