#!/bin/bash
# Zig musl compiler/linker wrapper for elodin-db dist builds.
#
# Required env:
#   ELODIN_ZIG_BIN     path to zig
#   ELODIN_ZIG_TARGET  zig triple, e.g. aarch64-linux-musl
#   ELODIN_ZIG_CMD     cc or c++
#
# Compile (-c): zig cc/c++. Link: zig cc, unless rustc emitted
# --fix-cortex-a53-843419, in which case we unwrap -Wl, and exec zig ld.lld.
# zig cc 0.16 rejects that flag even with -fuse-ld=lld / -Wl,; LLVM LLD
# implements it. Fail closed if rustc asked for the errata rewrite and LLD
# would not run with the flag.
#
# zig ld.lld does not know Zig's C++ runtime. rustc passes -lstdc++ (openh264).
# Rewrite that to Zig's cached libc++.a / libc++abi.a for this target.
set -eo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

zig_bin="${ELODIN_ZIG_BIN:?ELODIN_ZIG_BIN is not set}"
zig_target="${ELODIN_ZIG_TARGET:?ELODIN_ZIG_TARGET is not set}"
zig_cmd="${ELODIN_ZIG_CMD:?ELODIN_ZIG_CMD is not set}"

# Print libc++abi.a then libc++.a (Zig's link order) for $zig_target.
# Materializes them via a cached dummy zig c++ link if needed.
resolve_zig_libcxx() {
  local cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/elodin-zig-libcxx"
  mkdir -p "$cache_root"
  local zig_ver
  zig_ver=$("$zig_bin" version)
  local stamp="$cache_root/${zig_ver}-${zig_target}.txt"
  if [ -f "$stamp" ]; then
    local abi cxx
    abi=$(sed -n '1p' "$stamp")
    cxx=$(sed -n '2p' "$stamp")
    if [ -f "$abi" ] && [ -f "$cxx" ]; then
      printf '%s\n%s\n' "$abi" "$cxx"
      return
    fi
  fi

  local probe
  probe=$(mktemp -d)
  printf 'int main(void){return 0;}\n' >"$probe/p.cpp"
  local verbose
  if ! verbose=$(ZIG_VERBOSE_LINK=1 "$zig_bin" c++ -target "$zig_target" -static \
    "$probe/p.cpp" -o "$probe/p" 2>&1); then
    rm -rf "$probe"
    printf '%s\n' "$verbose" >&2
    die "failed to materialize zig libc++ for $zig_target"
  fi
  rm -rf "$probe"

  local abi cxx
  abi=$(printf '%s\n' "$verbose" | tr ' ' '\n' | grep '/libc++abi.a$' | tail -1 || true)
  cxx=$(printf '%s\n' "$verbose" | tr ' ' '\n' | grep '/libc++.a$' | tail -1 || true)
  if [ ! -f "$abi" ] || [ ! -f "$cxx" ]; then
    die "could not find zig libc++ archives for $zig_target"
  fi
  printf '%s\n%s\n' "$abi" "$cxx" >"$stamp"
  printf '%s\n%s\n' "$abi" "$cxx"
}

append_zig_libcxx() {
  local abi cxx
  abi=$(printf '%s\n' "$1" | sed -n '1p')
  cxx=$(printf '%s\n' "$1" | sed -n '2p')
  lld_args+=("$abi" "$cxx")
}

rust_wants_cortex=0
is_compile=0
for a in "$@"; do
  case "$a" in
    -c) is_compile=1 ;;
    --fix-cortex-a53-843419) rust_wants_cortex=1 ;;
    -Wl,*)
      case ",${a#-Wl,}," in
        *,--fix-cortex-a53-843419,*) rust_wants_cortex=1 ;;
      esac
      ;;
  esac
done

filter_zig_cc_args() {
  args=()
  for a in "$@"; do
    case "$a" in
      --target=*) ;;
      */self-contained/*crt*.o) ;;
      *) args+=("$a") ;;
    esac
  done
}

if [ "$is_compile" -eq 1 ]; then
  if [ "$rust_wants_cortex" -eq 1 ]; then
    die "rustc emitted --fix-cortex-a53-843419 on a compile (-c) invocation; zig ld.lld would not run"
  fi
  filter_zig_cc_args "$@"
  exec "$zig_bin" "$zig_cmd" -target "$zig_target" "${args[@]}"
fi

if [ "$rust_wants_cortex" -eq 1 ]; then
  lld_args=()
  skip_next=
  lld_has_cortex=0
  libcxx_paths=
  for a in "$@"; do
    if [ -n "$skip_next" ]; then
      skip_next=
      continue
    fi
    case "$a" in
      --target=*|-nodefaultlibs|-nostartfiles|-no-pie|-pie|-pthread|-flavor)
        [ "$a" = "-flavor" ] && skip_next=1
        ;;
      -lstdc++|-lc++)
        if [ -z "$libcxx_paths" ]; then
          libcxx_paths=$(resolve_zig_libcxx)
        fi
        append_zig_libcxx "$libcxx_paths"
        ;;
      -Wl,*)
        spec="${a#-Wl,}"
        IFS=',' read -ra parts <<< "$spec"
        for p in "${parts[@]}"; do
          [ -z "$p" ] && continue
          case "$p" in
            -lstdc++|-lc++)
              if [ -z "$libcxx_paths" ]; then
                libcxx_paths=$(resolve_zig_libcxx)
              fi
              append_zig_libcxx "$libcxx_paths"
              ;;
            *)
              lld_args+=("$p")
              if [ "$p" = "--fix-cortex-a53-843419" ]; then
                lld_has_cortex=1
              fi
              ;;
          esac
        done
        ;;
      --fix-cortex-a53-843419)
        lld_args+=("$a")
        lld_has_cortex=1
        ;;
      *)
        lld_args+=("$a")
        ;;
    esac
  done
  if [ "$lld_has_cortex" -ne 1 ]; then
    die "rustc emitted --fix-cortex-a53-843419 but it is missing from the zig ld.lld argument list"
  fi
  exec "$zig_bin" ld.lld "${lld_args[@]}"
fi

filter_zig_cc_args "$@"
exec "$zig_bin" "$zig_cmd" -target "$zig_target" "${args[@]}"
