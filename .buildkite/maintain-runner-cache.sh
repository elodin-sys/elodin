#!/usr/bin/env bash
set -euo pipefail

mode=dry-run
case "${1:-}" in
    --apply) mode=apply ;;
    --dry-run | "") ;;
    --help)
        echo "usage: $0 [--dry-run|--apply]"
        exit 0
        ;;
    *)
        echo "unknown argument: $1" >&2
        exit 2
        ;;
esac

builds_root="${BUILDKITE_BUILDS_ROOT:-/var/lib/buildkite-agent/builds}"
trigger_gib="${CACHE_GC_TRIGGER_GIB:-500}"
target_gib="${CACHE_GC_TARGET_GIB:-1024}"
min_retained="${CACHE_GC_MIN_RETAINED:-4}"
min_retained_per_repo="${CACHE_GC_MIN_RETAINED_PER_REPO:-1}"
nix_shell_tmp_root="${NIX_SHELL_TMP_ROOT:-/tmp}"
nix_shell_min_age_minutes="${NIX_SHELL_GC_MIN_AGE_MINUTES:-1440}"

[[ -d "$builds_root" ]] || {
    echo "Buildkite builds root not found: $builds_root" >&2
    exit 1
}
[[ -d "$nix_shell_tmp_root" ]] || {
    echo "Nix shell temporary root not found: $nix_shell_tmp_root" >&2
    exit 1
}
builds_root="$(realpath -e "$builds_root")"
nix_shell_tmp_root="$(realpath -e "$nix_shell_tmp_root")"
(( trigger_gib > 0 && target_gib > trigger_gib && min_retained > 0 && min_retained_per_repo > 0 && nix_shell_min_age_minutes >= 0 )) || {
    echo "invalid cache-maintenance thresholds" >&2
    exit 1
}
cd "$builds_root"

available_kib() {
    df -Pk "$builds_root" | awk 'END {print $4}'
}

available_gib() {
    echo $(( $(available_kib) / 1024 / 1024 ))
}

if [[ "$mode" == apply ]]; then
    command -v flock >/dev/null || {
        echo "flock is required for apply mode" >&2
        exit 1
    }
    exec 9>"$builds_root/.runner-cache-maintenance.lock"
    flock -n 9 || {
        echo "another runner cache maintenance process is active" >&2
        exit 1
    }
fi

stale_trash=()
mapfile -d '' -t stale_trash < <(
    find "$builds_root" -mindepth 4 \
        -type d -name .git -prune -o \
        -type d -name target -prune -o \
        -type d -name 'target.gc.*' -print0 -prune
)
if (( ${#stale_trash[@]} > 0 )); then
    if [[ "$mode" == dry-run ]]; then
        echo "Found ${#stale_trash[@]} interrupted eviction(s); apply mode would remove them"
    else
        for trash in "${stale_trash[@]}"; do
            echo "Removing interrupted eviction: $trash"
            nice -n 19 ionice -c 3 rm -rf --one-file-system -- "$trash"
        done
    fi
fi

declare -A active_nix_shells=()
nix_shell_root_for_path() {
    local path relative root_name
    path="$(realpath -m "$1")"
    [[ "$path" == "$nix_shell_tmp_root"/nix-shell.* ]] || return 1
    relative="${path#"$nix_shell_tmp_root"/}"
    root_name="${relative%%/*}"
    [[ "$root_name" == nix-shell.* ]] || return 1
    printf '%s/%s\n' "$nix_shell_tmp_root" "$root_name"
}

collect_active_nix_shells() {
    active_nix_shells=()
    local environ value shell
    for environ in /proc/[0-9]*/environ; do
        [[ -r "$environ" ]] || continue
        while IFS= read -r value; do
            [[ -n "$value" ]] || continue
            shell="$(nix_shell_root_for_path "$value" || true)"
            [[ -n "$shell" ]] && active_nix_shells["$shell"]=1
        done < <(
            tr '\0' '\n' 2>/dev/null < "$environ" \
                | awk -F= '$1 ~ /^(TMPDIR|TMP|TEMP|TEMPDIR|NIX_BUILD_TOP)$/ {sub(/^[^=]*=/, ""); print}' \
                || true
        )
    done
    for value in "${TMPDIR:-}" "${TMP:-}" "${TEMP:-}" "${TEMPDIR:-}" "${NIX_BUILD_TOP:-}"; do
        [[ -n "$value" ]] || continue
        shell="$(nix_shell_root_for_path "$value" || true)"
        [[ -n "$shell" ]] && active_nix_shells["$shell"]=1
    done
}

stale_nix_shell_trash=()
mapfile -d '' -t stale_nix_shell_trash < <(
    find "$nix_shell_tmp_root" -xdev -mindepth 1 -maxdepth 1 \
        -type d -name '.nix-shell.gc.*' -print0
)
if (( ${#stale_nix_shell_trash[@]} > 0 )); then
    if [[ "$mode" == dry-run ]]; then
        echo "Found ${#stale_nix_shell_trash[@]} interrupted Nix shell cleanup(s); apply mode would remove them"
    else
        for trash in "${stale_nix_shell_trash[@]}"; do
            echo "Removing interrupted Nix shell cleanup: $trash"
            nice -n 19 ionice -c 3 rm -rf --one-file-system -- "$trash"
        done
    fi
fi

collect_active_nix_shells
stale_nix_shells=()
stale_nix_shell_bytes=0
skipped_active_nix_shells=0
while IFS= read -r -d '' shell; do
    if [[ -n "${active_nix_shells[$shell]:-}" ]]; then
        (( skipped_active_nix_shells += 1 ))
        continue
    fi
    stale_nix_shells+=("$shell")
    shell_bytes="$(du -sx -B1 "$shell" 2>/dev/null | awk 'END {print $1 + 0}' || true)"
    stale_nix_shell_bytes=$(( stale_nix_shell_bytes + shell_bytes ))
done < <(
    find "$nix_shell_tmp_root" -xdev -mindepth 1 -maxdepth 1 \
        -type d -name 'nix-shell.*' -mmin "+$nix_shell_min_age_minutes" -print0
)
stale_nix_shell_size="$(numfmt --to=iec --suffix=B "$stale_nix_shell_bytes")"
echo "Nix shell cleanup: mode=$mode stale=${#stale_nix_shells[@]} allocated=$stale_nix_shell_size active-skipped=$skipped_active_nix_shells min-age=${nix_shell_min_age_minutes}m"

if [[ "$mode" == apply ]]; then
    shell_index=0
    removed_nix_shells=0
    for shell in "${stale_nix_shells[@]}"; do
        [[ -d "$shell" ]] || continue
        collect_active_nix_shells
        if [[ -n "${active_nix_shells[$shell]:-}" ]]; then
            echo "Skipping newly active Nix shell: $shell"
            continue
        fi
        trash="$nix_shell_tmp_root/.nix-shell.gc.$(date +%s).$$.$shell_index"
        (( shell_index += 1 ))
        mv -- "$shell" "$trash"
        nice -n 19 ionice -c 3 rm -rf --one-file-system -- "$trash"
        (( removed_nix_shells += 1 ))
    done
    echo "Removed $removed_nix_shells stale Nix shell(s)"
fi

declare -A active_checkouts=()
collect_active_checkouts() {
    active_checkouts=()
    local environ checkout canonical
    for environ in /proc/[0-9]*/environ; do
        [[ -r "$environ" ]] || continue
        checkout="$(
            tr '\0' '\n' 2>/dev/null < "$environ" \
                | awk -F= '$1 == "BUILDKITE_BUILD_CHECKOUT_PATH" {sub(/^[^=]*=/, ""); print; exit}' \
                || true
        )"
        [[ -n "$checkout" ]] || continue
        canonical="$(realpath -m "$checkout")"
        active_checkouts["$canonical"]=1
    done
    if [[ -n "${BUILDKITE_BUILD_CHECKOUT_PATH:-}" ]]; then
        canonical="$(realpath -m "$BUILDKITE_BUILD_CHECKOUT_PATH")"
        active_checkouts["$canonical"]=1
    fi
}

is_active() {
    [[ -n "${active_checkouts[$1]:-}" ]]
}

last_used_epoch() {
    local checkout="$1" targets="$2"
    local newest=0 candidate timestamp
    for candidate in "$checkout/.git/index"; do
        [[ -e "$candidate" ]] || continue
        timestamp="$(stat -c %Y "$candidate")"
        (( timestamp > newest )) && newest="$timestamp"
    done
    while IFS= read -r target; do
        [[ -n "$target" ]] || continue
        for candidate in "$target/.ci-last-used" "$target"; do
            [[ -e "$candidate" ]] || continue
            timestamp="$(stat -c %Y "$candidate")"
            (( timestamp > newest )) && newest="$timestamp"
        done
    done <<< "$targets"
    echo "$newest"
}

declare -A checkout_targets=()
while IFS= read -r -d '' target; do
    relative="${target#"$builds_root"/}"
    IFS=/ read -r agent organization repository _ <<< "$relative"
    [[ -n "$agent" && -n "$organization" && -n "$repository" ]] || continue
    checkout="$(realpath -e "$builds_root/$agent/$organization/$repository")"
    target="$(realpath -e "$target")"
    checkout_targets["$checkout"]+="$target"$'\n'
done < <(
    find "$builds_root" -mindepth 4 \
        -type d -name .git -prune -o \
        -type d -name 'target.gc.*' -prune -o \
        -type d -name target -print0 -prune
)

records=()
declare -A repo_counts=()
declare -A target_counts=()
for checkout in "${!checkout_targets[@]}"; do
    cache="${checkout#"$builds_root"/}"
    repo="${cache#*/}"
    targets="${checkout_targets["$checkout"]}"
    target_count=0
    while IFS= read -r target; do
        [[ -n "$target" ]] && (( target_count += 1 ))
    done <<< "$targets"
    target_counts["$checkout"]="$target_count"
    records+=("$(last_used_epoch "$checkout" "$targets")"$'\t'"$cache"$'\t'"$repo"$'\t'"$checkout")
    repo_counts["$repo"]=$(( ${repo_counts["$repo"]:-0} + 1 ))
done

free_gib="$(available_gib)"
echo "Runner cache maintenance: mode=$mode free=${free_gib}GiB trigger=${trigger_gib}GiB target=${target_gib}GiB"
echo "Found ${#records[@]} caches; retaining at least $min_retained total and $min_retained_per_repo per repository"

if (( free_gib >= trigger_gib )); then
    echo "No checkout cache cleanup needed"
    exit 0
fi

collect_active_checkouts
mapfile -t ordered < <(printf '%s\n' "${records[@]}" | sort -n)

if [[ "$mode" == dry-run ]]; then
    remaining=${#records[@]}
    (( remaining > min_retained )) || {
        echo "No cache can be evicted without violating the retention floor" >&2
        exit 1
    }
    declare -A remaining_by_repo=()
    for repo in "${!repo_counts[@]}"; do
        remaining_by_repo["$repo"]="${repo_counts["$repo"]}"
    done
    echo "Eviction order:"
    for record in "${ordered[@]}"; do
        (( remaining > min_retained )) || break
        IFS=$'\t' read -r last_used cache repo checkout <<< "$record"
        if (( remaining_by_repo["$repo"] <= min_retained_per_repo )); then
            echo "  skip repository floor: $cache"
            continue
        fi
        if is_active "$checkout"; then
            echo "  skip active: $cache"
            continue
        fi
        printf '  %s last-used=%s targets=%s\n' \
            "$cache" "$(date -d "@$last_used" --iso-8601=seconds)" "${target_counts["$checkout"]}"
        (( remaining -= 1 ))
        remaining_by_repo["$repo"]=$(( remaining_by_repo["$repo"] - 1 ))
    done
    echo "Dry run only; apply mode would evict in this order until ${target_gib}GiB is free"
    exit 0
fi

LOCK_FDS=()
release_target_locks() {
    local fd
    for fd in "${LOCK_FDS[@]}"; do
        eval "exec ${fd}>&-"
    done
    LOCK_FDS=()
}

acquire_checkout_locks() {
    local targets="$1"
    local lock fd
    LOCK_FDS=()
    while IFS= read -r target; do
        [[ -d "$target" ]] || continue
        while IFS= read -r -d '' lock; do
            exec {fd}<>"$lock"
            if ! flock -n "$fd"; then
                eval "exec ${fd}>&-"
                release_target_locks
                return 1
            fi
            LOCK_FDS+=("$fd")
        done < <(find "$target" -xdev -maxdepth 4 -type f -name .cargo-lock -print0)
    done <<< "$targets"
}

retained=${#records[@]}
for record in "${ordered[@]}"; do
    free_gib="$(available_gib)"
    (( free_gib < target_gib )) || break
    (( retained > min_retained )) || break

    IFS=$'\t' read -r last_used cache repo checkout <<< "$record"
    targets="${checkout_targets["$checkout"]}"
    has_target=false
    while IFS= read -r target; do
        [[ -d "$target" ]] && has_target=true
    done <<< "$targets"
    $has_target || continue

    if (( repo_counts["$repo"] <= min_retained_per_repo )); then
        echo "Skipping repository retention floor: $cache"
        continue
    fi
    collect_active_checkouts
    if is_active "$checkout"; then
        echo "Skipping active cache: $cache"
        continue
    fi
    if ! acquire_checkout_locks "$targets"; then
        echo "Skipping locked cache: $cache"
        continue
    fi

    collect_active_checkouts
    if is_active "$checkout"; then
        echo "Skipping newly active cache: $cache"
        release_target_locks
        continue
    fi

    trash_dirs=()
    stamp="$(date +%s).$$"
    echo "Evicting $cache (${target_counts["$checkout"]} targets, last used $(date -d "@$last_used" --iso-8601=seconds))"
    while IFS= read -r target; do
        [[ -d "$target" ]] || continue
        trash="$target.gc.$stamp"
        mv -- "$target" "$trash"
        trash_dirs+=("$trash")
    done <<< "$targets"
    release_target_locks
    for trash in "${trash_dirs[@]}"; do
        nice -n 19 ionice -c 3 rm -rf --one-file-system -- "$trash"
    done
    (( retained -= 1 ))
    repo_counts["$repo"]=$(( repo_counts["$repo"] - 1 ))
    echo "Free space after eviction: $(available_gib)GiB"
done

free_gib="$(available_gib)"
if (( free_gib < target_gib )); then
    if (( free_gib >= trigger_gib )); then
        echo "Stopped at the retention floor with ${free_gib}GiB free; the runner is above the ${trigger_gib}GiB trigger"
        exit 0
    fi
    echo "Could not reach ${target_gib}GiB free without touching active caches or dropping below $min_retained caches" >&2
    exit 1
fi
echo "Runner cache maintenance complete: ${free_gib}GiB free"
