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

[[ -d "$builds_root" ]] || {
    echo "Buildkite builds root not found: $builds_root" >&2
    exit 1
}
(( trigger_gib > 0 && target_gib > trigger_gib && min_retained > 0 && min_retained_per_repo > 0 )) || {
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
    local checkout="$1"
    local newest=0 candidate timestamp
    for candidate in \
        "$checkout/target/.ci-last-used" \
        "$checkout/.git/index" \
        "$checkout/target"
    do
        [[ -e "$candidate" ]] || continue
        timestamp="$(stat -c %Y "$candidate")"
        (( timestamp > newest )) && newest="$timestamp"
    done
    echo "$newest"
}

shopt -s nullglob
records=()
declare -A repo_counts=()
for target in "$builds_root"/*/*/*/target; do
    [[ -d "$target" ]] || continue
    checkout="${target%/target}"
    checkout="$(realpath -e "$checkout")"
    cache="${checkout#"$builds_root"/}"
    repo="${cache#*/}"
    records+=("$(last_used_epoch "$checkout")"$'\t'"$cache"$'\t'"$repo"$'\t'"$checkout")
    repo_counts["$repo"]=$(( ${repo_counts["$repo"]:-0} + 1 ))
done

free_gib="$(available_gib)"
echo "Runner cache maintenance: mode=$mode free=${free_gib}GiB trigger=${trigger_gib}GiB target=${target_gib}GiB"
echo "Found ${#records[@]} caches; retaining at least $min_retained total and $min_retained_per_repo per repository"

if (( free_gib >= trigger_gib )); then
    echo "No cleanup needed"
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
        printf '  %s last-used=%s target=%s\n' \
            "$cache" "$(date -d "@$last_used" --iso-8601=seconds)" "$checkout/target"
        (( remaining -= 1 ))
        remaining_by_repo["$repo"]=$(( remaining_by_repo["$repo"] - 1 ))
    done
    echo "Dry run only; apply mode would evict in this order until ${target_gib}GiB is free"
    exit 0
fi

command -v flock >/dev/null || {
    echo "flock is required for apply mode" >&2
    exit 1
}

exec 9>"$builds_root/.runner-cache-maintenance.lock"
flock -n 9 || {
    echo "another runner cache maintenance process is active" >&2
    exit 1
}

LOCK_FDS=()
release_target_locks() {
    local fd
    for fd in "${LOCK_FDS[@]}"; do
        eval "exec ${fd}>&-"
    done
    LOCK_FDS=()
}

acquire_target_locks() {
    local target="$1"
    local lock fd
    LOCK_FDS=()
    while IFS= read -r -d '' lock; do
        exec {fd}<>"$lock"
        if ! flock -n "$fd"; then
            eval "exec ${fd}>&-"
            release_target_locks
            return 1
        fi
        LOCK_FDS+=("$fd")
    done < <(find "$target" -xdev -maxdepth 4 -type f -name .cargo-lock -print0)
}

retained=${#records[@]}
for record in "${ordered[@]}"; do
    free_gib="$(available_gib)"
    (( free_gib < target_gib )) || break
    (( retained > min_retained )) || break

    IFS=$'\t' read -r last_used cache repo checkout <<< "$record"
    target="$checkout/target"
    [[ -d "$target" ]] || continue

    if (( repo_counts["$repo"] <= min_retained_per_repo )); then
        echo "Skipping repository retention floor: $cache"
        continue
    fi
    collect_active_checkouts
    if is_active "$checkout"; then
        echo "Skipping active cache: $cache"
        continue
    fi
    if ! acquire_target_locks "$target"; then
        echo "Skipping locked cache: $cache"
        continue
    fi

    collect_active_checkouts
    if is_active "$checkout"; then
        echo "Skipping newly active cache: $cache"
        release_target_locks
        continue
    fi

    trash="$checkout/target.gc.$(date +%s).$$"
    echo "Evicting $cache (last used $(date -d "@$last_used" --iso-8601=seconds))"
    mv -- "$target" "$trash"
    release_target_locks
    nice -n 19 ionice -c 3 rm -rf --one-file-system -- "$trash"
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
