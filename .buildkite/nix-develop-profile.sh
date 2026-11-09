#!/usr/bin/env bash
set -euo pipefail

flake=.
if [[ "${1:-}" != -- ]]; then
    flake="${1:-}"
    shift || true
fi
[[ "${1:-}" == -- ]] || {
    echo "usage: $0 [flake] -- command [args...]" >&2
    exit 2
}
shift
(( $# > 0 )) || {
    echo "a command is required" >&2
    exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
commit="$(git rev-parse HEAD)"
system="$(uname -m)-linux"
case "$flake" in
    . | .#) profile_name=default ;;
    .#*) profile_name="${flake#.#}" ;;
    *) profile_name="$(printf '%s' "$flake" | sha256sum | cut -c1-16)" ;;
esac
profile_name="${profile_name//[^a-zA-Z0-9._-]/_}"
profile_base="${NIX_DEVELOP_PROFILE_ROOT:-"$repo_root/target/nix-profiles"}"
profile_root="$profile_base/$system/$profile_name"
profile="$profile_root/$commit"
retain="${NIX_DEVELOP_PROFILE_RETAIN:-3}"
(( retain > 0 )) || {
    echo "NIX_DEVELOP_PROFILE_RETAIN must be positive" >&2
    exit 2
}

mkdir -p "$profile_root"
exec {lock_fd}>"$profile_root/.lock"
flock "$lock_fd"
if [[ ! -e "$profile" ]]; then
    rm -f -- "$profile" "$profile"-*-link
    echo "Creating Nix develop profile: $profile_name at $commit"
    nix develop "$flake" --profile "$profile" --command true
fi

profile_records=()
for candidate in "$profile_root"/*; do
    [[ -L "$candidate" ]] || continue
    candidate_name="${candidate##*/}"
    [[ "$candidate_name" =~ ^[0-9a-f]{40,64}$ ]] || continue
    profile_records+=("$(stat -c %Y "$candidate")"$'\t'"$candidate_name")
done
mapfile -t ordered_profiles < <(printf '%s\n' "${profile_records[@]}" | sort -rn)
kept=1
for record in "${ordered_profiles[@]}"; do
    candidate_name="${record#*$'\t'}"
    [[ "$candidate_name" != "$commit" ]] || continue
    if (( kept < retain )); then
        (( kept += 1 ))
        continue
    fi
    rm -f -- "$profile_root/$candidate_name" "$profile_root/$candidate_name"-*-link
done
flock -u "$lock_fd"
eval "exec ${lock_fd}>&-"

exec nix develop "$profile" --command "$@"
