# shellcheck shell=bash
# Shared by Linux runner scripts. Source from the same directory as this file.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${here}/../versions.env"

lanes_ok='ci-linux-x64 ci-linux-arm64'

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

detect_runtime() {
    case "$(uname -s)" in
        Darwin)
            command -v container >/dev/null || die 'Apple Container (`container`) is required on macOS'
            printf '%s\n' container
            ;;
        Linux)
            command -v docker >/dev/null || die 'Docker Engine is required on Linux'
            printf '%s\n' docker
            ;;
        *)
            die "unsupported host OS: $(uname -s)"
            ;;
    esac
}

require_lane() {
    local lane="${1:-}"
    case " ${lanes_ok} " in
        *" ${lane} "*) ;;
        *)
            die "usage: $0 ci-linux-x64|ci-linux-arm64"
            ;;
    esac
}

image_tag() {
    printf '%s:%s\n' "${IMAGE_NAME}" "${RUNNER_VERSION}"
}

host_docker_arch() {
    case "$(uname -m)" in
        x86_64 | amd64) printf '%s\n' amd64 ;;
        aarch64 | arm64) printf '%s\n' arm64 ;;
        *) die "unsupported host arch: $(uname -m)" ;;
    esac
}

lane_arch() {
    case "$1" in
        ci-linux-x64) printf '%s\n' amd64 ;;
        ci-linux-arm64) printf '%s\n' arm64 ;;
        *) die "unknown lane $1" ;;
    esac
}

runner_exec() {
    local lane="$1"
    shift
    case "$(detect_runtime)" in
        docker)
            if [[ -n "${RUNNER_TOKEN:-}" ]]; then
                docker exec -u runner -e RUNNER_TOKEN="${RUNNER_TOKEN}" "$lane" "$@"
            else
                docker exec -u runner "$lane" "$@"
            fi
            ;;
        container)
            if [[ -n "${RUNNER_TOKEN:-}" ]]; then
                container exec --user runner --env "RUNNER_TOKEN=${RUNNER_TOKEN}" "$lane" "$@"
            else
                container exec --user runner "$lane" "$@"
            fi
            ;;
    esac
}
