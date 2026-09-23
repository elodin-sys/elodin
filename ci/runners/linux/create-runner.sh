#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

lane="${1:-}"
require_lane "${lane}"
runtime="$(detect_runtime)"
tag="$(image_tag)"
volume="${lane}-work"

if [[ "${runtime}" == docker && "$(lane_arch "${lane}")" != "$(host_docker_arch)" ]]; then
    die "${lane} does not match this host arch ($(uname -m))"
fi
if [[ "${runtime}" == container && "${lane}" != ci-linux-arm64 ]]; then
    die 'Apple Container on the Studio hosts ci-linux-arm64 only'
fi

echo "Creating ${lane} from ${tag} (${runtime})"

case "${runtime}" in
    docker)
        if docker inspect "${lane}" >/dev/null 2>&1; then
            die "container ${lane} already exists; remove it first if you intend to recreate"
        fi
        docker volume create "${volume}" >/dev/null
        docker run --rm --user 0 \
            --mount "type=volume,source=${volume},target=/work" \
            --entrypoint /bin/sh \
            "${tag}" -c 'chown 1001:1001 /work'
        docker run --detach \
            --name "${lane}" \
            --restart unless-stopped \
            --env "CARGO_BUILD_JOBS=$(nproc)" \
            --mount "type=volume,source=${volume},target=/opt/actions-runner/_work" \
            "${tag}"
        ;;
    container)
        if container inspect "${lane}" >/dev/null 2>&1; then
            die "container ${lane} already exists; remove it first if you intend to recreate"
        fi
        container volume create --opt size=128g "${volume}"
        container run --rm --user 0 \
            --mount "type=volume,source=${volume},target=/work" \
            --entrypoint /bin/sh \
            "${tag}" -c 'chown 1001:1001 /work'
        container run --detach \
            --name "${lane}" \
            --arch arm64 \
            --cpus 6 \
            --memory 16G \
            --env CARGO_BUILD_JOBS=6 \
            --mount "type=volume,source=${volume},target=/opt/actions-runner/_work" \
            "${tag}"
        ;;
esac

echo "Created ${lane}. Register it with: RUNNER_TOKEN=... ${here}/register.sh ${lane}"
