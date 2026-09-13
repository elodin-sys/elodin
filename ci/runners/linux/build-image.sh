#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

runtime="$(detect_runtime)"
tag="$(image_tag)"
arch="$(host_docker_arch)"

# On the Studio, always build the ARM64 image used by ci-linux-arm64.
if [[ "${runtime}" == container ]]; then
    arch=arm64
fi

echo "Building ${tag} for linux/${arch} with ${runtime}"

case "${runtime}" in
    docker)
        docker build \
            --platform "linux/${arch}" \
            --build-arg "TARGETARCH=${arch}" \
            --build-arg "RUNNER_VERSION=${RUNNER_VERSION}" \
            --build-arg "RUNNER_SHA256_LINUX_X64=${RUNNER_SHA256_LINUX_X64}" \
            --build-arg "RUNNER_SHA256_LINUX_ARM64=${RUNNER_SHA256_LINUX_ARM64}" \
            --build-arg "RUST_TOOLCHAIN=${RUST_TOOLCHAIN}" \
            --build-arg "DIST_VERSION=${DIST_VERSION}" \
            --tag "${tag}" \
            "${here}"
        ;;
    container)
        container build \
            --arch arm64 \
            --cpus 4 \
            --memory 4G \
            --build-arg "TARGETARCH=arm64" \
            --build-arg "RUNNER_VERSION=${RUNNER_VERSION}" \
            --build-arg "RUNNER_SHA256_LINUX_X64=${RUNNER_SHA256_LINUX_X64}" \
            --build-arg "RUNNER_SHA256_LINUX_ARM64=${RUNNER_SHA256_LINUX_ARM64}" \
            --build-arg "RUST_TOOLCHAIN=${RUST_TOOLCHAIN}" \
            --build-arg "DIST_VERSION=${DIST_VERSION}" \
            --tag "${tag}" \
            "${here}"
        ;;
esac

echo "Built ${tag}"
