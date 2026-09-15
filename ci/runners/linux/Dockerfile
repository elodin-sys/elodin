# Runner image for ci-linux-x64 (Docker) and ci-linux-arm64 (Apple Container).
# Ubuntu 24.04 keeps the same glibc baseline as the previous hosted runners.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG RUNNER_VERSION=2.337.0
ARG RUNNER_SHA256_LINUX_X64=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
ARG RUNNER_SHA256_LINUX_ARM64=9b1dc70626422526e3c94767cf024896beb15da5342a3f4819bf2feac13e0393
ARG RUST_TOOLCHAIN=1.98.0
ARG DIST_VERSION=0.28.0
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CARGO_HOME=/home/runner/.cargo \
    RUSTUP_HOME=/home/runner/.rustup \
    PATH=/home/runner/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs jq unzip zip xz-utils tar bash tini \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake ninja-build pkg-config file patchelf \
    libssl-dev libopenblas-dev liblzma-dev \
    g++ \
    libx11-dev libasound2-dev libudev-dev libxkbcommon-x11-0 \
    libwayland-dev libxkbcommon-dev \
    gfortran gfortran-12 libgfortran-12-dev \
    protobuf-compiler \
    libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev \
    libavfilter-dev libswscale-dev libswresample-dev libclang-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1001 --shell /bin/bash runner \
    && install -d -o runner -g runner /opt/actions-runner

WORKDIR /opt/actions-runner
RUN case "${TARGETARCH}" in \
        amd64) runner_arch=linux-x64; runner_sha="${RUNNER_SHA256_LINUX_X64}" ;; \
        arm64) runner_arch=linux-arm64; runner_sha="${RUNNER_SHA256_LINUX_ARM64}" ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && [[ "${RUNNER_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    && curl --fail --show-error --location --retry 3 \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-${runner_arch}-${RUNNER_VERSION}.tar.gz" \
        --output /tmp/runner.tar.gz \
    && printf '%s  %s\n' "${runner_sha}" /tmp/runner.tar.gz | sha256sum --check - \
    && tar -xzf /tmp/runner.tar.gz -C /opt/actions-runner \
    && ./bin/installdependencies.sh \
    && rm -f /tmp/runner.tar.gz \
    && rm -rf /var/lib/apt/lists/* \
    && printf '%s\n' /home/runner/.cargo/bin > /opt/actions-runner/.path \
    && chown -R runner:runner /opt/actions-runner

USER runner
ARG TARGETARCH
ARG RUST_TOOLCHAIN
ARG DIST_VERSION
WORKDIR /home/runner
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}" --profile default \
    && rustup target add \
        "$( [ "${TARGETARCH}" = amd64 ] && echo x86_64-unknown-linux-musl || echo aarch64-unknown-linux-musl )" \
    && cargo install cargo-deb \
    && curl --proto '=https' --tlsv1.2 -LsSf \
        "https://github.com/axodotdev/cargo-dist/releases/download/v${DIST_VERSION}/cargo-dist-installer.sh" \
    | sh \
    && git lfs install --skip-repo

USER root
COPY entrypoint.sh /usr/local/bin/ci-runner-entrypoint
RUN chmod 0755 /usr/local/bin/ci-runner-entrypoint \
    && chown runner:runner /usr/local/bin/ci-runner-entrypoint
USER runner
WORKDIR /opt/actions-runner
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/usr/local/bin/ci-runner-entrypoint"]
