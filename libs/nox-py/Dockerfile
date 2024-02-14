From ubuntu:22.04

Run apt-get update && \
    apt-get install -y python3 python3-venv python3-pip && \
    apt-get install -y curl

Run curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --profile default --target wasm32-unknown-unknown --default-toolchain stable -y

Run pip install maturin patchelf jax jaxlib typing_extensions

Env PATH /root/.cargo/bin:$PATH
