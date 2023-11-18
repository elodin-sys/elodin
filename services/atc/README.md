# (ATC) Air Traffic Control

ATC manages everything to do with running distributed simulations. It is responsible for spinning up containers, managing telemetry, managing permissions, and everything else.

# Usage

You can run ATC by running `cargo run`. There is an [example config file](./config.toml) in this directory

# Build Docker Image

To build the Docker image for this service you will need Nix installed. Then you will need to run one of the following commands:

- `nix run .#docker.x86_64.copyToDockerDaemon` for building an x86_64 image
- `nix run .#docker.aarch64.copyToDockerDaemon` for building an aarch64 image

If you wish to push the image to a registry you can run `nix run .#docker.x86_64.copyTo docker://registry.example.com/atc:latest`
