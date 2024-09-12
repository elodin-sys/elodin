# (ATC) Air Traffic Control

ATC manages everything to do with running distributed simulations. It is responsible for spinning up containers, managing telemetry, managing permissions, and everything else.

# Usage

## Requirements

- [`Rust`](https://www.rust-lang.org/tools/install)
- [`protoc`](https://grpc.io/docs/protoc-installation/)

## Run locally
*Note:* local development in the follow manner doesn't work on Apple Silicon based Macs, use Windows:
- Kube config for the `vms` pods requires kvm, which doesn't exist in orbstack yet (coming soon)
- XLA doesn’t support running on Linux VMs on Apple Silicon

1. Start and set up the `PostgreSQL`, create the `atc_dev` database (`psql -c 'create database atc_dev;'`)
2. Start the `Redis`.
3. Create a local Kubernetes cluster: [`kind`](https://kind.sigs.k8s.io/)(windows) / [`orbstack`](https://orbstack.dev/)(mac)
4. Ceate the `vms` namespace in your new cluster
5. Set up the [`config file`](config.toml). Make sure that you've properly specified the database_url, redis_url and pod_name fields.
6. Make sure you have set up the [`GOOGLE_APPLICATION_CREDENTIALS env`](https://cloud.google.com/docs/authentication/provide-credentials-adc).
7. Run `cargo run` in this directory.

## Build Docker Image

To build the Docker image for this service you will need Nix installed. Then you can run:

``` sh
docker load -i $(nix build .#packages.aarch64-linux.docker.image --print-out-paths) # for arm
docker load -i $(nix build .#packages.x86_64-linux.docker.image --print-out-paths) # for x86
```

To build on macOS you will need to follow the instructions in [docs/nix.md](../../docs/internal/nix.md) under macOS VM
