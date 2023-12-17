# Elodin Web Runner

## Build Docker Image

To build the Docker image for this service you will need Nix installed. Then you can run:

``` sh
docker load -i $(nix build .#packages.aarch64-linux.docker.image --print-out-paths) # for arm
docker load -i $(nix build .#packages.x86_64-linux.docker.image --print-out-paths) # for x86
```

To build on macOS you will need to follow the instructions in [docs/nix.md](../../docs/nix.md) under macOS VM
