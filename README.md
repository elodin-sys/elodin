# Elodin

### Directory Layout
- `apps` - contains binaries that users run
- `.buildkite` - contains everything related to CI pipeline
- [`kubernetes`](kubernetes/README.md) - contains kubernetes manifests describing the cluster
- `docs/public` - public facing docs
- `docs/internal` & `rfcs` - rfcs, private docs that determine the future of the project
- `libs` - public and private libraries
- `services` - hosted services, anything that runs in the cloud

### Supported Operating Systems
- Debian 12+
- Ubuntu 22.04+
- NixOS unstable
- macOS 14 (Sonoma)
