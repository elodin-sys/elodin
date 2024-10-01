# Elodin

## Setup

```sh
# Enable repo-specific git hooks
git config --local core.hooksPath .githooks/
```

### Directory Layout
- `apps/` - contains binaries that users run
- `.buildkite/` - contains everything related to CI pipeline
- [`kubernetes/`](kubernetes/README.md) - contains kubernetes manifests describing the cluster
- `docs/`
  - `public/` - public facing docs
  - `rfcs/` - RFC documents used to propose changes to the project
  - `internal/` - private docs for developers
    - [`release.md`](docs/internal/release.md) - release process
- `libs/` - public and private libraries
- `services/` - hosted services, anything that runs in the cloud

### Supported Operating Systems
- Debian 12+
- Ubuntu 22.04+
- NixOS unstable
- macOS 14 (Sonoma)
