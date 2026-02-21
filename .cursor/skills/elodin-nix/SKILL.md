---
name: elodin-nix
description: Work with the Nix development environment and NixOS configurations in Elodin. Use when troubleshooting nix develop, modifying flake.nix, adding Nix packages, setting up OrbStack VMs for Linux builds on macOS, or developing Aleph NixOS modules.
---

# Elodin Nix Development

Elodin uses Nix flakes for reproducible builds, CI dependencies, Docker images, and the Aleph NixOS flight computer configuration.

## Development Shell

The unified dev shell includes all tools for Rust, Python, C/C++, cloud operations, documentation, and git-lfs:

```bash
nix develop                              # Enter interactive shell
nix develop --command "cargo build"      # One-off command
```

No need to switch shells for different tasks — everything is in one environment.

## Nix Installation

Use the [Determinate Systems installer](https://zero-to-nix.com/start/install) (recommended):

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

### Trusted Users

Add your username to trusted users for binary cache access:

**Determinate Nix** — edit `/etc/nix/nix.custom.conf`:
```
trusted-users = root <your_username>
```

**Upstream Nix** — edit `/etc/nix/nix.conf`:
```
experimental-features = nix-command flakes
trusted-users = root <your_username>
```

Restart the daemon:
```bash
# macOS
sudo launchctl kickstart -k system/systems.determinate.nix-daemon
# Linux
sudo systemctl restart nix-daemon.service
```

## Binary Cache

Elodin maintains a binary cache to avoid rebuilding common dependencies:

```
elodin-nix-cache.s3.us-west-2.amazonaws.com
```

This is configured in `flake.nix` and used automatically by `nix develop` and `nix build`.

## Flake Structure

### Root `flake.nix`

Provides:
- **Dev shell**: `nix develop` — unified development environment
- **Packages**: `elodin-py`, `elodin-cli`, `elodin-db`, `elodinsink`
- **Overlay**: `elodinOverlay` with all Elodin packages
- **NixOS configs**: NixOS 25.05 based

Key inputs: nixpkgs, rust-overlay (for toolchain from `rust-toolchain.toml`), crane (Rust builds).

### `aleph/flake.nix`

NixOS configuration for the Aleph flight computer. Composes modules from `aleph/modules/` into a complete system. See the `elodin-aleph` skill for details.

### `aleph/template/flake.nix`

Template for users to create their own Aleph configurations. Imports Elodin's NixOS modules and configures services.

## macOS Linux Builds (OrbStack VM)

Build Linux binaries on macOS using an OrbStack NixOS VM for remote builds:

1. Install [OrbStack](https://orbstack.dev)
2. Create a NixOS VM named `nixos`
3. Configure SSH — add to `/var/root/.ssh/config`:
   ```
   Host orb
     Hostname 127.0.0.1
     Port 32222
   ```
4. Register as build machine — add to `/etc/nix/machines`:
   ```
   ssh://user@orb x86_64-linux,aarch64-linux /Users/user/.orbstack/ssh/id_ed25519 20 20 nixos-test,benchmark,big-parallel,kvm - -
   ```
5. Inside the VM, add to `/etc/nixos/configuration.nix`:
   ```nix
   nix.settings.trusted-users = ["root" "@wheel"];
   ```
6. Rebuild: `sudo nixos-rebuild switch`
7. Test: `nix build --impure --expr '(with import <nixpkgs> { system = "x86_64-linux"; }; runCommand "foo" {} "uname > $out")'`

### OrbStack Disk Exhaustion

Qt builds can overflow `/build` tmpfs. Fix by using disk-backed build directory:

```nix
# In /etc/nixos/configuration.nix on the VM
nix.settings.build-dir = "/var/cache/nix-build";
systemd.tmpfiles.rules = [ "d /var/cache/nix-build 1777 root root -" ];
```

Then:
```bash
sudo install -d -m 1777 /var/cache/nix-build
sudo nixos-rebuild switch
```

## Adding Packages to the Dev Shell

Edit the `devShells` section in the root `flake.nix`. The shell is built from a single unified definition that includes:
- Rust toolchain (from `rust-toolchain.toml`)
- Python 3.12 + uv + maturin
- C/C++ compilers and system libraries
- GStreamer (for video streaming)
- git-lfs
- Cloud operation tools

## Nix Formatting

All `.nix` files are formatted with Alejandra (no configuration):

```bash
alejandra
```

This is enforced by CI.

## Troubleshooting

### `nix develop` hangs or is slow
- First run downloads and builds many dependencies — this is normal
- Check binary cache connectivity
- Ensure `trusted-users` is configured

### Build failures referencing missing system libraries
- These are provided by the Nix shell; make sure you're inside `nix develop`
- For macOS-specific issues, check that Xcode Command Line Tools are installed

### Hash mismatch errors
- Run `nix flake update` to refresh the lock file
- Or pin specific inputs in `flake.lock`

## Key References

- Internal Nix docs: [docs/internal/nix.md](../../../docs/internal/nix.md)
- Root flake: [flake.nix](../../../flake.nix)
- Aleph flake: [aleph/flake.nix](../../../aleph/flake.nix)
- Nix overlay: defined in root `flake.nix` as `elodinOverlay`
