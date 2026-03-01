---
name: elodin-dev
description: Develop and contribute to the Elodin codebase. Use when building Elodin from source, running tests, modifying core libraries, working on the Rust workspace, or onboarding as a contributor.
---

# Elodin Development

Elodin is a monorepo for aerospace simulation and flight software. The stack:

- **nox-py** — Python SDK (JAX + PyO3 bindings; includes ECS in `src/`)
- **nox** — Tensor compiler (→ IREE / JAX)
- **Impeller2** — High-performance pub-sub telemetry protocol
- **Elodin-DB** — Time-series telemetry database
- **Elodin Editor** — 3D viewer and graphing tool (Bevy + Egui)
- **Roci** — Reactive flight software framework
- **Aleph** — NixOS configuration for Jetson Orin flight computers

## Architecture

```
Python Simulations (nox-py)
        │
   ┌────┴────┬──────────────┐
   │         │              │
 NOX      Impeller2     Elodin-DB
Compiler  (Telemetry)   (Storage)
   │         │              │
 IREE /   Stellarator    Elodin
  JAX     (Async RT)     Editor
                │
         ┌──────┴──────┐
       Roci          Aleph
    (Flight SW)   (Hardware)
```

Key integration points:
1. nox-py → nox → IREE (default) or JAX (simulation compilation)
2. nox-py → impeller2 → elodin-db (telemetry)
3. elodin-editor → impeller2 → elodin-db (visualization)
4. roci → impeller2 → elodin-db (flight software telemetry)

## Prerequisites

- [Determinate Systems Nix](https://determinate.systems/nix-installer/)
- [Just](https://just.systems/) (`brew install just` / `apt install just`)
- [git-lfs](https://git-lfs.com/) (`git lfs install` globally)
- Arm macOS strongly preferred for build speed

## Development Environment

**Always work inside the Nix shell.** It provides Rust, Python, C/C++ toolchains, cloud tools, and git-lfs.

```bash
nix develop                              # Enter unified dev shell
nix develop --command "cargo build"      # One-off command
```

## Build Commands

```bash
# Editor + DB (installs to ~/.nix-profile/bin)
just install

# Python SDK wheel (for running/testing simulations)
uv venv --python 3.12
source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml

# Run an example
elodin editor examples/three-body/main.py
```

## CI Checks

All changes must pass these before merge. See [ci-checks.md](ci-checks.md) for details.

```bash
cargo fmt                                 # Rust formatting
cargo test                                # Rust tests
cargo clippy -- -Dwarnings                # Rust lints (warnings = errors)
ruff format --check && ruff check --fix   # Python formatting + lints
alejandra                                 # Nix formatting
```

## Workspace Structure

The Cargo workspace has 57 members. Key crates by area:

| Area | Crates |
|------|--------|
| Simulation | `nox`, `elodin-macros`, `nox-py`, `nox-frames`, `iree-runtime` |
| Database | `db`, `db/cli`, `db/eql`, `db/tests` |
| Telemetry | `impeller2`, `impeller2/{bevy,stellar,bbq,frame,kdl,wkt}` |
| Editor | `elodin-editor`, `apps/elodin` |
| Runtime | `stellarator`, `stellarator/{buf,macros,maitake}` |
| Flight SW | `roci`, `roci/{macros,adcs}` |
| FSW Apps | `serial-bridge`, `mekf`, `msp-osd`, `lqr`, `blackbox`, `gstreamer`, `video-streamer` |
| Utilities | `wmm`, `s10`, `video-toolbox`, `muxide` |

## Working from Repository Root

Always run commands from the repo root. This prevents path confusion across the many workspace members and aligns with how CI runs.

## Component-Specific Skills

For deeper work on specific areas, see:
- **Editor**: `.cursor/skills/elodin-editor-dev/`
- **Python SDK**: `.cursor/skills/nox-py-dev/`
- **Database**: `.cursor/skills/elodin-db/`
- **Aleph/NixOS**: `.cursor/skills/elodin-aleph/`
- **Nix environment**: `.cursor/skills/elodin-nix/`
