# Elodin

Elodin is an open-source platform for rapid design, testing, and simulation of aerospace and physical systems — aerospace's answer to ROS. This monorepo contains the Elodin Editor (3D viewer/graphing), Elodin DB (time-series telemetry database), the Python SDK (nox-py, JAX-based simulation), flight software components, and the Aleph flight computer NixOS configuration (NVIDIA Jetson Orin).

## Rules

- Always use the `nix develop` shell when developing changes
- Always use `uv` inside the nix shell for Python everything
- Don't commit changes to git — that's for the developer to do
- When suggesting new dependencies, check they are well supported and maintained
- Never use unsafe Rust code
- Run all activities from the repository root

## Quick Start

```bash
nix develop                          # Enter unified dev shell (Rust, Python, C/C++, git-lfs)
just install                         # Build and install Elodin Editor + Elodin DB
```

### Python SDK

```bash
uv venv --python 3.12 && source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
```

### CI Checks

```bash
cargo fmt && cargo test && cargo clippy -- -Dwarnings   # Rust
ruff format --check && ruff check --fix                  # Python
alejandra                                                # Nix
```

## Detailed Guidance

For in-depth instructions, read the relevant skill file below when working in that area:

- **Creating simulations** (Python SDK, components, systems, 6DOF, visualization, SITL/HITL): `.cursor/skills/elodin-simulation/SKILL.md`
- **Contributing to the codebase** (building from source, architecture, workspace structure, testing): `.cursor/skills/elodin-dev/SKILL.md`
- **Aleph flight computer** (AlephOS deployment, NixOS modules, flight software services, firmware): `.cursor/skills/elodin-aleph/SKILL.md`
- **Elodin DB** (running the database, client integrations, replication/follow mode, Lua REPL): `.cursor/skills/elodin-db/SKILL.md`
- **Elodin Editor** (Bevy/Egui architecture, hot-reload, viewport, telemetry graphs, KDL schematics): `.cursor/skills/elodin-editor-dev/SKILL.md`
- **Python SDK internals** (PyO3 bindings, nox-py, JAX integration, adding components/systems): `.cursor/skills/nox-py-dev/SKILL.md`
- **Nix environment** (dev shell troubleshooting, OrbStack VMs, flake.nix, binary cache): `.cursor/skills/elodin-nix/SKILL.md`
