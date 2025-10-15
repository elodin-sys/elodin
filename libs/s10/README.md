# S10

A process orchestrator and build runner for managing Elodin simulations, flight software, and development workflows.

## Overview

S10 is a task runner designed to streamline the development experience when working with Elodin projects. It provides:
- **Recipe-Based Configuration** - Define reusable tasks in TOML
- **Watch Mode** - Auto-restart on file changes for rapid iteration
- **Process Groups** - Run multiple interdependent processes together
- **Colored Output** - Track multiple process outputs with clear visual distinction
- **Liveness Monitoring** - Ensure processes stay healthy
- **Python/Rust Integration** - Seamlessly run both Python simulations and Rust binaries

## Configuration

S10 uses TOML files to define recipes. It searches for configuration in this order:
1. Command-line specified path (`-c` flag)
2. Environment variable `S10_CONFIG`
3. `./s10.toml` in the current directory
4. `/etc/elodin/s10.toml` system-wide

### Example Configuration

```toml
# Default recipe when none specified
[default]
type = "sim"
path = "../../examples/cube-sat/main.py"

# Cargo project with auto-rebuild
[flight-software]
type = "cargo"
path = "../fsw/sensor-fw"
package = "sensor-fw"
bin = "main"
features = ["telemetry"]
args = ["--port", "8080"]
env = { RUST_LOG = "debug" }
restart_policy = "instant"

# Generic process
[ground-station]
type = "process"
cmd = "python3"
args = ["ground_station.py"]
cwd = "./ground"
no_watch = false
restart_policy = "never"

# Group multiple recipes
[full-stack]
type = "group"
refs = ["flight-software", "ground-station", "default"]

# Direct process definition in group
[hardware-test]
type = "group"
recipes = { sensor = { type = "process", cmd = "./sensor_sim" } }
refs = ["flight-software"]
```

## Recipe Types

### Sim Recipe

Runs Python-based Elodin simulations with automatic `uv` or `python3` detection:

```toml
[my-sim]
type = "sim"
path = "examples/drone/main.py"
addr = "0.0.0.0:2240"  # Optional, default shown
optimize = false       # Optional optimization flag
```

Features:
- Automatic Python environment detection (`uv` preferred, falls back to `python3`)
- Liveness monitoring to detect hangs
- Process group management for clean shutdown

### Cargo Recipe

Builds and runs Rust projects with cargo:

```toml
[my-app]
type = "cargo"
path = "path/to/project"     # Path to project or Cargo.toml
package = "my-package"        # Optional for workspaces
bin = "my-bin"               # Optional binary name
features = ["feature1"]      # Optional features
args = ["--arg1", "value"]  # Runtime arguments
env = { KEY = "value" }      # Environment variables
restart_policy = "instant"   # "instant" or "never"
```

Features:
- Automatic rebuild in watch mode
- Release mode support (`--release` flag)
- Workspace-aware with package selection
- Inherits cargo environment variables

### Process Recipe

Runs arbitrary commands:

```toml
[my-process]
type = "process"
cmd = "my-command"
args = ["arg1", "arg2"]
cwd = "/path/to/dir"        # Working directory
env = { KEY = "value" }     # Environment variables
no_watch = false            # Disable watch mode for this recipe
restart_policy = "instant"  # "instant" or "never"
```

### Group Recipe

Orchestrates multiple recipes:

```toml
[my-group]
type = "group"
refs = ["recipe1", "recipe2"]  # Reference other recipes

# Or define inline
[my-group-inline]
type = "group"
recipes = {
    proc1 = { type = "process", cmd = "echo", args = ["hello"] },
    proc2 = { type = "cargo", path = "./my-app" }
}
```

Features:
- Parallel execution of all recipes in the group
- Terminates all when any recipe fails
- Combines output with colored prefixes

## Command Line Usage

```bash
# Run default recipe
s10

# Run specific recipe
s10 my-recipe

# Run in watch mode (auto-restart on file changes)
s10 --watch

# Build in release mode (for cargo recipes)
s10 --release

# Use custom config file
s10 -c custom.toml my-recipe

# Combine flags
s10 --watch --release flight-software
```

## Features

### Watch Mode

When `--watch` is enabled, S10 monitors the filesystem and automatically restarts recipes when files change:
- Uses debouncing (200ms default) to batch rapid changes
- Respects `.gitignore` patterns
- Recursively watches project directories
- Can be disabled per-recipe with `no_watch = true`

### Colored Output

Each process gets a unique colored prefix in the output:
- Blue for stdout
- Red for stderr
- Process names are bold for clarity
- Exit codes displayed on termination

### Liveness Monitoring

For `sim` recipes, S10 implements a heartbeat system:
- Server sends periodic heartbeats to connected clients
- Automatically terminates hung processes
- Ensures clean shutdown of process groups

### Restart Policies

- **`instant`** - Immediately restart on exit (default)
- **`never`** - Don't restart, useful for one-shot tasks

## Integration with Elodin

S10 is designed to work seamlessly with the Elodin ecosystem:

### Running Simulations

```toml
[drone-sim]
type = "sim"
path = "examples/drone/main.py"
```

This automatically:
1. Detects Python environment (`uv` or `python3`)
2. Passes `--no-s10` flag to prevent recursion
3. Sets up liveness monitoring
4. Manages process groups for clean shutdown

### Flight Software Development

```toml
[fsw]
type = "cargo"
path = "../fsw/my-controller"
args = ["--sim-addr", "localhost:2240"]

[sim]
type = "sim"
path = "sim.py"

[dev]
type = "group"
refs = ["sim", "fsw"]
```

Run with `s10 dev --watch` for a complete development loop with auto-reload.

## Environment Variables

- `S10_CONFIG` - Default config file path
- `RUST_LOG` - Control logging verbosity
- Standard cargo environment variables for Cargo recipes

## Error Handling

S10 provides detailed error messages with helpful diagnostics:
- Missing config files suggest default locations
- Cargo workspace issues suggest adding `package` or `bin`
- Python not found suggests installing `uv` or Python 3.10+
- Process spawn failures show the failing command and recipe

## Examples

### Basic Simulation

```toml
# s10.toml
[default]
type = "sim"
path = "my_sim.py"
```

Run: `s10 --watch`

### Multi-Process Development

```toml
[backend]
type = "cargo"
path = "./backend"
args = ["--port", "8080"]

[frontend]
type = "process"
cmd = "npm"
args = ["run", "dev"]
cwd = "./frontend"

[dev]
type = "group"
refs = ["backend", "frontend"]
```

Run: `s10 dev --watch`

### Hardware-in-the-Loop Testing

```toml
[simulator]
type = "sim"
path = "physics_sim.py"

[hardware-bridge]
type = "cargo"
path = "../hw-bridge"
features = ["serial"]

[controller]
type = "cargo"
path = "../controller"
release = true  # Always build optimized

[hil-test]
type = "group"
refs = ["simulator", "hardware-bridge", "controller"]
```

Run: `s10 hil-test --release`

## Design Philosophy

S10 follows these principles:

1. **Configuration as Code** - TOML recipes are version-controlled and shareable
2. **Fail Fast** - Clear errors with actionable diagnostics
3. **Developer First** - Optimized for the edit-compile-test loop
4. **Composable** - Small recipes combine into complex workflows
5. **Cross-Platform** - Works on Linux, macOS, and Windows (with limitations)

## Limitations

- Process group management (`process_group(0)`) is disabled on Linux due to compatibility issues
- Windows support is limited (no `sim` recipe type)
- Watch mode may not detect all filesystem events on network drives

## Contributing

When adding new recipe types:
1. Define the recipe struct in `src/recipe.rs`
2. Implement `run()` and `watch()` methods
3. Add to the `Recipe` enum
4. Update this README with examples

## Current Usage in Elodin

S10 is integrated throughout the Elodin ecosystem:

### Elodin Editor
The primary consumer of S10. When running `elodin editor main.py`:
- Automatically generates S10 recipes from Python simulations via the `plan` command
- Runs in watch mode by default for live development
- Coordinates simulation and visualization processes

### Python Simulations
Python simulations using `nox-py` can:
- Define recipes programmatically via `WorldBuilder.recipe()`
- Generate S10 configs with `python main.py plan <output_dir>`
- Run with integrated liveness monitoring
- Use `--no-s10` flag to prevent recursive spawning

### Typical Workflow
```bash
# Run simulation with editor (auto-generates S10 recipe)
elodin editor examples/drone/main.py

# Or run standalone with S10
s10 --watch  # Uses default recipe from s10.toml

# Python simulation generates its own recipe
python sim.py plan ./build/
s10 -c ./build/s10.toml sim --watch
```

### Key Integration Points
- **Auto-generation**: Python files can generate S10 recipes on-the-fly
- **Watch by Default**: Editor runs with file watching for rapid iteration
- **Process Coordination**: Manages simulation, visualization, and telemetry together
- **Liveness Monitoring**: Heartbeat system detects and recovers from hung processes
- **Colored Output**: Multiplexes outputs from all processes with clear visual separation

While S10 supports complex orchestration scenarios, its primary role in Elodin is streamlining the simulation development workflow with automatic rebuilding, restarting, and process management.

## License

See the repository's LICENSE file for details.
