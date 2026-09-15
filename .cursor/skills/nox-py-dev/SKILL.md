---
name: nox-py-dev
description: Contribute to the Elodin Python SDK (nox-py). Use when editing PyO3 bindings in libs/nox-py/, adding new Python API surface, modifying the JAX integration, working on component/system compilation, or changing the Python package in python/elodin/.
---

# nox-py Development

nox-py is the Elodin Python SDK — PyO3 bindings that bridge Python simulations to the Rust ECS engine (in nox-py/src/), the NOX tensor compiler (→ cranelift / JAX), and Impeller2 telemetry.

## Build & Test

```bash
just install py

# Run tests
pytest libs/nox-py/tests/

# Quick verification with an example
elodin editor examples/three-body/main.py
```

Rebuild the wheel after any Rust changes. Python-only changes in `python/elodin/` are picked up immediately.

## Architecture

```
Python user code
    │
    ▼
python/elodin/__init__.py    ← Python API surface, decorators, Query/GraphQuery
    │
    ▼
libs/nox-py/src/lib.rs       ← PyO3 module registration
    │
    ├── world_builder.rs      ← World creation, spawn, run
    ├── system.rs             ← System compilation pipeline
    ├── component.rs          ← Component types and metadata
    ├── archetype.rs          ← Archetype definitions
    ├── query.rs              ← Query system (map, map_seq)
    ├── graph.rs              ← GraphQuery and edge_fold
    ├── spatial.rs            ← SpatialTransform, SpatialMotion, SpatialForce, SpatialInertia
    ├── entity.rs             ← EntityId management
    ├── exec.rs               ← WorldExec enum (Iree/Jax), profiling, DB integration
    ├── jax_exec.rs           ← JaxExec, JaxWorldExec (JAX JIT per-tick execution)
    ├── step_context.rs       ← StepContext for pre/post step callbacks
    ├── impeller_client.rs    ← Impeller2 client for DB connection
    ├── asset.rs              ← Mesh, Material, GLB asset handling
    ├── linalg.rs             ← Linear algebra utilities
    ├── ukf.rs                ← Unscented Kalman Filter
    ├── s10.rs                ← S10 recipe integration
    └── error.rs              ← Error types and Python exception mapping
    │
    ▼
libs/nox/                     ← Tensor library, symbolic backend
    │
    ▼
libs/cranelift-mlir/          ← Pure Rust StableHLO runtime
```

## Python API Surface

### `python/elodin/__init__.py`

The main API module. Exports:
- Decorators: `@system`, `@map`, `@map_seq`
- Types: `World`, `Query`, `GraphQuery`, `Component`, `Archetype`, `Body`
- Spatial types: `SpatialTransform`, `SpatialMotion`, `SpatialForce`, `SpatialInertia`, `Quaternion`
- Built-in components: `WorldPos`, `WorldVel`, `Inertia`, `Force`, `WorldAccel`
- Utilities: `six_dof()`, `Panel`, `GraphEntity`, `Mesh`, `Material`, `Edge`

### `python/elodin/jaxsim.py`

JAX-only execution mode. Compiles the simulation world into pure JAX functions for RL training and `jax.vmap` batching.

### `python/elodin/egm08.py` / `python/elodin/j2.py`

Earth gravity models. EGM08 is a high-fidelity spherical harmonics model; J2 is simple oblate Earth.

## Key Rust Modules

### `world_builder.rs`
Central orchestrator. Handles `World.spawn()`, `World.insert()`, `World.run()`, `World.build()`, `World.to_jax()`. This is where simulation execution modes branch.

### `system.rs`
Compiles Python-defined systems into executable computations via cranelift (default) or JAX. Handles the `@system`, `@map`, `@map_seq` decorator logic on the Rust side. System composition (pipe `|`) is implemented here.

### `exec.rs`
Defines `WorldExec` enum with `Cranelift(CraneliftWorldExec)` and `Jax(JaxWorldExec)` variants. Both implement the same interface for tick execution, profiling, and DB integration. The `backend` parameter in `w.run()` / `w.build()` selects between the cranelift and JAX execution modes.

### `jax_exec.rs`
JAX backend: compiles Noxpr graph → `jax.jit()` callable, then executes each tick by calling the JAX function via PyO3. Slower than cranelift but supports all JAX operations and the GPU.

### `component.rs`
Maps Python `Component` annotations to component schemas. Handles type inference, metadata, and the `ComponentType` / `PrimitiveType` hierarchy.

### `spatial.rs`
PyO3 bindings for spatial vector algebra types. Each type wraps a nox tensor and exposes Python-friendly constructors, accessors, and arithmetic operators.

### `graph.rs`
Graph query implementation. `edge_fold` is the core operation — it iterates edges, queries left/right entity components, and accumulates results.

## Adding a New Feature

### New Component Type

1. Define the Rust type in `component.rs` or a new module
2. Add PyO3 `#[pyclass]` bindings
3. Export in `lib.rs` module registration
4. Add Python-side type alias in `python/elodin/__init__.py`
5. Test: spawn an entity with the component, verify it appears in the editor

### New System Decorator

1. Implement the Rust compilation logic in `system.rs`
2. Add the Python decorator in `python/elodin/__init__.py`
3. Ensure it composes with `|` (pipe operator)
4. Test: write a simulation using the new decorator, verify output

### New Execution Mode

1. Add the mode in `exec.rs` (or `world_builder.rs` for world-level API)
2. Wire CLI support in `apps/elodin/` if needed
3. Add Python API in `world_builder.rs`

## Dependencies

| Crate | Purpose |
|-------|---------|
| `pyo3` | Python ↔ Rust bindings |
| `numpy` | NumPy array interop (via pyo3-numpy) |
| nox-py (Rust core) | ECS world, component storage, system execution |
| `nox` | Tensor library, spatial math |
| `impeller2` | Telemetry protocol |
| `stellarator` | Async runtime for DB connections |
| `tokio` | Async runtime (for some I/O paths) |

## Key References

- Full SDK documentation: [libs/nox-py/README.md](../../../libs/nox-py/README.md)
- Python API reference: [docs/public/content/reference/python-api.md](../../../docs/public/content/reference/python-api.md)
- nox-py Rust source: [libs/nox-py/src/](../../../libs/nox-py/src/)
- Impeller2 protocol: [libs/impeller2/](../../../libs/impeller2/)
