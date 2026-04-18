---
name: elodin-simulation
description: Create and modify physics simulations using the Elodin Python SDK. Use when writing or editing simulation Python files, defining components or systems, spawning entities, configuring 6DOF physics, setting up visualization, or integrating with SITL/HITL workflows.
---

# Elodin Simulation

Elodin is a JAX-based simulation platform for aerospace and physical systems. Simulations are Python scripts that define a `World`, spawn entities with components, compose systems, and run.

## Installation

```bash
pip install -U elodin          # Released SDK
elodin editor sim.py           # Run with 3D visualization
elodin run sim.py              # Headless execution
python sim.py bench --profile  # Performance profiling
```

## Simulation Structure

Every simulation follows this pattern:

```python
import elodin as el
import jax.numpy as jnp

# 1. Create world
w = el.World()

# 2. Spawn entities with archetypes
w.spawn(el.Body(
    world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 10.0])),
    inertia=el.SpatialInertia(mass=1.0),
), name="ball")

# 3. Define systems
@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=inertia.mass() * jnp.array([0.0, 0.0, -9.81]))

# 4. Compose and run
sys = el.six_dof(sys=gravity, integrator=el.Integrator.Rk4)
w.run(sys, simulation_rate=120.0)
```

## Core Concepts

### Components

Data containers defined with `typing.Annotated` + `el.Component`:

```python
import typing as ty

Wind = ty.Annotated[
    jax.Array,
    el.Component("wind", el.ComponentType(el.PrimitiveType.F64, (3,)),
                 metadata={"element_names": "x,y,z"}),
]
```

Built-in spatial types (`WorldPos`, `WorldVel`, `Force`, `Inertia`, `WorldAccel`) already carry component metadata — no `ComponentType` needed.

### Archetypes

Group components into spawnable bundles:

```python
@el.dataclass
class Satellite(el.Archetype):
    world_pos: el.WorldPos
    world_vel: el.WorldVel
    inertia: el.Inertia
    reaction_wheels: ReactionWheelCmd
```

`el.Body` is the built-in archetype providing `WorldPos`, `WorldVel`, `Inertia`, `Force`, `WorldAccel`.

### Systems

Three decorator levels — choose the simplest that fits:

| Decorator | Use when | Graph queries? |
|-----------|----------|----------------|
| `@el.map` | Simple per-entity transform, vectorized | No |
| `@el.map_seq` | Need `jax.lax.cond` short-circuit behavior | No |
| `@el.system` | Need `Query.map`, `GraphQuery.edge_fold`, or multi-query | Yes |

```python
@el.map
def drag(vel: el.WorldVel) -> el.Force:
    return el.SpatialForce(linear=-0.01 * vel.linear())

@el.system
def gravity(graph: el.GraphQuery[GravityEdge],
            q: el.Query[el.WorldPos, el.Inertia]) -> el.Query[el.Force]:
    return graph.edge_fold(q, q, el.Force, el.SpatialForce(), compute_gravity)
```

### System Composition

Chain systems with the pipe operator — order matters:

```python
sys = sensors | kalman_filter | control | el.six_dof(sys=effectors)
```

### 6DOF Physics

`el.six_dof()` integrates forces/torques into position and velocity:

```python
el.six_dof(
    sys=effectors,                    # Systems computing el.Force
    integrator=el.Integrator.Rk4,     # or Integrator.SemiImplicit
    time_step=1/300.0,                # Optional: override simulation step
)
```

**Inertia is body-frame; all other quantities are world-frame.**

### Graph Queries

Model relationships (gravity, constraints, springs) between entities:

```python
GravityEdge = ty.Annotated[el.Edge, el.Component("gravity_edge")]

w.spawn(el.Archetype(edge=GravityEdge(el.Edge(body_a_id, body_b_id))), name="a_to_b")

@el.system
def gravity(graph: el.GraphQuery[GravityEdge],
            q: el.Query[el.WorldPos, el.Inertia]) -> el.Query[el.Force]:
    return graph.edge_fold(
        left_query=q, right_query=q,
        return_type=el.Force, init_value=el.SpatialForce(),
        fold_fn=lambda acc, pos_a, m_a, pos_b, m_b: acc + compute(pos_a, m_a, pos_b, m_b),
    )
```

## Spatial Vector Algebra

Elodin uses Featherstone spatial vectors. Key types:

| Type | Shape | Represents |
|------|-------|------------|
| `SpatialTransform` | (7,) | Quaternion (4) + position (3) |
| `SpatialMotion` | (6,) | Angular vel (3) + linear vel (3) |
| `SpatialForce` | (6,) | Torque (3) + force (3) |
| `SpatialInertia` | (7,) | Inertia diagonal (3) + mass (1) + padding (3) |

Quaternion operations: `Quaternion.from_axis_angle()`, `q @ vector` (rotate), `q.inverse()`, `q.integrate_body(omega)`.

## Visualization

### KDL Schematics

Define 3D objects and camera views in KDL files or inline:

```kdl
object_3d ball.world_pos {
    sphere radius=0.2 { color 25 50 255 }
}
object_3d aircraft.world_pos {
    glb path="f22.glb" scale=0.01 rotate="(0, 90, 0)"
}
viewport name=Chase pos="drone.world_pos.translate(-5, -5, 3)" look_at="drone.world_pos"
```

### Panel Layout (Python API)

```python
cam = el.Panel.viewport(track_entity=sat_id, fov=45.0, hdr=True, name="3D")
graph = el.Panel.graph(el.GraphEntity(sat_id, *el.Component.index(el.WorldPos)[:4]), name="Position")
w.spawn(el.Panel.vsplit(cam, graph), name="main_view")
```

## SITL/HITL Integration

Use `pre_step`/`post_step` callbacks with `StepContext` for lockstep synchronization:

```python
def post_step(tick: int, ctx: el.StepContext):
    data = ctx.component_batch_operation(reads=["drone.accel", "drone.gyro"])
    motors = flight_controller.step(accel=data["drone.accel"], gyro=data["drone.gyro"])
    ctx.write_component("drone.motor_command", motors)

w.run(sys, simulation_rate=1000.0, post_step=post_step, db_path="sitl_data")
```

Mark components as externally controlled to prevent simulation overwrite:

```python
ThrustCmd = ty.Annotated[jax.Array,
    el.Component("thrust_cmd", el.ComponentType.F64, metadata={"external_control": "true"})]
```

## Execution Modes

| Mode | Command | Backend | Use |
|------|---------|---------|-----|
| Editor (GUI) | `elodin editor sim.py` | cranelift (default) | Development with 3D visualization |
| Headless | `elodin run sim.py` | cranelift (default) | CI/CD, batch processing |
| JAX backend | `w.run(sys, backend="jax-cpu")` | JAX | When cranelift doesn't support certain JAX ops |
| GPU backend | `w.run(sys, backend="jax-gpu")` | JAX GPU | Large parallel workloads |
| JAX-only | `w.to_jax(sys)` | JAX | RL training, `jax.vmap` batching |
| Compiled | `w.build(sys)` | cranelift (default) | Maximum performance |
| Real-time | `w.run(sys, simulation_rate=120.0, generate_real_time=True)` | cranelift (default) | Match wall-clock time |
| DB-connected | `w.run(sys, db_addr="0.0.0.0:2240")` | cranelift (default) | External clients + Editor |

**Backend selection:** The `backend` parameter defaults to `"cranelift"` — a pure-Rust StableHLO JIT that runs the entire tick as a single native function call, with no Python in the hot loop. Use `"jax-gpu"` for high-parallelism workloads that benefit from GPU execution. For tiny worlds, the CPU `cranelift` backend is usually fastest because kernel launch and device-transfer overhead dominates compute.

Use `examples/n-body/main.py` as the canonical benchmark. It runs the supported backends (`cranelift`, `jax-cpu`, `jax-gpu`) side-by-side:

```bash
nix develop --command ELODIN_BACKEND=jax-gpu elodin run examples/n-body/main.py
```

To compare backends, run the same command with `ELODIN_BACKEND` set to each of:
`cranelift`, `jax-cpu`, `jax-gpu`.

## Earth Gravity Models

```python
from elodin.j2 import J2              # Simple oblate Earth
from elodin.egm08 import EGM08        # High-fidelity spherical harmonics

model = EGM08(max_degree=64)          # <2.5ms at degree <=250
force = model.compute_field(x, y, z, mass)
```

## Physics Regression Testing

When changes to the simulation pipeline (Noxpr graph, cranelift-mlir compilation, shape
handling, etc.) might alter numeric output, use a database-export diff to detect
regressions.  The process:

### 1. Capture a baseline on main

```bash
git stash && git checkout main
nix develop
just install

# Run the sim, writing to a dedicated DB path
BALL_DB_PATH=dbs/ball-main uv run examples/ball/main.py bench --ticks 2000

# Export to flat CSVs (--flatten splits vector columns)
elodin-db export --format csv --flatten --output exports/ball-main dbs/ball-main
```

The `BALL_DB_PATH` env var is read by `examples/ball/main.py` and passed to
`world().run(..., db_path=...)`.  Other examples can be wired the same way.

### 2. Capture the branch under test

```bash
git checkout <branch> && git stash pop
nix develop
just install
BALL_DB_PATH=dbs/ball-branch uv run examples/ball/main.py bench --ticks 2000
elodin-db export --format csv --flatten --output exports/ball-branch dbs/ball-branch
```

### 3. Diff component-by-component (ignoring timestamps)

```bash
# Quick pass/fail for every physics component:
for f in ball.world_pos.csv ball.world_vel.csv ball.force.csv ball.wind.csv ball.world_accel.csv; do
    echo -n "$f: "
    diff <(cut -d',' -f2- exports/ball-main/$f) \
         <(cut -d',' -f2- exports/ball-branch/$f) | wc -l
done
```

Zero diff lines = bit-for-bit identical physics.  Non-zero tells you which
component diverged.  Inspect the first differing row to find the tick where
divergence starts and whether it is a large discrete jump (logic bug) or
gradual drift (floating-point).

### 4. Interpret results

| Pattern | Likely cause |
|---------|-------------|
| All zeros | Physics preserved -- safe to land |
| One component diverges at tick 1 | Compilation or shape bug -- the compiled VMFB computes something different |
| Gradual drift accumulating over ticks | Floating-point evaluation order changed (e.g., different StableHLO structure) |
| Only `wind` diverges | Random-key generation changed (seed dtype, PRNG semantics) |

### Tips

- Use `--ticks 2000` (not 1200) to expose drifts that accumulate.
- Run all three canonical benchmarks to cover every code path:
  - `ball` -- single-entity (batch1 path), uses `el.Seed` (U64) + `random.key`
  - `drone` -- multi-entity, uses `sensor_tick` (U64)
  - `cube-sat` -- JAX backend (`backend="jax-cpu"`), covers the non-IREE path
- If you need to dump the StableHLO MLIR for comparison, set
  `ELODIN_IREE_DUMP_DIR=/tmp/debug` before running.
- Clean up temp databases after: `rm -rf dbs/ball-main dbs/ball-branch exports/`.

## Additional Resources

- For the full Python API reference, see [api-reference.md](api-reference.md)
- For simulation code patterns from examples, see [examples.md](examples.md)
- Online docs: https://docs.elodin.systems
