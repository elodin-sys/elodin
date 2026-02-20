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
w.run(sys, sim_time_step=1.0 / 120.0)
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
    time_step=1/300.0,                # Optional: override sim_time_step
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

w.run(sys, sim_time_step=1/1000.0, post_step=post_step, db_path="sitl_data")
```

Mark components as externally controlled to prevent simulation overwrite:

```python
ThrustCmd = ty.Annotated[jax.Array,
    el.Component("thrust_cmd", el.ComponentType.F64, metadata={"external_control": "true"})]
```

## Execution Modes

| Mode | Command | Use |
|------|---------|-----|
| Editor (GUI) | `elodin editor sim.py` | Development with 3D visualization |
| Headless | `elodin run sim.py` | CI/CD, batch processing |
| JAX-only | `w.to_jax(sys)` | RL training, `jax.vmap` batching |
| Compiled | `w.build(sys, optimize=True)` | Maximum performance |
| Real-time | `w.run(sys, run_time_step=1/120.0)` | Match wall-clock time |
| DB-connected | `w.run(sys, db_addr="0.0.0.0:2240")` | External clients + Editor |

## Earth Gravity Models

```python
from elodin.j2 import J2              # Simple oblate Earth
from elodin.egm08 import EGM08        # High-fidelity spherical harmonics

model = EGM08(max_degree=64)          # <2.5ms at degree <=250
force = model.compute_field(x, y, z, mass)
```

## Additional Resources

- For the full Python API reference, see [api-reference.md](api-reference.md)
- For simulation code patterns from examples, see [examples.md](examples.md)
- Online docs: https://docs.elodin.systems
