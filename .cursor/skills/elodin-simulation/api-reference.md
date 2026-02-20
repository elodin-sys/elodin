# Elodin Python API Quick Reference

## World

```python
w = el.World()
w.spawn(archetypes, name="entity_name")        # Returns EntityId
w.insert(entity_id, more_archetypes)           # Add components to existing entity
w.insert_asset(el.Mesh.cuboid(1, 1, 1))        # Returns asset handle
w.shape(mesh_handle, material_handle)           # Create Shape archetype
w.glb("model.glb")                              # Load GLB as Scene archetype
w.run(system, sim_time_step=1/120.0, ...)       # Execute simulation
w.build(system, optimize=True)                  # Build without running
w.to_jax(system, sim_time_step=1/120.0)         # Export as JAX function
```

### `w.run()` parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `system` | System | required | Composed system pipeline |
| `sim_time_step` | float | 1/120.0 | Simulated seconds per tick |
| `run_time_step` | float/None | None (max speed) | Real seconds per tick |
| `default_playback_speed` | float | 1.0 | Editor playback rate |
| `max_ticks` | int | None | Stop after N ticks |
| `optimize` | bool | False | Enable XLA optimizations |
| `pre_step` | Callable | None | `(tick, StepContext) -> None` before each tick |
| `post_step` | Callable | None | `(tick, StepContext) -> None` after each tick |
| `db_path` | str | None | Database directory path |
| `db_addr` | str | None | External DB address (e.g. `"0.0.0.0:2240"`) |
| `interactive` | bool | True | Keep running after max_ticks |
| `start_timestamp` | int | None | Starting timestamp in microseconds |
| `log_level` | str | "info" | Embedded DB log level |

## StepContext (SITL/HITL)

```python
def callback(tick: int, ctx: el.StepContext):
    ctx.tick                                         # Current tick (0-indexed)
    ctx.timestamp                                    # Current timestamp (microseconds)
    ctx.read_component("entity.component")           # -> numpy.ndarray
    ctx.write_component("entity.component", data)    # Write numpy array
    ctx.component_batch_operation(                   # Batch read/write (one lock)
        reads=["drone.accel", "drone.gyro"],
        writes={"drone.motor_cmd": motor_array},
    )
    ctx.truncate()                                   # Clear all data, reset tick to 0
    ctx.stop_recipes()                               # Gracefully stop s10 processes
```

## Components

```python
# Custom scalar
Thrust = ty.Annotated[jax.Array, el.Component("thrust", el.ComponentType.F64)]

# Custom vector
Wind = ty.Annotated[jax.Array,
    el.Component("wind", el.ComponentType(el.PrimitiveType.F64, (3,)),
                 metadata={"element_names": "x,y,z"})]

# External control (simulation won't overwrite)
ExtInput = ty.Annotated[jax.Array,
    el.Component("ext_input", el.ComponentType.F64,
                 metadata={"external_control": "true"})]

# Edge (for graph queries)
MyEdge = ty.Annotated[el.Edge, el.Component("my_edge")]
```

### Built-in Components

| Component | Type | Frame | Shape |
|-----------|------|-------|-------|
| `el.WorldPos` | SpatialTransform | World | (7,) |
| `el.WorldVel` | SpatialMotion | World | (6,) |
| `el.Inertia` | SpatialInertia | **Body** | (7,) |
| `el.Force` | SpatialForce | World | (6,) |
| `el.WorldAccel` | SpatialMotion | World | (6,) |

## Spatial Vector Algebra

```python
# SpatialTransform — quaternion + position
pos = el.SpatialTransform(angular=el.Quaternion.identity(), linear=jnp.zeros(3))
pos.angular()   # -> Quaternion
pos.linear()    # -> jax.Array (3,)

# SpatialMotion — angular velocity + linear velocity
vel = el.SpatialMotion(angular=jnp.zeros(3), linear=jnp.array([0, 7.5e3, 0]))
vel.angular()   # -> jax.Array (3,)
vel.linear()    # -> jax.Array (3,)

# SpatialForce — torque + force
f = el.SpatialForce(torque=jnp.zeros(3), force=jnp.array([0, 0, -9.81]))
f.torque()      # -> jax.Array (3,)
f.force()       # -> jax.Array (3,)

# SpatialInertia — mass + inertia diagonal
inertia = el.SpatialInertia(mass=10.0, inertia=jnp.array([1.0, 1.0, 2.0]))
inertia.mass()          # -> jax.Array scalar
inertia.inertia_diag()  # -> jax.Array (3,)

# Quaternion
q = el.Quaternion.from_axis_angle(jnp.array([0, 0, 1]), jnp.pi / 4)
q.inverse()
q @ vector              # Rotate a 3D vector or spatial type
q.integrate_body(omega) # Quaternion integration with body angular velocity
```

## Systems

```python
# Simple per-entity map (vectorized, no graph queries)
@el.map
def my_system(pos: el.WorldPos, vel: el.WorldVel) -> el.Force:
    return el.SpatialForce(linear=compute(pos, vel))

# Sequential map (preserves jax.lax.cond short-circuit)
@el.map_seq
def conditional_system(pos: el.WorldPos) -> el.Force:
    return jax.lax.cond(pos.linear()[2] < 0, bounce, no_op, pos)

# Low-level system with queries
@el.system
def multi_query(q1: el.Query[el.WorldPos], q2: el.Query[Wind]) -> el.Query[el.Force]:
    return q1.map(el.Force, lambda pos: compute_with_wind(pos, q2))

# Composition
pipeline = system_a | system_b | el.six_dof(sys=system_c)
```

## Query Operations

```python
query.map(ReturnType, lambda comp_a, comp_b: ...)     # Vectorized transform
query.map((TypeA, TypeB), lambda comp: ...)            # Multi-component return
query.map_seq(ReturnType, lambda comp: ...)            # Sequential (cond-safe)
```

## GraphQuery Operations

```python
graph.edge_fold(
    left_query=q,           # Query for left entity of each edge
    right_query=q,          # Query for right entity of each edge
    return_type=el.Force,   # Output component (written to LEFT entity)
    init_value=el.SpatialForce(),
    fold_fn=lambda acc, pos_l, inertia_l, pos_r, inertia_r: acc + compute(...),
)
```

## Visualization

```python
# Meshes and materials
mesh = w.insert_asset(el.Mesh.cuboid(x, y, z))   # or Mesh.sphere(radius)
mat = w.insert_asset(el.Material.color(r, g, b))
w.spawn([el.Body(), el.Shape(mesh, mat)], name="box")

# GLB models
glb = w.insert_asset(el.Glb("model.glb"))
w.spawn([el.Body(), el.Scene(glb)], name="vehicle")

# Panel layout
viewport = el.Panel.viewport(track_entity=eid, fov=45.0, hdr=True, name="3D")
graph = el.Panel.graph(el.GraphEntity(eid, *el.Component.index(el.WorldPos)[:3]), name="Position")
w.spawn(el.Panel.vsplit(viewport, graph), name="layout")
```

## CLI Commands

```bash
elodin editor sim.py                    # GUI with 3D visualization
elodin editor 127.0.0.1:2240            # Connect to running DB
elodin run sim.py                       # Headless execution
python sim.py bench --ticks 100         # Runtime benchmark
python sim.py bench --ticks 100 --profile  # Full complexity analysis
python sim.py components                # Discover components as JSON
```
