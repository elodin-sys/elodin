# Elodin Simulation Examples

Code patterns extracted from the `examples/` directory. Run any example with:

```bash
elodin editor examples/<name>/main.py
```

## Bouncing Ball — Basic Physics

`examples/ball/` — Gravity, wind, bounce detection with `jax.lax.cond`.

```python
@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())

@el.map_seq
def bounce(pos: el.WorldPos, vel: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        pos.linear()[2] < 0.0,
        lambda v: el.SpatialMotion(linear=v.linear() * jnp.array([1, 1, -0.8])),
        lambda v: v,
        vel,
    )

sys = bounce | el.six_dof(sys=gravity)
w.run(sys, sim_time_step=1.0 / 120.0)
```

Key patterns:
- `@el.map_seq` for conditional logic (`jax.lax.cond` short-circuit)
- `@el.map` for simple vectorized transforms
- Pipe composition: non-effectors before `six_dof`, effectors inside it

## Three-Body Orbit — Graph Queries

`examples/three-body/` — N-body gravitational interaction using `edge_fold`.

```python
GravityEdge = ty.Annotated[el.Edge, el.Component("gravity_edge")]

# Spawn bodies and edges (bidirectional)
for i, j in [(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)]:
    w.spawn(el.Archetype(edge=GravityEdge(el.Edge(i, j))), name=f"{i}_to_{j}")

@el.system
def gravity(graph: el.GraphQuery[GravityEdge],
            q: el.Query[el.WorldPos, el.Inertia]) -> el.Query[el.Force]:
    return graph.edge_fold(
        left_query=q, right_query=q,
        return_type=el.Force, init_value=el.SpatialForce(),
        fold_fn=lambda f, pos_a, m_a, pos_b, m_b: f + gravitational_force(pos_a, m_a, pos_b, m_b),
    )

sys = el.six_dof(sys=gravity)
```

Key patterns:
- Bidirectional edges for symmetric interactions
- `edge_fold` accumulates forces from all connected entities
- `init_value` is the identity element for accumulation

## Rocket — 6DOF with Aerodynamics

`examples/rocket/` — Thrust curves, drag, fin control, PID autopilot.

```python
# Custom components for rocket state
ThrustCurve = ty.Annotated[jax.Array, el.Component("thrust_curve", ...)]
FinDeflect = ty.Annotated[jax.Array, el.Component("fin_deflect", ...)]
AeroCoefs = ty.Annotated[jax.Array, el.Component("aero_coefs", ...)]

# Separate effectors from non-effectors
non_effectors = pid_controller | fin_actuator
effectors = thrust | aerodynamics | gravity

sys = non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.Rk4)
```

Key patterns:
- Effectors (systems producing `el.Force`) go inside `six_dof(sys=...)`
- Non-effectors (controllers, sensors, state machines) go before `six_dof`
- RK4 integrator for higher accuracy in dynamic flight

## CubeSat — Attitude Determination & Control

`examples/cube-sat/` — MEKF estimation, LQR control, reaction wheels, sensors.

```python
# Sensor → estimator → controller → actuator pipeline
sys = (
    sun_sensor | magnetometer | gyroscope     # Sensors
    | mekf_update                              # Kalman filter
    | lqr_controller                           # Control law
    | reaction_wheel_actuator                  # Actuators
    | el.six_dof(sys=gravity_effector)
)

# EGM08 gravity for orbital accuracy
from elodin.egm08 import EGM08
gravity_model = EGM08(max_degree=64)
```

Key patterns:
- Full GNC (Guidance, Navigation, Control) pipeline
- High-fidelity gravity model for LEO
- Kalman filter state stored as custom archetypes

## CubeSat PySim — JAX-Only Mode

`examples/cube-sat-pysim/` — Pure JAX execution for RL training.

```python
sim = w.to_jax(system, sim_time_step=1/120.0)
sim.step(100)
state = sim.get_state("attitude", "satellite")
sim.set_state("attitude", "satellite", new_quaternion)

# Compatible with JAX transformations
jax.vmap(sim.step)(batch_inputs)
```

Key patterns:
- `w.to_jax()` for pure JAX execution (no Rust runtime)
- Direct state get/set for RL reward computation
- `jax.vmap` for parallel environment batching

## Betaflight SITL — Flight Controller Integration

`examples/betaflight-sitl/` — Real Betaflight PID loop at 8kHz.

```python
def post_step(tick: int, ctx: el.StepContext):
    sensor_data = ctx.component_batch_operation(
        reads=["drone.accel", "drone.gyro"]
    )
    motors = betaflight.step(
        accel=sensor_data["drone.accel"],
        gyro=sensor_data["drone.gyro"],
        timestamp=ctx.timestamp,
    )
    ctx.write_component("drone.motor_command", motors)

w.run(sys, sim_time_step=1/8000.0, post_step=post_step)
```

Key patterns:
- `StepContext` for lockstep synchronization
- `component_batch_operation` for efficient multi-read
- High tick rate matching real flight controller frequency

## Drone — Quadcopter Dynamics

`examples/drone/` — Motor mixing, INDI control, multi-rate sensors.

Key patterns:
- Motor mixing matrix for quadcopter torque allocation
- Inner/outer control loops at different rates
- Coordinate system documentation in README

## RC Jet — Full Aircraft Simulation

`examples/rc-jet/` — Polynomial aerodynamics, turbine propulsion, RC controller input.

Key patterns:
- Aerodynamic coefficient lookup tables
- Real-time gamepad/keyboard input via external control
- High-fidelity 6DOF aircraft dynamics

## Video Stream — GStreamer Integration

`examples/video-stream/` — H.264 video streaming to Elodin-DB.

Key patterns:
- GStreamer pipeline for video capture
- Annex B NAL unit streaming to database
- Video overlay in Elodin Editor viewport

## Common Patterns Across Examples

### Entity Naming Convention
Names create hierarchical component paths: spawning with `name="drone"` creates components like `drone.world_pos`, `drone.world_vel`.

### Configuration Pattern
Most examples use a `config.py` for tunable parameters, keeping `main.py` and `sim.py` clean.

### File Organization
```
example/
├── main.py      # Entry point: world setup, spawn, run
├── sim.py       # Physics systems (optional, for larger sims)
├── config.py    # Constants and parameters (optional)
└── README.md    # Usage and background
```
