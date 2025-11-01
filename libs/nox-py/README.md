# nox-py (Elodin Python SDK)

The Python interface for Elodin - a platform for rapid design, testing, and simulation of aerospace control systems.

## Overview

nox-py provides a high-performance Python API for building aerospace simulations that leverage:
- **JAX Integration** - GPU-accelerated, differentiable physics simulations
- **Rust Performance** - Critical paths compiled through the NOX backend for maximum speed
- **Flexible Execution** - Multiple runtime modes from development to production
- **Real-Time Visualization** - Live 3D rendering and telemetry through the Elodin Editor
- **Distributed Systems** - Connect to hardware-in-the-loop testing and flight computers

## Architecture

```
┌─────────────────────────────────────────┐
│           Python User Code              │
│  (Systems, Components, World)           │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│          nox-py (PyO3 Bindings)         │
│  - Hierarchical Component System        │
│  - Graph-based Systems                  │
│  - Spatial Math (Quaternions, etc.)     │
└────────────┬────────────────────────────┘
             │
      ┌──────┴─────┬───────────────┐
      │            │               │
┌─────▼─────┐ ┌────▼────┐ ┌────────▼─────┐
│    NOX    │ │   JAX   │ │  Impeller2   │
│  Compiler │ │  (XLA)  │ │  (Pub-Sub)   │
└───────────┘ └─────────┘ └───────┬──────┘
                                  │
                          ┌───────▼───────┐
                          │  Elodin-DB    │
                          │ (Time-Series) │
                          └───────────────┘
```

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/elodin-sys/elodin
cd elodin/libs/nox-py

# Create virtual environment
uv venv
source .venv/bin/activate  # or activate.fish for fish shell

# Install with maturin
uvx maturin develop --uv
```

### Quick Start Examples
Once installed, try these examples:
```bash
# From repository root
elodin editor examples/three-body/main.py  # N-body orbital mechanics
elodin editor examples/cube-sat/main.py    # Satellite attitude control
elodin editor examples/rocket/main.py      # 6DOF rocket simulation
elodin editor examples/ball/main.py        # Physics with collisions
elodin editor examples/drone/main.py       # Quadcopter dynamics
```

## Execution Modes

nox-py supports multiple execution modes to fit different workflows:

### 1. Standard Python Script
Run simulations like any Python script:
```python
# sim.py
import elodin as el

w = el.World()
# ... build simulation ...
w.run(system, sim_time_step=1/120.0)
```

```bash
python sim.py
```

### 2. Elodin Editor (With GUI)
Launch with real-time 3D visualization and telemetry:
```bash
elodin editor sim.py
```

Features:
- Live 3D rendering with camera controls
- Real-time telemetry graphs
- Component inspector
- Simulation controls (play/pause/step)
- Hot-reload on file changes

### 3. Headless Mode
Run simulations without GUI (for CI/CD, batch processing):
```bash
elodin run sim.py
```

### 4. JAX-Only Mode (Pure Python)
Bypass the Rust NOX compiler for JAX-native execution:
```python
# Useful for JAXMarl integration and custom JAX transformations
sim = w.to_jax(system, sim_time_step=1/120.0)

# Step simulation
sim.step(100)

# Get/set component states
state = sim.get_state("attitude", "satellite")
sim.set_state("attitude", "satellite", new_quaternion)

# Compatible with JAX transformations
jax.vmap(sim.step)(batch_inputs)
```
*See full JAX-only example: [../../examples/cube-sat-pysim/main.py](../../examples/cube-sat-pysim/main.py)*

### 5. Compiled Mode
Pre-compile for maximum performance:
```python
exec = w.build(system, optimize=True)
exec.run(1000)  # Run 1000 ticks
df = exec.history(["satellite.world_pos", "satellite.world_vel"])
```

### 6. External Control Mode (Real-Time with External Inputs)
Enable external clients to control simulation components:
```python
# Define components with external control metadata
ControlInput = ty.Annotated[
    jax.Array, 
    el.Component("control_input", el.ComponentType.F64, 
                 metadata={"external_control": "true"})
]

# Run simulation in real-time with DB connection
w.run(system, 
      sim_time_step=1/120.0,     # Simulation timestep
      run_time_step=1/120.0,      # Real-time execution rate
      db_addr="0.0.0.0:2240")    # Listen for external connections
```

External clients can then connect and control these components in real-time. See the [Rust client example](../db/examples/rust_client/README.md) for a complete implementation.

## Performance Profiling

Analyze and optimize your simulation's computational complexity with the built-in profiler:

```bash
# Quick profile - static analysis only
python sim.py profile

# With runtime metrics
python sim.py profile --ticks 100

# Deep analysis - optimization opportunities
python sim.py profile --deep

# Generate interactive HTML visualization
python sim.py profile --html --deep
```

### Profile Output

The profiler provides:
- **Compilation metrics** - Build and compile times
- **HLO instruction analysis** - Operation counts and types
- **Memory footprint** - Per-component breakdown
- **Hot spot identification** - Python lines generating the most operations (with `--deep`)
- **Actionable recommendations** - Ranked optimization opportunities (with `--deep`)

Example output:
```
[Hot Spots in Python Code]
  main.py:339 - 504 ops
    Code: coefs = jnp.array([map_coordinates(coef, coords, 1, mode="nearest") for coef in aero])

[Detected Optimization Patterns]
1. [HIGH] Heavy shape manipulation (21.6% of ops)
   • Review tensor shape consistency across function boundaries
   • Consider using einsum instead of reshape + matmul chains
```

### When to Use

- **Development**: Quick `profile` to check complexity of new systems
- **Optimization**: `profile --deep` to identify bottlenecks and get specific recommendations
- **Debugging**: `profile --html` for visual inspection of the computation graph
- **Benchmarking**: `profile --deep --ticks 1000` for complete static + runtime analysis

Output files are saved to `profile_output/` directory next to your simulation file for easy access.

## Database Integration

### Embedded Database (Default)
Simulations automatically create a temporary database:
```python
w.run(system)  # Creates temp DB in system temp directory
```

### Connect to Existing Database
```python
# Start external database
# $ elodin-db run [::]:2240 ~/.elodin/db

# Connect from simulation
w.run(system, db_addr="127.0.0.1:2240")
```

### Save Simulation Data
```python
exec = w.build(system)
exec.run(1000)
exec.save_archive("simulation.arrow", format="arrow")  # Also: "parquet", "csv"
```

## Component Discovery

Discover all components and entities in a simulation without running it:

```bash
# Output JSON with all components and entities
python examples/rocket/main.py components
```

**Output Example:**
```json
{
  "components": [
    {
      "name": "world_pos",
      "type": "f64",
      "shape": [7],
      "metadata": {
        "element_names": "q0,q1,q2,q3,x,y,z",
        "priority": "5"
      }
    },
    {
      "name": "thrust",
      "type": "f64",
      "metadata": {
        "priority": "17"
      }
    }
    // ... more components
  ],
  "entities": [
    {
      "id": 1,
      "name": "rocket",
      "components": ["world_pos", "thrust", "fin_control", ...]
    },
    {
      "id": 0,
      "name": "Globals",
      "components": ["tick", "simulation_time_step"]
    }
  ],
  "total_components": 26,
  "total_entities": 2
}
```

This feature is useful for:
- Understanding simulation structure before running
- Integration with external tools (pipe to `jq` for queries)
- Generating documentation of simulation interfaces
- Debugging component registration and entity relationships
- Discovering which components belong to which entities

## Core Concepts

### World
Container for all components and systems:
```python
w = el.World()
```

### Hierarchical Components & Archetypes
Components are organized hierarchically using dot notation. When you spawn an archetype with a name, that name becomes the root of the component tree:
```python
import elodin as el
import jax.numpy as jnp

@el.dataclass
class Satellite(el.Archetype):
    world_pos: el.WorldPos
    world_vel: el.WorldVel
    inertia: el.Inertia
    
# Creates components: "satellite.world_pos", "satellite.world_vel", "satellite.inertia"
w.spawn(Satellite(
    world_pos=el.SpatialTransform(linear=jnp.array([7000e3, 0, 0])),
    world_vel=el.SpatialMotion(linear=jnp.array([0, 7.5e3, 0])),
    inertia=el.SpatialInertia(mass=500.0)
), name="satellite")
```

### Systems
Transform components each tick:
```python
@el.system
def gravity(q: el.Query[el.WorldPos, el.Inertia]) -> el.Query[el.Force]:
    return q.map(el.Force, lambda pos, inertia: 
        calculate_gravity(pos, inertia))

@el.map
def drag(vel: el.WorldVel) -> el.Force:
    return el.Force(linear=-0.01 * vel.linear())
```

### System Composition
Combine systems with pipes:
```python
physics = gravity | drag | el.six_dof(integrator=el.Integrator.Rk4)
w.run(physics, sim_time_step=1/120.0)
```

### Graph Systems
Handle relationships between named components:
```python
Edge = el.Annotated[el.Edge, el.Component("constraint")]

@el.system
def spring_force(
    graph: el.GraphQuery[Edge],
    q: el.Query[el.WorldPos]
) -> el.Query[el.Force]:
    return graph.edge_fold(
        left_query=q,
        right_query=q,
        return_type=el.Force,
        init_value=el.Force(),
        fold_fn=spring_calculation
    )
```

## External Control & Hardware-in-the-Loop

nox-py supports external control of simulation components, enabling hardware-in-the-loop testing and real-time control system development.

### Setting Up External Control

1. **Define External Control Components**:
```python
import typing as ty
import elodin as el
import jax.numpy as jnp

# Component with external_control metadata won't be written back by simulation
ThrustCommand = ty.Annotated[
    jax.Array,
    el.Component("thrust_command", el.ComponentType.F64,
                 metadata={"external_control": "true"})
]

@el.dataclass
class Spacecraft(el.Archetype):
    # External control input
    thrust_command: ThrustCommand = field(default_factory=lambda: jnp.float64(0.0))
    # Regular simulation state
    world_pos: el.WorldPos
    world_vel: el.WorldVel
```

2. **Use in Physics Systems**:
```python
@el.map
def apply_thrust(thrust_cmd: ThrustCommand, force: el.Force) -> el.Force:
    # External control directly affects physics
    return force + el.Force(linear=jnp.array([thrust_cmd, 0, 0]))
```

3. **Run in Real-Time Mode**:
```python
# Match simulation rate to real-time for responsive control
w.run(physics_system,
      sim_time_step=1/120.0,     # 120Hz simulation
      run_time_step=1/120.0,      # Real-time playback
      db_addr="0.0.0.0:2240")     # Accept external connections
```

### Connecting External Controllers

External controllers (flight computers, ground stations, test harnesses) can connect via:
- **TCP/UDP**: Using Impeller2 protocol
- **Shared Memory**: For local hardware-in-the-loop
- **Serial**: For embedded systems

Example with Rust client:
```rust
// Connect to simulation
let client = Client::connect("127.0.0.1:2240").await?;

// Send control commands
loop {
    let thrust = calculate_thrust_command();
    client.set_component("thrust_command", thrust).await?;
    sleep(Duration::from_millis(10)).await;
}
```

### Real-World Example: Rocket Fin Control

The `rocket.py` example demonstrates external trim control:
```python
# External trim control for rocket fins
FinControlTrim = ty.Annotated[
    jax.Array,
    el.Component("fin_control_trim", el.ComponentType.F64,
                 metadata={"external_control": "true"})
]

@el.map
def aero_coefs(fin_deflect: FinDeflect, fin_trim: FinControlTrim) -> AeroCoefs:
    # Combine autonomous control with external trim
    effective_deflect = jnp.clip(fin_deflect + fin_trim, -40.0, 40.0)
    # ... calculate aerodynamics
```

Run the complete example:
```bash
# Terminal 1: Simulation
python examples/rocket/main.py run 0.0.0.0:2240

# Terminal 2: External controller
cargo run --release -p elodin-db-rust-client

# Terminal 3: Visualization
elodin editor 127.0.0.1:2240
```

### Best Practices

1. **Initialize External Components**: Provide default values for operation before external control connects
2. **Validate Inputs**: Clamp or validate external inputs to prevent instability
3. **Rate Limiting**: External controllers should respect simulation timestep
4. **Failsafe**: Design systems to handle loss of external control gracefully

## Advanced Features

### S10 Process Orchestration
Define complex simulation pipelines:
```python
def plan(out_dir):
    return el.Recipe(
        name="drone_sim",
        path="drone.py",
        addr="[::]:2240",
        optimize=True
    )
```
*Learn more about S10: [../s10/README.md](../s10/README.md)*

### Custom Components
```python
Telemetry = el.Annotated[
    jax.Array,
    el.Component("telemetry", el.ComponentType(el.PrimitiveType.F64, (10,)))
]
```

### Impeller2 Streaming
Real-time telemetry to external systems:
```python
# Automatic streaming when connected to Elodin-DB
w.run(system, db_addr="127.0.0.1:2240")
```

### GPU Acceleration
```python
# Automatically uses GPU if available via JAX
exec = w.build(system, optimize=True)
```

### Earth Gravity Models

nox-py includes high-performance Earth gravity models for spacecraft simulations:

#### J2 Model
Simple oblate Earth model for quick orbital mechanics:
```python
from elodin.j2 import J2

gravity_model = J2()
# Compute gravitational acceleration at position (x, y, z) for given mass
force = gravity_model.compute_field(x=7000e3, y=0, z=0, mass=500.0)
```
*Implementation: [python/elodin/j2.py](python/elodin/j2.py)*

#### EGM08 Model
High-fidelity spherical harmonic gravity model with real-time performance:
```python
from elodin.egm08 import EGM08

# Initialize with desired degree (higher = more accurate, slower)
# Degree 10: ~0.1ms, Degree 100: ~1ms, Degree 250: ~2.5ms
gravity_model = EGM08(max_degree=64)

# Compute gravitational acceleration
force = gravity_model.compute_field(x=7000e3, y=0, z=0, mass=500.0)
```
*Implementation: [python/elodin/egm08.py](python/elodin/egm08.py)*

**Key Features:**
- Fully vectorized JAX implementation for GPU acceleration
- Pre-computed coefficients downloaded on first use
- Real-time capable: <50ms at full fidelity (degree 2190)
- <2.5ms for typical operational degrees (≤250)

*Read about the optimization approach: [FSW Workshop 2025 Abstract](python/elodin/FSW%20Workshop%202025%20abstract.md)*

This model enables:
- Entire satellite constellation simulations at high fidelity
- Accurate orbit propagation for mission planning
- Real-time conjunction analysis
- Precise landing trajectory calculations

#### Usage in Simulations
```python
@el.map
def gravity_effector(pos: el.WorldPos, mass: el.Inertia) -> el.Force:
    # Use EGM08 for high-fidelity gravity
    gravity_model = EGM08(max_degree=100)
    force = gravity_model.compute_field(
        pos.linear()[0], pos.linear()[1], pos.linear()[2], 
        mass.mass()
    )
    return el.Force(linear=force)

# Add to your system pipeline
sys = gravity_effector | el.six_dof()
```

*See real usage in spacecraft simulations:*
- [Cube-sat with EGM08 gravity](../../examples/cube-sat/main.py#L25) - Degree 64 for accurate ADCS
- [Cube-sat JAX simulation](../../examples/cube-sat-pysim/main.py#L26) - Degree 10 for faster training

## Examples

### Three-Body Problem
*Full implementation: [../../examples/three-body/main.py](../../examples/three-body/main.py)*
```python
import elodin as el
import jax.numpy as jnp

w = el.World()

# Spawn three bodies - names become component hierarchy roots
w.spawn(el.Body(
    world_pos=el.WorldPos(linear=jnp.array([1.0, 0.0, 0.0])),
    world_vel=el.WorldVel(linear=jnp.array([0.0, 0.5, 0.0])),
    inertia=el.Inertia(1.0)
), name="body_a")  # Creates: body_a.world_pos, body_a.world_vel, body_a.inertia

# Define gravity system
@el.system
def gravity(graph: el.GraphQuery[Edge], q: el.Query[el.WorldPos, el.Inertia]) -> el.Query[el.Force]:
    # ... implement N-body gravity ...
    
w.run(el.six_dof(sys=gravity))
```

### Spacecraft ADCS
*Full implementation: [../../examples/cube-sat/main.py](../../examples/cube-sat/main.py) | JAX-only version: [../../examples/cube-sat-pysim/main.py](../../examples/cube-sat-pysim/main.py)*
```python
# Complete MEKF, LQR control, reaction wheels, and sensor simulation
sys = sensors | kalman_filter | control | actuators | el.six_dof()
w.run(sys, sim_time_step=1/120.0)
```

### Rocket with Aerodynamics
*Full implementation: [../../examples/rocket/main.py](../../examples/rocket/main.py)*
```python
# 6DOF rocket simulation with drag, thrust curves, and fin control
sys = non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.Rk4)
```

### Additional Examples
- **Ball with Bounce**: [../../examples/ball/sim.py](../../examples/ball/sim.py) - Simple physics with collision detection
- **Drone Control**: [../../examples/drone/sim.py](../../examples/drone/sim.py) - Quadcopter dynamics and control

## Integration with Other Tools

### JAXMarl
Complete example of training a multi-agent system with JAXMarl:
```python
import elodin as el
import jax
import jax.numpy as jnp
from jaxmarl import make

# Define Elodin simulation for drone swarm
def build_drone_swarm():
    w = el.World()
    
    # Spawn multiple drones
    for i in range(4):
        w.spawn(
            DroneArchetype(
                world_pos=el.WorldPos(linear=jnp.array([i*2.0, 0.0, 1.0])),
                world_vel=el.WorldVel(),
                control=DroneControl()
            ),
            name=f"drone_{i}"
        )
    
    # Define control and physics systems
    system = control_system | el.six_dof(sys=collision_avoidance)
    return w, system

# Wrap Elodin sim as JAXMarl environment
class ElodinMARLEnv:
    def __init__(self):
        world, system = build_drone_swarm()
        self.sim = world.to_jax(system, sim_time_step=1/50.0)
        self.agents = [f"drone_{i}" for i in range(4)]
        self.num_agents = len(self.agents)
    
    def reset(self, key):
        # Reset simulation state
        obs = {
            agent: self.sim.get_state("sensors", agent)
            for agent in self.agents
        }
        state = self.sim.state  # Internal state for simulator
        return obs, state
    
    def step(self, key, state, actions):
        # Apply actions to each drone
        for agent, action in actions.items():
            self.sim.set_state("control", agent, action)
        
        # Step simulation
        self.sim.step(1)
        
        # Get observations, rewards, dones
        obs = {agent: self.sim.get_state("sensors", agent) 
               for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) 
                  for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        dones["__all__"] = self._check_episode_done()
        
        return obs, self.sim.state, rewards, dones, {}
    
    def action_space(self, agent):
        return self.sim.get_action_space(agent)

# Training loop with JAXMarl
env = ElodinMARLEnv()
key = jax.random.PRNGKey(0)

# JIT-compile the entire training step for performance
@jax.jit
def train_step(state, actions):
    obs, state, rewards, dones, info = env.step(key, state, actions)
    # Your RL algorithm update here (PPO, QMIX, etc.)
    return state, obs, rewards

# Train agents
obs, state = env.reset(key)
for episode in range(1000):
    actions = {agent: policy(obs[agent]) for agent in env.agents}
    state, obs, rewards = train_step(state, actions)
```

This enables:
- 14x faster training than CPU-based simulators
- Seamless integration with JAXMarl's IPPO, QMIX, and other algorithms
- GPU-accelerated physics and control
- Fully differentiable simulation for policy gradient methods
- Vectorized environments via `jax.vmap` for parallel training

Key integration points:
- `w.to_jax()` converts Elodin simulation to pure JAX operations
- Component states map directly to agent observations/actions
- JAX JIT compilation works across the entire train/sim loop
- Compatible with JAXMarl's centralized training patterns

### Basilisk
```python
# Import Basilisk components via roci
from roci.adcs import mekf, triad
```

### OpenAI Gym
```python
class ElodinEnv(gym.Env):
    def __init__(self):
        self.sim = build_world().to_jax(system)
    
    def step(self, action):
        self.sim.set_state("control", "agent", action)
        self.sim.step(1)
        return self.sim.get_state("observation", "agent")
```
*JAXSim implementation: [python/elodin/jaxsim.py](python/elodin/jaxsim.py)*

## Performance Tips

1. **Use compiled mode** for production: `w.build(system, optimize=True)`
2. **Batch operations** in JAX mode: `jax.vmap(sim.step)`
3. **Profile bottlenecks**: `exec.profile()` returns timing breakdowns
4. **Minimize Python callbacks** in hot loops
5. **Use graph systems** for many-to-many relationships

## Architecture Details

### PyO3 Bindings
The Rust core provides Python bindings for:
- Hierarchical component operations
- Spatial math (quaternions, spatial vectors)
- Graph operations for relationships between components
- NOX compiler interface

### NOX Compiler
Transforms Python-defined systems into optimized native code:
- Automatic vectorization
- Memory layout optimization
- Multi-threading
- Optional GPU compilation

### Impeller2 Protocol
High-performance pub-sub for telemetry:
- Zero-copy serialization
- Automatic component discovery
- Time-synchronized streams
- Multiple transport backends (TCP, UDP, shared memory)

## Development

### Running Tests
```bash
pytest tests/
```

### Building Documentation
```bash
# Documentation at https://docs.elodin.systems
```

## Architecture Notes: Hierarchical Components

### No Entity-Component-System (ECS)

Unlike traditional game engines, Elodin uses **hierarchical components** without entities. This design decision ([PR #88](https://github.com/elodin-sys/elodin/pull/88)) simplifies integration with flight software:

**Traditional ECS:**
```python
# Would have entities with IDs
entity_id = w.spawn(components)  # Returns EntityId(42)
w.add_component(entity_id, new_component)
```

**Elodin's Hierarchical Components:**
```python
# Components organized by hierarchical names
w.spawn(DroneBody(...), name="drone")  # Creates "drone.world_pos", "drone.world_vel", etc.
w.spawn(IMUSensor(...), name="drone.imu")  # Creates "drone.imu.accel", "drone.imu.gyro", etc.
```

### Benefits for SITL/HITL Testing

This architecture makes flight software integration straightforward:

1. **Natural Namespacing** - Components like `drone.motors.0.rpm` match how engineers think about systems
2. **Direct Mapping** - Flight software components map directly to simulation components
3. **No Entity Management** - Flight software doesn't need to track entity IDs
4. **Simple Telemetry** - Component paths are human-readable in logs and debugging

### Accessing Components

Components are referenced by their full hierarchical path:
```python
# In simulation
exec.history(["drone.world_pos", "drone.imu.accel", "drone.motors.0.rpm"])

# Flight software sees the same paths via Impeller2
// Component: "drone.imu.accel" -> [ax, ay, az]
```

## Related Projects

- [Elodin Editor](../../apps/elodin) - 3D visualization and debugging
- [Elodin-DB](../db) - Time-series telemetry database
- [Impeller2](../impeller2) - High-performance pub-sub protocol
- [NOX](../nox) - Tensor compiler backend
- [Roci](../roci) - Flight software framework

## License

See the repository's LICENSE file for details.
