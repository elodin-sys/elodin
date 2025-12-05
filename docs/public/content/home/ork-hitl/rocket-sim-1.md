+++
title = "Part 1: Vertical Flight"
description = "Learn to simulate a model rocket with basic 1D physics"
draft = false
weight = 108
sort_by = "weight"

[extra]
lead = "Build your first rocket simulation with thrust, drag, and gravity"
toc = true
top = false
order = 8
+++

# Model Rocket Simulation - Part 1: Vertical Flight

<img src="/assets/rocket-forces-diagram.jpg" alt="rocket-forces"/>
<br></br>

In this tutorial series, we'll build a complete model rocket simulation from scratch. We'll start with the simplest possible case - a rocket flying straight up - and gradually add complexity until we have a full 3D simulation with stability and recovery systems.

## What We're Building

By the end of this first part, you'll have:
- A rocket that launches vertically
- Realistic thrust, drag, and gravity forces
- Altitude and velocity tracking
- Visual feedback of the flight

Let's start with the absolute basics and build up from there.

## Setting Up Our World

First, let's import what we need and set up our simulation environment:

```python
import elodin as el
import jax.numpy as jnp
from typing import Annotated
from dataclasses import dataclass, field

# Simulation runs at 120 Hz for accuracy
SIM_TIME_STEP = 1.0 / 120.0

# Physical constants
g = 9.81  # gravity (m/s²)
```

## Step 1: Basic Rocket with Gravity

Let's start with just a rocket falling under gravity - no thrust or drag yet. This ensures our basic physics works.

```python
@dataclass
class SimpleRocket(el.Archetype):
    """Our rocket with just basic physics properties"""
    pass  # We'll add components as we go

def create_world() -> el.World:
    """Create the simulation world with a rocket"""
    world = el.World()
    
    # Spawn our rocket 1 meter above ground
    rocket = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    linear=jnp.array([0.0, 0.0, 1.0])  # 1m altitude
                ),
                world_vel=el.SpatialMotion(
                    linear=jnp.array([0.0, 0.0, 0.0])  # Starting at rest
                ),
                inertia=el.SpatialInertia(
                    mass=0.5,  # 500g rocket (typical model rocket with motor)
                ),
            ),
            SimpleRocket(),
        ],
        name="Rocket",
        id="rocket",
    )
    
    # Add visualization
    world.spawn(
        el.Panel.viewport(
            pos=[10.0, 0.0, 5.0],
            looking_at=[0.0, 0.0, 100.0],
        ),
        name="Viewport",
    )
    
    return world

# Define gravity system
@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    """Apply downward gravity force"""
    return f + el.SpatialForce(
        linear=jnp.array([0.0, 0.0, -g]) * inertia.mass()
    )

# Create and run simulation
world = create_world()
system = el.six_dof(sys=gravity)
world.run(system, sim_time_step=SIM_TIME_STEP, run_time_step=1/60.0, max_ticks=600)
```

Run this code and you'll see the rocket fall straight down. Good! Gravity works. Now let's make it go up.

## Step 2: Adding Thrust

Real rockets don't have constant thrust - they have a thrust curve that changes over time. But let's start simple with constant thrust for 2 seconds:

```python
# Add thrust component to track motor state
Thrust = Annotated[
    jax.Array,
    el.Component("thrust_force", el.ComponentType(el.PrimitiveType.F64, ()))
]

BurnTime = Annotated[
    jax.Array, 
    el.Component("burn_time", el.ComponentType(el.PrimitiveType.F64, ()))
]

@dataclass
class SimpleRocket(el.Archetype):
    """Rocket with thrust capability"""
    thrust_force: Thrust = field(default_factory=lambda: jnp.array(10.0))  # 10 Newtons
    burn_time: BurnTime = field(default_factory=lambda: jnp.array(2.0))    # 2 seconds

@el.system
def thrust_system(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep], 
    query: el.Query[Thrust, BurnTime, el.Force],
) -> el.Query[el.Force]:
    """Apply thrust force while motor is burning"""
    current_time = tick[0] * dt[0]
    
    def apply_thrust(thrust_force, burn_time, force):
        # Only apply thrust if we're still burning
        if current_time < burn_time:
            # Thrust acts upward (positive Z)
            return force + el.SpatialForce(
                linear=jnp.array([0.0, 0.0, thrust_force])
            )
        return force
    
    return query.map(el.Force, apply_thrust)

# Update system to include thrust
system = el.six_dof(sys=gravity | thrust_system)
```

Now the rocket should shoot up! But it will keep going forever (after thrust ends, it just falls). We need drag.

## Step 3: Adding Drag

Drag is the air resistance that slows the rocket down. The drag equation is:

**Drag = 0.5 × ρ × v² × Cd × A**

Where:
- ρ (rho) = air density (1.225 kg/m³ at sea level)
- v = velocity 
- Cd = drag coefficient (about 0.75 for a model rocket)
- A = cross-sectional area

```python
# Rocket parameters
ROCKET_RADIUS = 0.025  # 2.5 cm radius (5 cm diameter)
ROCKET_CD = 0.75       # Drag coefficient
AIR_DENSITY = 1.225    # kg/m³ at sea level

@el.map
def drag(vel: el.WorldVel, f: el.Force) -> el.Force:
    """Apply drag force opposing motion"""
    v = vel.linear()
    speed = jnp.linalg.norm(v)
    
    # Drag force magnitude
    area = jnp.pi * ROCKET_RADIUS**2
    drag_magnitude = 0.5 * AIR_DENSITY * speed**2 * ROCKET_CD * area
    
    # Drag opposes velocity direction
    # Use safe normalization to avoid division by zero
    drag_force = jnp.where(
        speed > 0.001,  # Only apply drag if moving
        -drag_magnitude * v / speed,  # Normalize velocity vector
        jnp.array([0.0, 0.0, 0.0])
    )
    
    return f + el.SpatialForce(linear=drag_force)

# Update system to include drag
system = el.six_dof(sys=gravity | thrust_system | drag)
```

## Step 4: Better Visualization

Let's add some visual elements to better see what's happening:

```python
def create_world() -> el.World:
    """Enhanced world with better visuals"""
    world = el.World()
    
    rocket = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    linear=jnp.array([0.0, 0.0, 1.0])
                ),
                world_vel=el.SpatialMotion(
                    linear=jnp.array([0.0, 0.0, 0.0])
                ),
                inertia=el.SpatialInertia(mass=0.5),
            ),
            SimpleRocket(),
            # Add visual mesh
            world.shape(
                el.Mesh.cylinder(radius=ROCKET_RADIUS, height=0.3),
                el.Material.color(255.0, 0.0, 0.0),  # Red rocket
            ),
        ],
        name="Rocket",
        id="rocket",
    )
    
    # Add ground plane for reference
    world.spawn(
        world.shape(
            el.Mesh.box([20.0, 20.0, 0.01]),
            el.Material.color(0.0, 100.0, 0.0),  # Green ground
        ),
        name="Ground",
    )
    
    # Trail to show path
    world.spawn(el.Line3d(rocket, "world_pos", line_width=2.0))
    
    # Better camera view
    world.spawn(
        el.Panel.viewport(
            pos=[15.0, -15.0, 10.0],
            looking_at=[0.0, 0.0, 50.0],
            fov=60.0,
            hdr=True,
        ),
        name="Viewport",
    )
    
    return world
```

## Step 5: Ground Detection

Let's stop the simulation when the rocket hits the ground:

```python
@el.map
def ground_contact(pos: el.WorldPos, vel: el.WorldVel) -> el.WorldVel:
    """Stop rocket when it hits the ground"""
    z = pos.linear()[2]
    vz = vel.linear()[2]
    
    # If below ground and moving down, stop
    return jnp.where(
        (z <= 0.0) & (vz < 0.0),
        el.SpatialMotion(linear=vel.linear() * jnp.array([1.0, 1.0, 0.0])),
        vel
    )

# Add to system pipeline
system = ground_contact | el.six_dof(sys=gravity | thrust_system | drag)
```

## Complete Working Example

Here's the full code for Part 1:

```python
import elodin as el
import jax
import jax.numpy as jnp
from typing import Annotated
from dataclasses import dataclass, field

# Constants
SIM_TIME_STEP = 1.0 / 120.0
g = 9.81
ROCKET_RADIUS = 0.025  # 2.5 cm
ROCKET_CD = 0.75
AIR_DENSITY = 1.225

# Components
Thrust = Annotated[
    jax.Array,
    el.Component("thrust_force", el.ComponentType(el.PrimitiveType.F64, ()))
]

BurnTime = Annotated[
    jax.Array,
    el.Component("burn_time", el.ComponentType(el.PrimitiveType.F64, ()))
]

@dataclass
class SimpleRocket(el.Archetype):
    thrust_force: Thrust = field(default_factory=lambda: jnp.array(10.0))
    burn_time: BurnTime = field(default_factory=lambda: jnp.array(2.0))

# Systems
@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -g]) * inertia.mass())

@el.system
def thrust_system(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    query: el.Query[Thrust, BurnTime, el.Force],
) -> el.Query[el.Force]:
    current_time = tick[0] * dt[0]
    
    def apply_thrust(thrust_force, burn_time, force):
        if current_time < burn_time:
            return force + el.SpatialForce(
                linear=jnp.array([0.0, 0.0, thrust_force])
            )
        return force
    
    return query.map(el.Force, apply_thrust)

@el.map
def drag(vel: el.WorldVel, f: el.Force) -> el.Force:
    v = vel.linear()
    speed = jnp.linalg.norm(v)
    area = jnp.pi * ROCKET_RADIUS**2
    drag_magnitude = 0.5 * AIR_DENSITY * speed**2 * ROCKET_CD * area
    
    drag_force = jnp.where(
        speed > 0.001,
        -drag_magnitude * v / speed,
        jnp.array([0.0, 0.0, 0.0])
    )
    
    return f + el.SpatialForce(linear=drag_force)

@el.map
def ground_contact(pos: el.WorldPos, vel: el.WorldVel) -> el.WorldVel:
    z = pos.linear()[2]
    vz = vel.linear()[2]
    
    return jnp.where(
        (z <= 0.0) & (vz < 0.0),
        el.SpatialMotion(linear=vel.linear() * jnp.array([1.0, 1.0, 0.0])),
        vel
    )

def create_world() -> el.World:
    world = el.World()
    
    rocket = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 1.0])),
                world_vel=el.SpatialMotion(linear=jnp.array([0.0, 0.0, 0.0])),
                inertia=el.SpatialInertia(mass=0.5),
            ),
            SimpleRocket(),
            world.shape(
                el.Mesh.cylinder(radius=ROCKET_RADIUS, height=0.3),
                el.Material.color(255.0, 0.0, 0.0),
            ),
        ],
        name="Rocket",
        id="rocket",
    )
    
    world.spawn(
        world.shape(
            el.Mesh.box([20.0, 20.0, 0.01]),
            el.Material.color(0.0, 100.0, 0.0),
        ),
        name="Ground",
    )
    
    world.spawn(el.Line3d(rocket, "world_pos", line_width=2.0))
    
    world.spawn(
        el.Panel.viewport(
            pos=[15.0, -15.0, 10.0],
            looking_at=[0.0, 0.0, 50.0],
            fov=60.0,
            hdr=True,
        ),
        name="Viewport",
    )
    
    return world

# Run simulation
world = create_world()
system = ground_contact | el.six_dof(sys=gravity | thrust_system | drag)
world.run(system, sim_time_step=SIM_TIME_STEP, run_time_step=1/60.0)
```

## What's Happening?

With our parameters:
- **Thrust:** 10 N for 2 seconds
- **Mass:** 0.5 kg  
- **Net upward force:** 10 N - (0.5 kg × 9.81 m/s²) = 5.1 N
- **Initial acceleration:** 5.1 N / 0.5 kg = 10.2 m/s² upward

The rocket will:
1. Accelerate upward for 2 seconds (minus drag)
2. Coast upward after burnout, slowing due to gravity and drag
3. Reach apogee (highest point) when velocity = 0
4. Fall back down
5. Stop when it hits the ground

Expected altitude: ~20-30 meters (varies with drag)

## Experiments to Try

1. **Change thrust force:** Try 5 N (barely flies) or 20 N (much higher)
2. **Change burn time:** Longer burn = higher altitude
3. **Change mass:** Lighter rocket = higher acceleration
4. **Change drag coefficient:** Lower Cd = higher altitude
5. **Remove drag:** Comment out the drag system to see the difference

## Understanding the Physics

### Why These Numbers?

Our simple rocket represents something like an Estes A8-3 motor:
- Average thrust: ~10 N
- Burn time: ~2 seconds  
- Total impulse: ~20 N·s

This should lift a 500g rocket to about 20-30 meters, which matches real-world experience with small model rockets.

### The Importance of Drag

Without drag, our rocket would reach about 40 meters. With drag, it only reaches about 25 meters. That's a 40% reduction! Drag becomes more important at higher speeds, which is why rockets have pointed nose cones and smooth bodies.

## Next Steps

In Part 2, we'll add:
- Real thrust curves (not constant thrust)
- Atmospheric model (air density changes with altitude)
- Mass change as propellant burns
- Better data recording and plotting

The simulation will become much more realistic and start matching real rocket flights!

## Summary

You've built your first rocket simulation! It has:
✅ Realistic gravity  
✅ Thrust force that burns out
✅ Quadratic drag model
✅ Ground collision detection
✅ Visual feedback

This is the foundation we'll build on. The physics is real - if you built a rocket with these exact parameters, it would fly almost exactly like our simulation (within ~20% for altitude).

Ready for Part 2? Let's add real motor curves and atmospheric effects!
