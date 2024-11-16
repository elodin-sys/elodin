+++
title = "Bouncing Ball"
description = "Simulate the model of a bouncing ball in windy conditions."
draft = false
weight = 103
sort_by = "weight"

[extra]
lead = "Simulate the model of a bouncing ball in windy conditions."
toc = true
top = false
order = 3
icon = ""
+++

<img src="/assets/ball-screenshot.jpg" alt="ball-screenshot"/>
<br></br>

In this tutorial, we'll model a bouncing ball in a windy environment. This will demonstrate how to:
- Set up a basic physics simulation
- Add multiple interacting systems
- Break code into multiple files for better organization

### Basic Ball Simulation Setup

As a starting point, let's first setup a world with just a ball and gravity.

#### Import Elodin and JAX
First, let's import our required libraries:
```python
import typing
from dataclasses import field

import elodin as el
import jax
from jax import numpy as jnp
from jax import random
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0
```
Notice we're importing typing and dataclasses, we'll use these later.

#### Create the World
Now let's set up our simulation world:
```python
BALL_RADIUS = 0.2

def world() -> el.World:
    world = el.World()
    ball = world.spawn(
        [
            el.Body(world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 6.0]))),
            world.shape(el.Mesh.sphere(BALL_RADIUS), el.Material.color(12.7, 9.2, 0.5)),
        ],
        name="Ball",
    )
    world.spawn(
        el.Panel.viewport(
            track_rotation=False,
            active=True,
            pos=[8.0, 2.0, 4.0],
            looking_at=[0.0, 0.0, 3.0],
            show_grid=True,
            hdr=True,
        ),
        name="Viewport",
    )
    world.spawn(el.Line3d(ball, "world_pos", index=[4, 5, 6], line_width=2.0))
    return world
```

We create a new elodin world, spawn an entity named "Ball" with a sphere mesh shape component, and a body archetype
which provides the ball with a position, velocity, and other aspects related to the Elodin 6DoF system
(see the [6DoF reference for more info](http://127.0.0.1:1111/reference/python-api/#6-degrees-of-freedom-model)).
We also spawn a viewport and a line to visualize the ball's position.


#### Define gravity
Let's bring in our system for basic gravity:
```python
@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())
```

{% alert(kind="info") %}
Let's take a moment to understand the use of `@el.map`. This decorator creates an Elodin system from a function.
The inputs of the function acts a filter for which entities this applies to.  In this case, when the system is
integrated, the entities that have a `Force` and `Inertia` component will have this function applied to them. The function returns the
updated force, which is then used to update the velocity and position of the ball during integration in the system function below.
{% end %}

#### Define the System
Let's combine our systems into one complete simulation:
```python
def system() -> el.System:
    sys = el.six_dof(sys=gravity)
    return sys
```

#### Running the Simulation
With everything set up, we can now run the simulation:
```python
world().run(system(), sim_time_step=SIM_TIME_STEP, max_ticks=1200)
```

{% alert(kind="info") %}
The `max_ticks` parameter is used to limit the number of simulation steps. This is useful for debugging and testing. Notice how in this
case the simulation only runs for 10 seconds (1200 ticks at 120 ticks per second).
{% end %}

<video autoplay loop muted playsinline style="width: 100%; height: auto;">
  <source src="/assets/falling-ball.av1.mp4" type="video/mp4; codecs=av01.0.05M.08">
  <source src="/assets/falling-ball.h264.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Bouncing off the Ground
You'll notice that the ball falls beautifully, and right through the "ground" into infinity. Let's introduce
a system to handle bouncing off the ground instead:
```python
@el.map
def bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[2], v.linear()[2]) < 0.0,
        lambda _: el.SpatialMotion(linear=v.linear() * jnp.array([1.0, 1.0, -1.0])),
        lambda _: v,
        operand=None,
    )
```
The JAX syntax can be visually daunting at first; let's break this down:

The JAX `lax.cond` function is used to conditionally apply the bounce, if the position of the ball is below the ground.
The first argument is the condition, the second is the function to apply if the condition is true, and the third
is the function to apply if the condition is false. The `operand` argument is not used in this case, so it's set to `None`.

Our bounce is applied by reversing the velocity in the z-direction. We'll add "bounciness" later.

#### Update the System
We now have two systems, gravity and bounce, that we want to apply to our simulation. We can combine them into a single system
with the concept of pipelining systems. Let's update our system function:
```python
def system() -> el.System:
    sys = bounce | el.six_dof(sys=gravity)
    return sys
```
{% alert(kind="notice") %}
Notice we use a pipe `|` to combine the systems. This is a powerful concept in Elodin that allows you to chain systems together.
{% end %}

{% alert(kind="info") %}
But why is gravity supplied to `six_dof`, while bounce is not? You'll notice bounce returns the resulting velocity, while gravity only
supplies forces that still need to be applied by an integrator. The `six_dof` system is an integrator that applies forces to update the
velocity and position of the ball. See the [six_dof reference documentation](reference/python-api/#6-degrees-of-freedom-model)
for more information.
{% end %}

With bounce applied, let's try running it again:

<video autoplay loop muted playsinline style="width: 100%; height: auto;">
  <source src="/assets/bouncing-ball.av1.mp4" type="video/mp4; codecs=av01.0.05M.08">
  <source src="/assets/bouncing-ball.h264.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Let's Make it Windy

Having a fully customizable ball bouncing physics simulation is great, but let's make it more interesting by adding some
randomized wind forces into the mix. We'll need to make a few changes to our simulation to accommodate this:

#### Create the Wind Component
We'll need a component to represent the wind force in our simulation:
```python
Wind = typing.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]
```

#### Add Global Wind Data
We'll use a custom `Archetype` called "WindData" to maintain our random number generator seed and wind state:

```python
@el.dataclass
class WindData(el.Archetype):
    seed: el.Seed = field(default_factory=lambda: jnp.int64(0))
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
```

Also make sure to update your world function to add a `WindData` instance:
```python
def world(seed: int = 0) -> el.World:
    world = el.World()
    world.spawn(WindData(seed=jnp.int64(seed)), name="WindData")
    ball = world.spawn(
    ...
```

{% alert(kind="info") %}
This is a simple technique to add global data into your simulation for easy reference.
<br></br>
<img src="/assets/winddata.jpg" alt="winddata"/>
{% end %}

We'll need to set a value for the wind force in our `WindData` instance. We can do this by creating a system that generates
a random wind force vector and sets it in the `Wind` component of our `WindData` instance.
```python
@el.system
def sample_wind(s: el.Query[el.Seed], q: el.Query[Wind]) -> el.Query[Wind]:
    return q.map(
        Wind,
        lambda _w: random.normal(random.key(s[0]), shape=(3,)),
    )
```
This system queries for Archetypes with `Seed` and `Wind` components, which will only be our
single "WindData" Archetype entity, and then uses the value of the seed with the random key
function to generate a random wind force vector to set for the `Wind` component.

#### Add the Wind Physics System

Now we need a system to apply our wind force to objects. We could just naiively apply the force of the wind
to our ball, but this wouldn't create a physically realistic simulation. Instead we can borrow from
[fluid dynamics](https://en.wikipedia.org/wiki/Fluid_dynamics) and leverage the [Simple Drag Equation](https://en.wikipedia.org/wiki/Drag_equation):
$$
F_{d}=c_{d}r\frac{v^2}{2}A
$$
The drag equation states that drag force `Fd` is equal to the drag coefficient `Cd` times the density `r`
times half of the velocity `V squared` times the reference area `A`

```python
def calculate_drag(Cd, r, V, A):
    return 0.5 * (Cd * r * V**2 * A)

@el.system
def apply_drag(w: el.Query[Wind],
                q: el.Query[el.Force, el.WorldVel]) -> el.Query[el.Force]:
    def apply_drag_inner(f, v):
        # the Wind entity is a singleton; use the 0th entry of the query result
        fluid_movement_vector = w[0]

        # ball shape generic value (https://en.wikipedia.org/wiki/Drag_coefficient)
        ball_drag_coefficient = 0.5
        # 1.225kg/m^3, density of air at sea level
        fluid_density = 1.225
        # magnitude of the wind velocity vector is the scalar fluid velocity value
        fluid_velocity = la.norm(fluid_movement_vector)
        # Area of a hemisphere = 2 * pi * r^2
        ball_surface_area = 2 * 3.1415 * BALL_RADIUS**2

        drag_force = calculate_drag(
            ball_drag_coefficient,
            fluid_density,
            fluid_velocity,
            ball_surface_area
        )

        fluid_vector_direction = fluid_movement_vector / fluid_velocity
        return el.SpatialForce(linear=f.force() + drag_force * fluid_vector_direction)

    return q.map(
        el.Force,
        apply_drag_inner,
    )
```

{% alert(kind="info") %}
Let's take a moment to understand `@el.system`, a lower level API for composing raw systems in Elodin:
```python
def apply_drag(w: el.Query[Wind],
                q: el.Query[el.Force, el.WorldVel]) -> el.Query[el.Force]
```
When this system is run, it will query for entities with the `Wind` component and entities with both `Force` and `WorldVel` components. These
are provided as arrays of matching entities to the function body as `w` and `q` respectively. The function body is then expected to return a
new Query of `Force` component attached entities, which in this case the Query.map function provides. See
[Query.map](reference/python-api/#class-elodin-query) for more details.
{% end %}

#### Update the System Function

Finally, we need to update our system function to include the wind and drag systems:
```python
def system() -> el.System:
    effectors = gravity | apply_drag
    sys = sample_wind | bounce | el.six_dof(sys=effectors)
    return sys
```

{% alert(kind="info") %}
Notice we've made a distinction now of "effectors" as systems which return forces that will be integrated in the 6DoF model.
{% end %}

<video autoplay loop muted playsinline style="width: 100%; height: auto;">
  <source src="/assets/windy-bounce.av1.mp4" type="video/mp4; codecs=av01.0.05M.08">
  <source src="/assets/windy-bounce.h264.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Improving the Wind Model

You'll notice that we're currently only considering the velocity of the fluid (wind) in the direction of the wind vector.
This is a simplification of the actual drag force due to wind that would be applied to the ball, because the ball's movement
through the air also creates a movement of air fluid experienced by the ball.

{% image(href="/assets/wind-drag-force") %}Wind Drag Force{% end %}

Let's update our apply_drag system to consider this additional aspect of air fluid movement:

```python
@el.system
def apply_drag(w: el.Query[Wind],
                q: el.Query[el.Force, el.WorldVel]) -> el.Query[el.Force]:
    def apply_drag_inner(f, v):
        # the Wind entity is a singleton; use the 0th entry of the query result
        fluid_movement_vector = w[0]
        # combine with the ball's velocity to get the relative velocity of fluid movement
        fluid_movement_vector -= v.linear()
        ...
```

And voila! We now have a more accurate wind model that accounts for the drag force
on the ball as it moves through the air, notice how the ball now loses energy as it struggles against
the drag force along it's direction of movement.

Speaking of losing energy, there is one more thing we can do to improve the simulation: Add a coefficient of restitution
to represent the "bounciness" of the ball. This will allow us to model the ball losing energy on each bounce.

```python
BOUNCINESS = 0.85

@el.map
def bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    return jax.lax.cond(
        jax.lax.max(p.linear()[2], v.linear()[2]) < 0.0,
        lambda _: el.SpatialMotion(linear=
            v.linear() * jnp.array([1.0, 1.0, -1.0]) * BOUNCINESS),
        lambda _: v,
        operand=None,
    )
```

<video autoplay loop muted playsinline style="width: 100%; height: auto;">
  <source src="/assets/bounciness.av1.mp4" type="video/mp4; codecs=av01.0.05M.08">
  <source src="/assets/bounciness.h264.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

This bounce system uses a [coefficient of restitution](https://en.wikipedia.org/wiki/Coefficient_of_restitution) of 0.85,
meaning the ball will lose 15% of its energy on each bounce.

### Make it Windy, Simpler

Crafting a simulation in Elodin allows for approaching a problem in multiple ways. We can simplify our wind simulation
by considering the wind not as a global constant, but instead as a force that affects the ball directly, as experienced from the perspective
of the ball entity. This will allow us to use the same `apply_drag` system, but using the `@el.map` decorator instead of `@el.system`,
allowing for a simpler implementation.

#### Update the Ball Entity

First remove the global spawn of WindData and instead add a WindData component to the ball entity:
```python
def world(seed: int = 0) -> el.World:
    world = el.World()
    # world.spawn(WindData(seed=jnp.int64(seed)), name="WindData")
    ball = world.spawn(
        [
            el.Body(world_pos=el.SpatialTransform(linear=jnp.array([0.0, 0.0, 6.0]))),
            world.shape(el.Mesh.sphere(BALL_RADIUS), el.Material.color(12.7, 9.2, 0.5)),
            WindData(seed=jnp.int64(seed)),

        ],
        name="Ball",
    )
```

#### Convert use of @el.System

You can now convert the `sample_wind` system to use the simpler `@el.map` decorator:
```python
@el.map
def sample_wind(s: el.Seed, _w: Wind) -> Wind:
    return random.normal(random.key(s), shape=(3,))
```

And likewise the `apply_drag` system can be simplified to directly query the wind component from the ball entity,
resulting in much simpler syntax and conceptual model:

```python
@el.map
def apply_drag(w: Wind, v: el.WorldVel, f: el.Force) -> el.Force:
    fluid_movement_vector = w
    fluid_movement_vector -= v.linear()

    ball_drag_coefficient = 0.5
    fluid_density = 1.225
    fluid_velocity = la.norm(fluid_movement_vector)
    ball_surface_area = 2 * 3.1415 * BALL_RADIUS**2

    drag_force = calculate_drag(
        ball_drag_coefficient,
        fluid_density,
        fluid_velocity,
        ball_surface_area
    )

    fluid_vector_direction = fluid_movement_vector / fluid_velocity
    return el.SpatialForce(linear=f.force() + drag_force * fluid_vector_direction)
```

You should be able to run the simulation and see the same behavior as before, but with a simpler implementation.

#### Checking your work
Success! We've added a wind force to our simulation. The ball now bounces around the world with the wind
affecting its trajectory, steadily blowing the ball in a single direction, losing energy as it bounces.
If you'd like to check your work, you can use the following command to generate the matching template code:
```bash
elodin create --template ball
```

{% alert(kind="notice") %}
You'll notice that the template code is broken into multiple files, this is meant to serve as an example of how you can organize your code as it grows.
{% end %}


## Next Steps

In the next section, learn about simulation state with the Tao of Elodin:

{% cardlink(title="The Tao of Elodin", icon="book", href="/get-started/tao/introduction") %}
Learn about the design principles behind Elodin.
{% end %}
