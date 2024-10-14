+++
title = "Three-Body Orbit"
description = "Example using a classic physics problem."
draft = false
weight = 102
sort_by = "weight"
template = "get-started/page.html"

[extra]
toc = true
top = false
order = 2
icon = ""
+++

The [three-body problem](https://en.wikipedia.org/wiki/Three-body_problem) is a classic orbital dynamics situation.
You have three bodies, each with significant mass, all interacting gravitationally.
This turns out to be a [chaotic system](https://en.wikipedia.org/wiki/Chaos_theory), with no general closed-form solution. There are however
a few stable configurations of the three-body problem.

In this tutorial, we will model one of the stable configurations from
R. A. Broucke's technical report ["Period Orbits in the Restricted Three-Body Problem with Earth Moon Masses"](https://ntrs.nasa.gov/api/citations/19680013800/downloads/19680013800.pdf).

#### Import Elodin and JAX
Our first step is to import Elodin and Jax into our environment:
```python
import elodin as el
from jax import numpy as np
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0
```
`SIM_TIME_STEP` value is the duration of each tick in the simulation.

#### Setup Gravity Constraints
With all dependencies prepared, we can set Gravity Constraints:
```python
GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]
G = 6.6743e-11


@el.dataclass
class GravityConstraint(el.Archetype):
    a: GravityEdge

    def __init__(self, a: el.EntityId, b: el.EntityId):
        self.a = el.Edge(a, b)

@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_inner(force, a_pos, a_inertia, b_pos, b_inertia):
        r = a_pos.linear() - b_pos.linear()
        m = a_inertia.mass()
        M = b_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return el.SpatialForce(linear=force.force() - f)

    return graph.edge_fold(query, query, el.Force, el.SpatialForce.zero(), gravity_inner)
```

#### Add 1st & 2nd Body
Before we can do anything we'll need an instance of a WorldBuilder, and with that we can spawn our first entities:
```python
w = el.World()
mesh = w.insert_asset(el.Mesh.sphere(0.2))
a = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=np.array([0.8822391241, 0, 0])),
            world_vel=el.WorldVel(linear=np.array([0, 1.0042424155, 0])),
            inertia=el.Inertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(25.3, 18.4, 1.0))),
    ],
    name="A",
)
b = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=np.array([-0.6432718586, 0, 0])),
            world_vel=el.WorldVel(linear=np.array([0, -1.6491842814, 0])),
            inertia=el.Inertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(10.0, 0.0, 10.0))),
    ],
    name="B",
)

w.spawn(GravityConstraint(a, b), name="A -> B")
w.spawn(GravityConstraint(b, a), name="B -> A")
```

`GravityConstraint` tells the simulation to calculate gravity between the two objects.


#### Let's try running the simulation.
But first, let's add a view port so we can observe the world:
```python
w.spawn(
    el.Panel.viewport(
        pos=[0.0, -3.0, 3.0],
        looking_at=[0.0, 0.0, 0.0],
        hdr=True,
    ),
    name="Viewport 1",
)
```

Now, we're ready to start simulating:
```python
sys = el.six_dof(sys=gravity)
sim = w.run(sys, SIM_TIME_STEP)
```
At this moment, bodies will be flying off into space, so feel free to remove these last 2 lines for now.

#### Add the Third Body
And last but not least, we will add the third body which will make this a stable orbit.
```python
c = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=np.array([-0.2389672654, 0, 0])),
            world_vel=el.WorldVel(linear=np.array([0, 0.6449418659, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(00.0, 1.0, 10.0))),
    ],
    name="C",
)

w.spawn(GravityConstraint(a, c), name="A -> C")
w.spawn(GravityConstraint(b, c), name="B -> C")

w.spawn(GravityConstraint(c, a), name="C -> A")
w.spawn(GravityConstraint(c, b), name="C -> B")
```

#### Start the Simulation!
You can now update the simulation by pressing `Update Sim` or hitting `Cmd-Enter`.
```python
sys = six_dof(gravity)
sim = w.run(sys, SIM_TIME_STEP)
```

#### Add some gizmos
Sometimes it can be helpful to visualize the forces acting on the bodies and their movements in 3D space. Add these two
lines to see a few options in action:
```python
w.spawn(el.VectorArrow(a, "world_vel", offset=3, body_frame=False, scale=1.0))

w.spawn(el.Line3d(b, "world_pos", index=[4, 5, 6], line_width=10.0))
```

#### Checking your work
And that's it! You can now run the simulation and see the three bodies orbiting around each other in a stable configuration.
If you'd like to check your work, you can use the following command to generate the matching template code:
```bash
elodin create --template three-body
```
