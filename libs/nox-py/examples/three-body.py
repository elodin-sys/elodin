import elodin as el
from jax import numpy as jnp
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0
# Set the gravitational constant for Newton's law of universal gravitation
G = 6.6743e-11

w = el.World()
mesh = w.insert_asset(el.Mesh.sphere(0.2))

a = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.8920281421, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.9957939373, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(25.3, 18.4, 1.0))),
    ],
    name="A",
)
b = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([-0.6628498947, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, -1.6191613336, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(10.0, 0.0, 10.0))),
    ],
    name="B",
)
c = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([-0.2291782474, 0, 0])),
            world_vel=el.WorldVel(linear=jnp.array([0, 0.6233673964, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(0.0, 1.0, 10.0))),
    ],
    name="C",
)

w.spawn(
    el.Panel.viewport(
        pos=[0.0, -3.0, 3.0],
        looking_at=[0.0, 0.0, 0.0],
        hdr=True,
    ),
    name="Viewport 1",
)

# Define a new "gravity edge" component type
GravityEdge = el.Annotated[el.Edge, el.Component("gravity_edge", el.ComponentType.Edge)]


# Define a new "gravity constraint" archetype using the gravity edge component
@el.dataclass
class GravityConstraint(el.Archetype):
    a: GravityEdge

    def __init__(self, a: el.EntityId, b: el.EntityId):
        self.a = GravityEdge(a, b)


# Define a new system to apply gravity by iterating over all gravity edge components
@el.system
def gravity(
    graph: el.GraphQuery[GravityEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    # Create a fold function to take an accumulator and the query results for the
    # left and right entities, and apply Netwon's law of universal gravitation:
    def gravity_fn(force, a_pos, a_inertia, b_pos, b_inertia):
        r = a_pos.linear() - b_pos.linear()
        m = a_inertia.mass()
        M = b_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        # returns the updated force value applied to the left entity this tick
        return el.Force(linear=force.force() - f)

    return graph.edge_fold(
        left_query=query,  # i.e. fetches WorldPos and Inertia components from A
        right_query=query,  # reusing the same query for B
        return_type=el.Force,  # matching the system query return type
        init_value=el.Force(),  # initial value for the fold function accumulator
        fold_fn=gravity_fn,  # the fold function to apply to each edge, that you defined above
    )


# Add the gravity constraint entities to the world
w.spawn(GravityConstraint(a, b), name="A -> B")
w.spawn(GravityConstraint(b, a), name="B -> A")

w.spawn(GravityConstraint(a, c), name="A -> C")
w.spawn(GravityConstraint(b, c), name="B -> C")

w.spawn(GravityConstraint(c, a), name="C -> A")
w.spawn(GravityConstraint(c, b), name="C -> B")

w.spawn(el.VectorArrow(a, "world_vel", offset=3, body_frame=False, scale=1.0))

w.spawn(el.Line3d(b, "world_pos", index=[4, 5, 6], line_width=10.0))

sys = el.six_dof(sys=gravity)
sim = w.run(sys, SIM_TIME_STEP)
