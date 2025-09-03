import elodin as el
from jax import numpy as jnp
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0
# Set the gravitational constant for Newton's law of universal gravitation
G = 6.6743e-11

w = el.World()

a = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.8920281421, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.9957939373, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
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
    ],
    name="C",
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
    # left and right entities, and apply Newton's law of universal gravitation:
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

w.schematic("""
    hsplit {
        tabs share=0.2 {
            hierarchy
            schematic_tree
        }
        tabs share=0.6 {
            viewport name=Viewport pos="(0,0,0,0,0,0,3)" look_at="(0,0,0,0,0,0,0)" hdr=#true
            graph "a.world_pos" name=Graph
        }
        tabs share=0.2 {
            inspector
        }
    }
    object_3d a.world_pos {
        sphere radius=0.2 r=10.0 g=10.0 b=0.0
    }
    object_3d b.world_pos {
        sphere radius=0.2 r=10.0 g=0.0 b=10.0
    }
    object_3d c.world_pos {
        sphere radius=0.2 r=0.0 g=1.0 b=10.0
    }
    line_3d b.world_pos line_width=10.0 color="yolk" perspective=#false
""")

sys = el.six_dof(sys=gravity)
sim = w.run(sys, SIM_TIME_STEP, run_time_step=1 / 120.0)
