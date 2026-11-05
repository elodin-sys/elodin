from dataclasses import field

import elodin as el
import elodin.ui as ui
import jax
from jax import numpy as jnp
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0
# Set the gravitational constant for Newton's law of universal gravitation
G = 6.6743e-11

# Axis-major 3×N world points for A, B, C.
TrailPositions = el.Annotated[
    jax.Array,
    el.Component(
        "trail_positions",
        el.ComponentType(el.PrimitiveType.F64, (9,)),
        metadata={"element_names": "x0,x1,x2,y0,y1,y2,z0,z1,z2"},
    ),
]


@el.dataclass
class TrailBuf(el.Archetype):
    trail_positions: TrailPositions = field(default_factory=lambda: jnp.zeros(9))


w = el.World()

a = w.spawn(
    [
        el.Body(
            world_pos=el.WorldPos(linear=jnp.array([0.8920281421, 0.0, 0.0])),
            world_vel=el.WorldVel(linear=jnp.array([0.0, 0.9957939373, 0.0])),
            inertia=el.Inertia(1.0 / G),
        ),
        TrailBuf(),
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


@el.system
def pack_trails(
    dest: el.Query[TrailPositions],
    bodies: el.Query[el.WorldPos],
) -> el.Query[TrailPositions]:
    pts = jnp.stack(
        [bodies[0].linear(), bodies[1].linear(), bodies[2].linear()],
        axis=1,
    )
    packed = pts.reshape(-1)
    return dest.map(TrailPositions, lambda _: packed)


# Add the gravity constraint entities to the world
w.spawn(GravityConstraint(a, b), name="A -> B")
w.spawn(GravityConstraint(b, a), name="B -> A")

w.spawn(GravityConstraint(a, c), name="A -> C")
w.spawn(GravityConstraint(b, c), name="B -> C")

w.spawn(GravityConstraint(c, a), name="C -> A")
w.spawn(GravityConstraint(c, b), name="C -> B")

w.schematic(
    ui.schematic(
        ui.hsplit(
            ui.tabs(
                ui.hierarchy(),
                ui.schematic_tree(),
                share=0.2,
            ),
            ui.tabs(
                ui.viewport(
                    name="Viewport",
                    pos="(0,0,0,0,0,0,3)",
                    look_at="(0,0,0,0,0,0,0)",
                    hdr=True,
                ),
                ui.graph("a.world_pos", name="Graph"),
                share=0.6,
            ),
            ui.tabs(
                ui.inspector(),
                share=0.2,
            ),
        ),
        ui.object_3d(
            "a.world_pos",
            mesh=ui.sphere(radius=0.2, emissivity=1.0, color="yellow"),
        ),
        ui.object_3d(
            "b.world_pos",
            mesh=ui.sphere(radius=0.2, emissivity=1.0, color="pink"),
        ),
        ui.object_3d(
            "c.world_pos",
            mesh=ui.sphere(radius=0.2, emissivity=1.0, color="cyan"),
        ),
        ui.line_3d("b.world_pos", line_width=10.0, perspective=False, color="yolk"),
        ui.point_trails(
            "a.trail_positions",
            head_shape="sphere",
            head_size=0.12,
            line_width=5.0,
            max_length=2.0,
            color="white",
        ),
        # This is not exactly correct. HCI might be more appropriate.
        coordinate=ui.coordinate(frame="ECEF"),
    ),
)


sys = el.six_dof(sys=gravity) | pack_trails
sim = w.run(
    sys,
    simulation_rate=1.0 / SIM_TIME_STEP,
    generate_real_time=True,
)
