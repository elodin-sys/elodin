import elodin as el
from jax import numpy as np
from jax.numpy import linalg as la

SIM_TIME_STEP = 1.0 / 120.0

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
        return el.Force(linear=force.force() - f)

    return graph.edge_fold(query, query, el.Force, el.Force.zero(), gravity_inner)


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

w.spawn(
    el.Panel.viewport(
        pos=[0.0, -3.0, 3.0],
        looking_at=[0.0, 0.0, 0.0],
        hdr=True,
    ),
    name="Viewport 1",
)

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


w.spawn(el.VectorArrow(a, "world_vel", offset=3, body_frame=False, scale=1.0))

w.spawn(el.Line3d(b, "world_pos", index=[4, 5, 6], line_width=10.0))

sys = el.six_dof(sys=gravity)
sim = w.run(sys, SIM_TIME_STEP)
