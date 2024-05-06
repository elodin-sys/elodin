from jax import numpy as np
from jax.numpy import linalg as la
import elodin as el

TIME_STEP = 1.0 / 120.0

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
        return el.SpatialForce.from_linear(force.force() - f)

    return graph.edge_fold(
        query, query, el.Force, el.SpatialForce.zero(), gravity_inner
    )


w = el.World()
a = w.spawn(
    el.Body(
        world_pos=el.WorldPos.from_linear(np.array([0.8822391241, 0, 0])),
        world_vel=el.WorldVel.from_linear(np.array([0, 1.0042424155, 0])),
        inertia=el.SpatialInertia(1.0 / G),
        pbr=w.insert_asset(
            el.Pbr(el.Mesh.sphere(0.2), el.Material.color(25.3, 18.4, 1.0))
        ),
    ),
    name="A",
)
b = w.spawn(
    el.Body(
        world_pos=el.WorldPos.from_linear(np.array([-0.6432718586, 0, 0])),
        world_vel=el.WorldVel.from_linear(np.array([0, -1.6491842814, 0])),
        inertia=el.SpatialInertia(1.0 / G),
        pbr=w.insert_asset(
            el.Pbr(el.Mesh.sphere(0.2), el.Material.color(10.0, 0.0, 10.0))
        ),
    ),
    name="B",
)
c = w.spawn(
    el.Body(
        world_pos=el.WorldPos.from_linear(np.array([-0.2389672654, 0, 0])),
        world_vel=el.WorldVel.from_linear(np.array([0, 0.6449418659, 0.0])),
        inertia=el.SpatialInertia(1.0 / G),
        pbr=w.insert_asset(
            el.Pbr(el.Mesh.sphere(0.2), el.Material.color(0.0, 10.0, 10.0))
        ),
    ),
    name="C",
)
w.spawn(GravityConstraint(a, b), name="A -> B")
w.spawn(GravityConstraint(a, c), name="A -> C")

w.spawn(GravityConstraint(b, c), name="B -> C")
w.spawn(GravityConstraint(b, a), name="B -> A")

w.spawn(GravityConstraint(c, a), name="C -> A")
w.spawn(GravityConstraint(c, b), name="C -> B")

w.spawn(
    el.Panel.viewport(
        track_rotation=False,
        active=True,
        pos=[0.0, 0.0, 5.0],
        looking_at=[0.0, 0.0, 0.0],
        hdr=True,
    ),
    name="Viewport 1",
)

sys = el.six_dof(TIME_STEP, gravity)
sim = w.run(sys, TIME_STEP)
