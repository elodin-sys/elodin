from jax import numpy as np
from jax.numpy import linalg as la
import elodin as el

TIME_STEP = 1.0 / 120.0

G = 6.6743e-11


@el.system
def gravity(
    graph: el.GraphQuery[el.TotalEdge],
    query: el.Query[el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_inner(force, a_pos, a_inertia, b_pos, b_inertia):
        r = a_pos.linear() - b_pos.linear()
        m = a_inertia.mass()
        M = b_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return el.SpatialForce(linear=force.force() - f)

    return graph.edge_fold(
        query, query, el.Force, el.SpatialForce.zero(), gravity_inner
    )


w = el.World()
mesh = w.insert_asset(el.Mesh.sphere(0.2))
a = w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.8822391241, 0, 0])),
            world_vel=el.SpatialMotion(linear=np.array([0, 0, 1.0042424155])),
            inertia=el.SpatialInertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(25.3, 18.4, 1.0))),
    ],
    name="A",
)
b = w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([-0.6432718586, 0, 0])),
            world_vel=el.SpatialMotion(linear=np.array([0, 0, -1.6491842814])),
            inertia=el.SpatialInertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(10.0, 0.0, 10.0))),
    ],
    name="B",
)
c = w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([-0.2389672654, 0, 0])),
            world_vel=el.SpatialMotion(linear=np.array([0, 0, 0.6449418659])),
            inertia=el.SpatialInertia(1.0 / G),
        ),
        el.Shape(mesh, w.insert_asset(el.Material.color(0.0, 1.0, 10.0))),
    ],
    name="C",
)

w.spawn(
    el.Panel.viewport(
        track_rotation=False,
        active=True,
        pos=[0.0, 5.0, 0.0],
        looking_at=[0.0, 0.0, 0.0],
        hdr=True,
    ),
    name="Viewport 1",
)


w.spawn(el.VectorArrow(a, "world_vel", offset=3, body_frame=False, scale=1.0))

sys = el.six_dof(TIME_STEP, gravity)
sim = w.run(sys, TIME_STEP)
