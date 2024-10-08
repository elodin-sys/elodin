import elodin as el
import jax
from jax import numpy as np
from jax.numpy import linalg as la

TIME_STEP = 1.0 / 30.0

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

    return graph.edge_fold(query, query, el.Force, el.SpatialForce(), gravity_inner)


w = el.World()
mesh = w.insert_asset(el.Mesh.sphere(0.2))
for i in range(1, 200):
    key = jax.random.key(i)
    pos = jax.random.uniform(key, shape=(3,), minval=-10.0, maxval=10.0)
    # vel = jax.random.uniform(key, shape=(3,), minval=-5.0, maxval=5.0)
    vel = np.zeros(3)
    [r, g, b] = jax.random.uniform(key, shape=(3,), minval=0.0, maxval=1.0) * 2.0
    color = w.insert_asset(el.Material.color(r, g, b))
    body = w.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=pos),
                world_vel=el.SpatialMotion(linear=vel),
                inertia=el.SpatialInertia(1.0 / G),
            ),
            el.Shape(mesh, color),
        ],
        name=f"Body {i}",
    )
w.spawn(
    el.Panel.viewport(
        track_rotation=False,
        active=True,
        pos=[100.0, 5.0, 100.0],
        looking_at=[0.0, 0.0, 0.0],
        hdr=True,
    ),
    name="Viewport 1",
)

sys = el.six_dof(sys=gravity)
sim = w.run(sys, 1 / 240.0)
