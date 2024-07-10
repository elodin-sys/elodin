from dataclasses import dataclass
import jax
import elodin as el
from jax import numpy as np
from jax.numpy import linalg as la

TIME_STEP = 1.0 / 240.0


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=np.array([0.0, 0.0, -9.81]) * inertia.mass())


elasticity = 0.85


@el.map
def bounce(p: el.WorldPos, v: el.WorldVel, inertia: el.Inertia) -> el.WorldVel:
    return v + jax.lax.cond(
        p.linear()[2] <= 0.4,
        lambda: collison_impulse_static(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.4]),
            v,
            inertia,
        ),
        lambda: el.SpatialMotion.zero(),
    )


@dataclass
class Wall:
    normal: jax.Array
    pos: jax.Array
    pos_dim: int
    leq: bool = False


@el.map
def walls(p: el.WorldPos, v: el.WorldVel, inertia: el.Inertia) -> el.WorldVel:
    walls = [
        Wall(np.array([1.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), 0),
        Wall(np.array([1.0, 0.0, 0.0]), np.array([-5.0, 0.0, 0.0]), 0, True),
        Wall(np.array([0.0, 1.0, 0.0]), np.array([0.0, 5.0, 0.0]), 2),
        Wall(np.array([0.0, 1.0, 0.0]), np.array([0.0, -5.0, 0.0]), 2, True),
    ]
    for wall in walls:
        if wall.leq:
            v += jax.lax.cond(
                p.linear()[wall.pos_dim] <= wall.pos[wall.pos_dim],
                lambda: collison_impulse_static(
                    wall.normal,
                    wall.normal * 0.4,
                    v,
                    inertia,
                ),
                lambda: el.SpatialMotion.zero(),
            )
        else:
            v += jax.lax.cond(
                p.linear()[wall.pos_dim] >= wall.pos[wall.pos_dim],
                lambda: collison_impulse_static(
                    wall.normal,
                    wall.normal * 0.4,
                    v,
                    inertia,
                ),
                lambda: el.SpatialMotion.zero(),
            )
    return v


def collison_impulse_static(
    norm: jax.Array,
    r_a: jax.Array,
    vel_a: el.WorldVel,
    inertia_a: el.Inertia,
) -> el.SpatialMotion:
    mass_a = inertia_a.mass()
    inverse_mass_a = 1.0 / mass_a
    inverse_inertia_a = 1.0 / inertia_a.inertia_diag()
    v = vel_a.linear()
    jr_top = np.dot(-1 * (1 + elasticity) * v, norm)
    jr_bottom = inverse_mass_a + (
        inverse_inertia_a * np.cross(np.cross(r_a, norm), r_a)
    ).dot(norm)
    jr = jr_top / jr_bottom
    impulse = jr / mass_a * norm
    return el.SpatialMotion(linear=impulse)


def collison_impulse(
    norm: jax.Array,
    r_a: jax.Array,
    r_b: jax.Array,
    vel_a: el.WorldVel,
    inertia_a: el.Inertia,
    vel_b: el.WorldVel,
    inertia_b: el.Inertia,
) -> el.SpatialMotion:
    mass_a = inertia_a.mass()
    mass_b = inertia_b.mass()
    inverse_mass_a = 1.0 / mass_a
    inverse_mass_b = 1.0 / mass_b
    inverse_inertia_a = 1.0 / inertia_a.inertia_diag()
    inverse_inertia_b = 1.0 / inertia_b.inertia_diag()
    v = vel_a.linear() - vel_b.linear()
    jr_top = np.dot(-1 * (1 + elasticity) * v, norm)
    jr_bottom = inverse_mass_a + inverse_mass_b
    jr_bottom = (
        inverse_mass_a
        + inverse_mass_b
        + (
            inverse_inertia_a * np.cross(np.cross(r_a, norm), r_b)
            + inverse_inertia_b * np.cross(np.cross(r_a, norm), r_b)
        ).dot(norm)
    )
    jr = jr_top / jr_bottom
    impulse = jr / mass_a * norm
    return el.SpatialMotion(linear=impulse)


@el.system
def collide(
    graph: el.GraphQuery[el.TotalEdge],
    query: el.Query[el.WorldPos, el.WorldVel, el.Inertia],
    vel: el.Query[el.WorldVel],
) -> el.Query[el.WorldVel]:
    def collide_inner(acc, pos_a, vel_a, inertia_a, pos_b, vel_b, inertia_b):
        r = pos_a.linear() - pos_b.linear()
        dist = la.norm(r)
        norm = r / dist
        r_a = norm * 0.4
        r_b = norm * 0.4
        impulse = jax.lax.cond(
            dist <= 0.8,
            lambda: collison_impulse(
                norm, r_a, r_b, vel_a, inertia_a, vel_b, inertia_b
            ),
            lambda: el.SpatialMotion.zero(),
        )
        return acc + impulse

    impulse = graph.edge_fold(
        query, query, el.WorldVel, el.SpatialMotion.zero(), collide_inner
    )
    return vel.join(impulse).map(el.WorldVel, lambda vel, impulse: vel + impulse)


world = el.World()

balls = []
mesh = world.insert_asset(el.Mesh.sphere(0.2))
for i in range(1, 200):
    key = jax.random.key(i)
    pos = jax.random.uniform(key, shape=(3,), minval=-5.0, maxval=5.0) + np.array(
        [0.0, 0.0, 7.0]
    )
    [r, g, b] = jax.random.uniform(key, shape=(3,), minval=0.0, maxval=1.0) * 2.0
    color = world.insert_asset(el.Material.color(r, g, b))
    ball = world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(linear=pos),
                world_vel=el.SpatialMotion(
                    linear=jax.random.normal(key, shape=(3,)) * 3.0
                ),
            ),
            el.Shape(mesh, color),
        ],
        name=f"Ball {i}",
    )
    balls.append(ball)


world.spawn(
    el.Panel.viewport(
        track_rotation=False,
        active=True,
        pos=[6.0, 6.0, 3.0],
        looking_at=[0.0, 0.0, 1.0],
        show_grid=True,
        hdr=True,
    ),
    name="Viewport",
)
world.spawn(
    el.Panel.graph(
        [el.GraphEntity(b, el.Component.index(el.WorldPos)[4:]) for b in balls]
    ),
)


sys = bounce.pipe(collide).pipe(walls).pipe(el.six_dof(TIME_STEP, gravity))
world.run(sys, TIME_STEP)
