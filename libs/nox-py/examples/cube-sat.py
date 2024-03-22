import jax
import jax.numpy as np
from jax.numpy import linalg as la
from dataclasses import dataclass
import elodin as el
from elodin import Annotated

initial_angular_vel = np.array([-2.0, 3.0, 1.0])
rw_force_clamp = 0.2
G = 6.6743e-11  #
M = 5.972e24  # mass of the Earth
earth_radius = 6378.1 * 1000
altitude = 800 * 1000
radius = earth_radius + altitude
velocity = np.sqrt(G * M / radius)

# control system
Goal = Annotated[el.Quaternion, el.Component("goal", el.ComponentType.Quaternion)]


@dataclass
class ControlInput(el.Archetype):
    goal: Goal


RWAxis = Annotated[
    jax.Array, el.Component("rw_axis", el.ComponentType(el.PrimitiveType.F64, (3,)))
]

RWForce = Annotated[
    el.SpatialForce, el.Component("rw_force", el.ComponentType.SpatialMotionF64)
]


def lqr_control_mat(j, q, r):
    d_diag = np.array(
        [
            np.sqrt(q1i / ri + ji * np.sqrt(q2i / ri))
            for (q1i, q2i, ri, ji) in zip(q[:3], q[3:], r, j)
        ]
    )
    k_diag = np.array([np.sqrt(q2i / ri) for (q2i, ri) in zip(q[3:], r)])
    return (d_diag, k_diag)


j = np.array([0.13, 0.10, 0.05])
q = np.array([5, 5, 5, 5, 5, 5])
r = np.array([8.0, 8.0, 8.0])
(d, k) = lqr_control_mat(j, q, r)


@el.system
def earth_point(sensor: el.Query[el.WorldPos, Goal]) -> el.Query[Goal]:
    def earth_point_inner(pos, _):
        linear = pos.linear()
        r = linear / la.norm(linear)
        body_axis = np.array([0.0, 0.0, -1.0])
        a = np.cross(body_axis, r)
        w = 1 + np.dot(body_axis, r)
        return el.Quaternion(np.array([a[0], a[1], a[2], w])).normalize()

    return sensor.map(Goal, earth_point_inner)


@el.system
def control(
    sensor: el.Query[el.WorldPos, el.WorldVel, Goal], rw: el.Query[RWAxis, RWForce]
) -> el.Query[RWForce]:
    control_force = sensor.map(
        el.Force,
        lambda p, v, i: el.SpatialForce.from_torque(
            -1.0 * v.angular() * d + -1.0 * (p.angular().vector() - i.vector())[:3] * k
        ),
    ).bufs[0][0]
    return rw.map(
        RWForce,
        lambda axis, _: el.SpatialForce.from_torque(
            np.clip(
                np.dot(control_force[:3], axis) * axis, -rw_force_clamp, rw_force_clamp
            )
        ),
    )


# effectors
@dataclass
class ReactionWheel(el.Archetype):
    rw_force: RWForce
    axis: RWAxis


@el.system
def rw_effector(
    rw_force: el.Query[RWForce], torque: el.Query[Goal]
) -> el.Query[el.Force]:
    force = np.sum(rw_force.bufs[0], 0)
    return torque.map(el.Force, lambda _: el.SpatialForce.from_torque(force[:3]))


@el.system
def gravity_effector(
    q: el.Query[Goal, el.Force, el.WorldPos, el.Inertia],
) -> el.Query[el.Force]:
    def gravity_inner(_, force, a_pos, a_inertia):
        r = a_pos.linear()
        m = a_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return force + el.SpatialForce.from_linear(-f)

    return q.map(el.Force, gravity_inner)


w = el.World()

b = el.Body(
    world_pos=el.SpatialTransform.from_linear(np.array([1.0, 0.0, 0.0]) * radius),
    world_vel=el.SpatialMotion(
        initial_angular_vel, np.array([0.0, 0.0, -1.0]) * velocity
    ),
    inertia=el.SpatialInertia(12.0, j),
    pbr=w.insert_asset(
        el.Pbr.from_url(
            "https://storage.googleapis.com/elodin-marketing/models/oresat-low.glb"
        )
    ),
    # Credit to the OreSat program https://www.oresat.org for the model above
)
w.spawn(
    ReactionWheel(
        rw_force=el.SpatialForce.zero(),
        axis=np.array([1.0, 0.0, 0.0]),
    )
).name("Reaction Wheel 1")

w.spawn(
    ReactionWheel(
        rw_force=el.SpatialForce.zero(),
        axis=np.array([0.0, 1.0, 0.0]),
    )
).name("Reaction Wheel 2")

w.spawn(
    ReactionWheel(
        rw_force=el.SpatialForce.zero(),
        axis=np.array([0.0, 0.0, 1.0]),
    )
).name("Reaction Wheel 3")

sat = (
    w.spawn(b)
    .metadata(el.EntityMetadata("OreSat"))
    .insert(
        ControlInput(
            el.Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(0))
        )
    )
)

w.spawn(
    el.Panel.viewport(
        track_entity=sat.id(),
        track_rotation=False,
        active=True,
        pos=[7.0, 0.0, 0.0],
        looking_at=[0.0, 0.0, 0.0],
    )
).name("Viewport 2")

w.spawn(
    el.Panel.viewport(
        track_entity=sat.id(),
        track_rotation=False,
        pos=[7.0, -3.0, 0.0],
        fov=20.0,
        looking_at=[0.0, 0.0, 0.0],
    )
).name("Viewport 1")

w.spawn(
    el.Body(
        world_pos=el.SpatialTransform.from_linear(np.array([0.0, 0.0, 0.0])),
        world_vel=el.SpatialMotion.from_angular(
            np.array([0.0, 1.0, 0.0]) * 7.2921159e-5
        ),
        inertia=el.SpatialInertia.from_mass(1.0),
        pbr=w.insert_asset(
            el.Pbr.from_url(
                "https://storage.googleapis.com/elodin-marketing/models/earth.glb"
            )
        ),
    )
).name("Earth")


exec = w.run(
    system=el.six_dof(
        1 / 30.0, control.pipe(rw_effector).pipe(gravity_effector).pipe(earth_point)
    ),
    time_step=1.0 / 240.0,
)
