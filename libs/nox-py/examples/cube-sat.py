import jax
import jax.numpy as np
from jax.numpy import linalg as la
import numpy
from elodin import *

initial_angular_vel = np.array([-0.0, 0.0, 0.0])
rw_force_clamp = 0.2
G = 6.6743e-11 #
M = 5.972e24  # mass of the Earth
earth_radius = 6378.1 * 1000
altitude = 800 * 1000
radius = earth_radius + altitude
velocity = np.sqrt(G * M / radius)


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

Goal = Annotated[Quaternion, Component("goal", ComponentType.Quaternion)]


@dataclass
class ControlInput(Archetype):
    goal: Goal


RWAxis = Annotated[
    jax.Array, Component("rw_axis", ComponentType(PrimitiveType.F64, (3,)))
]
RWForce = Annotated[SpatialForce, Component("rw_force", ComponentType.SpatialMotionF64)]


@dataclass
class ReactionWheel(Archetype):
    rw_force: RWForce
    axis: RWAxis


@system
def earth_point(sensor: Query[WorldPos, Goal]) -> Query[Goal]:
    def earth_point_inner(pos, goal):
        linear = pos.linear()
        r = linear / la.norm(linear)
        body_axis = np.array([0.0, 0.0, -1.0])
        a = np.cross(body_axis, r)
        w = 1 + np.dot(body_axis, r)
        return Quaternion(np.array([a[0], a[1], a[2], w])).normalize()
    return sensor.map(Goal, earth_point_inner)


@system
def control(
    sensor: Query[WorldPos, WorldVel, Goal], rw: Query[RWAxis, RWForce]
) -> Query[RWForce]:
    control_force = sensor.map(
        Force,
        lambda p, v, i: Force.from_torque(
            -1.0 * v.angular() * d + -1.0 * (p.angular().vector() - i.vector())[:3] * k
        ),
    ).bufs[0][0]
    return rw.map(
        RWForce,
        lambda axis, _torque: RWForce.from_torque(
            np.clip(
                np.dot(control_force[:3], axis) * axis, -rw_force_clamp, rw_force_clamp
            )
        ),
    )


@system
def rw_effector(
    rw_force: Query[RWForce], torque: Query[Goal, WorldPos, WorldVel, Force]
) -> Query[Force]:
    force = np.sum(rw_force.bufs[0], 0)
    return torque.map(Force, lambda _g, _p, _v, _torque: Force.from_torque(force[:3]))


@system
def gravity_effector(q: Query[Goal, Force, WorldPos, Inertia]) -> Query[Force]:
    def gravity_inner(_, force, a_pos, a_inertia):
        r = a_pos.linear()
        m = a_inertia.mass()
        norm = la.norm(r)
        f = G * M * m * r / (norm * norm * norm)
        return force + Force.from_linear(-f)

    return q.map(Force, gravity_inner)



w = World()
b = Body(
    world_pos=SpatialTransform.from_linear(np.array([1.0, 0.0, 0.0]) * radius),
    world_vel=WorldVel(initial_angular_vel, np.array([0.0, 0.0, -1.0]) * velocity),
    inertia=Inertia(12.0, j),
    pbr=w.insert_asset(
        Pbr.from_url(
            "https://storage.googleapis.com/elodin-marketing/models/oresat-low.glb"
        )
    ),
    # Credit to the OreSat program https://www.oresat.org for the model above
)
w.spawn(
    ReactionWheel(
        rw_force=SpatialForce.zero(),
        axis=np.array([1.0, 0.0, 0.0]),
    )
).metadata(EntityMetadata("RW1"))
w.spawn(
    ReactionWheel(
        rw_force=SpatialForce.zero(),
        axis=np.array([0.0, 1.0, 0.0]),
    )
).metadata(EntityMetadata("RW2"))
w.spawn(
    ReactionWheel(
        rw_force=SpatialForce.zero(),
        axis=np.array([0.0, 0.0, 1.0]),
    )
).metadata(EntityMetadata("RW3"))
w.spawn(b).metadata(EntityMetadata("OreSat")).insert(
    ControlInput(Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(0)))
)

w.spawn(
    Body(
        world_pos=WorldPos.from_linear(np.array([0.0, 0.0, 0.0])),
        world_vel=WorldVel.from_linear(np.array([0.0, 0.0, 0.0])),
        inertia=Inertia.from_mass(1.0),
        pbr=w.insert_asset(Pbr.from_url("https://storage.googleapis.com/elodin-marketing/models/earth.glb"))
    )).metadata(EntityMetadata("Earth"))

exec = w.run(
    six_dof(1/10.0, control.pipe(rw_effector).pipe(gravity_effector).pipe(earth_point)), 1.0/240.0)
