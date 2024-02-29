import elodin
import jax
import jax.numpy as np
import numpy
from elodin import *


def axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    return np.array([s * axis[0], s * axis[1], s * axis[2], c, 0.0, 0.0, 0.0])


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

print(axis_angle(np.array([0.0, 0.0, 1.0]), np.radians(45))[:3] * k)

Goal = Component[SpatialTransform, "goal", ComponentType.SpatialPosF64]

@dataclass
class ControlInput(Archetype):
    goal: Goal


RWSpeed = Component[jax.Array, "rw_speed", ComponentType.F64]
RWAxis = Component[jax.Array, "rw_axis", ComponentType(PrimitiveType.F64, (3,))]
RWForce = Component[SpatialForce, "rw_force", ComponentType.SpatialMotionF64]


@dataclass
class ReactionWheel(Archetype):
    rw_force: RWForce
    speed: RWSpeed
    axis: RWAxis


@system
def control(
    sensor: Query[WorldPos, WorldVel, Goal], rw: Query[RWAxis, RWForce]
) -> ComponentArray[RWForce]:
    control_force = sensor.map(
        Force,
        lambda p, v, i: Force.from_torque(
            -1.0 * v.angular() * d
            + -1.0 * (p.angular().vector() - i.angular().vector())[:3] * k
        ),
    ).buf[0]
    for buf in rw.bufs:
        print(buf.shape)
    return rw.map(
        RWForce,
        lambda axis, _torque: RWForce.from_torque(
            np.clip(np.dot(control_force[:3], axis) * axis, -0.5, 0.5)
        ),
    )


@system
def rw_effector(
    rw_force: ComponentArray[RWForce], torque: Query[WorldPos, WorldVel, Force]
) -> ComponentArray[Force]:
    force = np.sum(rw_force.buf, 0)
    return torque.map(Force, lambda _p, _v, _torque: Force.from_torque(force[:3]))


w = WorldBuilder()
b = Body(
    world_pos=SpatialTransform.from_linear(np.array([0.0, 0.0, 0.0])),
    world_vel=WorldVel.from_angular(np.array([0.0, 0.0, 0.0])),
    world_accel=WorldVel.from_linear(np.array([0.0, 0.0, 0.0])),
    force=Force.zero(),
    inertia=Inertia(12.0, j),
    pbr=w.insert_asset(Pbr.from_path("examples/oresat-low.glb")),
)
w.spawn(
    ReactionWheel(
        rw_force=SpatialForce.zero(),
        speed=np.array(0.0),
        axis=np.array([1.0, 0.0, 0.0]),
    )
).metadata(EntityMetadata("RW1"))
w.spawn(
    ReactionWheel(
        rw_force=SpatialForce.zero(),
        speed=np.array(0.0),
        axis=np.array([0.0, 1.0, 0.0]),
    )
).metadata(EntityMetadata("RW2"))
w.spawn(
    ReactionWheel(
        rw_force=SpatialForce.zero(),
        speed=np.array(0.0),
        axis=np.array([0.0, 0.0, 1.0]),
    )
).metadata(EntityMetadata("RW3"))
w.spawn(b).metadata(EntityMetadata("OreSat")).insert(
    ControlInput(
        SpatialTransform(axis_angle(np.array([1.0, 1.0, 0.0]), np.radians(45)))
    )
)

exec = w.run(six_dof(1.0 / 60.0, control.pipe(rw_effector)), 1.0 / 60.0)
