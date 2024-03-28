from elodin.elodin import Quaternion
import jax
import jax.numpy as np
from jax.numpy import linalg as la
from dataclasses import dataclass
import elodin as el
from typing import Annotated

initial_angular_vel = np.array([2.0, 3.0, 4.0])
rw_force_clamp = 0.2
G = 6.6743e-11  #
M = 5.972e24  # mass of the Earth
earth_radius = 6378.1 * 1000
altitude = 400 * 1000
radius = earth_radius + altitude
velocity = np.sqrt(G * M / radius)

# sensors
GyroOmega = Annotated[
    jax.Array, el.Component("gyro_omega", el.ComponentType(el.PrimitiveType.F64, (3,)))
]

MagReadingBody = Annotated[
    jax.Array,
    el.Component("mag_reading_body", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
MagReadingRef = Annotated[
    jax.Array,
    el.Component("mag_reading_ref", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

StarReadingBody = Annotated[
    jax.Array,
    el.Component("star_reading_body", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
StarReadingRef = Annotated[
    jax.Array,
    el.Component("star_reading_ref", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


@dataclass
class Sensors(el.Archetype):
    gyro_omega: GyroOmega
    mag_reading_body: MagReadingBody
    mag_reading_ref: MagReadingRef
    star_reading_body: StarReadingBody
    star_reading_ref: StarReadingRef


@el.map
def fake_magnetometer_ref(pos: el.WorldPos) -> MagReadingRef:
    key = jax.random.key(jax.lax.convert_element_type(pos.linear()[0], "int64"))
    noise = 0.01 * jax.random.normal(key, shape=(4,))
    return el.Quaternion(pos.angular().vector() + noise).normalize() @ np.array(
        [1.0, 0.0, 0.0]
    )


@el.map
def fake_magnetometer_body(_: el.WorldPos) -> MagReadingBody:
    return np.array([1.0, 0.0, 0.0])


@el.map
def fake_star_ref(pos: el.WorldPos) -> StarReadingRef:
    key = jax.random.key(jax.lax.convert_element_type(pos.linear()[1], "int64"))
    noise = 0.01 * jax.random.normal(key, shape=(4,))
    return el.Quaternion(pos.angular().vector() + noise).normalize() @ np.array(
        [0.0, 0.0, 1.0]
    )


@el.map
def fake_star_body(_: el.WorldPos) -> StarReadingBody:
    return np.array([0.0, 0.0, 1.0])


@el.map
def gyro_omega(vel: el.WorldVel) -> GyroOmega:
    key = jax.random.key(jax.lax.convert_element_type(vel.linear()[0], "int64"))
    noise = 3.16e-7 * jax.random.normal(key, shape=(3,))
    return vel.angular() + noise


sensors = (
    fake_magnetometer_body.pipe(fake_magnetometer_ref)
    .pipe(fake_star_body)
    .pipe(fake_star_ref)
    .pipe(gyro_omega)
)

# attitude det - mekf
# source: Optimal Estimation of Dynamic Systems, 2nd Edition - Chapter 7


def calculate_covariance(
    sigma_g: jax.Array, sigma_b: jax.Array, dt: float
) -> jax.Array:
    variance_g = np.diag(sigma_g * sigma_g * dt)
    variance_b = np.diag(sigma_b * sigma_b * dt)
    Q_00 = variance_g + variance_b * dt**2 / 3
    Q_01 = variance_b * dt / 2
    Q_10 = Q_01
    Q_11 = variance_b
    return np.block([[Q_00, Q_01], [Q_10, Q_11]])


Q = calculate_covariance(
    np.array([0.1, 0.1, 0.1]), np.array([0.01, 0.01, 0.01]), 1 / 120.0
)
Y = np.diag(np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
YQY = Y @ Q @ Y.T
P = Annotated[
    jax.Array, el.Component("P", el.ComponentType(el.PrimitiveType.F64, (6, 6)))
]
AttEst = Annotated[
    el.Quaternion, el.Component("att_est", el.ComponentType.Quaternion)
]  # q_hat
AngVelEst = Annotated[
    jax.Array, el.Component("ang_vel_est", el.ComponentType(el.PrimitiveType.F64, (3,)))
]  # omega_hat
BiasEst = Annotated[
    jax.Array, el.Component("bias_est", el.ComponentType(el.PrimitiveType.F64, (3,)))
]  # b_hat
sensor_count = 2


# returns q_hat
def propogate_quaternion(q: Quaternion, omega: jax.Array, dt: float) -> Quaternion:
    omega_norm = la.norm(omega)
    c = np.cos(0.5 * omega_norm * dt)
    s = np.sin(0.5 * omega_norm * dt) / omega_norm
    omega_s = s * omega
    [x, y, z] = omega_s
    big_omega = np.array([[c, z, -y, x], [-z, c, x, y], [y, -x, c, z], [-x, -y, -z, c]])
    o = big_omega @ q.vector()
    return Quaternion(jax.lax.select(omega_norm > 1e-5, o, q.vector()))


# returns q_hat
def update_quaternion(q: Quaternion, delta_alpha: jax.Array) -> Quaternion:
    a = 0.5 * delta_alpha
    qa = Quaternion(np.array([a[0], a[1], a[2], 0.0]))
    q_hat = q + qa * q
    return q_hat.normalize()


def propogate_state_covariance(
    big_p: jax.Array, omega: jax.Array, dt: float
) -> jax.Array:
    omega_norm = la.norm(omega)
    s = np.sin(omega_norm * dt)
    c = np.cos(omega_norm * dt)
    p = s / omega_norm
    q = (1 - c) / (omega_norm**2)
    r = (omega_norm * dt - s) / (omega_norm**3)
    omega_cross = skew_symmetric_cross(omega)
    omega_cross_square = omega_cross @ omega_cross
    phi_00 = np.eye(3) - omega_cross * p + omega_cross_square * q
    phi_01 = omega_cross * q - np.eye(3) * dt - omega_cross_square * r
    phi_10 = np.zeros((3, 3))
    phi_11 = np.eye(3)
    phi = np.block([[phi_00, phi_01], [phi_10, phi_11]])
    # TODO(sphw): handle steady state eq 7.51b
    return (phi @ big_p @ phi.T) + YQY


def estimate_attitude(
    q_hat: Quaternion,
    b_hat: jax.Array,
    omega: jax.Array,
    p: jax.Array,
    measured_bodys: jax.Array,
    measured_references: jax.Array,
    dt: float,
) -> tuple[Quaternion, jax.Array, jax.Array]:
    q_hat = propogate_quaternion(q_hat, omega, dt)
    p = propogate_state_covariance(p, omega, dt)
    delta_x_hat = np.zeros(6)
    var_r = np.eye(3) * 0.001
    for i in range(0, sensor_count):
        measured_reference = measured_references[i]
        measured_body = measured_bodys[i]
        body_r = q_hat @ measured_reference
        H = np.block([skew_symmetric_cross(body_r), np.zeros((3, 3))])
        H_trans = H.T
        K = p @ H_trans @ la.inv(H @ p @ H_trans + var_r)
        e = measured_body - body_r
        delta_x_hat = delta_x_hat + K @ (e - H @ delta_x_hat)
        p = (np.eye(6) - K @ H) @ p
    delta_alpha = delta_x_hat[0:3]
    delta_beta = delta_x_hat[3:6]
    q_hat = update_quaternion(q_hat, delta_alpha)
    b_hat = b_hat + delta_beta
    return (q_hat, b_hat, p)


def skew_symmetric_cross(a: jax.Array) -> jax.Array:
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


@el.map
def kalman_filter(
    omega: GyroOmega,
    mag_body: MagReadingBody,
    mag_ref: MagReadingRef,
    star_body: StarReadingBody,
    star_ref: StarReadingRef,
    att_est: AttEst,
    b_hat: BiasEst,
    p: P,
) -> tuple[AttEst, AngVelEst, BiasEst, P]:
    q_hat, b_hat, big_p = estimate_attitude(
        att_est,
        b_hat,
        omega,
        p,
        np.array([mag_body, star_body]),
        np.array([mag_ref, star_ref]),
        1 / 120.0,
    )
    omega_hat = omega
    return (q_hat, omega_hat, b_hat, big_p)


@dataclass
class KalmanFilter(el.Archetype):
    p: P
    att_est: AttEst
    ang_vel_est: AngVelEst
    bias_est: BiasEst


# control system
Goal = Annotated[el.Quaternion, el.Component("goal", el.ComponentType.Quaternion)]
UserGoal = Annotated[
    el.Quaternion, el.Component("rot_deg", el.ComponentType.Quaternion)
]


@dataclass
class ControlInput(el.Archetype):
    goal: Goal


@dataclass
class UserInput(el.Archetype):
    deg: UserGoal


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


@el.map
def earth_point(pos: el.WorldPos, deg: UserGoal) -> Goal:
    linear = pos.linear()
    r = linear / la.norm(linear)
    body_axis = np.array([0.0, 0.0, -1.0])
    a = np.cross(body_axis, r)
    w = 1 + np.dot(body_axis, r)
    return deg * el.Quaternion(np.array([a[0], a[1], a[2], w])).normalize()


@el.system
def control(
    sensor: el.Query[AttEst, AngVelEst, Goal], rw: el.Query[RWAxis, RWForce]
) -> el.Query[RWForce]:
    control_force = sensor.map(
        el.Force,
        lambda p, v, i: el.SpatialForce.from_torque(
            -1.0 * v * d + -1.0 * (p.inverse().vector() - i.vector())[:3] * k
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


@el.map
def gravity_effector(
    _: Goal,
    force: el.Force,
    a_pos: el.WorldPos,
    a_inertia: el.Inertia,
) -> el.Force:
    r = a_pos.linear()
    m = a_inertia.mass()
    norm = la.norm(r)
    f = G * M * m * r / (norm * norm * norm)
    return force + el.SpatialForce.from_linear(-f)


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
        ),
    )
    .insert(UserInput(el.Quaternion(np.array([0.0, 1.0, 0.0, 0.0]))))
    .insert(Sensors(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)))
    .insert(
        KalmanFilter(np.identity(6), el.Quaternion.identity(), np.zeros(3), np.zeros(3))
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
        1 / 120.0,
        sensors
        | kalman_filter
        | control
        | rw_effector
        | gravity_effector
        | earth_point,
    ),
    time_step=1.0 / 240.0,
)
