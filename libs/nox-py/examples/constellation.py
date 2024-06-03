from elodin.elodin import Quaternion
import jax
import jax.numpy as np
from jax.numpy import linalg as la
from dataclasses import dataclass
import elodin as el
from typing import Annotated

initial_angular_vel = np.array([0.5, 0.7, 0.6])
rw_force_clamp = 0.02
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
AttEst = Annotated[el.Quaternion, el.Component("att_est")]  # q_hat
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
Goal = Annotated[el.Quaternion, el.Component("goal")]
UserGoal = Annotated[
    jax.Array, el.Component("euler_input", el.ComponentType(el.PrimitiveType.F64, (3,)))
]

RWAxis = Annotated[
    jax.Array, el.Component("rw_axis", el.ComponentType(el.PrimitiveType.F64, (3,)))
]

RWForce = Annotated[el.SpatialForce, el.Component("rw_force")]

ControlForce = Annotated[el.SpatialForce, el.Component("control_force")]


@dataclass
class ControlInput(el.Archetype):
    goal: Goal
    control_force: ControlForce


@dataclass
class UserInput(el.Archetype):
    deg: UserGoal


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


def euler_to_quat(angles: jax.Array) -> el.Quaternion:
    [roll, pitch, yaw] = np.deg2rad(angles)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return el.Quaternion(np.array([x, y, z, w]))


@el.map
def earth_point(pos: el.WorldPos, deg: UserGoal) -> Goal:
    linear = pos.linear()
    r = linear / la.norm(linear)
    body_axis = np.array([0.0, 0.0, -1.0])
    a = np.cross(body_axis, r)
    w = 1 + np.dot(body_axis, r)
    return (
        euler_to_quat(deg) * el.Quaternion(np.array([a[0], a[1], a[2], w])).normalize()
    )


@el.system
def control(
    sensor: el.Query[AttEst, AngVelEst, Goal], rw: el.Query[RWAxis, RWForce]
) -> el.Query[ControlForce]:
    return sensor.map(
        ControlForce,
        lambda p, v, i: el.SpatialForce.from_torque(
            -1.0 * v * d + -1.0 * (p.inverse().vector() - i.vector())[:3] * k
        ),
    )


RWEdge = Annotated[el.Edge, el.Component("rw_edge")]


@el.system
def actuator_allocator(
    q: el.GraphQuery[Annotated[RWEdge, el.RevEdge]],
    rw_query: el.Query[RWAxis],
    control_query: el.Query[ControlForce],
) -> el.Query[RWForce]:
    return q.edge_fold(
        rw_query,
        control_query,
        RWForce,
        el.SpatialForce.zero(),
        lambda xs, axis, control_force: xs
        + el.SpatialForce.from_torque(
            np.clip(
                np.dot(control_force.torque(), axis) * axis,
                -rw_force_clamp,
                rw_force_clamp,
            )
        ),
    )


# effectors


@dataclass
class RWRel(el.Archetype):
    edge: RWEdge


@dataclass
class ReactionWheel(el.Archetype):
    rw_force: RWForce
    axis: RWAxis


@el.system
def rw_effector(
    rw_force: el.GraphQuery[RWEdge],
    force_query: el.Query[el.Force],
    rw_query: el.Query[RWForce],
) -> el.Query[el.Force]:
    return rw_force.edge_fold(
        force_query,
        rw_query,
        el.Force,
        el.SpatialForce.zero(),
        lambda f, _, force: f + force,
    )


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

sat_ids = []

scene = w.glb("https://storage.googleapis.com/elodin-marketing/models/oresat-low.glb")


def spawn_sat(x, y, w: el.World):
    sat_num = x + y * 100
    rand_key = jax.random.key(sat_num)
    rw_1 = w.spawn(
        ReactionWheel(
            rw_force=el.SpatialForce.zero(),
            axis=np.array([1.0, 0.0, 0.0]),
        ),
        name=f"Sat {sat_num} Reaction Wheel 1",
    )

    rw_2 = w.spawn(
        ReactionWheel(
            rw_force=el.SpatialForce.zero(),
            axis=np.array([0.0, 1.0, 0.0]),
        ),
        name=f"Sat {sat_num} Reaction Wheel 2",
    )

    rw_3 = w.spawn(
        ReactionWheel(
            rw_force=el.SpatialForce.zero(),
            axis=np.array([0.0, 0.0, 1.0]),
        ),
        name=f"Sat {sat_num} Reaction Wheel 3",
    )

    rot_x = el.Quaternion.from_axis_angle(
        np.array([0.0, 1.0, 0.0]), np.radians(0.00003 * x)
    )
    rot_y = el.Quaternion.from_axis_angle(
        np.array([0.0, 0.0, 1.0]), np.radians(0.00003 * y)
    )
    rot = rot_x * rot_y
    ang_vel = jax.random.normal(rand_key, shape=(3,))
    ang_vel = ang_vel / la.norm(ang_vel) * 3.0
    b = el.Body(
        world_pos=el.SpatialTransform.from_linear(
            rot @ np.array([1.0, 0.0, 0.0]) * radius
        ),
        world_vel=el.SpatialMotion(
            ang_vel, rot @ np.array([0.0, 0.0, -1.0]) * velocity
        ),
        inertia=el.SpatialInertia(12.0, j),
        # Credit to the OreSat program https://www.oresat.org for the model above
    )

    sat = w.spawn(
        [
            b,
            ControlInput(
                el.Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(0)),
                el.SpatialForce.zero(),
            ),
            UserInput(np.array([0.0, 0.0, 0.0])),
            Sensors(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
            KalmanFilter(
                np.identity(6), el.Quaternion.identity(), np.zeros(3), np.zeros(3)
            ),
            scene,
        ],
        name=f"OreSat {sat_num}",
    )

    w.spawn(RWRel(el.Edge(sat, rw_1)), name=f"Sat {sat_num}  -> RW 1")
    w.spawn(RWRel(el.Edge(sat, rw_2)), name=f"Sat {sat_num} -> RW 2")
    w.spawn(RWRel(el.Edge(sat, rw_3)), name=f"Sat {sat_num} -> RW 3")
    sat_ids.append(sat)
    return sat


sat = spawn_sat(0, 0, w)
count = 80
bound = int(count**0.5 / 2)
for x in range(-bound, bound):
    for y in range(-bound, bound):
        spawn_sat(x, y, w)


w.spawn(
    el.Panel.hsplit(
        [
            el.Panel.viewport(
                track_entity=sat,
                track_rotation=False,
                pos=[100.0, 0.0, 0.0],
                looking_at=[0.0, 0.0, 0.0],
                active=True,
            ),
            el.Panel.graph(
                [
                    el.GraphEntity(sat_id, el.Component.index(el.WorldPos)[:4])
                    for sat_id in sat_ids
                ]
            ),
        ],
        active=True,
    ),
    name="Viewport and Graph",
)

w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform.from_linear(np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion.from_angular(
                np.array([0.0, 1.0, 0.0]) * 7.2921159e-5
            ),
            inertia=el.SpatialInertia(1.0),
        ),
        w.glb("https://storage.googleapis.com/elodin-marketing/models/earth.glb"),
    ],
    name="Earth",
)


exec = w.run(
    system=el.six_dof(
        1 / 120.0,
        sensors
        | kalman_filter
        | control
        | actuator_allocator
        | rw_effector
        | gravity_effector
        | earth_point,
        el.Integrator.SemiImplicit,
    ),
    time_step=1.0 / 240.0,
)
