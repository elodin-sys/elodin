from dataclasses import dataclass, field
from typing import Annotated

import elodin as el
import jax
import jax.numpy as np
from elodin.elodin import Quaternion
from jax.numpy import linalg as la

angular_vel_axis = np.array([1.0, 1.0, 1.0])
angular_vel_axis = angular_vel_axis / la.norm(angular_vel_axis)
initial_angular_vel = angular_vel_axis * np.radians(80)
rw_force_clamp = 0.002
G = 6.6743e-11  # gravitational constant
M = 5.972e24  # mass of the Earth
earth_radius = 6378.1 * 1000
altitude = 400 * 1000
radius = earth_radius + altitude
velocity = np.sqrt(G * M / radius)
SIM_TIME_STEP = 1.0 / 20.0

# sensors
GyroOmega = Annotated[
    jax.Array, el.Component("gyro_omega", el.ComponentType(el.PrimitiveType.F64, (3,)))
]

MagReadingBody = Annotated[
    jax.Array,
    el.Component("mag_value", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
MagReadingRef = Annotated[
    jax.Array,
    el.Component("mag_ref", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

CssReading = Annotated[
    jax.Array,
    el.Component("css_reading", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

SunPos = Annotated[
    jax.Array,
    el.Component("sun_pos", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


@dataclass
class Sensors(el.Archetype):
    gyro_omega: GyroOmega
    mag_reading_body: MagReadingBody
    mag_reading_ref: MagReadingRef
    star_reading_ref: CssReading
    sun_pos: SunPos


CssValue = Annotated[
    jax.Array,
    el.Component("css_value", el.ComponentType(el.PrimitiveType.F64, ())),
]

CssFov = Annotated[
    jax.Array,
    el.Component("css_fov", el.ComponentType(el.PrimitiveType.F64, (1,))),
]

CssNormal = Annotated[
    jax.Array,
    el.Component("css_normal", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

CSSEdge = Annotated[el.Edge, el.Component("css_edge")]


@dataclass
class CSSRel(el.Archetype):
    edge: CSSEdge


@dataclass
class SunSensor(el.Archetype):
    value: CssValue
    fov: CssFov
    normal: CssNormal


@el.map
def sun_pos(pos: el.WorldPos) -> SunPos:
    pos = pos.linear()
    return pos / la.norm(pos)


@el.system
def sun_sensor(
    sensor: el.GraphQuery[CSSEdge],
    css_normal: el.Query[CssNormal, CssFov],
    sun_pos: el.Query[SunPos, el.WorldPos],
) -> el.Query[CssValue]:
    def inner(acc, css_normal, fov, sun_pos, world_pos):
        key = jax.random.key(jax.lax.convert_element_type(world_pos.linear()[1], "int64"))
        noise = 0.01 * jax.random.normal(key, shape=())
        sun_pos_b = world_pos.angular().inverse() @ sun_pos
        cos = np.dot(css_normal, sun_pos_b)
        return acc + jax.lax.select((np.abs(np.acos(cos)) < fov).all(), cos, 0.0) + noise

    return sensor.edge_fold(css_normal, sun_pos, CssValue, np.array(0.0), inner)


@el.system
def sun_sensor_value(
    graph: el.GraphQuery[Annotated[CSSEdge, el.RevEdge]],
    css: el.Query[CssValue, CssNormal],
    sat: el.Query[el.WorldPos],
) -> el.Query[CssReading]:
    value = graph.edge_fold(
        sat,
        css,
        CssReading,
        np.array([0.0, 0.0, 0.0]),
        lambda acc, _, value, norm: acc + value * norm,
    )
    return value.map(CssReading, lambda x: x / la.norm(x))


sun_sensor_sys = sun_pos.pipe(sun_sensor).pipe(sun_sensor_value)

k_0 = np.array(
    [
        -30926.00e-9,
        5817.00e-9,
        -2318.00e-9,
    ]
)


@el.map
def fake_magnetometer_ref(pos: el.WorldPos) -> MagReadingRef:
    pos = pos.linear()
    pos_norm = la.norm(pos)
    e_hat = pos / pos_norm
    B = ((earth_radius / pos_norm) ** 3) * (3 * np.dot(k_0, e_hat) * e_hat - k_0)
    return B / la.norm(B)


@el.map
def fake_magnetometer_body(pos: el.WorldPos, mag_ref: MagReadingRef) -> MagReadingBody:
    key = jax.random.key(jax.lax.convert_element_type(pos.linear()[0], "int64"))
    noise = 0.01 * jax.random.normal(key, shape=(3,))
    return pos.angular().inverse() @ mag_ref + noise


@el.map
def gyro_omega(pos: el.WorldPos, vel: el.WorldVel) -> GyroOmega:
    key = jax.random.key(jax.lax.convert_element_type(vel.linear()[0], "int64"))
    noise = 3.16e-7 * jax.random.normal(key, shape=(3,))
    return (pos.angular().inverse() @ vel.angular()) + noise + 2.0


sensors = sun_sensor_sys.pipe(fake_magnetometer_body).pipe(fake_magnetometer_ref).pipe(gyro_omega)

# attitude det - mekf
# source: Optimal Estimation of Dynamic Systems, 2nd Edition - Chapter 7


def calculate_covariance(sigma_g: jax.Array, sigma_b: jax.Array, dt: float) -> jax.Array:
    variance_g = np.diag(sigma_g * sigma_g * dt)
    variance_b = np.diag(sigma_b * sigma_b * dt)
    Q_00 = variance_g + variance_b * dt**2 / 3
    Q_01 = variance_b * dt / 2
    Q_10 = Q_01
    Q_11 = variance_b
    return np.block([[Q_00, Q_01], [Q_10, Q_11]])


Q = calculate_covariance(np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), SIM_TIME_STEP)
Y = np.diag(np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]))
YQY = Y @ Q @ Y.T
P = Annotated[jax.Array, el.Component("P", el.ComponentType(el.PrimitiveType.F64, (6, 6)))]
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
    q_hat = q + q * qa
    return q_hat.normalize()


def propogate_state_covariance(big_p: jax.Array, omega: jax.Array, dt: float) -> jax.Array:
    omega_norm = la.norm(omega)
    s = np.sin(omega_norm * dt)
    c = np.cos(omega_norm * dt)
    p = s / omega_norm
    q = (1 - c) / (omega_norm**2)
    r = (omega_norm * dt - s) / (omega_norm**3)
    omega_cross = el.skew(omega)
    omega_cross_square = omega_cross @ omega_cross
    phi_00 = jax.lax.select(
        omega_norm > 1e-5,
        np.eye(3) - omega_cross * p + omega_cross_square * q,
        np.eye(3),
    )
    phi_01 = jax.lax.select(
        omega_norm > 1e-5,
        omega_cross * q - np.eye(3) * dt - omega_cross_square * r,
        np.eye(3) * -SIM_TIME_STEP,
    )
    phi_10 = np.zeros((3, 3))
    phi_11 = np.eye(3)
    phi = np.block([[phi_00, phi_01], [phi_10, phi_11]])
    return (phi @ big_p @ phi.T) + YQY


def estimate_attitude(
    q_hat: Quaternion,
    b_hat: jax.Array,
    omega: jax.Array,
    p: jax.Array,
    measured_bodys: jax.Array,
    measured_references: jax.Array,
    dt: float,
) -> tuple[Quaternion, jax.Array, jax.Array, jax.Array]:
    omega = omega - b_hat
    q_hat = propogate_quaternion(q_hat, omega, dt)
    p = propogate_state_covariance(p, omega, dt)
    delta_x_hat = np.zeros(6)
    var_r = np.eye(3) * 0.001
    for i in range(0, sensor_count):
        measured_reference = measured_references[i]
        measured_body = measured_bodys[i]
        body_r = q_hat.inverse() @ measured_reference
        e = measured_body - body_r
        H = np.block([el.skew(body_r), np.zeros((3, 3))])
        H_trans = H.T
        K = p @ H_trans @ la.inv(H @ p @ H_trans + var_r)
        p = (np.eye(6) - K @ H) @ p
        delta_x_hat = delta_x_hat + K @ (e - H @ delta_x_hat)
    delta_alpha = delta_x_hat[0:3]
    delta_beta = delta_x_hat[3:6]
    q_hat = update_quaternion(q_hat, delta_alpha)
    b_hat = b_hat + delta_beta
    return (q_hat, b_hat, p, omega)


@el.map
def kalman_filter(
    omega: GyroOmega,
    mag_body: MagReadingBody,
    mag_ref: MagReadingRef,
    sun_body: CssReading,
    sun_ref: SunPos,
    att_est: AttEst,
    b_hat: BiasEst,
    p: P,
) -> tuple[AttEst, AngVelEst, BiasEst, P]:
    q_hat, b_hat, big_p, omega_hat = estimate_attitude(
        att_est,
        b_hat,
        omega,
        p,
        np.array([mag_body, sun_body]),
        np.array([mag_ref, sun_ref]),
        SIM_TIME_STEP,
    )
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


j = np.array([15204079.70002, 14621352.61765, 6237758.3131]) * 1e-9
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
    body_axis = np.array([0.0, -1.0, 0.0])
    a = np.cross(body_axis, r)
    w = 1 + np.dot(body_axis, r)
    return euler_to_quat(deg) * el.Quaternion(np.array([*a, w])).normalize()


@el.map
def control(att_est: AttEst, ang_vel: AngVelEst, goal: Goal) -> ControlForce:
    error = (att_est.inverse() * goal).vector()
    sign = np.sign(error[3])
    error = error[:3]
    return el.SpatialForce(torque=-1.0 * ang_vel * d + sign * error * k)


# effectors

RWEdge = Annotated[el.Edge, el.Component("rw_edge")]

RWAxis = Annotated[jax.Array, el.Component("rw_axis", el.ComponentType(el.PrimitiveType.F64, (3,)))]

RWForce = Annotated[el.SpatialForce, el.Component("rw_force")]
RWAngMomentum = Annotated[
    jax.Array,
    el.Component("rw_ang_momentum", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
RWSpeed = Annotated[
    jax.Array,
    el.Component("rw_speed", el.ComponentType(el.PrimitiveType.F64, ())),
]
RWVoltage = Annotated[
    jax.Array,
    el.Component("rw_voltage", el.ComponentType(el.PrimitiveType.F64, ())),
]
RWFriction = Annotated[
    jax.Array,
    el.Component("rw_friction", el.ComponentType(el.PrimitiveType.F64, ())),
]


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
        el.SpatialForce(),
        lambda xs, axis, control_force: xs
        + el.SpatialForce(torque=np.dot(control_force.torque(), axis) * axis),
    )


@el.map
def calculate_speed(ang_momentum: RWAngMomentum) -> RWSpeed:
    i = 0.185 * (0.05 / 2) ** 2 / 2
    return np.array(la.norm(ang_momentum) / i)


# source: https://hanspeterschaub.info/basilisk/_downloads/17eeb82a3f1a8e0b8617c8b8284303ed/Basilisk-REACTIONWHEELSTATEEFFECTOR-20170816.pdf
@el.map
def rw_drag(speed: RWSpeed, force: RWForce, axis: RWAxis) -> tuple[RWForce, RWFriction]:
    static_fric = 0.0005
    columb_fric = 0.0005
    stribeck_coef = 0.0005
    cv = 0.00005
    omega_limit = 0.1

    stribeck_torque = (
        -np.sqrt(2 * np.exp(1.0))
        * (static_fric - columb_fric)
        * np.exp(-((speed / stribeck_coef) ** 2))
        - columb_fric * np.tanh(10 * speed / stribeck_coef)
        - cv * speed
    )
    use_stribeck = np.logical_and(
        np.abs(speed) < 0.01 * omega_limit,
        np.sign(speed) == np.sign(la.norm(force.torque())),
    )
    # use_stribeck = np.abs(speed) < (0.01 * omega_limit)
    torque = jax.lax.select(
        use_stribeck, stribeck_torque, -columb_fric * np.sign(speed) - cv * speed
    )
    return (force + el.SpatialForce(torque=torque * axis), torque)


@el.map
def saturate_force(force: RWForce, ang_momentum: RWAngMomentum) -> tuple[RWForce, RWAngMomentum]:
    new_ang_momentum = ang_momentum + force.torque() * SIM_TIME_STEP
    torque = jax.lax.select(np.abs(new_ang_momentum) < 0.04, force.torque(), np.zeros(3))
    torque = np.clip(torque, -rw_force_clamp, rw_force_clamp)
    return (el.SpatialForce(torque=torque), ang_momentum + torque * SIM_TIME_STEP)


@dataclass
class RWRel(el.Archetype):
    edge: RWEdge


@dataclass
class ReactionWheel(el.Archetype):
    axis: RWAxis
    rw_force: RWForce = field(default_factory=lambda: el.SpatialForce())
    ang_momentum: RWAngMomentum = field(default_factory=lambda: np.zeros(3))
    speed: RWSpeed = field(default_factory=lambda: np.float64(0.0))
    voltage: RWVoltage = field(default_factory=lambda: np.float64(0.0))
    friction: RWFriction = field(default_factory=lambda: np.float64(0.0))


@el.system
def rw_effector(
    rw_force: el.GraphQuery[RWEdge],
    force_query: el.Query[el.WorldPos],
    rw_query: el.Query[RWForce],
) -> el.Query[el.Force]:
    return rw_force.edge_fold(
        force_query,
        rw_query,
        el.Force,
        el.SpatialForce(),
        lambda f, pos, force: f + el.SpatialForce(torque=pos.angular() @ force.torque()),
    )


J2 = 1.08262668e-3


Radius = Annotated[
    jax.Array,
    el.Component("radius", el.ComponentType(el.PrimitiveType.F64, ())),
]


@el.map
def gravity_effector(
    _: Goal,
    force: el.Force,
    a_pos: el.WorldPos,
    a_inertia: el.Inertia,
) -> tuple[el.Force, Radius]:
    r = a_pos.linear()
    m = a_inertia.mass()
    norm = la.norm(r)
    e_r = r / norm
    mu = G * M
    f = -mu * m * r / (norm * norm * norm)
    z = r[2]
    e_z = np.array([0.0, 0.0, 1.0])
    j2 = (
        -mu
        * m
        * J2
        * earth_radius**2
        * (
            3 * z / (norm**5) * e_z
            + (3.0 / (2.0 * norm**4) - 15.0 * z**2 / (2.0 * norm**6.0)) * e_r
        )
    )
    # j2 = -mu * m * r / (norm ** 3) * ( 1 - (3/2) * J2 * (earth_radius / norm) ** 2 * (5 * z ** 2 / norm ** 2 - np.array([1, 1, 3])))
    return (force + el.SpatialForce(linear=f + j2), norm)


@dataclass
class Debug(el.Archetype):
    radius: Radius


w = el.World()

sat = w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([1.0, 0.0, 0.0]) * radius),
            world_vel=el.SpatialMotion(initial_angular_vel, np.array([0.0, 1.0, 0.0]) * velocity),
            inertia=el.SpatialInertia(2825.2 / 1000.0, j),
            # Credit to the OreSat program https://www.oresat.org for the model above
        ),
        ControlInput(
            el.Quaternion.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.radians(0)),
            el.SpatialForce(),
        ),
        UserInput(np.array([0.0, 0.0, 0.0])),
        Sensors(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)),
        KalmanFilter(np.identity(6), el.Quaternion.identity(), np.zeros(3), np.zeros(3)),
        Debug(np.float64(0.0)),
        w.glb("https://storage.googleapis.com/elodin-assets/oresat-low.glb"),
    ],
    name="OreSat",
)

# w.spawn(el.VectorArrow(sat, "control_force", color = el.Color(1.0, 0.0, 0.0), scale=2.0))
# w.spawn(el.VectorArrow(sat, "world_vel", offset=3, body_frame=False, scale=1/2000.0))
w.spawn(el.BodyAxes(sat, scale=1.0))

rw_1 = w.spawn(
    ReactionWheel(
        axis=np.array([1.0, 0.0, 0.0]),
    ),
    name="Reaction Wheel 1",
)


rw_2 = w.spawn(
    ReactionWheel(
        axis=np.array([0.0, 1.0, 0.0]),
    ),
    name="Reaction Wheel 2",
)

rw_3 = w.spawn(
    ReactionWheel(axis=np.array([0.0, 0.0, 1.0])),
    name="Reaction Wheel 3",
)


css_0 = w.spawn(
    SunSensor(
        value=0.0,
        fov=np.radians(90),
        normal=np.array([0.0, 0.0, 1.0]),
    ),
    name="Course Sun Sensor 0",
)

css_1 = w.spawn(
    SunSensor(
        value=0.0,
        fov=np.radians(90),
        normal=np.array([0.0, 1.0, 0.0]),
    ),
    name="Course Sun Sensor 1",
)

css_2 = w.spawn(
    SunSensor(
        value=0.0,
        fov=np.radians(90),
        normal=np.array([1.0, 0.0, 0.0]),
    ),
    name="Course Sun Sensor 2",
)
css_3 = w.spawn(
    SunSensor(
        value=0.0,
        fov=np.radians(90),
        normal=np.array([0.0, 0.0, -1.0]),
    ),
    name="Course Sun Sensor 3",
)

css_4 = w.spawn(
    SunSensor(
        value=0.0,
        fov=np.radians(90),
        normal=np.array([0.0, -1.0, 0.0]),
    ),
    name="Course Sun Sensor 4",
)

css_5 = w.spawn(
    SunSensor(
        value=0.0,
        fov=np.radians(90),
        normal=np.array([-1.0, 0.0, 0.0]),
    ),
    name="Course Sun Sensor 5",
)

w.spawn(RWRel(el.Edge(sat, rw_1)), name="Sat -> RW 1")
w.spawn(RWRel(el.Edge(sat, rw_2)), name="Sat -> RW 2")
w.spawn(RWRel(el.Edge(sat, rw_3)), name="Sat -> RW 3")

w.spawn(CSSRel(el.Edge(css_0, sat)), name="CSS 0 -> Sat")
w.spawn(CSSRel(el.Edge(css_1, sat)), name="CSS 1 -> Sat")
w.spawn(CSSRel(el.Edge(css_2, sat)), name="CSS 2 -> Sat")
w.spawn(CSSRel(el.Edge(css_3, sat)), name="CSS 3 -> Sat")
w.spawn(CSSRel(el.Edge(css_4, sat)), name="CSS 4 -> Sat")
w.spawn(CSSRel(el.Edge(css_5, sat)), name="CSS 5 -> Sat")

w.spawn(
    el.Panel.vsplit(
        el.Panel.hsplit(
            el.Panel.viewport(
                track_entity=sat,
                track_rotation=False,
                pos=[7.0, 0.0, 0.0],
                looking_at=[0.0, 0.0, 0.0],
            ),
            el.Panel.graph(
                *[
                    el.GraphEntity(css, CssValue)
                    for css in [css_0, css_1, css_2, css_3, css_4, css_5]
                ]
            ),
        ),
        el.Panel.graph(
            el.GraphEntity(
                sat,
                AttEst,
                *el.Component.index(el.WorldPos)[:4],
            )
        ),
        active=True,
    )
)

w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(linear=np.array([0.0, 0.0, 0.0])),
            world_vel=el.SpatialMotion(angular=np.array([0.0, 0.0, 1.0]) * 7.2921159e-5),
            inertia=el.SpatialInertia(1.0),
        ),
        w.glb("https://storage.googleapis.com/elodin-assets/earth.glb"),
    ],
    name="Earth",
)

w.spawn(
    el.Line3d(
        sat,
        "world_pos",
        line_width=10.0,
        perspective=False,
    )
)

exec = w.run(
    system=el.six_dof(
        sys=sensors
        | kalman_filter
        | control
        | actuator_allocator
        | rw_drag
        | saturate_force
        | calculate_speed
        | rw_effector
        | gravity_effector
        | earth_point,
        integrator=el.Integrator.SemiImplicit,
    ),
    sim_time_step=SIM_TIME_STEP,
    run_time_step=0.0 / 10.0,
    output_time_step=1 / 512.0,
    max_ticks=60 * 20 * 128,
)
