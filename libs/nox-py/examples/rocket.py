import typing as ty
import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la

TIME_STEP = 1.0 / 120.0
thrust_vector_body_frame = jnp.array([-1.0, 0.0, 0.0])

Wind = ty.Annotated[
    el.SpatialMotion, el.Component("wind", el.ComponentType.SpatialMotionF64)
]

EulerAngles = ty.Annotated[
    jax.Array,
    el.Component("euler_angles", el.ComponentType(el.PrimitiveType.F64, (3,))),
]

MotorThrust = ty.Annotated[
    jax.Array,
    el.Component("motor", el.ComponentType.F64),
]

AngleOfAttack = ty.Annotated[
    jax.Array,
    el.Component("angle_of_attack", el.ComponentType(el.PrimitiveType.F64, (3,))),
]


def euler_to_quat(angles: jax.Array) -> el.Quaternion:
    [roll, pitch, yaw] = jnp.deg2rad(angles)
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return el.Quaternion(jnp.array([x, y, z, w]))


def quat_to_euler(q: el.Quaternion) -> jax.Array:
    x, y, z, w = q.vector()
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = jnp.arcsin(2 * (w * y - z * x))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return jnp.rad2deg(jnp.array([roll, pitch, yaw]) * 2)


@el.dataclass
class Rocket(el.Archetype):
    motor_thrust: MotorThrust
    euler_pos: EulerAngles
    angle_of_attack: AngleOfAttack
    wind: Wind = el.SpatialMotion.zero()


@el.map
def apply_wind(w: Wind, f: el.Force, v: el.WorldVel) -> el.Force:
    # mass density
    p = 0.2
    # reference area
    A = 0.5
    # drag coefficient
    cd = 0.1
    # flow velocity relative to the object
    u = w.linear() - v.linear()
    u_dir = u / jnp.clip(la.norm(u), 1e-6)
    d = 0.5 * p * u**2 * cd * A
    return f + el.Force.from_linear(d * u_dir)


@el.map
def apply_aero_moments(aoa: AngleOfAttack, v: el.WorldVel, f: el.Force) -> el.Force:
    v = jnp.clip(la.norm(v.linear()), 1e-6)
    torque = aoa * 0.1 * v
    return f + el.Force.from_torque(torque)


@el.map
def apply_aero_moments_precomp(
    aoa: AngleOfAttack, v: el.WorldVel, f: el.Force
) -> el.Force:
    # maps pitch angle to pitch moment
    aero_data = jnp.array([0.0, 0.01, 0.02, 0.04, 0.07, 0.11, 0.15, 0.21, 0.27, 0.35])
    # map [0:pi] to [0:10]
    aoa_adj = (aoa[2] / jnp.pi) * len(aero_data)
    # interpolate aoa from aero_data
    pitch = jax.scipy.ndimage.map_coordinates(aero_data, [jnp.abs(aoa_adj)], 1)
    v = jnp.clip(la.norm(v.linear()), 1e-6)
    torque = jnp.array([0.0, 0.0, pitch * jnp.sign(aoa_adj)]) * v
    return f + el.Force.from_torque(torque)


@el.map
def angle_of_attack(p: el.WorldPos, v: el.WorldVel, w: Wind) -> AngleOfAttack:
    ang_pos = p.angular() @ thrust_vector_body_frame
    v_diff = v.linear() - w.linear()
    # handle the case where the velocity is zero
    v_dir = jax.lax.cond(
        la.norm(v_diff) < 1e-6,
        lambda _: ang_pos,
        lambda _: v_diff / la.norm(v_diff),
        operand=None,
    )
    # q = quaternion of rotation from ang_pos to v_dir:
    n = jnp.cross(ang_pos, v_dir)
    w = jnp.dot(ang_pos, ang_pos) * jnp.dot(v_dir, v_dir) + jnp.dot(ang_pos, v_dir)
    q = el.Quaternion.from_array(jnp.array([n[0], n[1], n[2], w])).normalize()
    e = jnp.deg2rad(quat_to_euler(q)) / 2
    return e


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.Force.from_linear(jnp.array([0.0, inertia.mass() * -9.81, 0.0]))


@el.map
def apply_thrust(m: MotorThrust, p: el.WorldPos, f: el.Force) -> el.Force:
    thrust = p.angular() @ thrust_vector_body_frame * m
    return f + el.Force.from_linear(thrust)


@el.map
def update_euler_angles(p: el.WorldPos) -> EulerAngles:
    return quat_to_euler(p.angular())


w = el.World()
rocket = (
    w.spawn(
        el.Body(
            world_pos=el.WorldPos.from_linear(jnp.array([0.0, 1.0, 0.0]))
            + el.WorldPos.from_angular(euler_to_quat(jnp.array([0.0, 0.0, -90.0]))),
            pbr=w.insert_asset(
                el.Pbr.from_url(
                    "https://storage.googleapis.com/elodin-marketing/models/rocket.glb"
                )
            ),
            inertia=el.Inertia.from_mass(2.0),
        )
    )
    .name("Rocket")
    .insert(
        Rocket(
            motor_thrust=jnp.float64(400.0),
            euler_pos=jnp.array([0.0, 0.0, 0.0]),
            angle_of_attack=jnp.array([0.0, 0.0, 0.0]),
            wind=el.SpatialMotion.zero(),
        )
    )
)
w.spawn(
    el.Panel.viewport(
        track_rotation=False,
        pos=[5.0, 2.0, 5.0],
        looking_at=[0.0, 0.0, 0.0],
        show_grid=True,
    )
).name("Viewport (Origin)")
w.spawn(
    el.Panel.viewport(
        track_entity=rocket.id(),
        track_rotation=False,
        active=True,
        pos=[5.0, 1.0, 0.0],
        looking_at=[0.0, 0.0, 0.0],
        show_grid=True,
    )
).name("Viewport (Follow)")

effectors = (
    gravity
    | apply_thrust
    | angle_of_attack
    | apply_aero_moments_precomp
    | apply_wind
    | update_euler_angles
)
sys = el.six_dof(TIME_STEP, effectors)
w.run(sys)
