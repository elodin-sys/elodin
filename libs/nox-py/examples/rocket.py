import typing as ty
import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from dataclasses import field
from jax.scipy.ndimage import map_coordinates
import polars as pl

TIME_STEP = 1.0 / 120.0
lp_sample_freq = round(1.0 / TIME_STEP)
lp_buffer_size = lp_sample_freq * 4
lp_cutoff_freq = 1

thrust_vector_body_frame = jnp.array([-1.0, 0.0, 0.0])
a_ref = 24.89130 / 100**2
l_ref = 5.43400 / 100
xmc = 0.40387

pitch_pid = [1.1, 0.8, 3.8]

Wind = ty.Annotated[
    jax.Array,
    el.Component(
        "wind",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z"},
    ),
]

AeroCoefs = ty.Annotated[
    jax.Array,
    el.Component(
        "aero_coefs",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "Cl,CnR,CmR,CA,CZR,CYR"},
    ),
]

AeroForce = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "aero_force",
        el.ComponentType.SpatialMotionF64,
        metadata={"element_names": "τx,τy,τz,x,y,z"},
    ),
]

CenterOfGravity = ty.Annotated[
    jax.Array, el.Component("center_of_gravity", el.ComponentType.F64)
]

DynamicPressure = ty.Annotated[
    jax.Array, el.Component("dynamic_pressure", el.ComponentType.F64)
]

AngleOfAttack = ty.Annotated[
    jax.Array, el.Component("angle_of_attack", el.ComponentType.F64)
]

Mach = ty.Annotated[jax.Array, el.Component("mach", el.ComponentType.F64)]

Motor = ty.Annotated[jax.Array, el.Component("rocket_motor", el.ComponentType.F64)]

FinControl = ty.Annotated[jax.Array, el.Component("fin_control", el.ComponentType.F64)]

FinDeflect = ty.Annotated[jax.Array, el.Component("fin_deflect", el.ComponentType.F64)]

VRelAccel = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 20},
    ),
]

VRelAccelBuffer = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel_buffer",
        el.ComponentType(el.PrimitiveType.F64, (lp_buffer_size, 3)),
        metadata={"priority": -1},
    ),
]

VRelAccelFiltered = ty.Annotated[
    jax.Array,
    el.Component(
        "v_rel_accel_filtered",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "x,y,z", "priority": 19},
    ),
]

PitchPID = ty.Annotated[
    jax.Array,
    el.Component(
        "pitch_pid",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "Kp,Ki,Kd"},
    ),
]

PitchPIDState = ty.Annotated[
    jax.Array,
    el.Component(
        "pitch_pid_state",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "e,i,d", "priority": 18},
    ),
]

AccelSetpoint = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 101},
    ),
]

AccelSetpointSmooth = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint_smooth",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 100},
    ),
]

Thrust = ty.Annotated[
    jax.Array,
    el.Component("thrust", el.ComponentType.F64, metadata={"priority": 17}),
]

aero_df = pl.from_dict({
    'Mach': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    'Alphac': [0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0],
    'Delta': [-40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, -40.0, -40.0, -40.0, -40.0, -20.0, -20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0],
    'CmR': [-5.997, -6.905, -8.235, -10.83, -5.315, -6.008, -5.918, -5.714, 0.0, 1.313, 2.335, 0.4163, 5.315, 3.642, 2.977, 1.061, 5.997, 5.372, 4.191, 1.882, -7.269, -8.373, -9.873, -12.93, -6.323, -7.255, -7.14, -6.846, 0.0, 1.486, 2.681, 0.445, 6.323, 4.263, 3.463, 1.222, 7.269, 6.396, 4.963, 2.27, -11.53, -12.49, -13.88, -15.71, -9.056, -8.891, -8.448, -8.155, 0.0, 1.921, 3.144, 1.169, 9.056, 8.419, 7.126, 4.228, 11.53, 10.14, 8.19, 4.94],
    'CA': [1.121, 1.028, 0.9495, 0.9803, 0.6405, 0.5852, 0.4342, 0.217, 0.2942, 0.2873, 0.2591, 0.2032, 0.6405, 0.5988, 0.635, 0.6333, 1.121, 1.215, 1.246, 1.267, 1.242, 1.137, 1.051, 1.095, 0.6902, 0.6278, 0.4588, 0.2184, 0.2924, 0.2856, 0.2577, 0.2025, 0.6902, 0.6434, 0.6895, 0.6967, 1.242, 1.351, 1.392, 1.425, 1.851, 1.747, 1.621, 1.48, 0.9888, 0.8509, 0.658, 0.4269, 0.448, 0.4446, 0.4345, 0.418, 0.9888, 1.06, 1.111, 1.154, 1.851, 1.961, 2.03, 2.098],
    'CZR': [-1.092, -0.3878, 0.3984, 1.141, -1.141, -0.4069, 0.7324, 2.176, 0.0, 1.061, 2.368, 3.494, 1.141, 1.561, 2.483, 3.64, 1.092, 1.789, 2.577, 3.68, -1.191, -0.4161, 0.4355, 1.252, -1.274, -0.4526, 0.8073, 2.408, 0.0, 1.178, 2.63, 3.88, 1.274, 1.736, 2.755, 4.043, 1.191, 1.973, 2.844, 4.07, -1.609, -0.8494, 0.1373, 1.323, -1.639, -0.5395, 0.9159, 2.704, 0.0, 1.304, 2.894, 4.443, 1.639, 2.532, 3.576, 4.981, 1.609, 2.483, 3.481, 4.811]
})  # fmt: skip

thrust_curve = {
    'time': [0.01, 0.67, 1.33, 1.99, 2.65, 3.31, 3.97, 4.63, 5.29, 5.95, 6.61, 7.27, 7.93, 8.59, 9.25, 9.91, 10.57, 11.23, 11.89, 12.55, 13.21, 13.87, 14.53, 15.19, 15.85, 16.51, 17.17, 17.83, 18.49, 19.15, 19.81, 20.47, 21.13, 21.79, 22.45, 23.11, 23.77, 24.43, 25.09, 25.75, 26.41, 27.07, 27.73, 28.39, 29.05, 29.71, 30.37, 31.03, 31.69, 32.15],
    'thrust': [322.148, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 88.426, 0.0],
}  # fmt: skip


def second_order_butterworth(
    signal: jax.Array, f_sampling: int, f_cutoff: int, method: str = "forward"
) -> jax.Array:
    "https://stackoverflow.com/questions/20924868/calculate-coefficients-of-2nd-order-butterworth-low-pass-filter"
    if method == "forward_backward":
        signal = second_order_butterworth(signal, f_sampling, f_cutoff, "forward")
        return second_order_butterworth(signal, f_sampling, f_cutoff, "backward")
    elif method == "forward":
        pass
    elif method == "backward":
        signal = jnp.flip(signal, axis=0)
    else:
        raise NotImplementedError
    ff = f_cutoff / f_sampling
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita**2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita**2) * b0

    def f(carry, x_i):
        x_im1, x_im2, y_im1, y_im2 = carry
        y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
        return (x_i, x_im1, y_i, y_im1), y_i

    init = (signal[1], signal[0]) * 2
    signal = jax.lax.scan(f, init, signal[2:])[1]
    signal = jnp.concatenate((signal[0:1],) * 2 + (signal,))
    if method == "backward":
        signal = jnp.flip(signal, axis=0)
    return signal


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
    pitch = (
        2
        * jnp.atan2(
            jnp.sqrt(1 + 2 * (w * y - x * z)), jnp.sqrt(1 - 2 * (w * y - x * z))
        )
        - jnp.pi / 2
    )
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return jnp.rad2deg(jnp.array([roll, pitch, yaw]))


def quat_from_vecs(v1: jax.Array, v2: jax.Array) -> el.Quaternion:
    v1 = v1 / la.norm(v1)
    v2 = v2 / la.norm(v2)
    n = jnp.cross(v1, v2)
    w = jnp.dot(v2, v2) * jnp.dot(v1, v1) + jnp.dot(v1, v2)
    q = el.Quaternion.from_array(jnp.array([*n, w])).normalize()
    return q


def aero_interp_table(df: pl.DataFrame) -> jax.Array:
    coefs = ["CmR", "CA", "CZR"]
    aero = jnp.array(
        [
            [
                df.group_by(["Alphac"], maintain_order=True)
                .agg(pl.col(coefs).min())
                .select(pl.col(coefs))
                .to_numpy()
                for _, df in df.group_by(["Delta"], maintain_order=True)
            ]
            for _, df in df.group_by(["Mach"], maintain_order=True)
        ]
    )
    aero = aero.transpose(3, 0, 1, 2)
    return aero


# coverts `val` to a coordinate along some series `s`
def to_coord(s: pl.Series, val: jax.Array) -> jax.Array:
    s_min = s.min()
    s_max = s.max()
    s_count = len(s.unique())
    return (val - s_min) * (s_count - 1) / jnp.clip(s_max - s_min, 1e-06)


@el.dataclass
class Rocket(el.Archetype):
    angle_of_attack: AngleOfAttack = field(default_factory=lambda: jnp.array([0.0]))
    aero_coefs: AeroCoefs = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    center_of_gravity: CenterOfGravity = field(default_factory=lambda: jnp.float64(0.2))
    mach: Mach = field(default_factory=lambda: jnp.float64(0.0))
    dynamic_pressure: DynamicPressure = field(default_factory=lambda: jnp.float64(0.0))
    aero_force: AeroForce = field(default_factory=lambda: el.SpatialForce.zero())
    wind: Wind = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    motor: Motor = field(default_factory=lambda: jnp.float64(0.0))
    fin_deflect: FinDeflect = field(default_factory=lambda: jnp.float64(0.0))
    fin_control: FinControl = field(default_factory=lambda: jnp.float64(0.0))
    v_rel_accel_buffer: VRelAccelBuffer = field(
        default_factory=lambda: jnp.zeros((lp_buffer_size, 3))
    )
    v_rel_accel: VRelAccel = field(default_factory=lambda: jnp.array([0.0, 0.0, 0.0]))
    v_rel_accel_filtered: VRelAccelFiltered = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    pitch_pid: PitchPID = field(default_factory=lambda: jnp.array(pitch_pid))
    pitch_pid_state: PitchPIDState = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    accel_setpoint: AccelSetpoint = field(default_factory=lambda: jnp.array([0.0, 0.0]))
    accel_setpoint_smooth: AccelSetpointSmooth = field(
        default_factory=lambda: jnp.array([0.0, 0.0])
    )
    thrust: Thrust = field(default_factory=lambda: jnp.float64(0.0))


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=jnp.array([0.0, 0.0, -9.81]) * inertia.mass())


@el.map
def mach(p: el.WorldPos, v: el.WorldVel, w: Wind) -> tuple[Mach, DynamicPressure]:
    atmosphere = {
        "h": jnp.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]),
        "T": jnp.array([15.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.2]),
        "p": jnp.array([101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.]),
        "d": jnp.array([1.225, 0.3639, 0.0880, 0.0132, 0.0014, 0.0009, 0.0001, 0.]),
    }  # fmt: skip
    altitude = p.linear()[2]
    temperature = jnp.interp(altitude, atmosphere["h"], atmosphere["T"]) + 273.15
    density = jnp.interp(altitude, atmosphere["h"], atmosphere["d"])
    specific_heat_ratio = 1.4
    specific_gas_constant = 287.05
    speed_of_sound = jnp.sqrt(specific_heat_ratio * specific_gas_constant * temperature)
    local_flow_velocity = la.norm(v.linear() - w)
    mach = local_flow_velocity / speed_of_sound
    dynamic_pressure = 0.5 * density * local_flow_velocity**2
    dynamic_pressure = jnp.clip(dynamic_pressure, 1e-6)
    return mach, dynamic_pressure


@el.map
def angle_of_attack(p: el.WorldPos, v: el.WorldVel, w: Wind) -> AngleOfAttack:
    # u = freestream velocity vector in body frame
    u = p.angular().inverse() @ (v.linear() - w)

    # angle of attack is the angle between the freestream velocity vector and the attitude vector
    angle_of_attack = jnp.dot(u, thrust_vector_body_frame) / jnp.clip(la.norm(u), 1e-6)
    angle_of_attack = jnp.rad2deg(jnp.arccos(angle_of_attack)) * -jnp.sign(u[2])
    return angle_of_attack


@el.map
def aero_coefs(
    mach: Mach,
    angle_of_attack: AngleOfAttack,
    fin_deflect: FinDeflect,
) -> AeroCoefs:
    aero = aero_interp_table(aero_df)
    aoa_sign = jax.lax.cond(
        jnp.abs(angle_of_attack) < 1e-6,
        lambda _: 1.0,
        lambda _: jnp.sign(angle_of_attack),
        operand=None,
    )
    # aoa_sign is used to negate fin deflection angle as a way to interpolate negative angles of attack
    fin_deflect *= aoa_sign
    coords = [
        to_coord(aero_df["Mach"], mach),
        to_coord(aero_df["Delta"], fin_deflect),
        to_coord(aero_df["Alphac"], jnp.abs(angle_of_attack)),
    ]
    coefs = jnp.array(
        [map_coordinates(coef, coords, 1, mode="nearest") for coef in aero]
    )
    coefs = jnp.array(
        [
            0.0,
            0.0,
            coefs[0] * aoa_sign,
            coefs[1],
            coefs[2] * aoa_sign,
            0.0,
        ]
    )
    return coefs


@el.map
def aero_forces(
    aero_coefs: AeroCoefs,
    xcg: CenterOfGravity,
    q: DynamicPressure,
) -> AeroForce:
    Cl, CnR, CmR, CA, CZR, CYR = aero_coefs

    # shift CmR, CnR from MC to CG
    CmR = CmR - CZR * (xcg - xmc) / l_ref
    CnR = CnR - CYR * (xcg - xmc) / l_ref

    f_aero_linear = jnp.array([CA, CYR, CZR]) * q * a_ref
    f_aero_torque = jnp.array([Cl, -CmR, CnR]) * q * a_ref * l_ref
    f_aero = el.SpatialForce(linear=f_aero_linear, torque=f_aero_torque)
    return f_aero


@el.map
def apply_aero_forces(
    p: el.WorldPos,
    f_aero: AeroForce,
    f: el.Force,
) -> el.Force:
    # convert from body frame to world frame
    return f + p.angular() @ f_aero


@el.system
def thrust(
    tick: el.Query[el.SimulationTick],
    dt: el.Query[el.SimulationTimeStep],
    q: el.Query[Motor],
) -> el.Query[Thrust]:
    t = tick[0] * dt[0]
    time = jnp.array(thrust_curve["time"])
    thrust = jnp.array(thrust_curve["thrust"])
    f_t = jnp.interp(t, time, thrust)
    return q.map(Thrust, lambda _: f_t)


@el.map
def apply_thrust(thrust: Thrust, f: el.Force, p: el.WorldPos) -> el.Force:
    return f + el.SpatialForce(linear=p.angular() @ thrust_vector_body_frame * thrust)


@el.map
def v_rel_accel(v: el.WorldVel, a: el.WorldAccel) -> VRelAccel:
    v = jax.lax.cond(
        la.norm(v.linear()) < 1e-6,
        lambda _: thrust_vector_body_frame,
        lambda _: v.linear(),
        operand=None,
    )
    v_rot = quat_from_vecs(thrust_vector_body_frame, v)
    a_rel = v_rot.inverse() @ a.linear()
    return a_rel


@el.map
def v_rel_accel_buffer(a_rel: VRelAccel, buffer: VRelAccelBuffer) -> VRelAccelBuffer:
    return jnp.concatenate((buffer[1:], a_rel.reshape(1, 3)))


@el.map
def v_rel_accel_filtered(s: VRelAccelBuffer) -> VRelAccelFiltered:
    return second_order_butterworth(s, lp_sample_freq, lp_cutoff_freq)[-1]


@el.map
def accel_setpoint_smooth(
    a: AccelSetpoint, a_s: AccelSetpointSmooth
) -> AccelSetpointSmooth:
    dt = TIME_STEP
    exp_decay_constant = 0.5
    return a_s + (a - a_s) * jnp.exp(-exp_decay_constant * dt)


@el.map
def pitch_pid_state(
    a_setpoint: AccelSetpointSmooth,
    a_rel: VRelAccelFiltered,
    s: PitchPIDState,
) -> PitchPIDState:
    e = a_rel[2] - a_setpoint[0]
    i = jnp.clip(s[1] + e * TIME_STEP * 2, -2.0, 2.0)
    d = e - s[0]
    pid_state = jnp.array([e, i, d])
    return pid_state


@el.map
def pitch_pid_control(pid: PitchPID, s: PitchPIDState) -> FinControl:
    Kp, Ki, Kd = pid
    e, i, d = s
    mv = Kp * e + Ki * i + Kd * d
    fin_control = mv * TIME_STEP
    return fin_control


@el.map
def fin_control(fd: FinDeflect, fc: FinControl, mach: Mach) -> FinDeflect:
    fc = fc / (0.1 + mach)
    fc = jnp.clip(fc, -0.2, 0.2)
    fd += fc
    fd = jnp.clip(fd, -40.0, 40.0)
    return fd


w = el.World()
rocket = w.spawn(
    [
        el.Body(
            world_pos=el.SpatialTransform(
                angular=euler_to_quat(jnp.array([0.0, 70.0, 0.0])),
                linear=jnp.array([0.0, 0.0, 1.0]),
            ),
            inertia=el.SpatialInertia(3.0, jnp.array([0.1, 1.0, 1.0])),
        ),
        Rocket(),
        w.glb("https://storage.googleapis.com/elodin-marketing/models/rocket.glb"),
    ],
    name="Rocket",
)

w.spawn(el.Line3d(rocket, line_width=11.0))

w.spawn(
    el.Panel.hsplit(
        el.Panel.vsplit(
            el.Panel.viewport(
                track_entity=rocket,
                track_rotation=False,
                pos=[5.0, 0.0, 1.0],
                looking_at=[0.0, 0.0, 0.0],
                show_grid=True,
            ),
        ),
        el.Panel.vsplit(
            el.Panel.graph(el.GraphEntity(rocket, FinDeflect)),
            el.Panel.graph(
                el.GraphEntity(
                    rocket,
                    AccelSetpointSmooth,
                    *el.Component.index(VRelAccel)[1:],
                    *el.Component.index(VRelAccelFiltered)[1:],
                )
            ),
        ),
        active=True,
    )
)

non_effectors = (
    mach
    | angle_of_attack
    | accel_setpoint_smooth
    | v_rel_accel
    | v_rel_accel_buffer
    | v_rel_accel_filtered
    | pitch_pid_state
    | pitch_pid_control
    | fin_control
    | aero_coefs
    | aero_forces
    | thrust
)
effectors = gravity | apply_thrust | apply_aero_forces
sys = non_effectors | el.six_dof(sys=effectors, integrator=el.Integrator.Rk4)
w.run(sys, time_step=TIME_STEP)
