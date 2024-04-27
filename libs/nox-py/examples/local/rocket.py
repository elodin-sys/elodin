import typing as ty
import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax.scipy.ndimage import map_coordinates
from aero import read_aero_csv, to_coord, aero_interp_table
from util import second_order_butterworth
from dataclasses import field
import pathlib

TIME_STEP = 1.0 / 60.0
thrust_vector_body_frame = jnp.array([-1.0, 0.0, 0.0])
a_ref = 24.89130 / 100**2
l_ref = 5.43400 / 100
xmc = 0.40387
aero_data_path = f"{pathlib.Path(__file__).parent}/aero_data.csv"
thrust_curve_path = f"{pathlib.Path(__file__).parent}/thrust_curve.csv"
lp_sample_freq = 60
lp_cutoff_freq = 0.5
pitch_pid = [0.6, 0.2, 6.8]

with open(aero_data_path, "r") as file:
    aero_data = file.read()
df = read_aero_csv(aero_data)
aero = aero_interp_table(df)

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

RollAngle = ty.Annotated[jax.Array, el.Component("roll_angle", el.ComponentType.F64)]

WorldEulerAngles = ty.Annotated[
    jax.Array,
    el.Component(
        "world_euler_angles",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
    ),
]

Mach = ty.Annotated[jax.Array, el.Component("mach", el.ComponentType.F64)]

RocketMotor = ty.Annotated[
    jax.Array, el.Component("rocket_motor", el.ComponentType.F64)
]

Control = ty.Annotated[
    jax.Array,
    el.Component(
        "control",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "φ,θ,ψ"},
    ),
]

FinDeflection = ty.Annotated[
    jax.Array,
    el.Component(
        "fin_deflection",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "d1,d2,d3,d4"},
    ),
]

FinControl = ty.Annotated[
    jax.Array,
    el.Component(
        "fin_control",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "d1,d2,d3,d4"},
    ),
]

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
        el.ComponentType(el.PrimitiveType.F64, (240, 3)),
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
        metadata={"element_names": "p,y", "priority": 17},
    ),
]

AccelSetpointSmooth = ty.Annotated[
    jax.Array,
    el.Component(
        "accel_setpoint_smooth",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "p,y", "priority": 16},
    ),
]


@el.dataclass
class Rocket(el.Archetype):
    time: el.Time = field(default=jnp.float64(0.0))
    angle_of_attack: AngleOfAttack = field(default=jnp.array([0.0]))
    roll_angle: RollAngle = field(default=jnp.array([0.0]))
    aero_coefs: AeroCoefs = field(default=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    center_of_gravity: CenterOfGravity = field(default=jnp.float64(0.2))
    world_euler_angles: WorldEulerAngles = field(default=jnp.array([0.0, 0.0, 0.0]))
    mach: Mach = field(default=jnp.float64(0.0))
    dynamic_pressure: DynamicPressure = field(default=jnp.float64(0.0))
    aero_force: AeroForce = field(default=el.Force.zero())
    wind: Wind = field(default=jnp.array([0.0, 0.0, 0.0]))
    rocket_motor: RocketMotor = field(default=jnp.float64(0.0))
    control: Control = field(default=jnp.array([0.0, 0.0, 0.0]))
    fin_deflection: FinDeflection = field(default=jnp.array([0.0, 0.0, 0.0, 0.0]))
    fin_control: FinControl = field(default=jnp.array([0.0, 0.0, 0.0, 0.0]))
    v_rel_accel_buffer: VRelAccelBuffer = field(default=jnp.zeros((240, 3)))
    v_rel_accel: VRelAccel = field(default=jnp.array([0.0, 0.0, 0.0]))
    v_rel_accel_filtered: VRelAccelFiltered = field(default=jnp.array([0.0, 0.0, 0.0]))
    pitch_pid: PitchPID = field(default=jnp.array(pitch_pid))
    pitch_pid_state: PitchPIDState = field(default=jnp.array([0.0, 0.0, 0.0]))
    accel_setpoint: AccelSetpoint = field(default=jnp.array([0.0, 0.0]))
    accel_setpoint_smooth: AccelSetpointSmooth = field(default=jnp.array([0.0, 0.0]))


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


def quat_from_vecs(v1: jax.Array, v2: jax.Array) -> el.Quaternion:
    v1 = v1 / la.norm(v1)
    v2 = v2 / la.norm(v2)
    n = jnp.cross(v1, v2)
    w = jnp.dot(v2, v2) * jnp.dot(v1, v1) + jnp.dot(v1, v2)
    q = el.Quaternion.from_array(jnp.array([n[0], n[1], n[2], w])).normalize()
    return q


drone_vector_body_frame = jnp.array([0.0, 0.0, 1.0])
drone_attitude = euler_to_quat(jnp.array([0.0, -90.0, 0.0]))
drone_thrust_dir = euler_to_quat(jnp.array([0.0, -45.0, 0.0])) @ drone_vector_body_frame


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.Force.from_linear(jnp.array([0.0, inertia.mass() * -9.81, 0.0]))


def thrust_curve() -> jax.Array:
    with open(thrust_curve_path, "r") as file:
        thrust_curve = file.read()
    return jnp.array(
        [[float(y) for y in x.split()] for x in thrust_curve.strip().split("\n")]
    ).transpose()


@el.map
def mach(p: el.WorldPos, v: el.WorldVel, w: Wind) -> tuple[Mach, DynamicPressure]:
    atmosphere = {
        "h": jnp.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]),
        "T": jnp.array([15.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.2]),
        "p": jnp.array([101325.0, 22632.0, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.]),
        "d": jnp.array([1.225, 0.3639, 0.0880, 0.0132, 0.0014, 0.0009, 0.0001, 0.]),
    }  # fmt: skip
    altitude = p.linear()[1]
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
def angle_of_attack(
    p: el.WorldPos, v: el.WorldVel, w: Wind
) -> tuple[AngleOfAttack, RollAngle]:
    # u = freestream velocity vector in body frame
    u = p.angular().inverse() @ (v.linear() - w)

    # angle of attack is the angle between the freestream velocity vector and the attitude vector
    angle_of_attack = jnp.dot(u, thrust_vector_body_frame) / jnp.clip(la.norm(u), 1e-6)
    angle_of_attack = jnp.rad2deg(jnp.arccos(angle_of_attack)) * -jnp.sign(u[1])

    # ignore axial component of u to calculate roll angle
    u_r = jnp.array([0.0, jnp.abs(u[1]), u[2]])
    u_r = u_r / jnp.clip(la.norm(u_r), 1e-6)
    roll_angle = jnp.rad2deg(jnp.arcsin(u_r[2]))
    return angle_of_attack, roll_angle


@el.map
def aerodynamic_coefs(
    mach: Mach,
    angle_of_attack: AngleOfAttack,
    roll_angle: RollAngle,
    fin_deflection: FinDeflection,
) -> AeroCoefs:
    aoa_sign = jax.lax.cond(
        jnp.abs(angle_of_attack) < 1e-6,
        lambda _: 1.0,
        lambda _: jnp.sign(angle_of_attack),
        operand=None,
    )
    roll_sign = jax.lax.cond(
        jnp.abs(roll_angle) < 1e-6,
        lambda _: 1.0,
        lambda _: jnp.sign(roll_angle),
        operand=None,
    )
    # aoa_sign and roll_sign are used to negate deflection angles
    # as a way to interpolate negative angles of attack and roll angles
    # as the aero data only covers positive angles
    d11, d12, d13, d14 = (
        fin_deflection[0] * roll_sign,
        fin_deflection[1] * aoa_sign,
        fin_deflection[2] * roll_sign,
        fin_deflection[3] * aoa_sign,
    )
    # generate interp coords for both 1/3 and 2/4 deflection scenarios
    coords = jnp.array(
        [
            [
                to_coord(df["Mach"], mach),
                to_coord(df["Delta12"], d12),
                to_coord(df["Delta14"], d14),
                to_coord(df["Phi"], jnp.abs(roll_angle)),
                to_coord(df["Alphac"], jnp.abs(angle_of_attack)),
            ],
            [
                to_coord(df["Mach"], mach),
                to_coord(df["Delta11"], d11),
                to_coord(df["Delta13"], d13),
                to_coord(df["Phi"], jnp.abs(roll_angle)),
                to_coord(df["Alphac"], jnp.abs(angle_of_attack)),
            ],
        ]
    )
    # if fin 1 deflection angle is set, assume 1/3
    # c_i = jax.lax.cond(
    #     jnp.abs(d11) < 1e-6,
    #     lambda _: 0,
    #     lambda _: 1,
    #     operand=None,
    # )
    p_coefs = jnp.array(
        [map_coordinates(coef[0], coords[0], 1, mode="nearest") for coef in aero[:6]]
    )
    y_coefs = jnp.array(
        [map_coordinates(coef[1], coords[1], 1, mode="nearest") for coef in aero[:6]]
    )
    coefs = jnp.array(
        [
            y_coefs[0] * roll_sign,
            y_coefs[1] * roll_sign,
            p_coefs[2] * aoa_sign,
            p_coefs[3],
            p_coefs[4] * aoa_sign,
            y_coefs[5] * roll_sign,
        ]
    )
    return coefs


@el.map
def aerodynamic_forces(
    aero_coefs: AeroCoefs,
    xcg: CenterOfGravity,
    q: DynamicPressure,
    p: el.WorldPos,
    f: el.Force,
) -> tuple[AeroForce, el.Force]:
    Cl, CnR, CmR, CA, CZR, CYR = aero_coefs

    # shift CmR, CnR from MC to CG
    CmR = CmR - CZR * (xcg - xmc) / l_ref
    CnR = CnR - CYR * (xcg - xmc) / l_ref

    f_aero_linear = jnp.array([CA, CZR, CYR]) * q * a_ref
    f_aero_torque = jnp.array([Cl, -CnR, CmR]) * q * a_ref * l_ref
    f_aero = el.Force.from_linear(f_aero_linear) + el.Force.from_torque(f_aero_torque)

    # convert from body frame to world frame
    rot_torque = p.angular() @ f_aero.torque()
    rot_torque = rot_torque.at[0].set(0.0)
    f_aero_rot = el.Force.from_linear(
        p.angular() @ f_aero.linear()
    ) + el.Force.from_torque(rot_torque)
    return f_aero, f + f_aero_rot


@el.map
def apply_thrust(t: el.Time, p: el.WorldPos, f: el.Force, _: RocketMotor) -> el.Force:
    tc = thrust_curve()
    f_t = jnp.interp(t, tc[0], tc[1], right=0.0)
    # convert from body frame to world frame
    thrust = p.angular() @ thrust_vector_body_frame * f_t
    return f + el.Force.from_linear(thrust)


@el.map
def control(control: Control) -> FinDeflection:
    control /= 100.0
    d11, d12, d13, d14 = (
        control[2] + control[0],
        control[1],
        control[2] - control[0],
        control[1],
    )
    return jnp.array([d11, d12, d13, d14])


@el.map
def world_euler_angles(pos: el.WorldPos) -> WorldEulerAngles:
    return quat_to_euler(pos.angular()) / 2.0


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
def v_rel_accel_buffer(
    time: el.Time, a_rel: VRelAccel, buffer: VRelAccelBuffer
) -> VRelAccelBuffer:
    slow_start_time = 3.0
    slow_start = jnp.minimum(time, slow_start_time) / slow_start_time
    sample_tick_index = 1.0 / lp_sample_freq
    a_rel *= slow_start
    return jax.lax.cond(
        time % sample_tick_index < TIME_STEP,
        lambda _: jnp.concatenate((buffer[1:], a_rel.reshape(1, 3))),
        lambda _: buffer,
        operand=None,
    )
    # return jnp.concatenate((buffer[1:], a_rel.reshape(1, 3)))


@el.map
def v_rel_accel_filtered(s: VRelAccelBuffer) -> VRelAccelFiltered:
    return second_order_butterworth(s, lp_sample_freq, lp_cutoff_freq)[-1]


@el.map
def accel_setpoing_smooth(
    a: AccelSetpoint, a_s: AccelSetpointSmooth
) -> AccelSetpointSmooth:
    diff = jnp.clip(a - a_s, -0.2, 0.2)
    return a_s + diff


@el.map
def pitch_pid_state(
    time: el.Time,
    a_setpoint: AccelSetpointSmooth,
    a_rel: VRelAccelFiltered,
    s: PitchPIDState,
) -> PitchPIDState:
    e = a_rel[1] - a_setpoint[0]
    # e = jnp.clip(e, -5.0, 5.0)
    i = jnp.clip(s[1] + e * TIME_STEP, -2.0, 2.0)
    d = e - s[0]
    pid_state = jnp.array([e, i, d])
    return pid_state


@el.map
def pitch_pid_control(pid: PitchPID, s: PitchPIDState) -> FinControl:
    Kp, Ki, Kd = pid
    e, i, d = s
    mv = Kp * e + Ki * i + Kd * d
    fin_control = jnp.array([0.0, mv, 0.0, mv]) * TIME_STEP
    return fin_control


@el.map
def fin_control(fd: FinDeflection, fc: FinControl, mach: Mach) -> FinDeflection:
    # fc = fc / (1e-6 + mach)
    return fd + fc


w = el.World()
rocket = (
    w.spawn(
        el.Body(
            world_pos=el.SpatialTransform.from_linear(jnp.array([0.0, 1.0, 0.0]))
            + el.SpatialTransform.from_angular(
                euler_to_quat(jnp.array([0.0, 0.0, -60.0]))
            ),
            pbr=w.insert_asset(
                el.Pbr.from_url(
                    "https://storage.googleapis.com/elodin-marketing/models/rocket.glb"
                )
            ),
            inertia=el.SpatialInertia(3.0, jnp.array([0.1, 3.0, 3.0])),
        )
    )
    .name("Rocket")
    .insert(Rocket())
)

w.spawn(
    el.Panel.viewport(
        track_entity=rocket.id(),
        track_rotation=False,
        active=True,
        pos=[5.0, 1.0, 0.0],
        looking_at=[0.0, 0.0, 0.0],
        show_grid=True,
    )
).name("Viewport (Rocket)")
# w.spawn(
#     el.Panel.hsplit(
#         [
#             el.Panel.vsplit(
#                 [
#                     el.Panel.viewport(
#                         track_entity=rocket.id(),
#                         track_rotation=False,
#                         pos=[5.0, 1.0, 0.0],
#                         looking_at=[0.0, 0.0, 0.0],
#                         show_grid=True,
#                     ),
#                     el.Panel.graph(
#                         [
#                             el.GraphEntity(
#                                 rocket.id(),
#                                 [
#                                     el.Component.index(FinDeflection)[1],
#                                     el.Component.index(FinControl)[1],
#                                 ],
#                             )
#                         ]
#                     ),
#                 ]
#             ),
#             el.Panel.vsplit(
#                 [
#                     el.Panel.graph(
#                         [
#                             el.GraphEntity(
#                                 rocket.id(),
#                                 [
#                                     el.Component.index(PitchPIDState),
#                                 ],
#                             )
#                         ]
#                     ),
#                     el.Panel.graph(
#                         [
#                             el.GraphEntity(
#                                 rocket.id(),
#                                 [
#                                     el.Component.index(VRelAccel)[1:2],
#                                     el.Component.index(VRelAccelFiltered)[1:2],
#                                     el.Component.index(AccelSetpoint)[:1],
#                                     el.Component.index(AccelSetpointSmooth)[:1],
#                                 ],
#                             )
#                         ]
#                     ),
#                 ]
#             ),
#         ],
#         active=True,
#     )
# )

effectors = (
    gravity
    | mach
    | angle_of_attack
    | world_euler_angles
    | accel_setpoing_smooth
    | v_rel_accel
    | v_rel_accel_buffer
    | v_rel_accel_filtered
    | pitch_pid_state
    | pitch_pid_control
    | fin_control
    | apply_thrust
    | aerodynamic_coefs
    | aerodynamic_forces
)
sys = el.advance_time(TIME_STEP) | el.six_dof(
    TIME_STEP, effectors, integrator=el.Integrator.SemiImplicit
)
w.run(sys, time_step=TIME_STEP)
