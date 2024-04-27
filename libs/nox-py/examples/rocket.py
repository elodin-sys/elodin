import typing as ty
import elodin as el
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jax.scipy.ndimage import map_coordinates
import polars as pl
import polars.selectors as cs
import io

TIME_STEP = 1.0 / 120.0
thrust_vector_body_frame = jnp.array([-1.0, 0.0, 0.0])

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
        metadata={"element_names": "Cl,Cn,Cm,CA,CN,CY"},
    ),
]

CenterOfPressure = ty.Annotated[
    jax.Array,
    el.Component(
        "center_of_pressure",
        el.ComponentType(el.PrimitiveType.F64, (2,)),
        metadata={"element_names": "n,y"},
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

AeroForce = ty.Annotated[
    el.SpatialForce, el.Component("aero_force", el.ComponentType.SpatialMotionF64)
]


@el.dataclass
class Rocket(el.Archetype):
    alphac: AngleOfAttack
    aero_coefs: AeroCoefs
    center_of_pressure: CenterOfPressure
    center_of_gravity: CenterOfGravity
    time: el.Time
    mach: Mach
    dynamic_pressure: DynamicPressure
    aero_force: AeroForce
    wind: Wind


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


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.Force.from_linear(jnp.array([0.0, inertia.mass() * -9.81, 0.0]))


def thrust_curve() -> jax.Array:
    thrust_curve = """
    0.01 642.879
    0.87 519.675
    1.73 47.188
    2.59 47.188
    3.45 47.188
    4.31 47.188
    5.17 47.188
    6.03 47.188
    6.89 47.188
    7.75 47.188
    8.61 47.188
    9.47 47.188
    10.33 47.188
    11.19 47.188
    12.05 47.188
    12.91 47.188
    13.77 47.188
    14.63 47.188
    15.49 47.188
    16.35 47.188
    17.21 47.188
    18.07 47.188
    18.93 47.188
    19.79 47.188
    20.65 47.188
    21.51 47.188
    22.37 47.188
    23.23 47.188
    24.09 47.188
    24.95 47.188
    25.81 47.188
    26.67 47.188
    27.53 47.188
    28.39 47.188
    29.25 47.188
    30.11 47.188
    30.97 47.188
    31.83 47.188
    32.69 47.188
    33.55 47.188
    34.41 47.188
    35.27 47.188
    36.13 47.188
    36.99 47.188
    37.85 47.188
    38.71 47.188
    39.57 47.188
    40.43 47.188
    41.29 47.188
    42.15 0
    """
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


def read_aero_csv(data: str) -> pl.DataFrame:
    """
    This function extracts the aerodynamics data from a CSV str and returns a Polars DataFrame.

    IJ = input body load, 1 ≤ IJ ≤ NMACH*NCOND
    Mach = Mach number
    Alphac = included angle of attack (degrees)
    CA = axial force coefficient
    CN = normal force coefficient
    CY = side force coefficient
    Cm = pitching moment coefficient
    Cn = yawing moment coefficient
    Cl = rolling moment coefficient
    XCP(CN)/L = center of pressure for normal force
    XCP(CY)/L = center of pressure for side force
    """
    clean_data = io.StringIO()
    for line in data.split("\n"):
        if "REFERENCE AREA" in line:
            line = line.strip('"').split()
            # Parse "REFERENCE AREA =   14.08130  REFERENCE LENGTH =    4.23400  XMC/LBASE =    0.44997"
            A, L, Xmc = float(line[3]), float(line[7]), float(line[10])
        else:
            clean_data.write(line + "\n")
    clean_data.seek(0)
    df = pl.read_csv(
        clean_data,
        comment_prefix="#",
        dtypes={
            "IJ": pl.Int32,
            "Mach": pl.Float64,
            "Alphac": pl.Float64,
            "CA": pl.Float64,
            "CN": pl.Float64,
            "CY": pl.Float64,
            "Cm": pl.Float64,
            "Cn": pl.Float64,
            "Cl": pl.Float64,
            "XCP(CN)/L": pl.Float64,
            "XCP(CY)/L": pl.Float64,
            "Delta11": pl.Float64,
            "Delta12": pl.Float64,
            "Delta13": pl.Float64,
            "Delta14": pl.Float64,
        },
    )
    df = df.select(cs.by_dtype([pl.Int32, pl.Float64])).drop_nulls()
    # TODO: support fin deflection angles
    df = df.filter(
        (df["Delta11"] == 0)
        & (df["Delta12"] == 0)
        & (df["Delta13"] == 0)
        & (df["Delta14"] == 0)
    )
    df = df.with_columns(
        pl.Series("A", [A] * len(df)),
        pl.Series("L", [L] * len(df)),
        pl.Series("Xmc", [Xmc] * len(df)),
    )
    df = df.sort(["Mach", "Alphac"])
    clean_data.close()
    return df


@el.map
def angle_of_attack(p: el.WorldPos, v: el.WorldVel, w: Wind) -> AngleOfAttack:
    ang_pos = p.angular() @ thrust_vector_body_frame
    # sign of angle of attack
    # u = freestream velocity vector
    u = v.linear() - w
    u = jax.lax.cond(
        la.norm(u) < 1e-6,
        lambda _: ang_pos,
        lambda _: u,
        operand=None,
    )
    q = quat_from_vecs(u, ang_pos)
    aoa = q @ thrust_vector_body_frame
    alphac = jnp.rad2deg(2 * jnp.arccos(q.vector()[3]))
    z_sign = jnp.sign(aoa[1])
    return alphac * z_sign


@el.map
def aerodynamic_coefs(
    mach: Mach, alphac: AngleOfAttack
) -> tuple[AeroCoefs, CenterOfPressure]:
    # prerequsite: steps in Mach, AngleOfAttack, Phi, Fin deflection angles must be uniform
    # TODO: replace with better reference data
    aero_data = """
"REFERENCE AREA =   24.89130  REFERENCE LENGTH =    5.43400  XMC/LBASE =    0.40387"
"IJ","Mach","Re/Length","Alphac","Phi","CA","CN","Cm","CY","Cn","Cl","XCP(CN)/L","XCP(CY)/L","CA w/BASE","Cmq+Cmad","Clp","Cmq","Cnr","CZq","CYr","pl/2V","ql/2V","rl/2V","Phif1","Delta11","Delta12","Delta13","Delta14","Phif2","Delta21","Delta22","Delta23","Delta24","Phif3","Delta31","Delta32","Delta33","Delta34",
  
  241,  0.10,  6.600E+06,   0.00,   0.00,    2.942E-01,  0.000E+00,  0.000E+00,    0.000E+00,  0.000E+00,  0.000E+00,    4.039E-01,  4.039E-01,   4.203E-01,   0.000E+00,   -2.146E+01, -5.281E+02, -5.359E+02,  6.856E+01,  7.281E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
  244,  0.10,  6.600E+06,   6.00,   0.00,    2.837E-01,  1.304E+00,  1.569E+00,    0.000E+00,  0.000E+00,  0.000E+00,    2.984E-01,  4.039E-01,   4.073E-01,   0.000E+00,   -2.323E+01, -6.251E+02, -5.720E+02,  6.944E+01,  7.702E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
  247,  0.10,  6.600E+06,  12.00,   0.00,    2.402E-01,  2.804E+00,  1.703E+00,    0.000E+00,  0.000E+00,  0.000E+00,    3.506E-01,  4.039E-01,   3.588E-01,   0.000E+00,   -2.167E+01, -6.578E+02, -6.308E+02,  8.128E+01,  8.501E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
  250,  0.10,  6.600E+06,  18.00,   0.00,    2.202E-01,  3.417E+00,  8.563E-01,    0.000E+00,  0.000E+00,  0.000E+00,    3.819E-01,  4.039E-01,   3.313E-01,   0.000E+00,   -1.659E+01, -4.698E+02, -5.561E+02,  7.451E+01,  7.432E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,

 1221,  0.30,  6.600E+06,   0.00,   0.00,    2.939E-01,  0.000E+00,  0.000E+00,    0.000E+00,  0.000E+00,  0.000E+00,    4.039E-01,  4.039E-01,   4.199E-01,   0.000E+00,   -2.146E+01, -5.241E+02, -5.319E+02,  6.812E+01,  7.237E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 1224,  0.30,  6.600E+06,   6.00,   0.00,    2.833E-01,  1.346E+00,  1.627E+00,    0.000E+00,  0.000E+00,  0.000E+00,    2.979E-01,  4.039E-01,   4.070E-01,   0.000E+00,   -2.301E+01, -6.188E+02, -5.691E+02,  6.798E+01,  7.656E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 1227,  0.30,  6.600E+06,  12.00,   0.00,    2.401E-01,  2.893E+00,  1.771E+00,    0.000E+00,  0.000E+00,  0.000E+00,    3.502E-01,  4.039E-01,   3.587E-01,   0.000E+00,   -2.168E+01, -6.578E+02, -6.305E+02,  8.076E+01,  8.499E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 1230,  0.30,  6.600E+06,  18.00,   0.00,    2.171E-01,  3.557E+00,  8.149E-01,    0.000E+00,  0.000E+00,  0.000E+00,    3.838E-01,  4.039E-01,   3.284E-01,   0.000E+00,   -1.660E+01, -4.799E+02, -5.592E+02,  7.550E+01,  7.471E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
  
 2201,  0.50,  6.600E+06,   0.00,   0.00,    2.924E-01,  0.000E+00,  0.000E+00,    0.000E+00,  0.000E+00,  0.000E+00,    4.039E-01,  4.039E-01,   4.185E-01,   0.000E+00,   -2.146E+01, -5.233E+02, -5.311E+02,  6.803E+01,  7.228E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 2204,  0.50,  6.600E+06,   6.00,   0.00,    2.820E-01,  1.448E+00,  1.779E+00,    0.000E+00,  0.000E+00,  0.000E+00,    2.962E-01,  4.039E-01,   4.057E-01,   0.000E+00,   -2.301E+01, -5.989E+02, -5.593E+02,  6.455E+01,  7.492E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 2207,  0.50,  6.600E+06,  12.00,   0.00,    2.391E-01,  3.110E+00,  1.961E+00,    0.000E+00,  0.000E+00,  0.000E+00,    3.486E-01,  4.039E-01,   3.579E-01,   0.000E+00,   -2.169E+01, -6.583E+02, -6.309E+02,  8.073E+01,  8.506E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 2210,  0.50,  6.600E+06,  18.00,   0.00,    2.116E-01,  3.879E+00,  7.611E-01,    0.000E+00,  0.000E+00,  0.000E+00,    3.867E-01,  4.039E-01,   3.230E-01,   0.000E+00,   -1.661E+01, -4.910E+02, -5.643E+02,  7.673E+01,  7.531E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
  
 3181,  0.70,  6.600E+06,   0.00,   0.00,    3.039E-01,  0.000E+00,  0.000E+00,    0.000E+00,  0.000E+00,  0.000E+00,    4.039E-01,  4.039E-01,   4.282E-01,   0.000E+00,   -2.181E+01, -5.285E+02, -5.354E+02,  6.749E+01,  7.127E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 3184,  0.70,  6.600E+06,   6.00,   0.00,    2.929E-01,  1.553E+00,  1.967E+00,    0.000E+00,  0.000E+00,  0.000E+00,    2.928E-01,  4.039E-01,   4.148E-01,   0.000E+00,   -2.355E+01, -6.038E+02, -5.627E+02,  6.269E+01,  7.370E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 3187,  0.70,  6.600E+06,  12.00,   0.00,    2.471E-01,  3.328E+00,  2.119E+00,    0.000E+00,  0.000E+00,  0.000E+00,    3.481E-01,  4.039E-01,   3.643E-01,   0.000E+00,   -2.143E+01, -6.567E+02, -6.375E+02,  8.356E+01,  8.443E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 3190,  0.70,  6.600E+06,  18.00,   0.00,    2.114E-01,  4.302E+00,  6.044E-01,    0.000E+00,  0.000E+00,  0.000E+00,    3.916E-01,  4.039E-01,   3.214E-01,   0.000E+00,   -1.539E+01, -4.861E+02, -5.729E+02,  8.461E+01,  7.467E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
  
 4161,  0.90,  6.600E+06,   0.00,   0.00,    4.480E-01,  0.000E+00,  0.000E+00,    0.000E+00,  0.000E+00,  0.000E+00,    4.039E-01,  4.039E-01,   5.694E-01,   0.000E+00,   -2.388E+01, -5.549E+02, -5.633E+02,  6.633E+01,  7.097E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 4164,  0.90,  6.600E+06,   6.00,   0.00,    4.431E-01,  1.602E+00,  2.213E+00,    0.000E+00,  0.000E+00,  0.000E+00,    2.828E-01,  4.039E-01,   5.624E-01,   0.000E+00,   -2.534E+01, -6.265E+02, -5.903E+02,  6.257E+01,  7.344E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 4167,  0.90,  6.600E+06,  12.00,   0.00,    4.286E-01,  3.505E+00,  2.642E+00,    0.000E+00,  0.000E+00,  0.000E+00,    3.378E-01,  4.039E-01,   5.433E-01,   0.000E+00,   -2.012E+01, -6.673E+02, -6.714E+02,  9.004E+01,  8.511E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
 4170,  0.90,  6.600E+06,  18.00,   0.00,    4.052E-01,  4.811E+00,  1.591E+00,    0.000E+00,  0.000E+00,  0.000E+00,    3.749E-01,  4.039E-01,   5.129E-01,   0.000E+00,   -1.830E+01, -5.577E+02, -6.278E+02,  8.622E+01,  7.826E+01,    0.000E+00,  0.000E+00,  0.000E+00,    0.00,   0.00,   0.00,   0.00,   0.00,    0.00,   0.00,   0.00,   0.00,   0.00,   45.00,   0.00,   0.00,   0.00,   0.00,
    """
    # Can also be read from a file:
    # with open("path/to/csv.csv", "r") as file:
    #     aero_data = file.read()
    df = read_aero_csv(aero_data)
    aero = {}
    for c in ["CA", "CN", "CY", "Cm", "Cn", "Cl", "XCP(CN)/L", "XCP(CY)/L"]:
        aero[c] = jnp.array(
            [
                df.group_by(["Alphac"], maintain_order=True)
                .agg(pl.col(c).min())[c]
                .to_list()
                for _, df in df.group_by(["Mach"], maintain_order=True)
            ]
        )

    def to_coord(s: pl.Series, val: jax.Array) -> jax.Array:
        s_min = s.min()
        s_max = s.max()
        s_count = len(s.unique())
        return (val - s_min) * (s_count - 1) / jnp.clip(s_max - s_min, 1e-06)

    interp_coords = [
        [to_coord(df["Mach"], mach)],
        [to_coord(df["Alphac"], jnp.abs(alphac))],
    ]
    coefs = jnp.array(
        [
            map_coordinates(aero[c], interp_coords, 1, mode="nearest")[0]
            for c in ["Cl", "Cn", "Cm", "CA", "CN", "CY"]
        ]
    )
    xcp = jnp.array(
        [
            map_coordinates(aero[c], interp_coords, 1, mode="nearest")[0]
            for c in ["XCP(CN)/L", "XCP(CY)/L"]
        ]
    )
    return coefs, xcp


@el.map
def aerodynamic_forces(
    aero_coefs: AeroCoefs,
    xcp: CenterOfPressure,
    xcg: CenterOfGravity,
    alphac: AngleOfAttack,
    q: DynamicPressure,
    p: el.WorldPos,
    f: el.Force,
) -> tuple[AeroForce, el.Force]:
    lbody = 62
    xcg /= lbody
    a_ref = 24.89130 / 100**2
    l_ref = 5.43400 / 100
    xmc = 0.40387
    xcp_cn, xcp_cy = xcp
    Cl, Cn, Cm, CA, CN, CY = aero_coefs

    # shift Cm from moment center to CG
    # Cm at moment center is Cm
    # Cm at xcp_cn is 0
    # TODO: there must be a better way to do this
    Cm = Cm - CN * (xcg - xmc) / l_ref
    Cn = Cn - CY * (xcg - xmc) / l_ref

    f_aero_linear = jnp.array([CA, CN * jnp.sign(alphac), CY]) * q * a_ref
    f_aero_torque = jnp.array([Cl, Cn, Cm * jnp.sign(alphac)]) * q * a_ref * l_ref
    # body-fixed frame:
    f_aero = el.Force.from_linear(f_aero_linear) + el.Force.from_torque(f_aero_torque)
    # world frame:
    f_aero_rot = el.Force.from_linear(
        p.angular() @ f_aero_linear
    ) + el.Force.from_torque(p.angular() @ f_aero_torque)

    return f_aero, f + f_aero_rot


@el.map
def apply_thrust(t: el.Time, p: el.WorldPos, f: el.Force) -> el.Force:
    tc = thrust_curve()
    f_t = jnp.interp(t, tc[0], tc[1], right=0.0)
    thrust = p.angular() @ thrust_vector_body_frame * f_t
    return f + el.Force.from_linear(thrust)


# @el.map
# def body_fixed_rot(p: el.WorldPos, v: el.WorldVel):
#     ang_pos = p.angular() @ thrust_vector_body_frame
#     # sign of angle of attack
#     # u = freestream velocity vector
#     u = v.linear() - w.linear()
#     u = jax.lax.cond(
#         la.norm(u) < 1e-6,
#         lambda _: ang_pos,
#         lambda _: u,
#         operand=None,
#     )
#     q = quat_from_vecs(u, ang_pos)
#     aoa = q @ thrust_vector_body_frame
#     alphac = jnp.rad2deg(2 * jnp.arccos(q.vector()[3]))
#     z_sign = jnp.sign(aoa[1])
#     return alphac * z_sign


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
            inertia=el.Inertia.from_mass(2.5),
        )
    )
    .name("Rocket")
    .insert(
        Rocket(
            alphac=jnp.array([0.0]),
            aero_coefs=jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            center_of_pressure=jnp.array([0.0, 0.0]),
            center_of_gravity=jnp.float64(18.6),
            time=jnp.float64(0.0),
            mach=jnp.float64(0),
            dynamic_pressure=jnp.float64(0.0),
            aero_force=el.Force.zero(),
            wind=jnp.array([0.0, 0.0, 0.0]),
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
    | mach
    | angle_of_attack
    | aerodynamic_coefs
    | apply_thrust
    | aerodynamic_forces
)
sys = el.advance_time(TIME_STEP) | el.six_dof(TIME_STEP, effectors)
w.run(sys)
