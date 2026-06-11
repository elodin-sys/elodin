import math
import typing as ty
from datetime import datetime, timezone
from pathlib import Path

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
from reference import build_reference

SIMULATION_RATE_HZ = 120.0
SIM_TIME_STEP = 1.0 / SIMULATION_RATE_HZ
TELEMETRY_RATE_HZ = 40.0
GUIDANCE_RATE_HZ = 24.0
SCHEMATIC_PATH = Path(__file__).with_name("apollo-lander.kdl")
PDI_START = datetime(1969, 7, 20, 20, 9, 53, 164000, tzinfo=timezone.utc)
START_TIMESTAMP_US = int(PDI_START.timestamp() * 1_000_000)

G0 = 9.80665
LUNAR_GRAVITY = 1.622
R_MOON_M = 1_737_400.0
DPS_MAX_THRUST_N = 45_040.0
DPS_MIN_THRUST_N = 4_670.0
DPS_FTP_THROTTLE = 0.925  # fixed throttle point flown through the braking phase
THROTTLE_MIN = DPS_MIN_THRUST_N / DPS_MAX_THRUST_N
THROTTLE_MAX = 1.0
RCS_THRUST_N = 445.0
RCS_ISP_S = 290.0
RCS_MOMENT_ARM_M = 2.0
RCS_AXIS_TORQUE_LIMIT_NM = 4.0 * RCS_THRUST_N * RCS_MOMENT_ARM_M
SOFT_VERTICAL_SPEED_MPS = 3.0
SOFT_HORIZONTAL_SPEED_MPS = 1.0
UPRIGHT_DOT_MIN = 0.94

# Smoothed Apollo 11 descent reference (true altitude / descent rate /
# approximate pitch / reconstructed horizontal profile) shared with the
# external guidance controller.
REFERENCE = build_reference()
REF_TIME_S = np.asarray(REFERENCE.time_s, dtype=np.float64)
REF_ALTITUDE_M = np.asarray(REFERENCE.altitude_m, dtype=np.float64)
REF_RATE_MPS = np.asarray(REFERENCE.descent_rate_mps, dtype=np.float64)
REF_PITCH_DEG = np.asarray(REFERENCE.pitch_deg, dtype=np.float64)
REF_SLANT_RANGE_M = np.asarray(REFERENCE.slant_range_m, dtype=np.float64)
REF_HSPEED_MPS = np.asarray(REFERENCE.horizontal_speed_mps, dtype=np.float64)
REF_DOWNRANGE_M = np.asarray(REFERENCE.downrange_m, dtype=np.float64)
REF_INIT_ALTITUDE = float(REF_ALTITUDE_M[0])
REF_INIT_RATE = float(REF_RATE_MPS[0])
REF_INIT_HSPEED = float(REF_HSPEED_MPS[0])
REF_INIT_DOWNRANGE = float(REF_DOWNRANGE_M[0])
# Run slightly past the data window so the gentle terminal descent reaches contact.
DEFAULT_MAX_TICKS = int(math.ceil((REFERENCE.t_end + 20.0) * SIMULATION_RATE_HZ)) + 1

# --- Visualization: GLB model scaling (units derived from the NASA assets) ---
# The Apollo LM GLB is modeled in meters (world AABB ~6.4 m footprint, ~5.0 m
# tall, Y-up), so render it at its modeled scale to be ~life-size.
LANDER_GLB_SCALE = 1.0
# The landing-site GLB is a 256-sample heightmap of a 30 km x 30 km region: its
# 255.5 native units span 30 km (~117.4 m/unit), it is Z-up, and elevation is
# exaggerated 60x. Scale to the true horizontal extent; the -90 deg X rotation in
# the schematic stands the Z-up tile upright in the editor's Y-up render space.
TERRAIN_NATIVE_SPAN = 255.5
TERRAIN_REGION_M = 30_000.0
TERRAIN_GLB_SCALE = TERRAIN_REGION_M / TERRAIN_NATIVE_SPAN  # ~117.4
# Tile-center (landing point) elevation in native units, sampled from the mesh.
# Seat the rendered surface at world z = 0 so the lander meets it at touchdown.
TERRAIN_CENTER_NATIVE_Z = 10.63
TERRAIN_SEAT_Z = -TERRAIN_GLB_SCALE * TERRAIN_CENTER_NATIVE_Z

PARAMS = el.monte_carlo.params_spec(
    # Radar-corrected altitude at landing-radar lock-on (~38,700 ft). The
    # navigation dispersions are tight: a saturated FTP braking burn cannot
    # recover large state errors, exactly as on the real mission.
    init_altitude_m=el.monte_carlo.Param(
        float, default=REF_INIT_ALTITUDE, min=11_650.0, max=11_950.0
    ),
    init_vertical_speed_mps=el.monte_carlo.Param(
        float, default=REF_INIT_RATE, min=-32.0, max=-18.0
    ),
    # Residual orbital velocity mid-way through the P63 braking phase
    # (reconstructed ~800 m/s at radar lock).
    init_downrange_speed_mps=el.monte_carlo.Param(
        float, default=REF_INIT_HSPEED, min=784.0, max=814.0
    ),
    init_crossrange_speed_mps=el.monte_carlo.Param(float, default=0.0, min=-3.0, max=3.0),
    # Sampled offset from the reconstructed window-start downrange position.
    init_downrange_offset_m=el.monte_carlo.Param(float, default=0.0, min=-1_500.0, max=1_500.0),
    # Retrograde braking attitude (~77 deg from vertical at radar lock).
    init_pitch_deg=el.monte_carlo.Param(float, default=-77.0, min=-80.0, max=-70.0),
    dry_mass_kg=el.monte_carlo.Param(float, default=6_853.0, min=6_750.0, max=6_950.0),
    # DPS propellant remaining at the start of the telemetry window. The full
    # load was ~8,212 kg; the mission-report fuel chart shows ~4,260 kg burned
    # by radar lock-on.
    propellant_kg=el.monte_carlo.Param(float, default=3_950.0, min=3_800.0, max=4_200.0),
    rcs_propellant_kg=el.monte_carlo.Param(float, default=240.0, min=180.0, max=320.0),
    thrust_scale=el.monte_carlo.Param(float, default=1.0, min=0.98, max=1.02),
    isp_s=el.monte_carlo.Param(float, default=311.0, min=308.0, max=314.0),
    gravity_scale=el.monte_carlo.Param(float, default=1.0, min=0.998, max=1.002),
    track_gain=el.monte_carlo.Param(float, default=0.04, min=0.02, max=0.06),
    horizontal_gain=el.monte_carlo.Param(float, default=0.05, min=0.02, max=0.10),
    vertical_gain=el.monte_carlo.Param(float, default=0.45, min=0.30, max=0.60),
    attitude_gain=el.monte_carlo.Param(float, default=0.040, min=0.030, max=0.060),
    throttle_response_hz=el.monte_carlo.Param(float, default=3.0, min=2.0, max=5.0),
)

Altitude = ty.Annotated[
    jax.Array,
    el.Component("altitude", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
VerticalSpeed = ty.Annotated[
    jax.Array,
    el.Component("vertical_speed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
HorizontalSpeed = ty.Annotated[
    jax.Array,
    el.Component("horizontal_speed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
Pitch = ty.Annotated[
    jax.Array,
    el.Component("pitch", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
Throttle = ty.Annotated[
    jax.Array,
    el.Component("throttle", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
ThrottleCmd = ty.Annotated[
    jax.Array,
    el.Component(
        "throttle_cmd",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"external_control": "true"},
    ),
]
AttitudeSetpoint = ty.Annotated[
    el.Quaternion,
    el.Component("attitude_setpoint", metadata={"external_control": "true"}),
]
Propellant = ty.Annotated[
    jax.Array,
    el.Component("propellant", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
Thrust = ty.Annotated[
    jax.Array,
    el.Component("thrust", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
RcsTorque = ty.Annotated[
    jax.Array,
    el.Component("rcs_torque", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
RcsPropellant = ty.Annotated[
    jax.Array,
    el.Component("rcs_propellant", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
MainThrustViz = ty.Annotated[
    jax.Array,
    el.Component("main_thrust_viz", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
RcsTorqueViz = ty.Annotated[
    jax.Array,
    el.Component("rcs_torque_viz", el.ComponentType(el.PrimitiveType.F64, (3,))),
]
# Per-nozzle cold-gas firing level (0..1) for editor particles. Index order matches
# `RCS_JETS` in libs/elodin-editor/src/plugins/thruster_particles/mod.rs.
RcsThrusterViz = ty.Annotated[
    jax.Array,
    el.Component("rcs_thruster_viz", el.ComponentType(el.PrimitiveType.F64, (16,))),
]
RCS_THRUSTER_AXIS = jnp.array([0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2], dtype=jnp.int32)
RCS_THRUSTER_SIGN = jnp.array(
    [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
    dtype=jnp.float64,
)
Landed = ty.Annotated[
    jax.Array,
    el.Component("landed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
TouchdownSpeed = ty.Annotated[
    jax.Array,
    el.Component("touchdown_speed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
TouchdownHorizontalSpeed = ty.Annotated[
    jax.Array,
    el.Component("touchdown_horizontal_speed", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# Landing-radar slant range replay (truth ghost only); during the pitched-back
# braking phase it visibly exceeds the true altitude.
SlantRange = ty.Annotated[
    jax.Array,
    el.Component("slant_range", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
# Marker scoping the truth-replay playback system to the lander_truth entity.
TruthMarker = ty.Annotated[
    jax.Array,
    el.Component("truth_marker", el.ComponentType(el.PrimitiveType.F64, (1,))),
]


@el.dataclass
class StaticSceneObject(el.Archetype):
    world_pos: el.WorldPos


def truth_altitude(t_s: float) -> float:
    return float(np.interp(t_s, REF_TIME_S, REF_ALTITUDE_M))


def truth_velocity(t_s: float) -> float:
    return float(np.interp(t_s, REF_TIME_S, REF_RATE_MPS))


def truth_pitch(t_s: float) -> float:
    """Smoothed, approximate LM pitch-from-vertical from the inner gimbal angle.

    The source CSV reports PGNS stable-member gimbal angles; a rigorous vehicle
    attitude reconstruction needs the mission REFSMMAT we do not have, so this is
    an illustrative attitude trend (the LM pitches back to brake, then uprights).
    """

    return abs(float(np.interp(t_s, REF_TIME_S, REF_PITCH_DEG)))


def truth_quat(t_s: float) -> np.ndarray:
    pitch = -math.radians(truth_pitch(t_s))
    s = math.sin(pitch * 0.5)
    c = math.cos(pitch * 0.5)
    return np.array([0.0, s, 0.0, c], dtype=np.float64)


def _quat_from_pitch_deg(deg: float) -> el.Quaternion:
    return el.Quaternion.from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.deg2rad(deg))


def build(params: el.monte_carlo.Params) -> tuple[el.World, el.System]:
    world = el.World()

    init_altitude = float(params.get("init_altitude_m", REF_INIT_ALTITUDE))
    init_vertical_speed = float(params.get("init_vertical_speed_mps", REF_INIT_RATE))
    init_downrange_speed = float(params.get("init_downrange_speed_mps", REF_INIT_HSPEED))
    init_crossrange_speed = float(params.get("init_crossrange_speed_mps", 0.0))
    init_downrange = REF_INIT_DOWNRANGE + float(params.get("init_downrange_offset_m", 0.0))
    init_pitch = float(params.get("init_pitch_deg", -77.0))
    dry_mass = float(params.get("dry_mass_kg", 6_853.0))
    propellant = float(params.get("propellant_kg", 3_950.0))
    rcs_propellant = float(params.get("rcs_propellant_kg", 240.0))
    thrust_scale = float(params.get("thrust_scale", 1.0))
    isp = float(params.get("isp_s", 311.0))
    gravity_scale = float(params.get("gravity_scale", 1.0))
    attitude_gain = float(params.get("attitude_gain", 0.040))
    throttle_response_hz = float(params.get("throttle_response_hz", 3.0))
    total_mass = dry_mass + propellant + rcs_propellant
    # Representative LM inertia tensor: roughly a 7m tall, 4m wide vehicle. The
    # DPS is treated as gimbaled through the moving CG; RCS provides control torque.
    base_inertia_diag = jnp.array([78_000.0, 72_000.0, 45_000.0], dtype=jnp.float64)
    landed_inertia = jnp.array([1.0e9, 1.0e9, 1.0e9], dtype=jnp.float64)
    rcs_k = jnp.array([4_500.0, 5_500.0, 4_500.0], dtype=jnp.float64) * (attitude_gain / 0.040)
    # Damping sized for ~0.85 damping ratio against the proportional gains so
    # the large braking-phase attitude commands do not overshoot.
    rcs_d = jnp.array([19_000.0, 21_000.0, 19_000.0], dtype=jnp.float64)
    rcs_limit = jnp.array([RCS_AXIS_TORQUE_LIMIT_NM] * 3, dtype=jnp.float64)
    response_alpha = min(max(throttle_response_hz * SIM_TIME_STEP, 0.0), 1.0)
    lunar_g = LUNAR_GRAVITY * gravity_scale

    initial_attitude = _quat_from_pitch_deg(init_pitch)
    world.spawn(
        [
            el.Body(
                world_pos=el.SpatialTransform(
                    angular=initial_attitude,
                    linear=jnp.array([init_downrange, 0.0, init_altitude], dtype=jnp.float64),
                ),
                world_vel=el.SpatialMotion(
                    angular=jnp.zeros(3, dtype=jnp.float64),
                    linear=jnp.array(
                        [init_downrange_speed, init_crossrange_speed, init_vertical_speed],
                        dtype=jnp.float64,
                    ),
                ),
                inertia=el.SpatialInertia(total_mass, base_inertia_diag),
            ),
            el.C(Altitude, jnp.array([init_altitude], dtype=jnp.float64)),
            el.C(VerticalSpeed, jnp.array([init_vertical_speed], dtype=jnp.float64)),
            el.C(
                HorizontalSpeed, jnp.array([jnp.hypot(init_downrange_speed, init_crossrange_speed)])
            ),
            # Pitch telemetry is unsigned tilt-from-vertical (see derive_telemetry).
            el.C(Pitch, jnp.array([abs(init_pitch)], dtype=jnp.float64)),
            # The window opens mid-braking-burn with the DPS at the fixed
            # throttle point.
            el.C(Throttle, jnp.array([DPS_FTP_THROTTLE], dtype=jnp.float64)),
            el.C(ThrottleCmd, jnp.array([DPS_FTP_THROTTLE], dtype=jnp.float64)),
            el.C(AttitudeSetpoint, initial_attitude),
            el.C(Propellant, jnp.array([propellant], dtype=jnp.float64)),
            el.C(RcsPropellant, jnp.array([rcs_propellant], dtype=jnp.float64)),
            el.C(Thrust, jnp.array([0.0], dtype=jnp.float64)),
            el.C(RcsTorque, jnp.zeros(3, dtype=jnp.float64)),
            el.C(MainThrustViz, jnp.zeros(3, dtype=jnp.float64)),
            el.C(RcsTorqueViz, jnp.zeros(3, dtype=jnp.float64)),
            el.C(RcsThrusterViz, jnp.zeros(16, dtype=jnp.float64)),
            el.C(Landed, jnp.array([0.0], dtype=jnp.float64)),
            el.C(TouchdownSpeed, jnp.array([0.0], dtype=jnp.float64)),
            el.C(TouchdownHorizontalSpeed, jnp.array([0.0], dtype=jnp.float64)),
        ],
        name="lander",
    )

    # The truth vehicle is a kinematic ghost: no el.Body, so gravity and the
    # six-dof integrator never touch it. truth_playback drives it every tick.
    world.spawn(
        [
            StaticSceneObject(
                el.WorldPos(
                    angular=el.Quaternion(truth_quat(0.0)),
                    linear=jnp.array(
                        [REF_INIT_DOWNRANGE, 30.0, REF_INIT_ALTITUDE], dtype=jnp.float64
                    ),
                )
            ),
            el.C(TruthMarker, jnp.array([1.0], dtype=jnp.float64)),
            el.C(Altitude, jnp.array([REF_INIT_ALTITUDE], dtype=jnp.float64)),
            el.C(VerticalSpeed, jnp.array([truth_velocity(0.0)], dtype=jnp.float64)),
            el.C(HorizontalSpeed, jnp.array([REF_INIT_HSPEED], dtype=jnp.float64)),
            el.C(SlantRange, jnp.array([float(REF_SLANT_RANGE_M[0])], dtype=jnp.float64)),
            el.C(Pitch, jnp.array([truth_pitch(0.0)], dtype=jnp.float64)),
        ],
        name="lander_truth",
    )
    world.spawn(
        StaticSceneObject(
            el.WorldPos(linear=jnp.array([0.0, 0.0, TERRAIN_SEAT_Z], dtype=jnp.float64))
        ),
        name="surface",
    )

    @el.map
    def engine_response(
        throttle: Throttle, throttle_cmd: ThrottleCmd, prop: Propellant, landed: Landed
    ) -> tuple[Throttle, Thrust]:
        cmd = jnp.clip(throttle_cmd[0], THROTTLE_MIN, THROTTLE_MAX)
        actual = throttle[0] + (cmd - throttle[0]) * response_alpha
        active = jnp.logical_and(prop[0] > 0.0, landed[0] < 0.5)
        actual = jnp.where(active, actual, 0.0)
        thrust = actual * DPS_MAX_THRUST_N * thrust_scale
        return jnp.array([actual], dtype=jnp.float64), jnp.array([thrust], dtype=jnp.float64)

    @el.map
    def mass_props(
        thrust: Thrust,
        torque: RcsTorque,
        prop: Propellant,
        rcs_prop: RcsPropellant,
        landed: Landed,
    ) -> tuple[Propellant, RcsPropellant, el.Inertia]:
        dps_burn = thrust[0] / (isp * G0) * SIM_TIME_STEP
        rcs_force_equivalent = jnp.sum(jnp.abs(torque)) / RCS_MOMENT_ARM_M
        rcs_burn = rcs_force_equivalent / (RCS_ISP_S * G0) * SIM_TIME_STEP
        next_prop = jnp.maximum(prop[0] - dps_burn, 0.0)
        next_rcs_prop = jnp.maximum(rcs_prop[0] - rcs_burn, 0.0)
        mass = dry_mass + next_prop + next_rcs_prop
        inertia_scale = mass / total_mass
        inertia = jnp.where(landed[0] > 0.5, landed_inertia, base_inertia_diag * inertia_scale)
        return (
            jnp.array([next_prop], dtype=jnp.float64),
            jnp.array([next_rcs_prop], dtype=jnp.float64),
            el.SpatialInertia(mass, inertia),
        )

    @el.map
    def attitude_control(
        pos: el.WorldPos, vel: el.WorldVel, setpoint: AttitudeSetpoint, landed: Landed
    ) -> RcsTorque:
        q_err = pos.angular().inverse() * setpoint
        err = q_err.vector()
        sign = jnp.where(err[3] >= 0.0, 1.0, -1.0)
        body_rate = pos.angular().inverse() @ vel.angular()
        torque = sign * err[:3] * rcs_k - body_rate * rcs_d
        torque = jnp.clip(torque, -rcs_limit, rcs_limit)
        return jnp.where(landed[0] > 0.5, jnp.zeros(3, dtype=jnp.float64), torque)

    @el.map
    def lunar_gravity(force: el.Force, inertia: el.Inertia, vel: el.WorldVel) -> el.Force:
        # Flat-world stand-in for the orbital centrifugal relief: a vehicle
        # moving horizontally at v needs v^2/R less thrust to hold altitude.
        # At the braking-phase ~830 m/s this is ~0.40 m/s^2 (a quarter of
        # lunar gravity) and it is what made the real P63 fuel budget close.
        v_h_sq = jnp.sum(vel.linear()[:2] ** 2)
        g_eff = jnp.maximum(lunar_g - v_h_sq / R_MOON_M, 0.0)
        return force + el.SpatialForce(
            linear=jnp.array([0.0, 0.0, -1.0], dtype=jnp.float64) * g_eff * inertia.mass()
        )

    @el.map
    def apply_main_thrust(thrust: Thrust, force: el.Force, pos: el.WorldPos) -> el.Force:
        thrust_body = jnp.array([0.0, 0.0, thrust[0]], dtype=jnp.float64)
        return force + el.SpatialForce(linear=pos.angular() @ thrust_body)

    @el.map
    def apply_rcs_torque(torque: RcsTorque, force: el.Force, pos: el.WorldPos) -> el.Force:
        return force + el.SpatialForce(torque=pos.angular() @ torque)

    @el.map
    def ground_contact(
        pos: el.WorldPos,
        vel: el.WorldVel,
        landed: Landed,
        touchdown_speed: TouchdownSpeed,
        touchdown_horizontal_speed: TouchdownHorizontalSpeed,
    ) -> tuple[el.WorldPos, el.WorldVel, Landed, TouchdownSpeed, TouchdownHorizontalSpeed]:
        altitude = pos.linear()[2]
        vertical_speed = vel.linear()[2]
        contact = altitude <= 0.0
        was_landed = landed[0] > 0.5
        landed_now = jnp.logical_or(was_landed, contact)
        # Latch impact speeds on the contact tick, before velocity is zeroed.
        first_contact = jnp.logical_and(~was_landed, contact)
        impact_speed = jnp.where(first_contact, jnp.abs(vertical_speed), touchdown_speed[0])
        impact_horizontal = jnp.where(
            first_contact,
            jnp.linalg.norm(vel.linear()[:2]),
            touchdown_horizontal_speed[0],
        )
        linear_pos = jnp.where(landed_now, pos.linear().at[2].set(0.0), pos.linear())
        linear_vel = jnp.where(landed_now, jnp.zeros(3, dtype=jnp.float64), vel.linear())
        angular_vel = jnp.where(landed_now, jnp.zeros(3, dtype=jnp.float64), vel.angular())
        return (
            el.SpatialTransform(linear=linear_pos, angular=pos.angular()),
            el.SpatialMotion(linear=linear_vel, angular=angular_vel),
            jnp.array([jnp.where(landed_now, 1.0, 0.0)], dtype=jnp.float64),
            jnp.array([impact_speed], dtype=jnp.float64),
            jnp.array([impact_horizontal], dtype=jnp.float64),
        )

    @el.map
    def derive_telemetry(
        pos: el.WorldPos, vel: el.WorldVel
    ) -> tuple[Altitude, VerticalSpeed, HorizontalSpeed, Pitch]:
        body_up = pos.angular() @ jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
        pitch = jnp.rad2deg(jnp.arccos(jnp.clip(body_up[2], -1.0, 1.0)))
        return (
            jnp.array([pos.linear()[2]], dtype=jnp.float64),
            jnp.array([vel.linear()[2]], dtype=jnp.float64),
            jnp.array([jnp.linalg.norm(vel.linear()[:2])], dtype=jnp.float64),
            jnp.array([pitch], dtype=jnp.float64),
        )

    @el.map
    def thrust_visualization(
        thrust: Thrust, torque: RcsTorque
    ) -> tuple[MainThrustViz, RcsTorqueViz, RcsThrusterViz]:
        torque_norm = torque / RCS_AXIS_TORQUE_LIMIT_NM
        per_thruster = jnp.maximum(0.0, torque_norm[RCS_THRUSTER_AXIS] * RCS_THRUSTER_SIGN)
        return (
            jnp.array([0.0, 0.0, thrust[0] / DPS_MAX_THRUST_N], dtype=jnp.float64),
            torque_norm,
            per_thruster,
        )

    ref_time = jnp.asarray(REF_TIME_S)
    ref_altitude = jnp.asarray(REF_ALTITUDE_M)
    ref_rate = jnp.asarray(REF_RATE_MPS)
    ref_pitch = jnp.asarray(REF_PITCH_DEG)
    ref_slant = jnp.asarray(REF_SLANT_RANGE_M)
    ref_hspeed = jnp.asarray(REF_HSPEED_MPS)
    ref_downrange = jnp.asarray(REF_DOWNRANGE_M)

    @el.system
    def truth_playback(
        tick: el.Query[el.SimulationTick],
        truth: el.Query[TruthMarker],
    ) -> el.Query[el.WorldPos, Altitude, VerticalSpeed, HorizontalSpeed, SlantRange, Pitch]:
        """Replay the recorded Apollo 11 descent on the kinematic truth ghost."""

        t_s = tick[0] * SIM_TIME_STEP
        altitude = jnp.interp(t_s, ref_time, ref_altitude)
        rate = jnp.interp(t_s, ref_time, ref_rate)
        hspeed = jnp.interp(t_s, ref_time, ref_hspeed)
        slant = jnp.interp(t_s, ref_time, ref_slant)
        downrange = jnp.interp(t_s, ref_time, ref_downrange)
        pitch_deg = jnp.abs(jnp.interp(t_s, ref_time, ref_pitch))
        attitude = el.Quaternion.from_axis_angle(
            jnp.array([0.0, 1.0, 0.0]), -jnp.deg2rad(pitch_deg)
        )
        pose = el.SpatialTransform(
            angular=attitude,
            linear=jnp.array([downrange, 30.0, altitude]),
        )
        return truth.map(
            (el.WorldPos, Altitude, VerticalSpeed, HorizontalSpeed, SlantRange, Pitch),
            lambda _marker: (
                pose,
                jnp.array([altitude]),
                jnp.array([rate]),
                jnp.array([hspeed]),
                jnp.array([slant]),
                jnp.array([pitch_deg]),
            ),
        )

    world.schematic(SCHEMATIC_PATH.read_text(), SCHEMATIC_PATH.name)

    non_effectors = engine_response | attitude_control | mass_props | thrust_visualization
    effectors = lunar_gravity | apply_main_thrust | apply_rcs_torque
    return (
        world,
        truth_playback
        | non_effectors
        | el.six_dof(sys=effectors, integrator=el.Integrator.SemiImplicit)
        | ground_contact
        | derive_telemetry,
    )
