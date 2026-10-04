#!/usr/bin/env python3
"""Scripted LWIR maneuver harness: mountains, sky-only, loop, ground return.

Reproduces the sensor-output failure modes that trim flight never hits:
the sky white band and the post-sky AGC latch. Run with:

    ./target/release/elodin run scripts/boson_ref/lwir_maneuver_sim.py

Outputs per-phase grayscale captures, a montage, metrics JSON, and prints a
"[MANEUVER] PASS" / "[MANEUVER] FAIL" verdict (see run_maneuver_check.sh).
"""

import json
import os
import sys
import typing as ty
from dataclasses import field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "examples" / "rc-jet"))

import elodin as el  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from extract_frames import write_grayscale_png  # noqa: E402
from frames import (  # noqa: E402
    geodetic_to_ecef,
    level_attitude_ecef,
    quaternion_xyzw_from_matrix,
)
from maneuver_report import evaluate_captures, write_montage  # noqa: E402

# Mojave RC field site (matches examples/rc-jet/bdx.kdl `coordinate`).
SITE_LAT_DEG = 35.350664
SITE_LON_DEG = -117.809027
SITE_ALT_M = 589.274
HEADING_DEG = 350.0
ALTITUDE_AGL_M = 300.0

WIDTH, HEIGHT = 640, 512
DT = 1.0 / 120.0
TOTAL_S = 24.0
MAX_TICKS = int(TOTAL_S / DT)

# Pitch profile: hold level, pitch to sky (90 deg), hold sky, then pull
# through the rest of the loop (another 270 deg: inverted, straight down,
# and back to level at 360 deg total), hold level.
PITCH_RAMP_START_S = 6.0
PITCH_RAMP_S = 3.0
LOOP_START_S = 12.0
LOOP_S = 6.0
PITCH_RAMP_DEG = 90.0
LOOP_DEG = 270.0

# (label, capture sim-time seconds). Retries extend past each time on missed
# frames but never across a phase boundary.
CAPTURES = [
    ("ground_initial_a", 4.5),
    ("ground_initial_b", 5.5),
    ("climb", 7.5),
    ("sky_hold_a", 10.0),
    ("sky_hold_b", 11.5),
    ("loop_inverted", 14.0),
    ("loop_down", 16.0),
    ("ground_return_a", 19.5),
    ("ground_return_b", 21.0),
    ("ground_return_c", 23.0),
]
CAPTURE_RETRY_S = 0.5
CAMERA_MSG = "cam.ir_cam"

OUT_DIR = Path(os.environ.get("ELODIN_LWIR_MANEUVER_OUT", "/tmp/lwir-maneuver"))

POS_ECEF = jnp.asarray(
    geodetic_to_ecef(
        jnp.deg2rad(SITE_LAT_DEG), jnp.deg2rad(SITE_LON_DEG), SITE_ALT_M + ALTITUDE_AGL_M
    )
)
Q_BASE = jnp.asarray(
    quaternion_xyzw_from_matrix(
        level_attitude_ecef(np.deg2rad(SITE_LAT_DEG), np.deg2rad(SITE_LON_DEG), HEADING_DEG)
    )
)

ManeuverT = ty.Annotated[jax.Array, el.Component("maneuver_t", el.ComponentType.F64)]


@el.dataclass
class ManeuverRig(el.Archetype):
    maneuver_t: ManeuverT = field(default_factory=lambda: jnp.float64(0.0))
    world_pos: el.WorldPos = field(default_factory=el.SpatialTransform)


def pitch_profile_rad(t: jax.Array) -> jax.Array:
    ramp = jnp.clip((t - PITCH_RAMP_START_S) / PITCH_RAMP_S, 0.0, 1.0)
    loop = jnp.clip((t - LOOP_START_S) / LOOP_S, 0.0, 1.0)
    return jnp.deg2rad(PITCH_RAMP_DEG) * ramp + jnp.deg2rad(LOOP_DEG) * loop


def quat_mul_xyzw(a: jax.Array, b: jax.Array) -> jax.Array:
    ax, ay, az, aw = a[0], a[1], a[2], a[3]
    bx, by, bz, bw = b[0], b[1], b[2], b[3]
    return jnp.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_axis_angle_xyzw(axis: jax.Array, angle: jax.Array) -> jax.Array:
    half = angle * 0.5
    return jnp.concatenate([axis * jnp.sin(half), jnp.cos(half)[None]])


@el.map
def advance_clock(t: ManeuverT) -> ManeuverT:
    return t + DT


@el.map
def drive_pose(t: ManeuverT) -> el.WorldPos:
    # Pitch about body -Y (left axis) rotates body +X (camera forward) toward
    # body +Z (camera up): a nose-up pull-through loop.
    pitch = quat_axis_angle_xyzw(jnp.array([0.0, -1.0, 0.0]), pitch_profile_rad(t))
    attitude = quat_mul_xyzw(Q_BASE, pitch)
    return el.SpatialTransform(angular=el.Quaternion(attitude), linear=POS_ECEF)


world = el.World()
rig = world.spawn(ManeuverRig(), name="cam")
world.sensor_camera(
    entity=rig,
    name="ir_cam",
    camera_model="boson640p",
    lens_hfov=18.0,
    effect="lwir",
    near=0.1,
    far=100_000.0,
    pos_offset=[0.0, 0.0, 0.0],
    rot_offset=[0.0, 0.0, 0.0],
    create_frustum=False,
)

# Terrain + ECEF anchor only; panels are irrelevant to the render-server.
world.schematic(
    f"""
coordinate frame="ECEF" lat={SITE_LAT_DEG} lon={SITE_LON_DEG} alt={SITE_ALT_M}
world_mesh "mojave_rc_field" frame="ENU" translate="(0.000, 0.000, 528.000)"
""",
    "lwir-maneuver.kdl",
)

terrain_atlas = REPO_ROOT / "assets" / "terrains" / "planar" / "mojave_rc_field"
if not terrain_atlas.exists():
    print(
        "[MANEUVER] WARNING: terrain atlas missing at "
        f"{terrain_atlas}; frames will show the fallback grid and metrics "
        "will not be meaningful"
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
captures: list[dict] = []
pending = [
    {"label": label, "t": t, "deadline": t + CAPTURE_RETRY_S, "next_try": t}
    for label, t in CAPTURES
]
evaluated = [False]


def capture_frame(ctx, entry: dict, t: float) -> bool:
    frame = ctx.read_msg(CAMERA_MSG)
    if frame is None:
        return False
    rgba = np.asarray(frame, dtype=np.uint8)
    if rgba.size != WIDTH * HEIGHT * 4:
        print(f"[MANEUVER] WARNING: unexpected frame size {rgba.size} for {entry['label']}")
        return False
    gray = np.ascontiguousarray(rgba.reshape(-1, 4)[:, 0])
    path = OUT_DIR / f"{entry['label']}.png"
    write_grayscale_png(path, gray.tobytes(), WIDTH, HEIGHT)
    captures.append({"label": entry["label"], "t": t, "gray": gray, "path": str(path)})
    print(f"[MANEUVER] capture {entry['label']} t={t:.2f}s mean={float(gray.mean()):.1f}")
    return True


def finish() -> None:
    report, failures = evaluate_captures(captures, WIDTH, HEIGHT)
    montage_path = OUT_DIR / "montage.png"
    write_montage(captures, WIDTH, HEIGHT, montage_path)
    report["montage"] = str(montage_path)
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[MANEUVER] report {report_path}")
    print(f"[MANEUVER] montage {montage_path}")
    for failure in failures:
        print(f"[MANEUVER] FAIL: {failure}")
    if not failures:
        print("[MANEUVER] PASS")


def post_step(tick: int, ctx) -> None:
    t = tick * DT
    for entry in list(pending):
        if t + 1e-9 < entry["next_try"]:
            continue
        if capture_frame(ctx, entry, t):
            pending.remove(entry)
        elif t >= entry["deadline"]:
            print(f"[MANEUVER] FAIL: no frame captured for {entry['label']}")
            pending.remove(entry)
        else:
            entry["next_try"] = t + 0.1
    if tick >= MAX_TICKS - 2 and not evaluated[0]:
        evaluated[0] = True
        finish()


world.run(
    advance_clock | drive_pose,
    simulation_rate=1.0 / DT,
    generate_real_time=True,
    max_ticks=MAX_TICKS,
    post_step=post_step,
    interactive=False,
    db_path=os.environ.get("ELODIN_DB_PATH"),
)
