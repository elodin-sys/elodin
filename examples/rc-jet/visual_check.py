#!/usr/bin/env python3
"""Isolated RGBA/gray8 LWIR parity check for the RC-jet scene."""

import json
import os
import sys
import typing as ty
from dataclasses import field
from pathlib import Path

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "boson_ref"))

from compare_rgba import evaluate_metrics  # noqa: E402
from extract_frames import frame_metrics, write_grayscale_png  # noqa: E402
from frames import (  # noqa: E402
    enu_basis,
    geodetic_to_ecef,
    level_attitude_ecef,
    quaternion_xyzw_from_matrix,
)
from maneuver_report import write_montage  # noqa: E402

SITE_LAT_DEG = 35.350664
SITE_LON_DEG = -117.809027
SITE_ALT_M = 589.274
HEADING_DEG = 350.0
LOW_ALTITUDE_AGL_M = 300.0
HIGH_ALTITUDE_AGL_M = 2500.0
TARGET_RANGE_M = 350.0
WIDTH, HEIGHT = 640, 512
DT = 1.0 / 120.0
TOTAL_S = 20.0
INITIAL_SKY_END_S = 4.0
SKY_START_S = 8.0
SKY_END_S = 11.0
HIGH_ALTITUDE_START_S = 15.0
FAR_M = float(os.environ.get("ELODIN_LWIR_VISUAL_FAR", "100000"))
OUT_DIR = Path(os.environ.get("ELODIN_LWIR_VISUAL_OUT", "/tmp/rc-jet-lwir-visual-check"))

CAPTURES = [
    ("ground", 6.0),
    ("sky", 9.5),
    ("ground_return", 13.0),
    ("high_altitude", 17.0),
]
TARGET_PHASES = {"ground", "ground_return", "high_altitude"}
PARITY_MAX_DN = 1
TARGET_CONTRAST_MIN_DN = 15.0
DYNAMIC_RANGE_MIN_DN = 50
TARGET_CHIP_HALF_PX = 16
TARGET_BACKGROUND_HALF_PX = 48

RigMarker = ty.Annotated[
    jax.Array,
    el.Component("visual_check_rig", el.ComponentType(el.PrimitiveType.F64, (1,))),
]
TargetMarker = ty.Annotated[
    jax.Array,
    el.Component("visual_check_target", el.ComponentType(el.PrimitiveType.F64, (1,))),
]


@el.dataclass
class VisualRig(el.Archetype):
    marker: RigMarker = field(default_factory=lambda: jnp.ones(1))
    world_pos: el.WorldPos = field(default_factory=el.SpatialTransform)


@el.dataclass
class VisualTarget(el.Archetype):
    marker: TargetMarker = field(default_factory=lambda: jnp.ones(1))
    world_pos: el.WorldPos = field(default_factory=el.SpatialTransform)


def quat_mul_xyzw(a: jax.Array, b: jax.Array) -> jax.Array:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
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


site_ecef = jnp.asarray(
    geodetic_to_ecef(jnp.deg2rad(SITE_LAT_DEG), jnp.deg2rad(SITE_LON_DEG), SITE_ALT_M)
)
basis = jnp.asarray(enu_basis(jnp.deg2rad(SITE_LAT_DEG), jnp.deg2rad(SITE_LON_DEG)))
up = basis[2]
heading = jnp.deg2rad(HEADING_DEG)
forward = jnp.sin(heading) * basis[0] + jnp.cos(heading) * basis[1]
base_attitude = jnp.asarray(
    quaternion_xyzw_from_matrix(
        level_attitude_ecef(
            np.deg2rad(SITE_LAT_DEG),
            np.deg2rad(SITE_LON_DEG),
            HEADING_DEG,
        )
    )
)


def altitude_at(t: jax.Array) -> jax.Array:
    return jnp.where(
        t >= HIGH_ALTITUDE_START_S,
        HIGH_ALTITUDE_AGL_M,
        LOW_ALTITUDE_AGL_M,
    )


@el.system
def drive_rig(
    tick: el.Query[el.SimulationTick],
    rigs: el.Query[RigMarker],
) -> el.Query[el.WorldPos]:
    t = tick[0] * DT
    sky = (t < INITIAL_SKY_END_S) | ((t >= SKY_START_S) & (t < SKY_END_S))
    pitch = jnp.where(sky, jnp.pi / 2.0, 0.0)
    attitude = quat_mul_xyzw(
        base_attitude,
        quat_axis_angle_xyzw(jnp.array([0.0, -1.0, 0.0]), pitch),
    )
    position = site_ecef + up * altitude_at(t)
    pose = el.SpatialTransform(
        angular=el.Quaternion(attitude),
        linear=position,
    )
    return rigs.map(el.WorldPos, lambda _marker: pose)


@el.system
def drive_target(
    tick: el.Query[el.SimulationTick],
    targets: el.Query[TargetMarker],
) -> el.Query[el.WorldPos]:
    t = tick[0] * DT
    position = site_ecef + up * altitude_at(t) + forward * TARGET_RANGE_M
    pose = el.SpatialTransform(
        angular=el.Quaternion(base_attitude),
        linear=position,
    )
    return targets.map(el.WorldPos, lambda _marker: pose)


initial_rig = el.SpatialTransform(
    angular=el.Quaternion(base_attitude),
    linear=site_ecef + up * LOW_ALTITUDE_AGL_M,
)
initial_target = el.SpatialTransform(
    angular=el.Quaternion(base_attitude),
    linear=site_ecef + up * LOW_ALTITUDE_AGL_M + forward * TARGET_RANGE_M,
)

world = el.World()
rig = world.spawn(VisualRig(world_pos=initial_rig), name="cam")
target = world.spawn(VisualTarget(world_pos=initial_target), name="target")
world.thermal_tag(target, temperature_c=18.0, emissivity=0.92)

camera_args = {
    "entity": rig,
    "camera_model": "boson640p",
    "lens_hfov": 18.0,
    "effect": "lwir",
    "effect_params": {
        "agc": {"low": 0.0, "smoothing": 0.0},
        "temporal_noise_sigma_dn": 0.0,
        "column_fpn_sigma_dn": 0.0,
        "dead_pixel_ppm": 0.0,
    },
    "near": 0.1,
    "far": FAR_M,
    "pos_offset": [0.0, 0.0, 0.0],
    "rot_offset": [0.0, 0.0, 0.0],
    "create_frustum": False,
}
world.sensor_camera(name="ir_rgba", format="rgba", **camera_args)
world.sensor_camera(name="ir_gray8", format="gray8", **camera_args)

kdl_path = Path(__file__).with_name("visual_check.kdl")
world.schematic(kdl_path.read_text(), kdl_path.name)

OUT_DIR.mkdir(parents=True, exist_ok=True)
pending = list(CAPTURES)
captures: list[dict] = []
finished = [False]


def frames_at(ctx, timestamp: int):
    rgba_latest = ctx.read_msg_latest("cam.ir_rgba")
    gray8_latest = ctx.read_msg_latest("cam.ir_gray8")
    if (
        rgba_latest is None
        or gray8_latest is None
        or rgba_latest[0] < timestamp
        or gray8_latest[0] < timestamp
    ):
        return None
    rgba = ctx.read_msg("cam.ir_rgba", timestamp)
    gray8 = ctx.read_msg("cam.ir_gray8", timestamp)
    if rgba is None or gray8 is None:
        return None
    rgba = np.asarray(rgba, dtype=np.uint8)
    gray8 = np.asarray(gray8, dtype=np.uint8)
    if rgba.size != WIDTH * HEIGHT * 4 or gray8.size != WIDTH * HEIGHT:
        return None
    rgba_r = np.ascontiguousarray(rgba.reshape(-1, 4)[:, 0])
    return timestamp, rgba_r, gray8


def target_contrast(gray: np.ndarray) -> float:
    image = gray.reshape(HEIGHT, WIDTH).astype(np.float32)
    cx, cy = WIDTH // 2, HEIGHT // 2
    chip = image[
        cy - TARGET_CHIP_HALF_PX : cy + TARGET_CHIP_HALF_PX,
        cx - TARGET_CHIP_HALF_PX : cx + TARGET_CHIP_HALF_PX,
    ]
    surround = image[
        cy - TARGET_BACKGROUND_HALF_PX : cy + TARGET_BACKGROUND_HALF_PX,
        cx - TARGET_BACKGROUND_HALF_PX : cx + TARGET_BACKGROUND_HALF_PX,
    ].copy()
    surround[
        TARGET_BACKGROUND_HALF_PX - TARGET_CHIP_HALF_PX : TARGET_BACKGROUND_HALF_PX
        + TARGET_CHIP_HALF_PX,
        TARGET_BACKGROUND_HALF_PX - TARGET_CHIP_HALF_PX : TARGET_BACKGROUND_HALF_PX
        + TARGET_CHIP_HALF_PX,
    ] = np.nan
    return float(abs(np.nanmean(surround) - np.percentile(chip, 5.0)))


def finish() -> list[str]:
    failures: list[str] = []
    report = {"far_m": FAR_M, "captures": []}
    montage = []
    for capture in captures:
        rgba_r = capture["rgba_r"]
        gray8 = capture["gray8"]
        delta = np.abs(gray8.astype(np.int16) - rgba_r.astype(np.int16))
        parity = {
            "max_abs_dn": int(delta.max()),
            "mean_abs_dn": round(float(delta.mean()), 6),
        }
        formats = {}
        for name, frame in (("rgba_r", rgba_r), ("gray8", gray8)):
            metrics = frame_metrics(frame.tobytes(), WIDTH, HEIGHT)
            formats[name] = {
                "target_contrast_dn": round(target_contrast(frame), 3),
                "metrics": metrics,
                "boson_reference_failures": evaluate_metrics(metrics),
            }
        capture_report = {
            "label": capture["label"],
            "timestamp_us": capture["timestamp"],
            "parity": parity,
            "formats": formats,
        }
        report["captures"].append(capture_report)
        montage.extend(
            [
                {"label": f"{capture['label']}_rgba_r", "gray": rgba_r},
                {"label": f"{capture['label']}_gray8", "gray": gray8},
            ]
        )
        if parity["max_abs_dn"] > PARITY_MAX_DN:
            failures.append(
                f"{capture['label']}: gray8/RGBA parity max "
                f"{parity['max_abs_dn']} DN > {PARITY_MAX_DN}"
            )
        if capture["label"] in TARGET_PHASES:
            for name, result in formats.items():
                metrics = result["metrics"]
                dynamic_range = metrics["max_dn"] - metrics["min_dn"]
                if dynamic_range < DYNAMIC_RANGE_MIN_DN:
                    failures.append(
                        f"{capture['label']} {name}: dynamic range "
                        f"{dynamic_range} DN < {DYNAMIC_RANGE_MIN_DN}"
                    )
                contrast = result["target_contrast_dn"]
                if contrast < TARGET_CONTRAST_MIN_DN:
                    failures.append(
                        f"{capture['label']} {name}: target contrast "
                        f"{contrast:.1f} DN < {TARGET_CONTRAST_MIN_DN:.1f}"
                    )
    report["failures"] = failures
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    write_montage(montage, WIDTH, HEIGHT, OUT_DIR / "montage.png")
    for failure in failures:
        print(f"[VISUAL] FAIL: {failure}")
    if not failures:
        print("[VISUAL] PASS")
    return failures


def post_step(tick: int, ctx) -> None:
    t = tick * DT
    if pending and t >= pending[0][1]:
        latest = frames_at(ctx, round(pending[0][1] * 1_000_000))
        if latest is not None:
            timestamp, rgba_r, gray8 = latest
            label, _ = pending.pop(0)
            write_grayscale_png(
                OUT_DIR / f"{label}_rgba_r.png",
                rgba_r.tobytes(),
                WIDTH,
                HEIGHT,
            )
            write_grayscale_png(
                OUT_DIR / f"{label}_gray8.png",
                gray8.tobytes(),
                WIDTH,
                HEIGHT,
            )
            captures.append(
                {
                    "label": label,
                    "timestamp": timestamp,
                    "rgba_r": rgba_r,
                    "gray8": gray8,
                }
            )
            print(f"[VISUAL] captured {label} at {timestamp / 1e6:.3f}s (far={FAR_M:.0f}m)")
    if tick >= int(TOTAL_S / DT) - 2 and not finished[0]:
        finished[0] = True
        if pending:
            raise RuntimeError(f"missing captures: {[label for label, _ in pending]}")
        failures = finish()
        if failures:
            raise RuntimeError(f"visual check failed with {len(failures)} issue(s)")


print(f"[VISUAL] output={OUT_DIR} far={FAR_M:.0f}m")
world.run(
    drive_rig | drive_target,
    simulation_rate=1.0 / DT,
    generate_real_time=True,
    max_ticks=int(TOTAL_S / DT),
    post_step=post_step,
    interactive=False,
    db_path=os.environ.get("ELODIN_DB_PATH", str(OUT_DIR / "db")),
    start_timestamp=0,
)
print(f"[VISUAL] captured {len(captures)}/{len(CAPTURES)} phases")
