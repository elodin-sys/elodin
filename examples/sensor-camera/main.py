#!/usr/bin/env python3
"""
Sensor Camera Test — Bouncing Balls Room

Multiple balls bounce around inside a walled room under gravity.
Two special balls carry sensor cameras:
  - Cyan ball:    RGB camera     (60 fps)
  - Magenta ball: Thermal camera (30 fps)

The cameras render continuously at their configured FPS; frames are pushed
to the DB by the headless render server. The simulation only *reads* frames
via ``ctx.read_msg(name, timestamp=...)`` — there is no blocking render
call. Pick the apparent camera latency at read time by passing a timestamp
offset.

Usage:
    elodin editor examples/sensor-camera/main.py
    elodin run examples/sensor-camera/main.py
"""

import elodin as el
import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import time

SIM_TIME_STEP = 1.0 / 120.0
MAX_TICKS = int(os.getenv("ELODIN_SENSOR_CAMERA_MAX_TICKS", "18000"))
EMIT_PERF = "run" in sys.argv
BALL_RADIUS = 0.3
BOUNDARY = 5.0
BOUNCINESS = 0.95
FRICTION = 0.05

SCENE_FPS = 60.0
THERMAL_FPS = 30.0
# Simulated camera latency applied at read time (matches a real FPV camera).
SCENE_LATENCY_US = 16_667  # ~16.7 ms (one frame at 60 fps)
THERMAL_LATENCY_US = 33_333  # ~33.3 ms (one frame at 30 fps)

DB_PATH = os.environ.get("ELODIN_SENSOR_CAMERA_DB")

# ── Systems ──────────────────────────────────────────────────────────────────


@el.map
def gravity(f: el.Force, inertia: el.Inertia) -> el.Force:
    return f + el.SpatialForce(linear=inertia.mass() * jnp.array([0.0, 0.0, -9.81]))


@el.map
def ground_bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    pos = p.linear()
    vel = v.linear()
    new_vz = jax.lax.cond(
        jnp.logical_and(pos[2] < BALL_RADIUS, vel[2] < 0.0),
        lambda _: -vel[2] * BOUNCINESS,
        lambda _: vel[2],
        operand=None,
    )
    return el.SpatialMotion(linear=jnp.array([vel[0], vel[1], new_vz]))


@el.map
def wall_bounce(p: el.WorldPos, v: el.WorldVel) -> el.WorldVel:
    pos = p.linear()
    vel = v.linear()

    vel_x = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(pos[0] > BOUNDARY, vel[0] > 0),
            jnp.logical_and(pos[0] < -BOUNDARY, vel[0] < 0),
        ),
        lambda _: -vel[0] * BOUNCINESS,
        lambda _: vel[0],
        operand=None,
    )
    vel_y = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_and(pos[1] > BOUNDARY, vel[1] > 0),
            jnp.logical_and(pos[1] < -BOUNDARY, vel[1] < 0),
        ),
        lambda _: -vel[1] * BOUNCINESS,
        lambda _: vel[1],
        operand=None,
    )

    return el.SpatialMotion(linear=jnp.array([vel_x, vel_y, vel[2]]))


@el.map
def damping(v: el.WorldVel, f: el.Force) -> el.Force:
    return el.SpatialForce(linear=f.force() - FRICTION * v.linear())


# ── World ────────────────────────────────────────────────────────────────────

world = el.World()

BALL_DEFS = [
    # (name, position, velocity)
    ("cam_ball_a", [-3.0, -3.0, 4.0], [2.0, 3.0, 1.0]),
    ("cam_ball_b", [3.0, 3.0, 5.0], [-1.5, -2.0, 0.5]),
    ("ball_1", [0.0, 0.0, 6.0], [1.0, -1.0, 0.0]),
    ("ball_2", [-2.0, 2.0, 3.0], [3.0, 1.0, 2.0]),
    ("ball_3", [1.0, -3.0, 7.0], [-2.0, 2.0, -1.0]),
]

entities = {}
for name, pos, vel in BALL_DEFS:
    e = world.spawn(
        el.Body(
            world_pos=el.SpatialTransform(linear=jnp.array(pos)),
            world_vel=el.SpatialMotion(linear=jnp.array(vel)),
            inertia=el.SpatialInertia(mass=1.0),
        ),
        name=name,
    )
    entities[name] = e

cam_ball_a = entities["cam_ball_a"]
cam_ball_b = entities["cam_ball_b"]

# ── Sensor Cameras ───────────────────────────────────────────────────────────

world.sensor_camera(
    entity=cam_ball_a,
    name="scene_cam",
    width=640,
    height=480,
    fov=90.0,
    near=0.02,
    far=0.65,
    pos_offset=[0.0, 0.0, 0.5],
    rot_offset=[0.0, 0.0, 45.0],
    format="rgba",
    fps=SCENE_FPS,
    create_frustum=True,
    frustums_color=[0.0, 1.0, 1.0, 1.0],
    projection_color=[0.0, 1.0, 1.0, 1.0],
)

world.sensor_camera(
    entity=cam_ball_b,
    name="thermal_cam",
    width=128,
    height=128,
    fov=90.0,
    near=0.02,
    far=0.65,
    pos_offset=[0.0, 0.0, 0.5],
    rot_offset=[0.0, 0.0, -135.0],
    format="rgba",
    effect="thermal",
    effect_params={"contrast": 1.5, "noise_sigma": 0.02},
    fps=THERMAL_FPS,
    create_frustum=True,
    frustums_color=[1.0, 0.0, 1.0, 1.0],
    projection_color=[1.0, 0.0, 1.0, 1.0],
)

# ── Schematic ────────────────────────────────────────────────────────────────

BALL_COLORS = {
    "cam_ball_a": "0 220 220",  # cyan
    "cam_ball_b": "220 0 220",  # magenta
    "ball_1": "255 140 0",  # orange
    "ball_2": "255 255 100",  # yellow
    "ball_3": "100 255 100",  # green
}

object_3d_lines = []
for name, color in BALL_COLORS.items():
    object_3d_lines.append(
        f"    object_3d {name}.world_pos {{ sphere radius={BALL_RADIUS} {{ color {color} }} }}"
    )

schematic = """
    timeline follow_latest=#true
    hsplit {{
        viewport name=Main pos="(0,0,0,0, 14,14,10)" look_at="(0,0,0,0, 0,0,1)" show_grid=#true show_frustums=#true
        vsplit {{
            sensor_view "cam_ball_a.scene_cam" name="RGB Camera (Cyan Ball)"
            sensor_view "cam_ball_b.thermal_cam" name="Thermal (Magenta Ball)"
        }}
    }}
{objects}
    object_3d "(0,0,0,1, 0,0,0)" {{
        plane width=12 depth=12 {{ color 60 120 60 }}
    }}
""".format(objects="\n".join(object_3d_lines))

world.schematic(schematic, "sensor-camera.kdl")

# ── System composition ───────────────────────────────────────────────────────

constraints = ground_bounce | wall_bounce
effectors = gravity | damping
system = constraints | el.six_dof(sys=effectors)

# ── Post-step: observe rendered frames via read_msg ──────────────────────────

scene_observations = [0]
thermal_observations = [0]
scene_first_logged = [False]
thermal_first_logged = [False]


def post_step(tick, ctx):
    if tick == 0:
        return

    # Sample the latest frame for each camera once per second of sim time.
    if tick % 120 == 0:
        scene_now = ctx.read_msg("cam_ball_a.scene_cam")
        thermal_now = ctx.read_msg("cam_ball_b.thermal_cam")
        if scene_now is not None:
            scene_observations[0] += 1
            if not scene_first_logged[0]:
                arr = np.asarray(scene_now)
                print(
                    f"[scene_cam] first frame seen at tick {tick}: "
                    f"{len(scene_now)} bytes, nonzero={np.count_nonzero(arr)}"
                )
                scene_first_logged[0] = True
        if thermal_now is not None:
            thermal_observations[0] += 1
            if not thermal_first_logged[0]:
                arr = np.asarray(thermal_now)
                print(
                    f"[thermal_cam] first frame seen at tick {tick}: "
                    f"{len(thermal_now)} bytes, nonzero={np.count_nonzero(arr)}"
                )
                thermal_first_logged[0] = True

    # Periodic heartbeat (so a long run shows progress).
    if tick > 0 and tick % 1200 == 0:
        print(
            f"  tick {tick}: scene_obs={scene_observations[0]} "
            f"thermal_obs={thermal_observations[0]} "
            f"sim_ts={ctx.timestamp}"
        )

    # Final-tick verification: walk the message log, count frames, compute
    # observed FPS, and demonstrate historical reads.
    if tick == MAX_TICKS - 1:
        _verify(ctx)


def _verify(ctx) -> None:
    print()
    print("─── Verification ────────────────────────────────────────────────")
    sim_seconds = MAX_TICKS * SIM_TIME_STEP
    print(f"sim_duration = {sim_seconds:.3f} s, final_sim_ts = {ctx.timestamp}")

    # Skip the renderer's warm-up window (first 2 s of sim time) when
    # estimating FPS — the GPU pipeline is still primed during this phase.
    warmup_us = 2_000_000
    sweep_end = ctx.timestamp - 100_000  # leave a small margin below latest
    sweep_start = max(warmup_us, sweep_end - int((sim_seconds - 2.0) * 1_000_000))
    sweep_window_us = max(sweep_end - sweep_start, 1)
    sweep_seconds = sweep_window_us / 1_000_000.0
    print(f"fps-sweep window = [{sweep_start}, {sweep_end}] ({sweep_seconds:.2f} s)")

    all_ok = True
    for cam_name, target_fps, latency_us in [
        ("cam_ball_a.scene_cam", SCENE_FPS, SCENE_LATENCY_US),
        ("cam_ball_b.thermal_cam", THERMAL_FPS, THERMAL_LATENCY_US),
    ]:
        latest = ctx.read_msg(cam_name)
        if latest is None:
            print(f"  [{cam_name}] FAIL: no frames in DB")
            all_ok = False
            continue

        # Estimate observed FPS by walking through the message log at twice
        # the camera's expected frame interval. Fingerprint each frame by
        # downsampling across the whole buffer (single-pixel slabs miss
        # variation if the chosen pixel is uniform sky or floor).
        step_us = max(int(1_000_000.0 / target_fps / 2.0), 100)
        seen_sig = None
        unique_frames = 0
        cursor = sweep_start
        while cursor <= sweep_end:
            frame = ctx.read_msg(cam_name, timestamp=cursor)
            if frame is not None:
                arr = np.asarray(frame)
                stride = max(len(arr) // 256, 1)
                sig = hash(arr[::stride].tobytes())
                if sig != seen_sig:
                    unique_frames += 1
                    seen_sig = sig
            cursor += step_us

        observed_fps = unique_frames / sweep_seconds if sweep_seconds > 0 else 0.0
        ratio = observed_fps / target_fps if target_fps > 0 else 0.0
        # The renderer can fall short of the configured fps when the GPU
        # can't keep up — that's a soft failure, not an error. Anything
        # under ~30% of target indicates a real bug.
        ok = 0.30 <= ratio <= 1.25
        marker = "OK" if ok else "FAIL"
        print(
            f"  [{cam_name}] {marker}: observed_fps≈{observed_fps:.2f} "
            f"target={target_fps:.0f} (ratio={ratio:.2f}) "
            f"unique_frames_in_window={unique_frames}"
        )
        if not ok:
            all_ok = False

        # Demonstrate the historical-read API: latest frame vs frame
        # `latency_us` µs in the past. With a 1-frame offset and balls in
        # motion these should differ; if not, sample further back.
        def _fingerprint(arr: np.ndarray) -> int:
            stride = max(len(arr) // 256, 1)
            return hash(arr[::stride].tobytes())

        latest_sig = _fingerprint(np.asarray(latest))
        for offset in (latency_us, latency_us * 4, latency_us * 16):
            past = ctx.read_msg(cam_name, timestamp=ctx.timestamp - offset)
            if past is None:
                continue
            past_sig = _fingerprint(np.asarray(past))
            if past_sig != latest_sig:
                print(
                    f"  [{cam_name}] OK: historical read at -{offset}us "
                    f"returned a different frame than latest"
                )
                break
        else:
            print(
                f"  [{cam_name}] WARN: historical reads up to "
                f"-{latency_us * 16}us all matched the latest frame"
            )

    print("─────────────────────────────────────────────────────────────────")
    print(f"verification: {'PASS' if all_ok else 'FAIL — see above'}")


# ── Run ──────────────────────────────────────────────────────────────────────

wall_start = time.perf_counter() if EMIT_PERF else None
world.run(
    system,
    simulation_rate=1.0 / SIM_TIME_STEP,
    generate_real_time=True,
    max_ticks=MAX_TICKS,
    post_step=post_step,
    interactive=False,
    db_path=DB_PATH,
)

print(
    f"\nrun complete: scene_observations={scene_observations[0]} "
    f"thermal_observations={thermal_observations[0]}"
)
if EMIT_PERF and wall_start is not None:
    wall_elapsed_s = time.perf_counter() - wall_start
    sim_elapsed_s = MAX_TICKS * SIM_TIME_STEP
    rtf = sim_elapsed_s / wall_elapsed_s if wall_elapsed_s > 0 else 0.0
    print(
        "PERF sensor_camera "
        f"max_ticks={MAX_TICKS} "
        f"elapsed_s={wall_elapsed_s:.3f} "
        f"sim_s={sim_elapsed_s:.3f} "
        f"rtf={rtf:.3f}"
    )
