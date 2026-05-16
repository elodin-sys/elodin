# Sensor Camera Example

This example demonstrates Elodin's **sensor camera** feature: synthetic camera imagery generated as first-class sensor data within a physics simulation. Virtual cameras are attached to simulation entities, rendered continuously by a headless GPU render-server at a configured frame rate, and made available to flight software as raw RGBA pixel arrays in the DB.

## Overview

Five colored balls bounce around a walled room under gravity. Two of them carry sensor cameras:

- **Cyan ball** — RGB camera (640×480) at **60 fps**
- **Magenta ball** — Thermal camera (128×128, iron-bow colormap) at **30 fps**

The simulation never blocks on rendering. The render server runs as a sibling process (managed by `s10`), subscribes to the DB for live world state, and emits one frame per camera every `1 / fps` µs of sim time. The simulation reads frames with `ctx.read_msg(name, timestamp=...)`.

The editor displays a 3D viewport alongside live sensor camera feeds using `sensor_view` panels.

```
┌──────────────────────┬──────────────────────┐
│                      │  RGB Camera          │
│   3D Viewport        │  (Cyan Ball)         │
│                      ├──────────────────────┤
│                      │  Thermal Camera      │
│                      │  (Magenta Ball)      │
└──────────────────────┴──────────────────────┘
```

## Quick Start

From the repository root:

```bash
nix develop
uv venv --python 3.12 && source .venv/bin/activate
uvx maturin develop --uv --manifest-path=libs/nox-py/Cargo.toml
just install

elodin editor examples/sensor-camera/main.py
```

For a headless run with an inspectable DB:

```bash
ELODIN_SENSOR_CAMERA_DB=dbs/sensor_camera_verify \
ELODIN_SENSOR_CAMERA_MAX_TICKS=2400 \
elodin run examples/sensor-camera/main.py
```

The example's final-tick verification block prints per-camera frame counts, observed FPS, and a historical-read sanity check.

## How It Works

### Architecture

Sensor camera rendering runs in a **separate headless process** managed by s10. The simulation, render-server, and editor are three distinct processes, communicating exclusively through Elodin DB:

```
Simulation (Python + nox-py)
  │
  │  write components, bump db.last_updated
  │
  └──── TCP ────────▶  Elodin DB
                         ▲   │
                         │   │  SubscribeLastUpdated (auto)
                         │   │  + TelemetryCache stream
                         │   ▼
                       Render-Server (headless Bevy)
                         • Receives LastUpdated(t)
                         • For each camera with elapsed
                           interval: renders at t, pushes
                           MsgWithTimestamp(t, camera_name, rgba)
                           back to DB

Simulation reads frames from the DB:
  ctx.read_msg("drone.scene_cam")                       # latest
  ctx.read_msg("drone.scene_cam", timestamp=t - 33_000)  # 33 ms ago
```

There is no UDS, no blocking call from Python into the renderer, and no "render this now" request — frame timing is purely configuration on the camera.

### Python API

#### Registering a camera

```python
world.sensor_camera(
    entity=drone,              # Entity the camera is attached to
    name="scene_cam",          # Becomes "drone.scene_cam" in the DB
    width=640, height=480,     # Resolution
    fov=90.0,                  # Field of view (degrees)
    fps=60.0,                  # Frames per second of sim time (default 30)
    pos_offset=[0, 0, 0.5],    # Body-frame offset from entity origin
    look_at_offset=[6, 6, 0],  # Body-frame look-at direction
    format="rgba",             # Pixel format
    effect="normal",           # "normal", "thermal", "night_vision", "depth"
    effect_params={},          # Effect-specific parameters
    create_frustum=True,       # Show this camera's frustum in 3D viewports
    show_ellipsoids=False,     # Hide ellipsoid debug objects from camera frames
)
```

The camera transform is computed every frame from the entity's `world_pos` plus the offsets, both rotated into the entity's body frame. As the entity moves and rotates, the camera follows.
Sensor camera frustums are drawn in viewports with `show_frustums=#true` and use the same coverage/projection controls as viewport frustums.

#### Reading frames in post_step

The renderer produces frames automatically. The sim only reads:

```python
def post_step(tick, ctx):
    # Latest frame the renderer has produced (may lag the current sim tick
    # by ~1 / fps µs in the steady state).
    rgb = ctx.read_msg("drone.scene_cam")

    # Pick the apparent camera latency at read time.
    # Returns the frame with the greatest timestamp <= `timestamp`
    # (floor / sample-and-hold). Past-the-end timestamps clamp to latest.
    rgb_33ms_ago = ctx.read_msg(
        "drone.scene_cam",
        timestamp=ctx.timestamp - 33_000,
    )

    if rgb is not None:
        rgba = np.asarray(rgb).reshape(480, 640, 4)
```

`read_msg` is a pure DB lookup — it never blocks the sim and never talks to the renderer.

#### Displaying in the editor

Add `sensor_view` panels to your KDL schematic:

```kdl
sensor_view "drone.scene_cam" name="RGB Camera"
sensor_view "drone.thermal_cam" name="Thermal"
```

### GPU Post-Processing Effects

The render-server applies optional GPU shader effects after the 3D render:

| Effect | Description | Parameters |
|--------|-------------|------------|
| `"normal"` | Standard RGB rendering | — |
| `"thermal"` | Iron-bow thermal colormap | `contrast` (default 1.5), `noise_sigma` (default 0.02) |
| `"night_vision"` | Green-tinted night vision | `gain` (default 2.0), `noise_sigma` (default 0.04) |
| `"depth"` | Depth buffer visualization | — |

```python
world.sensor_camera(
    entity=drone, name="thermal_cam",
    width=128, height=128, fov=90.0,
    fps=30.0,
    format="rgba",
    effect="thermal",
    effect_params={"contrast": 1.5, "noise_sigma": 0.02},
)
```

## Choosing FPS and latency

`fps` is the only timing knob on the camera. The renderer paces its work off sim-time deltas, so:

- In **real-time** mode (`generate_real_time=True`, the SITL/HITL default), `fps` maps directly to wall-clock fps.
- In **non-real-time** mode, the renderer races the sim at GPU speed. If it can't keep up at the requested `fps`, frames are simply spaced farther apart in sim time. There is no backpressure on the sim.

Apparent camera latency is **not** a config — it lives in the caller's `read_msg` argument:

```python
# Tight loop, no artificial latency:
frame = ctx.read_msg("drone.scene_cam")

# 33 ms of "camera latency" (one frame at 30 fps):
frame = ctx.read_msg("drone.scene_cam", timestamp=ctx.timestamp - 33_000)

# An entire pipeline of latency (sensor + ISP + transfer):
frame = ctx.read_msg("drone.scene_cam", timestamp=ctx.timestamp - 80_000)
```

This is how a real camera behaves — the frame your flight software reads was captured some time before "now", and `read_msg(timestamp=…)` lets you dial that delay in to match the real hardware you're emulating.

## Pairing with Betaflight SITL

The sensor camera feature is designed to complement the [Betaflight SITL example](../betaflight-sitl/) for comprehensive flight software testing. Sensor cameras add **synthetic vision** to the existing IMU/baro/mag sensor data, enabling simulation of vision-based algorithms that run alongside the flight controller.

```python
def post_step(tick, ctx):
    # 1. Read IMU sensor data from physics
    sensors = ctx.component_batch_operation(
        reads=["drone.accel", "drone.gyro", "drone.world_pos"]
    )

    # 2. Send sensor data to Betaflight, receive motor commands
    fdm = build_fdm_packet(sensors)
    motors = betaflight_bridge.step(fdm, rc_packet)
    ctx.write_component("drone.motor_command", motors)

    # 3. Read the latest camera frame for the vision pipeline.
    #    Apply a 33 ms latency to match a 30 fps FPV camera.
    rgb = ctx.read_msg("drone.forward_cam", timestamp=ctx.timestamp - 33_000)
    if rgb is not None:
        rgba = np.asarray(rgb).reshape(480, 640, 4)
        vision_result = run_vision_pipeline(rgba)
        ctx.write_component("drone.vision_target", vision_result)
```

This pattern gives you deterministic, reproducible testing of the full sensor stack: IMU + camera + flight controller + vision algorithms.

## Configuration Reference

### `world.sensor_camera()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity` | Entity | required | Entity the camera is attached to |
| `name` | str | required | Camera name (becomes `"{entity_name}.{name}"` in DB) |
| `width` | int | required | Frame width in pixels |
| `height` | int | required | Frame height in pixels |
| `fov` | float | 90.0 | Vertical field of view in degrees |
| `near` | float | 0.01 | Near clipping plane |
| `far` | float | 1000.0 | Far clipping plane |
| `pos_offset` | [f64; 3] | [0,0,0] | Camera position offset in entity body frame |
| `look_at_offset` | [f64; 3] | [0,0,-1] | Look-at target offset in entity body frame |
| `format` | str | "rgba" | Pixel format (`"rgba"`) |
| `effect` | str | "normal" | Post-process effect |
| `effect_params` | dict | {} | Effect-specific parameters |
| `fps` | float | 30.0 | Rendering rate in frames per second of sim time |
| `create_frustum` | bool | false | Create this sensor camera as a frustum source for 3D viewports |
| `show_ellipsoids` | bool | false | Render ellipsoid debug objects in this sensor camera |
| `frustums_color` | [f32; 3/4] | yellow | Frustum color, normalized RGBA |
| `projection_color` | [f32; 3/4] | white | 2D projection color, normalized RGBA |
| `frustums_thickness` | float | 0.006 | Frustum edge radius in world units |

### `ctx.read_msg()`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `msg_name` | str | required | Message name (e.g. `"drone.scene_cam"`) |
| `timestamp` | int | None | Optional sim time (µs). None ⇒ latest frame. Otherwise sample-and-hold: returns the frame with the greatest timestamp ≤ the requested one; past-the-end clamps to the latest. |

Returns a NumPy `uint8` array containing the raw RGBA bytes, or `None` if no frame exists at or before the requested timestamp.

### KDL Schematic

```kdl
sensor_view "entity.camera_name" name="Display Label"
```
