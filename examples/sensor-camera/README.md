# Sensor Camera Example

This example demonstrates Elodin's **sensor camera** feature: synthetic camera imagery generated as first-class sensor data within a physics simulation. Virtual cameras are attached to simulation entities, rendered on demand by a headless GPU render-server, and returned as raw RGBA pixel arrays — ready to feed into vision-based flight software, SLAM pipelines, or any algorithm that consumes video frames.

## Overview

Five colored balls bounce around a walled room under gravity. Two of them carry sensor cameras:

- **Cyan ball** — RGB camera (640×480)
- **Magenta ball** — Thermal camera (128×128, iron-bow colormap)

The `post_step` callback demonstrates **both** render API styles:

- **Every 4th tick (30 fps)** — `ctx.render_cameras()` batches both cameras in a single round-trip to the render-server.
- **Every other 2nd tick (the 2nd ticks that aren't 4th ticks)** — `ctx.render_camera()` renders only the RGB camera, returning the frame directly as a numpy array.

This means the RGB camera renders at 60 fps total (batch + single ticks) while the thermal camera renders at 30 fps (batch ticks only).

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

## How It Works

### Architecture

Sensor camera rendering runs in a **separate headless process** managed by s10. The simulation, render-server, and editor are three distinct processes:

```
Simulation (Python + nox-py)
  │
  │  ctx.render_cameras(["..scene_cam", "..thermal_cam"])   ← batch
  │  ctx.render_camera("..scene_cam")                       ← single
  │    └─── UDS request ──▶  Render-Server (headless Bevy)
  │                            • Receives RENDER / RENDER_BATCH request
  │                            • Sets entity transforms to request timestamp
  │                            • Runs app.update() (GPU render + readback)
  │    ◄─── UDS response ──   • Returns RGBA pixel bytes
  │
  │  Frame returned as numpy array to Python
  │  Frame also written to DB for editor display
  │
  └──── DB (TCP) ────────▶  Editor (Bevy + Egui)
                              • Receives frames via FixedRateMsgStream
                              • Displays in sensor_view panels
```

The blocking UDS round-trip ensures the simulation has the correct frame **before** advancing to the next tick — critical for SITL workflows where flight software needs the camera frame to compute the next control command.

### Python API

#### Registering a camera

```python
world.sensor_camera(
    entity=drone,              # Entity the camera is attached to
    name="scene_cam",          # Becomes "drone.scene_cam" in the DB
    width=640, height=480,     # Resolution
    fov=90.0,                  # Field of view (degrees)
    pos_offset=[0, 0, 0.5],   # Body-frame offset from entity origin
    look_at_offset=[6, 6, 0], # Body-frame look-at direction
    format="rgba",             # Pixel format
    effect="normal",           # "normal", "thermal", "night_vision", "depth"
    effect_params={},          # Effect-specific parameters
)
```

The camera transform is computed each frame from the entity's `world_pos` plus the offsets, both rotated into the entity's body frame. As the entity moves and rotates, the camera follows.

#### Rendering in post_step

The example uses both render APIs in the same `post_step` callback:

```python
def post_step(tick, ctx):
    # Every 4th tick: batch both cameras in one round-trip
    if tick % 4 == 0:
        ctx.render_cameras(["drone.scene_cam", "drone.thermal_cam"])
        rgb = ctx.read_msg("drone.scene_cam")
        thermal = ctx.read_msg("drone.thermal_cam")

    # Every other 2nd tick: single camera, frame returned directly
    elif tick % 2 == 0:
        frame = ctx.render_camera("drone.scene_cam")
        if frame is not None:
            rgba = np.asarray(frame).reshape(480, 640, 4)
```

**`render_cameras()`** renders multiple cameras in a single UDS round-trip. After it returns, use `ctx.read_msg()` to retrieve each frame from the database.

**`render_camera()`** renders one camera and returns the frame directly as a numpy array — no separate read needed.

Both calls block until the frame is ready, ensuring the simulation does not advance until the render-server has produced the requested frames.

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
    format="rgba",
    effect="thermal",
    effect_params={"contrast": 1.5, "noise_sigma": 0.02},
)
```

## Pairing with Betaflight SITL

The sensor camera feature is designed to complement the [Betaflight SITL example](../betaflight-sitl/) for comprehensive flight software testing. While the SITL example provides IMU, barometer, and magnetometer sensor data to Betaflight's flight controller, sensor cameras add **synthetic vision** — enabling simulation of vision-based algorithms that run alongside the flight controller.

### The SITL + Vision Pattern

In a real flight computer (like Elodin's Aleph), the flight controller and vision algorithms share the same hardware. The flight controller consumes IMU data at 8 kHz, while vision algorithms consume camera frames at 30-60 fps. Both produce outputs that affect the vehicle's behavior.

The `post_step` callback is the integration point for all of this:

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

    # 3. Render cameras and feed to vision algorithm
    if tick % 2 == 0:  # 60 fps at 120 Hz sim
        ctx.render_cameras(["drone.forward_cam", "drone.thermal_cam"])
        rgb = ctx.read_msg("drone.forward_cam")
        if rgb is not None:
            rgba = np.asarray(rgb).reshape(480, 640, 4)
            vision_result = run_vision_pipeline(rgba)
            ctx.write_component("drone.vision_target", vision_result)
```

This pattern gives you deterministic, reproducible testing of the full sensor stack: IMU + camera + flight controller + vision algorithms — all synchronized tick-by-tick.

### Why Blocking Matters

`render_camera()` blocks the simulation until the frame is ready. This is intentional:

1. **Determinism** — The same simulation produces the same frames every time. No race conditions between rendering and physics.
2. **Correctness** — The frame shows the scene at exactly the current simulation timestamp. The vision algorithm sees what the camera would see at that instant.
3. **SITL compatibility** — Flight software expects sensor data to arrive in order, at known timestamps. A frame from tick N must be available before tick N+1's control computation.

### Performance Budget

At a 120 Hz simulation with rendering every 2nd tick:

| Operation | Time | Budget |
|-----------|------|--------|
| Physics tick | ~0.5 ms | |
| Single render — `render_camera()` (RGB 640×480) | ~5-8 ms | |
| Batch render — `render_cameras()` (RGB + thermal) | ~8-12 ms | |
| **Per-tick budget** | **8.33 ms** | 1/120 s |

Single-camera ticks fit within the per-tick budget. Batch ticks slightly exceed it, but the simulation catches up on non-render ticks. At steady state, the simulation maintains real-time pace. The batch approach avoids the overhead of two separate round-trips on dual-camera ticks.

For higher-resolution cameras or more cameras, reduce the render frequency or increase `sim_time_step`.

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
| `look_at_offset` | [f64; 3] | [0,0,0] | Look-at target offset in entity body frame |
| `format` | str | "rgba" | Pixel format (`"rgba"`) |
| `effect` | str | "normal" | Post-process effect |
| `effect_params` | dict | {} | Effect-specific parameters |

### `ctx.render_camera()` / `ctx.render_cameras()`

| Method | Returns | Description |
|--------|---------|-------------|
| `ctx.render_camera("name")` | `numpy.ndarray` or `None` | Render one camera, return RGBA bytes |
| `ctx.render_cameras(["a", "b"])` | — | Render multiple cameras in one batch |

### KDL Schematic

```kdl
sensor_view "entity.camera_name" name="Display Label"
```
