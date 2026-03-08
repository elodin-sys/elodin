# Sensor Camera Example

This example demonstrates Elodin's **sensor camera** feature: synthetic camera imagery generated as first-class sensor data within a physics simulation. Virtual cameras are attached to simulation entities, rendered on demand by a headless GPU render-server, and returned as raw RGBA pixel arrays — ready to feed into vision-based flight software, SLAM pipelines, or any algorithm that consumes video frames.

## Overview

Five colored balls bounce around a walled room under gravity. Two of them carry sensor cameras:

- **Cyan ball** — RGB camera (640×480, rendered at 60 fps)
- **Magenta ball** — Thermal camera (128×128, rendered at 30 fps, iron-bow colormap)

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
  │  ctx.render_camera("cam_ball_a.scene_cam")
  │    └─── UDS request ──▶  Render-Server (headless Bevy)
  │                            • Receives RENDER request
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

```python
def post_step(tick, ctx):
    # Render at 60 fps (every 2nd tick of a 120 Hz sim)
    if tick % 2 == 0:
        frame = ctx.render_camera("drone.scene_cam")
        # frame is a numpy uint8 array of RGBA pixels (W × H × 4)
        if frame is not None:
            rgba = np.asarray(frame).reshape(480, 640, 4)
            # Feed to your vision algorithm, SLAM pipeline, etc.
```

`render_camera()` blocks until the frame is ready and returns the pixel data directly as a numpy array. This is the lockstep synchronization point — the simulation does not advance until the frame is produced.

For multiple cameras, you can render them sequentially:

```python
def post_step(tick, ctx):
    if tick % 2 == 0:
        rgb_frame = ctx.render_camera("drone.scene_cam")
    if tick % 4 == 0:
        thermal_frame = ctx.render_camera("drone.thermal_cam")
```

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

    # 3. Render camera and feed to vision algorithm
    if tick % 2 == 0:  # 60 fps at 120 Hz sim
        frame = ctx.render_camera("drone.forward_cam")
        if frame is not None:
            rgba = np.asarray(frame).reshape(480, 640, 4)
            # Run optical flow, SLAM, target detection, etc.
            vision_result = run_vision_pipeline(rgba)
            # Feed vision output back into the simulation
            ctx.write_component("drone.vision_target", vision_result)
```

This pattern gives you deterministic, reproducible testing of the full sensor stack: IMU + camera + flight controller + vision algorithms — all synchronized tick-by-tick.

### Why Blocking Matters

`render_camera()` blocks the simulation until the frame is ready. This is intentional:

1. **Determinism** — The same simulation produces the same frames every time. No race conditions between rendering and physics.
2. **Correctness** — The frame shows the scene at exactly the current simulation timestamp. The vision algorithm sees what the camera would see at that instant.
3. **SITL compatibility** — Flight software expects sensor data to arrive in order, at known timestamps. A frame from tick N must be available before tick N+1's control computation.

### Performance Budget

At a 120 Hz simulation with `render_camera()` called every 2nd tick (60 fps):

| Operation | Time | Budget |
|-----------|------|--------|
| Physics tick | ~0.5 ms | |
| RGB render (640×480) | ~5-8 ms | |
| Thermal render (128×128) | ~3-5 ms | |
| **Per-tick budget** | **8.33 ms** | 1/120 s |

Single-camera ticks (RGB only) fit within the per-tick budget. Dual-camera ticks (RGB + thermal) slightly exceed it, but the simulation catches up on non-render ticks. At steady state, the simulation maintains real-time pace.

For higher-resolution cameras or more cameras, reduce the render frequency or increase `sim_time_step`.

## Project Structure

```
examples/sensor-camera/
├── main.py    # Simulation: entities, cameras, physics, post_step rendering
└── README.md  # This file
```

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

## References

- [Betaflight SITL example](../betaflight-sitl/) — Full SITL integration with IMU, motor commands, and lockstep synchronization
- [Sensor camera architecture](../../ai-context/sensor-camera-final-summary.md) — Internal design and implementation details
- [Latency optimization](../../ai-context/sensor-camera-latency-optimization-2.md) — Render-server performance tuning
