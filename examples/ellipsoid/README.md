# Ellipsoid Frustum Intersection

Demonstrates sensor camera frustum/ellipsoid intersection: volume coverage and 2D projection on the far plane.

## Run

From the repository root, so the drone GLB resolves against the shared `assets/` directory:

```
elodin editor examples/ellipsoid/main.py
```

## KDL layout

A horizontal split with two 3D viewports and a sensor camera feed:

- **Viewport Source:** `create_frustum=#true` — creates a static viewport frustum.
- **Target View:** `show_frustums=#true` — displays both the viewport frustum and the sensor camera frustum.
- **Sensor Camera:** `sensor_view "drone.scene_cam"` — displays frames rendered from the drone-mounted camera.

The frustum source is registered in Python with `world.sensor_camera(..., create_frustum=True)`.
It is attached to a `drone` entity whose `world_pos` is updated in `pre_step`, so the sensor camera follows a drone GLB moving inside the ellipsoid.
The viewport frustum and sensor camera frustum use different colors so they can be compared in the target viewport.

Both frustums carry a camera-up marker, which tells you how the image is oriented on a frustum seen from outside.
They use `frustums_up_marker="highlight"`: the far-plane top edge is drawn thicker, with a ball on the corner holding the image origin, both in white so they read against the frustum's own color.
The same marker is painted along the top of each camera's own pane — the two 3D viewports and the **Sensor Camera** pane — so you can match what the frustum says against the image it describes.
Since the drone rolls and yaws continuously, watch the sensor camera's marker rotate with it in the **Target View**.
The other available marker is `frustums_up_marker="triangle"`, which stands a triangle on the middle of that edge instead.

The schematic embeds a smaller ellipsoid `object_3d` with `ellipsoid.world_pos`, plus a `talon-quad-v2.glb` drone that stays inside it. The camera is mounted close to the drone body so part of the drone remains visible in the sensor image. The sensor camera leaves `show_ellipsoids=False`, so it does not render the ellipsoid debug surface. The ellipsoid name (`ellipsoid`) is used for the `FrustumCoverage` component (`ellipsoid.frustum_coverage`).

## Inspector controls

On the **Frustum View** viewport, open the Inspector and enable:

- **SHOW FRUSTUMS** — required to see frustum overlays and intersection options.
- **COVERAGE** — volume ratio (%), `FrustumCoverage` write, monitor strip at bottom.
- **PROJ. 2D** — 2D silhouette on the far plane; the mesh color follows the sensor camera's **PROJ. 2D COLOR**.

Intersection toggles appear only when at least one ellipsoid is detected.

Open the **Sensor Camera** tile's Inspector to create/delete the sensor camera frustum or edit its frustum style.
