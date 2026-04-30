# Ellipsoid Frustum Intersection

Demonstrates sensor camera frustum/ellipsoid intersection: volume coverage and 2D projection on the far plane.

## Run

```
elodin editor main.py
```

## KDL layout

A horizontal split with two 3D viewports and a sensor camera feed:

- **Viewport Source:** `create_frustum=#true` — creates a static viewport frustum.
- **Target View:** `show_frustums=#true` — displays both the viewport frustum and the sensor camera frustum.
- **Sensor Camera:** `sensor_view "frustum_camera_rig.frustum_cam"` — displays frames rendered from the synthetic camera.

The frustum source is registered in Python with `world.sensor_camera(..., create_frustum=True)`.
It is attached to a `frustum_camera_rig` entity whose `world_pos` is updated in `pre_step`, so the sensor camera orbits the ellipsoid while continuously looking at it.
The viewport frustum and sensor camera frustum use different colors so they can be compared in the target viewport.

The schematic embeds an ellipsoid `object_3d` with `ellipsoid.world_pos`. The ellipsoid name (`ellipsoid`) is used for the `FrustumCoverage` component (`ellipsoid.frustum_coverage`).

## Inspector controls

On the **Frustum View** viewport, open the Inspector and enable:

- **SHOW FRUSTUMS** — required to see frustum overlays and intersection options.
- **COVERAGE** — volume ratio (%), `FrustumCoverage` write, monitor strip at bottom.
- **PROJ. 2D** — 2D silhouette on the far plane; the mesh color follows the sensor camera's **PROJ. 2D COLOR**.

Intersection toggles appear only when at least one ellipsoid is detected.

Open the **Sensor Camera** tile's Inspector to create/delete the sensor camera frustum or edit its frustum style.
