# Ellipsoid Frustum Intersection

Demonstrates frustum/ellipsoid intersection: volume coverage and 2D projection on the far plane.

## Run

```
elodin editor main.py
```

## KDL layout

Two viewports in a horizontal split:

- **Frustum Source:** `create_frustum=#true` — creates the frustum geometry from this camera.
- **Frustum View:** `show_frustums=#true` — displays the frustum and intersection overlays.

The schematic embeds an ellipsoid `object_3d` with `ellipsoid.world_pos`. The ellipsoid name (`ellipsoid`) is used for the `FrustumCoverage` component (`ellipsoid.frustum_coverage`).

## Inspector controls

On the **Frustum View** viewport, open the Inspector and enable:

- **SHOW FRUSTUMS** — required to see frustum overlays and intersection options.
- **COVERAGE** — volume ratio (%), `FrustumCoverage` write, monitor strip at bottom.
- **PROJ. 2D** — 2D silhouette on the far plane; **PROJ. COLOR** sets the mesh color.

Intersection toggles appear only when at least one ellipsoid is detected.
