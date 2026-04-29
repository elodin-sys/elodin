# frustum_intersection

Frustum/ellipsoid intersection: volume coverage and 2D projection.

## What it does

- Finds active viewport frustums from cameras with `create_frustum=#true` (set in Inspector).
- Finds ellipsoid `object_3d` meshes with received world position data.
- **Feature 1 – COVERAGE (Y/N):** Computes frustum∩ellipsoid volume ratio, writes `FrustumCoverage`, and displays `%` in the monitor strip at the bottom of the target viewport (no 3D mesh). Uses SDF grid sampling.
- **Feature 2 – PROJ. 2D (Y/N):** Renders a 2D projection mesh of frustum∩ellipsoid on the far plane. The projection is drawn on **target** viewports (those with `show_frustums=#true` and **PROJ. 2D** on), not on the source viewport that creates the frustum. Each target viewport shows the silhouette as seen from its own camera, using other viewports' frustums as the intersection volume. Projection color comes from the source viewport's **PROJ. 2D COLOR**.
- Both features are independent toggles in the Inspector.
- Intersection controls are available only when `show_frustums=#true` and at least one ellipsoid is detected.
- Renders only on viewports that enable `show_frustums=#true` (same layer-routing model as `frustum`).

## Inspector

Controls in the viewport Inspector:

- **SHOW FRUSTUMS:** Enables frustum overlays on this viewport.
- **FRUSTUM COLOR:** Color for this viewport's source frustum.
- **PROJ. 2D COLOR:** Color for this viewport's source frustum 2D projection in target viewports.
- If an ellipsoid is detected and **SHOW FRUSTUMS** is on:
  - **COVERAGE:** Toggle volume computation + `FrustumCoverage` write + bottom monitor display.
  - **PROJ. 2D:** Toggle 2D projection mesh on far plane.
- If prerequisites are not met, intersection toggles are hidden and forced off.

## FrustumCoverage component

When COVERAGE is enabled, the plugin writes a component per ellipsoid:

- **Component ID:** `{ellipsoid_name}.frustum_coverage` (e.g. `ellipsoid.frustum_coverage`)
- **Schema:** F32 array of length 1, value in `[0.0, 1.0]` (volume ratio)
- **Usage:** Query this component from Elodin DB, EQL, or other systems to get the fraction of each ellipsoid inside any frustum. The ratio is the maximum across all frustums that overlap the ellipsoid. Ellipsoids not in any frustum are reset to `0.0`.

## Notes

- Volume computation uses a fixed grid (`INTERSECTION_GRID`, 32³ cells) for bounded cost.
- 2D projection uses a grid on the far plane (`PROJECTION_GRID`, 160×160) for mesh generation.
