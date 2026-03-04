# frustum_intersection

Frustum/ellipsoid intersection: volume coverage and 2D projection.

## What it does

- Finds active viewport frustums from cameras with `create_frustum=#true` (set in Inspector).
- Finds ellipsoid `object_3d` meshes with received world position data.
- **Feature 1 – COVERAGE (Y/N):** Computes frustum∩ellipsoid volume ratio, writes `FrustumCoverage`, and displays `%` in the monitor strip at the bottom of the target viewport (no 3D mesh). Uses SDF grid sampling.
- **Feature 2 – PROJ. 2D (Y/N):** Renders a 2D projection mesh of frustum∩ellipsoid on the far plane.
- Both features are independent toggles in the Inspector.
- Intersection controls are available only when `show_frustums=#true` and at least one ellipsoid is detected.
- Renders only on viewports that enable `show_frustums=#true` (same layer-routing model as `frustum`).

## Inspector

Controls in the viewport Inspector:

- **SHOW FRUSTUMS:** Enables frustum overlays on this viewport.
- If an ellipsoid is detected and **SHOW FRUSTUMS** is on:
  - **COVERAGE:** Toggle volume computation + `FrustumCoverage` write + bottom monitor display.
  - **PROJ. 2D:** Toggle 2D projection mesh on far plane.
  - **PROJ. COLOR:** Color for the 2D projection mesh in this viewport (shown only when **PROJ. 2D** is on).
- If prerequisites are not met, intersection toggles are hidden and forced off.
- If no ellipsoid is detected, a helper message is shown.

## Notes

- Volume computation uses a fixed grid (`INTERSECTION_GRID`, 32³ cells) for bounded cost.
- 2D projection uses a grid on the far plane (`PROJECTION_GRID`, 80×80) for mesh generation.
