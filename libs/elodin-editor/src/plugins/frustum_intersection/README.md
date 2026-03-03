# frustum_intersection

Frustum/ellipsoid intersection: volume coverage and 2D projection.

## What it does

- Finds active viewport frustums from cameras with `create_frustum=#true` (set in Inspector).
- Finds ellipsoid `object_3d` meshes with received world position data.
- **Feature 1 – Coverage in viewport (Y/N):** Computes frustum∩ellipsoid volume ratio, creates `FrustumCoverage` component, displays ratio in viewport (no 3D mesh). Uses SDF grid sampling.
- **Feature 2 – Projection 2D (Y/N):** Renders 2D projection of frustum∩ellipsoid on far plane.
- Both features are independent toggles in the Inspector; they are shown when `show_frustums=#true`.
- Renders only on viewports that enable `show_frustums=#true` (same layer-routing model as `frustum`).

## Inspector

When **SHOW FRUSTUMS** is enabled on a viewport:

- **COVERAGE IN VIEWPORT:** Toggle volume computation + component + viewport display.
- **PROJECTION 2D:** Toggle 2D projection mesh on far plane.
- **PROJECTION COLOR:** Color of the 2D projection mesh in this viewport (each viewport can choose its own color).
- Both features are auto-enabled when SHOW FRUSTUMS is first toggled on.
- If no ellipsoid is detected, a message is shown but the toggles remain armed.

## Notes

- Volume computation uses a fixed grid (`INTERSECTION_GRID`, 32³ cells) for bounded cost.
- 2D projection uses a grid on the far plane (`PROJECTION_GRID`, 80×80) for mesh generation.
