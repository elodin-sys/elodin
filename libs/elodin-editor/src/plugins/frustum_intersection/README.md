# frustum_intersection

Frustum/ellipsoid intersection: volume coverage and 2D projection.

## What it does

- Finds active viewport frustums from cameras with `create_frustum=#true` (set in Inspector).
- Finds visible ellipsoid `object_3d` meshes.
- **Feature 1 – Coverage in viewport (Y/N):** Computes frustum∩ellipsoid volume ratio, creates `FrustumCoverage` component, displays ratio in viewport (no 3D mesh). Uses SDF grid sampling.
- **Feature 2 – Projection 2D (Y/N):** Renders 2D projection of frustum∩ellipsoid on far plane.
- Both features are independent toggles in the Inspector; they can be enabled together.
- Renders only on viewports that enable `show_frustums=#true` (same layer-routing model as `frustum`).

## Inspector

When a frustum is created (CREATE button in Inspector):

- **COVERAGE IN VIEWPORT:** Toggle volume computation + component + viewport display.
- **COVERAGE MONITOR:** When coverage is on, show ratio on other viewports with `show_frustums`.
- **PROJECTION 2D:** Toggle 2D projection mesh on far plane.
- **PROJECTION COLOR:** Color of the 2D projection mesh.

## Notes

- Volume computation uses a fixed grid (`32³` cells) for bounded cost.
- 2D projection uses a grid on the far plane for mesh generation.
