# frustum_intersection

Frustum/ellipsoid intersection mesh rendering.

## What it does
- Finds active viewport frustums from cameras with `create_frustum=#true`.
- Finds visible ellipsoid `object_3d` meshes.
- Builds a runtime mesh of `frustum ∩ ellipsoid` using SDF boolean intersection (`max(d_frustum, d_ellipsoid)`) and marching tetrahedra.
- Renders the result only on viewports that enable `show_frustums=#true` (same layer-routing model as `frustum`).

## Notes
- This is currently a CPU-generated POC mesh intended for visualization and validation.
- Mesh resolution is fixed (`32^3` cells) to keep update cost bounded.
