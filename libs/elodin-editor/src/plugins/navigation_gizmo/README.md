# navigation_gizmo

Navigation gizmo support and camera-overlay synchronization helpers.

## What it does
- Provides render-layer allocation (`RenderLayerAlloc`).
- Provides shared components (`NavGizmoParent`, `NavGizmoCamera`).
- Keeps overlay camera viewport/rotation synchronized with the main camera.
- Still contains legacy `spawn_gizmo` behavior for the previous 3D nav gizmo.

## Main API
- `NavigationGizmoPlugin`
- `RenderLayerAlloc`
- `NavGizmoParent`
- `NavGizmoCamera`
- `spawn_gizmo` (legacy)

## Status
Legacy transition: old navigation gizmo UX is obsolete and replaced by Cube-Viewer (`view_cube`).
Some types/systems are still used by current overlay integration.
