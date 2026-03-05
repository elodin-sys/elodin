# gizmos

Rendering systems for vector arrows, body axes, and related UI labels.

## What it does
- Registers `GizmoPlugin` systems for arrow/body-axis rendering.
- Manages arrow-label UI cameras and label placement.
- Defines shared render-layer constant `GIZMO_RENDER_LAYER`.

## Main API
- `GizmoPlugin`
- `GIZMO_RENDER_LAYER`
- `evaluate_vector_arrow`

## Status
Partially legacy: navigation cube UX is now handled by Cube-Viewer (`view_cube`).
This module remains active for vector arrows/body axes rendering.
