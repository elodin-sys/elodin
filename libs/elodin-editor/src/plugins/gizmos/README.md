# gizmos

Rendering systems for vector arrows, body axes, and related UI labels.

## What it does
- Registers `GizmoPlugin` systems for arrow/body-axis rendering.
- Evaluates arrow expressions into start/end `WorldPos` endpoint entities
  (`evaluate_vector_arrows`), which the canonical position pipeline
  (`sync_pos` -> `GeoPosition`/`GeoRotation` -> `Transform`) places in Bevy
  space; rendering then only reads the resulting endpoint poses.
- Manages arrow-label UI cameras and label placement.
- Defines shared render-layer constant `GIZMO_RENDER_LAYER`.

## Main API
- `GizmoPlugin`
- `GIZMO_RENDER_LAYER`
- `evaluate_vector_arrows`

## Status
Partially legacy: navigation cube UX is now handled by Cube-Viewer (`view_cube`).
This module remains active for vector arrows/body axes rendering.
