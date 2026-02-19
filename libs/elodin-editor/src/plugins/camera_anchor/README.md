# camera_anchor

Small helper for safe camera anchor computation.

## What it does
- Exposes `camera_anchor_from_transform(transform: &Transform) -> Option<DVec3>`.
- Returns `None` when inversion/values are invalid, to avoid propagating `NaN`s.

## Main API
- `camera_anchor_from_transform`

## Status
Active (shared utility).
