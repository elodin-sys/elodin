# editor_cam_touch

Touch input integration for `bevy_editor_cam::EditorCam`.

## What it does
- Tracks touch state through `TouchTracker`.
- Maps gestures to camera interactions:
  - 1 finger: orbit.
  - 2 fingers: pan + pinch zoom.
- Applies interactions only when the touch midpoint is inside the active viewport.

## Main API
- `EditorCamTouchPlugin`
- `touch_tracker`
- `touch_editor_cam`

## Status
Active.
