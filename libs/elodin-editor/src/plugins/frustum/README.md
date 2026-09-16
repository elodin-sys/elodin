# frustum

Viewport camera frustum overlay rendering.

## What it does
- Builds frustum geometry for `MainCamera` viewports with `create_frustum=#true`.
- Draws those frustums only on viewports with `show_frustums=#true`.
- Keeps frustums visible in other viewports when their source viewport is hidden behind a tab.
- Builds line-mesh frustums directly from each camera projection (`near`, `far`, `fov`, `aspect`).
- If viewport `near`/`far` are set in KDL, frustum rendering follows those values automatically.
- Supports per-viewport style via `frustums_color` and `frustums_thickness`.
- Marks the camera up direction via `frustums_up_marker`, so the image orientation can be read off the frustum.
  `highlight` thickens the far-plane top edge and balls the corner holding the image origin.
  The edge keeps `frustums_color`, which is what tells several frustums apart; the ball is opaque white, falling back to that color's complement when the frustum is itself near-white.
- Repeats the marker along the top of a sensor camera's own pane, so the image orientation reads the same there as on the frustum. 3D viewports can opt in with `frustums_up_marker_overlay`.
- Parents frustum visuals to the source camera, so motion/rotation stay exact.
- Renders frustums across viewport render layers.
- A viewport never renders its own frustum; it only renders frustums from other viewports.

## Viewport parameters
- `create_frustum` (bool): creates/publishes this viewport camera frustum.
- `show_frustums` (bool): shows frustums created by other viewports on this viewport.
- `near`/`far` (optional): override camera clipping planes. Defaults are `near=0.05`, `far=5.0`.
- `aspect` (optional): fixed camera aspect ratio. If omitted, aspect is derived from viewport size.
- `frustums_color` (optional): named color or tuple string like `"(255,255,0,200)"`.
- `frustums_thickness` (optional): edge radius in world units.
- `frustums_up_marker` (optional): `none` (default) or `highlight`, which thickens the far-plane top edge and balls the image-origin corner. The ball is white, or the complement of `frustums_color` when that color is near-white.
- `frustums_up_marker_overlay` (bool, default false): repeats the marker along the top of this viewport's own pane. Off by default, since a viewport is an interactive scene view and several panes carrying the marker at once read as clutter.

## KDL usage
```kdl
tabs {
    viewport name=ViewportA pos="(0,0,0,0, 8,2,4)" look_at="(0,0,0,0, 0,0,0)" create_frustum=#true frustums_color="yellow" frustums_thickness=0.008 frustums_up_marker="highlight" near=0.05 far=300.0 aspect=1.7778 active=#true
    viewport name=ViewportB pos="(0,0,0,0, 2,2,2)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true active=#true
}
```

## Main API
- `FrustumPlugin`

## Inspector UX
- In viewport inspector, frustum controls are contextual:
  - `create_frustum` is exposed as a toggle button (`CREATE` / `DELETE`).
  - `show_frustums` toggle controls whether this viewport renders frustums from other viewports.
  - `frustums_color`, `frustums_thickness`, and `frustums_up_marker` are editable when `create_frustum` is enabled, and `frustums_up_marker_overlay` once a marker is picked. The pane overlay is gated on `create_frustum` too, so deleting the frustum cannot strand a marker the inspector no longer exposes. The ball color is derived from `frustums_color` and is not separately editable.
