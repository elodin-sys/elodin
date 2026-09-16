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
  `highlight` thickens the far-plane top edge and balls the corner holding the image origin; `triangle` stands a triangle on that edge, using the edge itself as its base.
  Marker geometry is drawn opaque white so it separates from `frustums_color`, falling back to that color's complement when the frustum is itself near-white.
- Repeats the marker along the top of the camera's own pane — 3D viewports and sensor camera panes alike — so the image orientation reads the same there as on the frustum.
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
- `frustums_up_marker` (optional): `none` (default), `highlight` (thickens the far-plane top edge and balls the image-origin corner), or `triangle` (stands a triangle on the middle of that edge, closed by the edge itself). The marker is drawn in white, or in the complement of `frustums_color` when that color is near-white.

## KDL usage
```kdl
tabs {
    viewport name=ViewportA pos="(0,0,0,0, 8,2,4)" look_at="(0,0,0,0, 0,0,0)" create_frustum=#true frustums_color="yellow" frustums_thickness=0.008 frustums_up_marker="triangle" near=0.05 far=300.0 aspect=1.7778 active=#true
    viewport name=ViewportB pos="(0,0,0,0, 2,2,2)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true active=#true
}
```

## Main API
- `FrustumPlugin`

## Inspector UX
- In viewport inspector, frustum controls are contextual:
  - `create_frustum` is exposed as a toggle button (`CREATE` / `DELETE`).
  - `show_frustums` toggle controls whether this viewport renders frustums from other viewports.
  - `frustums_color`, `frustums_thickness`, and `frustums_up_marker` are editable when `create_frustum` is enabled. The up-marker color is derived from `frustums_color` and is not separately editable.
