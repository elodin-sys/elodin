# frustum

Viewport camera frustum overlay rendering.

## What it does
- Builds frustum geometry for `MainCamera` viewports with `create_frustum=#true`.
- Draws those frustums only on viewports with `show_frustums=#true`.
- Builds line-mesh frustums directly from each camera projection (`near`, `far`, `fov`, `aspect`).
- If viewport `near`/`far` are set in KDL, frustum rendering follows those values automatically.
- Supports per-viewport style via `frustums_color` and `frustums_thickness`.
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

## KDL usage
```kdl
tabs {
    viewport name=ViewportA pos="(0,0,0,0, 8,2,4)" look_at="(0,0,0,0, 0,0,0)" create_frustum=#true frustums_color="yalk" frustums_thickness=0.008 near=0.05 far=300.0 aspect=1.7778 active=#true
    viewport name=ViewportB pos="(0,0,0,0, 2,2,2)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true active=#true
}
```

## Main API
- `FrustumPlugin`

## Inspector UX
- In viewport inspector, frustum controls are contextual:
  - `create_frustum` is exposed as a button (`CREATE`, then disabled as `CREATED`).
  - `show_frustums` toggle controls whether this viewport renders frustums from other viewports.
  - `frustums_color` and `frustums_thickness` are editable when `create_frustum` is enabled.
