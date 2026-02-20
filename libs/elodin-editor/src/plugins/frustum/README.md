# frustum

Viewport camera frustum overlay rendering.

## What it does
- Watches `MainCamera` viewports with `show_frustums=#true`.
- Builds line-mesh frustums directly from each camera projection (`near`, `far`, `fov`, `aspect`).
- If viewport `near`/`far` are set in KDL, frustum rendering follows those values automatically.
- Supports per-viewport style via `frustums_color` and `frustums_thickness`.
- Parents frustum visuals to the source camera, so motion/rotation stay exact.
- Renders those frustums on viewport render layers (works across viewports).
- In practice, frustums are easiest to inspect from another viewport.

## Viewport parameters
- `show_frustums` (bool): enable frustum rendering for that viewport camera.
- `near`/`far` (optional): override camera clipping planes. Defaults are `near=0.05`, `far=5.0`.
- `frustums_color` (optional): named color or tuple string like `"(255,255,0,200)"`.
- `frustums_thickness` (optional): edge radius in world units.

## KDL usage
```kdl
tabs {
    viewport name=ViewportA pos="(0,0,0,0, 8,2,4)" look_at="(0,0,0,0, 0,0,0)" show_frustums=#true frustums_color="yalk" frustums_thickness=0.008 near=0.05 far=300.0 active=#true
    viewport name=ViewportB pos="(0,0,0,0, 2,2,2)" look_at="(0,0,0,0, 0,0,0)" active=#true
}
```

## Main API
- `FrustumPlugin`

## Inspector UX
- In viewport inspector, frustum controls are contextual:
  - `show_frustums` toggle is always visible.
  - When enabled, a color swatch opens a color picker and `frustums_thickness` is editable.
