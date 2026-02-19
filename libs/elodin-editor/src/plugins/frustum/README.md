# frustum

Viewport camera frustum overlay rendering.

## What it does
- Watches `MainCamera` viewports with `show_frustum=#true`.
- Builds line-mesh frustums directly from each camera projection (`near`, `far`, `fov`, `aspect`).
- Parents frustum visuals to the source camera, so motion/rotation stay exact.
- Renders those frustums on viewport render layers (works across viewports).
- In practice, frustums are easiest to inspect from another viewport.

## KDL usage
```kdl
tabs {
    viewport name=ViewportA pos="(0,0,0,0, 8,2,4)" look_at="(0,0,0,0, 0,0,0)" show_frustum=#true active=#true
    viewport name=ViewportB pos="(0,0,0,0, 2,2,2)" look_at="(0,0,0,0, 0,0,0)" active=#true
}
```

## Main API
- `FrustumPlugin`
