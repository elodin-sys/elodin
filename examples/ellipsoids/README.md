# Ellipsoids – Frustum/Ellipsoid Intersection Test Scenario

> **Note:** This example is a temporary fork of the `drone` example, modified to use
> an ellipsoid mesh instead of a GLB model. It exists solely to exercise the
> `frustum_intersection` editor plugin. Once the plugin is promoted out of POC
> status, this example should either be simplified to a minimal standalone
> scenario or folded back into the main drone example.

## What it does

Spawns a drone entity whose visual representation is an ellipsoid (instead of
the usual quadcopter GLB). Two viewports are configured:

- **FrustumSource** — a camera that can have its frustum projected.
- **IntersectionView** — a second camera showing the scene from a different
  angle, where the intersection mesh can be rendered.

The `frustum_intersection` plugin computes and renders `frustum ∩ ellipsoid`
in real time, and persists the coverage ratio as a `frustum_coverage` component
in the telemetry database.

## Run

```
elodin editor main.py
```

Then in the editor inspector, enable `create_frustum` on the FrustumSource
viewport and set the ellipsoid intersect mode to **3D** or **2D** to see the
intersection visualization.

## Differences from the drone example

| Area | Drone | Ellipsoids |
|------|-------|------------|
| 3D mesh | GLB model | `ellipsoid scale="(1.2, 0.7, 0.5)"` |
| Viewports | single viewport | dual viewports (frustum source + intersection view) |
| Flight code | identical | identical (full attitude + motor stack) |
