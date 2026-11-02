+++
title = "Terrain Heights"
description = "Prepare and use ground heights in an Elodin simulation."
draft = false
weight = 104
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
+++

Terrain heights give your simulation a ground surface. They work with or without
Photo 3D, and with or without the editor.

[Cesium](@/reference/cesium.md) controls the visible scenery. Switching Photo 3D
on or off does not change the heights used by the physics.

## Prepare the area in Python

Before running your simulation, prepare a height grid around its initial position:

```python
import elodin as el

terrain = el.terrain_heightmap(
    center_ecef=initial_position,
    dimensions=(24_000, 24_000),
    spacing=10,
    asset_id=1,
)
```

`initial_position` is the simulation’s initial WGS84 ECEF position, in meters.
This example covers a 24 × 24 km area with samples spaced about 10 m apart.
`asset_id` selects the terrain source in Cesium ion; it is independent of the
Photo 3D asset used for rendering.

The SDK downloads missing heights and caches the grid. Preparation happens when
this Python code runs, before the simulation starts. No editor, rendering window
or separate preparation command is needed. Downloads use the saved Cesium token
described in [Cesium setup](@/reference/cesium.md#set-up-access-once).

## Read heights during the simulation

```python
height, normal, valid = terrain.sample(position_ecef)
```

- `height` is meters above the WGS84 ellipsoid, not sea level.
- `normal` is the outward surface normal in ECEF coordinates.
- `valid` tells you whether the position is inside the prepared area.

Your model uses these values to calculate ground contact. Outside the area,
`valid` is false; choose how your simulation handles that case rather than
using the clamped edge height as real terrain.

Sampling uses JAX operations and makes no network or disk requests. Both
`jax.jit` and `jax.vmap` are supported.

## Reuse the grid offline

The cache lives in `~/.cache/elodin/terrain-heights`, or under `XDG_CACHE_HOME`
when set. Pass `cache_dir=` to choose another directory.

Use the same preparation call with `offline=True` to require an existing cached
grid. If it is missing, preparation fails rather than downloading it.

You can also load a specific grid directly:

```python
terrain = el.TerrainHeightmap.load("terrain.json")
```

Keep `terrain.path` and `terrain.sha256` with your simulation results to identify
the exact grid used. Cached grids are not automatically refreshed.

## Heights and visible scenery

A height grid contains elevations. It does not contain imagery, buildings or
3D meshes, and it does not provide building collisions. The terrain surface
can differ from the visible Photo 3D geometry.

An offline height grid is enough to run the ground calculations. Viewing the
same area offline also requires the visual tiles to be downloaded separately.
