+++
title = "Terrain"
description = "Add aerial imagery and ground heights to your Elodin simulation."
draft = false
weight = 104
sort_by = "weight"
aliases = ["/reference/cesium/", "/reference/terrain-heights/"]

[extra]
toc = true
top = false
icon = ""
order = 4
+++

Choose a real location for your simulation, display its surroundings, and use
ground heights in your physics model. Configure everything in Python, whether
you run headless or open the editor.

Data access is included in the demo through a shared demo account. There is no
account to create or access token to enter.

## Choose a location

Define the geographic origin in your schematic:

```python
import elodin.ui as ui

scene = ui.schematic(
    ui.viewport(name="World", frame="ENU"),
    coordinate=ui.coordinate(
        frame="ENU",
        lat=35.350664,
        lon=-117.809027,
        alt=589.274,
    ),
    terrain=ui.terrain(scenery="aerial"),
)

world.schematic(scene)
```

This example places the local origin at the Mojave RC field. Latitude and
longitude are in degrees; altitude is meters above the WGS84 ellipsoid.
Use the same origin when placing simulated objects and preparing ground heights.

## Aerial imagery and Photo 3D

Choose what to display with the `scenery` setting:

| `scenery` | What you see |
| --- | --- |
| `"local"` | Your existing local scene, without streaming. |
| `"aerial"` | Terrain with aerial imagery. |
| `"photo3d"` | Photorealistic 3D scenery, including buildings where available. |

Elodin streams the visible area as you move. New areas may take a moment to load,
and Photo 3D coverage varies by location.

The scenery setting controls the view. Switching Photo 3D on or off does not
change the ground heights used by the physics.

## Height maps

Prepare a grid around your simulation’s initial position in Python:

```python
import elodin as el

terrain = el.terrain_heightmap(
    center_ecef=initial_position,
    dimensions=(24_000, 24_000),
    spacing=10,
)
```

`initial_position` is a WGS84 ECEF position in meters. This example covers a
24 × 24 km area with samples spaced about 10 m apart.

The SDK prepares and caches missing heights before the simulation starts.
There is no separate preparation command and no editor window is needed.

During the simulation, read the grid:

```python
height, normal, valid = terrain.sample(position_ecef)
```

`height` is meters above the WGS84 ellipsoid. `normal` is the surface normal in
ECEF coordinates. Check `valid` before using a result outside the prepared area.
Sampling uses JAX operations, without network or disk requests.

Your model defines how ground contact works. Height maps describe the terrain
surface; visible Photo 3D buildings do not automatically become collision surfaces.

## Run your simulation

Run headless:

```sh
elodin run my_simulation.py
```

Or open the editor:

```sh
elodin editor my_simulation.py
```

Both use your Python configuration. You can prepare and use heights without
rendering the scenery. A cached height map contains elevations, not imagery or
3D buildings.
