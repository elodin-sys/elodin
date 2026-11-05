+++
title = "Terrain"
description = "Place your simulation, display scenery and query ground heights."
weight = 104
aliases = ["/reference/cesium/", "/reference/terrain-heights/"]

[extra]
toc = true
order = 4
+++

Define the location and terrain in your Python schematic. The same terrain
provides scenery for the editor and heights for your simulation.

## Location and scenery

```python
import elodin as el
import elodin.ui as ui

location = ui.coordinate(
    frame="ECEF",
    lat=28.60839, lon=-80.60433,
    alt=-7.063,
)
terrain = el.Terrain()

scene = ui.schematic(
    ui.viewport(name="Flight"),
    coordinate=location,
    terrain=terrain,
)
world = el.World()
world.schematic(scene)
```

Latitude and longitude are in degrees; altitude is in meters above the WGS84
ellipsoid.

`Terrain()` selects Google Photorealistic 3D Tiles, streamed through Bevy-Cesium.
Omit `terrain=terrain` to use the location on its own.

## Terrain Heights

Prepare a grid before running the simulation, using the same `terrain` object:

```python
import jax.numpy as jnp
from elodin.geodesy import geodetic_to_ecef

landing = geodetic_to_ecef(
    jnp.deg2rad(28.485738),
    jnp.deg2rad(-80.542948),
    0.0,
)
heights = terrain.heightmap(
    landing, dimensions=(200.0, 200.0), spacing=10.0,
)
```

Dimensions and spacing are in meters. The grid is cached and reused on later
runs. Use `offline=True` to read from the cache. No editor is needed.

During the simulation:

```python
sample = heights.sample(position_ecef)
# sample.height_m, sample.normal_ecef, sample.valid
```

Use `sample.height_m` and `sample.normal_ecef` in your ground-contact model,
after checking `sample.valid`. Heights are above the WGS84 ellipsoid; normals
are in ECEF. Queries work inside JAX systems.

## Run Falcon 9

```sh
elodin editor examples/falcon9/main.py
```

Falcon 9 flies from LC-39A to Landing Zone 1 at Cape Canaveral. See its
[README](https://github.com/elodin-sys/elodin/tree/main/examples/falcon9)
for the simulation and reference flights.
