+++
title = "Cesium"
description = "Choose a real location for your Elodin simulation."
draft = false
weight = 104
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
+++

Configure Cesium in your simulation’s Python schematic. The same configuration
works when you run headless or open the editor.

## Set up access once

In [Cesium ion](https://ion.cesium.com/), create a read-only access token for the
terrain, imagery or Photo 3D you want to use.

Save the token in `~/.config/elodin/cesium-ion-token.txt`. Elodin reuses it across
runs, from both the CLI and the Python SDK. You do not need to open the editor
or enter it again each time. Keep the token outside your simulation and recordings.

## Describe the scene in Python

Add the geographic origin and Cesium scenery to your schematic:

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
    cesium=ui.cesium(scenery="photo3d"),
)

world.schematic(scene)
```

This places the local origin at the Mojave RC field. Latitude and longitude are
in degrees; altitude is meters above the WGS84 ellipsoid. Use this same origin
when placing your simulated objects and preparing ground heights.

Choose the scenery in `ui.cesium(...)`:

| `scenery` | What you see |
| --- | --- |
| `"local"` | Your existing local terrain, without streaming. |
| `"terrain"` | Cesium World Terrain with aerial imagery. |
| `"photo3d"` | Google Photorealistic 3D Tiles, including buildings where available. |

Photo 3D coverage varies by location. Changing the scenery changes the rendering;
it does not change your model’s ground-contact behavior.

## Run with or without the editor

Run the simulation headless:

```sh
elodin run my_simulation.py
```

Or open it in the editor:

```sh
elodin editor my_simulation.py
```

Both use the configuration from your Python file. There is no separate Cesium
launcher and no editor setup to repeat.

## How it works

The schematic describes the visible scene. Bevy-Cesium streams terrain, imagery
or 3D tiles and renders them in Elodin as you move through the area.

The simulation’s ground heights are configured separately in Python. They work
with or without Photo 3D and do not require the editor. See
[Terrain Heights](@/reference/terrain-heights.md) for preparation, sampling and
offline use. Changing the scenery does not change the physics.
