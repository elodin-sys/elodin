+++
title = "Cesium terrain"
description = "Place an Elodin simulation at a real location and stream its surroundings."
draft = false
weight = 104
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 4
+++

You have a Cesium ion account and a place in mind. You want your aircraft to fly
over that part of the desert, with the surrounding terrain visible in Elodin.
Start with the RC-jet example, then use the same steps for your own simulation.

The desktop example runs the simulation and streams its surroundings. The
current browser demo replays a recorded flight while streaming the scenery;
publishing a different flight or location still requires developer setup.

![RC-jet in the WASM editor: streamed desert terrain, aircraft, color camera and telemetry.](../../assets/cesium-rc-jet-wasm.jpg)

*A recorded RC-jet flight in the browser, with scenery streamed as needed.*

## 1. Choose your scenery

In [Cesium ion](https://ion.cesium.com/), enable the assets you want to use:

| What you want to see | Assets used by the RC-jet launcher |
| --- | --- |
| Terrain with aerial imagery | Cesium World Terrain (`1`) and imagery (`2`) |
| Photo 3D, including buildings where covered | Google Photorealistic 3D Tiles (`2275207`) |

Photo 3D is the launcher's default. Coverage and detail vary by location. For
ground contact, the example also needs World Terrain (`1`), even when Photo 3D
supplies the visible scenery.

Create an ion token with read access to those assets. Save its value in
`$HOME/.config/elodin/cesium-ion-token.txt`, outside the repository. For a public browser deployment, create a separate
read-only token restricted to the required assets and your site's browser origin.
A browser token is visible to visitors; do not use a personal administrative token.

## 2. Try a flight at Mojave

From the Elodin repository root, build the Python SDK and desktop editor:

```bash
export ELODIN_SHELL_ID=cesium
nix develop
just install py
just install editor
```

Inside that shell, point the launcher at your token file:

```bash
export CESIUM_ION_TOKEN_FILE="$HOME/.config/elodin/cesium-ion-token.txt"
bash examples/cesium/run_rc_jet_terrain.sh
```

The example starts at **35.350664° N, 117.809027° W**. Bevy-Cesium selects,
downloads and renders terrain tiles for the editor and its sensor views as the
flight progresses.

To use terrain and aerial imagery instead of Photo 3D:

```bash
bash examples/cesium/run_rc_jet_terrain.sh --photo3d off
```

The original local terrain is still available without streaming:

```bash
unset CESIUM_ION_ASSET_ID CESIUM_ION_TOKEN
ELODIN_RC_JET_TERRAIN=planar elodin editor examples/rc-jet/main.py
```

## 3. Put the flight at your coordinates

Choose a latitude, longitude and reference altitude. Keep that same site in
the simulation's initial conditions, the heightmap request and the editor's
geographic configuration.

For the current RC-jet example:

- Set the demo site's latitude, longitude and field elevation in
  `examples/rc-jet/scenario.py` (`MOJAVE_RC_FIELD`). The scenario derives its
  initial ECEF position and attitude from that site.
- Set the matching `ui.coordinate(...)` values in
  `examples/rc-jet/bdx_schematic.py`. Its `lat` and `lon` values are degrees;
  `alt` is meters above the WGS84 ellipsoid.
- Disable the bundled Mojave-specific OSM overlay when using another location:
  `export ELODIN_CESIUM_OVERLAY_FILE=''`.
- Run the Cesium launcher again. It prepares a height grid around the new
  initial position before starting the simulation.

There is currently no latitude/longitude switch in the RC-jet launcher. Changing
the editor's origin alone does **not** move the simulation or relocate a recorded
flight. For your own model, use the existing
[coordinate-system primitives](@/reference/coords.md) to keep the simulation and
the displayed terrain in the same frame.

## 4. Give the physics a ground surface

The streamed scene and the physical ground are separate. Elodin's
`el.terrain_heightmap()` API asks Bevy-Cesium to prepare and cache heights before
the run. During simulation, JAX samples that fixed grid without network access.

- Set the grid's center and extent to cover the planned flight.
- Choose an initial altitude above the sampled ground. Heights use the WGS84
  ellipsoid, not mean sea level.
- Check the lookup's `valid` result when the aircraft approaches the edge.
- Implement ground contact in your model. Visible Photo 3D buildings do not
  automatically become collision geometry.

See [Terrain heights](@/reference/python-api.md#terrain-heights) for the API,
units, cache behavior and an example you can adapt.

## 5. View the flight in a browser

[Open the RC-jet WASM demo](https://medialoco.github.io/rc-jet-wasm/).
The browser runs the Elodin editor, including its timeline, graphs and camera
controls. The flight is recorded; the terrain and Photo 3D tiles stream as needed.
The Overlay button shows or hides the experimental OpenStreetMap layer.

The demo uses Bevy-Cesium's renderer with an experimental Rust streaming backend
adapted for WASM. The desktop integration uses Cesium Native. The browser demo
has a color camera; its infrared view has not been ported.

To publish your own location and flight today:

- Run your Python/JAX simulation outside the browser and keep its
  [recording](@/reference/replays.md).
- Have the recording, geographic configuration and assets integrated into a
  browser build, with a dedicated ion token for its deployment origin.
- Rebuild and publish that application. The current RC-jet demo embeds a prepared
  recording; it does not yet have a general recording upload or location picker.

## If the first view is slow or blurry

- Allow the initial terrain to load. Turning toward a new area requests more tiles.
- For the desktop example, try `--photo3d off` for a lighter scene, or increase
  `CESIUM_MAXIMUM_SCREEN_ERROR` to request less detail (the launcher defaults to `48`).
- If loading fails, check that the token can read every selected asset. For a
  browser token, also check the allowed origin.
- A cached physics heightmap does not cache the visual terrain or imagery.
  Streaming the scenery still requires network access.
