+++
title = "Cesium terrain"
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

Use Cesium to run your simulation at a real location, with terrain, aerial imagery
or photorealistic 3D scenery streamed into Elodin.

## Connect Cesium

In [Cesium ion](https://ion.cesium.com/), add the assets you want to use and create
a read-only access token for them.

Open the Command Palette, choose **Cesium: Connect account**, and paste your token.

## Choose where to simulate

Choose **Scene: Set location** in the Command Palette. Enter the latitude,
longitude and altitude of your simulation’s origin.

For the Mojave RC field, use **35.350664°, −117.809027°, 589.274 m**.
Latitude and longitude are in degrees; altitude is above the WGS84 ellipsoid.

Elodin saves this origin with the simulation and uses it for both the scene and
ground heights. There is only one location to configure.

## Choose what to see

Choose **Cesium: Choose scenery** in the Command Palette.

| Scenery | What you see |
| --- | --- |
| Local terrain | Your local scene, without streaming. |
| Terrain and imagery | Cesium World Terrain with aerial imagery. |
| Photo 3D | Google Photorealistic 3D Tiles, including buildings where available. |

Select the assets from your ion account. Photo 3D coverage varies by location.

## Run your simulation

Elodin prepares and caches ground heights automatically before the run.
Your model reads them during the simulation without waiting for the network.
The model defines ground contact; visible Photo 3D buildings do not automatically
provide collisions.

**Bevy-Cesium** streams and renders the scenery as you move. New areas may take
a moment to load. Lower the scene detail if you need a smoother view.

## How it works

- **One geographic origin.** The location you choose anchors local simulation
  coordinates on Earth. The scene and ground-height queries use the same origin.
- **Scenery streams as you move.** Bevy-Cesium loads terrain, imagery or 3D tiles
  from Cesium ion and renders them in Elodin. It refines the visible area as more
  detail arrives.
- **Heights are prepared before the run.** Elodin’s Python SDK, nox-py, builds a
  height grid through Bevy-Cesium and caches it. During the simulation, JAX samples
  that grid locally; a slow connection does not slow down the physics.
- **Your model handles contact.** The grid describes the terrain surface.
  Photo 3D supplies the visible scenery, including buildings, but does not add
  building collisions to your model.
