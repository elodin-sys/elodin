+++
title = "Command Palette"
description = "Command palette reference"
draft = false
weight = 107
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 7
+++

## Command palette

### Create
- `Window`: prompts for a title, then prompts for the first tab (viewport/monitor/graph). Creates a secondary window when a path is provided in the schematic, otherwise configures the primary window.
- `Viewport`: inserts a 3D viewport tile.
- `Graph`: inserts a graph tile (prompts for component/fields).
- `Monitor`: inserts a monitor tile (prompts for component).
- `Query Table` / `Query Plot`: inserts the corresponding query tile.
- `Action`: inserts an action tile (prompts for label and Lua payload).
- `Video Stream`: inserts a video stream tile (prompts for message name).
- `Dashboard`: inserts a dashboard tile (empty layout).

### Viewport & render toggles
- `Toggle Wireframe`
- `Toggle HDR`
- `Toggle Grid`
- `Reset Cameras`: reset all or a selected viewport camera to defaults.
- `Toggle Body Axes`: currently disabled (placeholder prompt).

## Simulation / time
- `Toggle Recording`
- `Set Playback Speed`: choose from predefined speeds.
- `Goto Tick`: jump to a specific tick/timestamp.
- `Fix Current Time Range`: freezes the current time range.
- `Set Time Range`: prompts for start/end offsets.

### Schematic / persistence
- `Save Schematic`, `Save Schematic As`
- `Save Schematic to DB`
- `Load Schematic`
- `Clear Schematic`
- `Set Color Scheme`

### 3D objects
- `Create 3D Object`: create a basic mesh driven by an EQL expression; supports sphere/box/cylinder/plane/ellipsoid/GLB with optional emissivity and color.

+++
