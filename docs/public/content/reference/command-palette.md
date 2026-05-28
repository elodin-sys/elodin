+++
title = "Command Palette"
description = "Command palette reference"
draft = false
weight = 108
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 8
+++

## Usage

- Open (fresh start): `Cmd+P` on macOS, `Ctrl+P` on Windows/Linux. This clears any existing
  palette nesting and starts at the top level.
- Open (resume): `Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux. This reopens the
  palette with any existing nested state.
- Type to filter; use Up/Down arrows to move selection.
- Enter to run the selected command; Escape to close.
- Backspace on an empty input returns to the previous page when available.

## Create

- `Create Window`: prompt for a title, then opens a new secondary window.
- `Create Viewport`: add a viewport pane to the current container.
- `Create Graph`: pick a component and create a graph pane for it.
- `Create Monitor`: pick a component and create a component monitor pane.
- `Create Query Table`: add an empty query table pane.
- `Create Query Plot`: add an empty query plot pane.
- `Create Action`: prompt for a name, then choose a `send_msg` preset or enter a custom Lua
  command to create an action pane.
- `Create Video Stream`: prompt for the message name, then create a video stream pane.
- `Create Schematic Tree`: add a schematic tree pane.
- `Create Data Overview`: add a data overview pane.
- `Create 3D Object`: prompt for an EQL expression, pick a mesh type (GLTF or primitives), then
  optionally enter dimensions and color.
- `Hierarchy` and `Inspector` are built-in sidebars; they are always present and do not appear in
  the command palette.

## Viewport

- `Toggle Wireframe`: enable/disable global wireframe.
- `Toggle HDR`: enable/disable HDR.
- `Toggle Grid`: show/hide the infinite grid.
- `Reset Cameras`: reset all viewports or pick a specific viewport to reset.

## Skybox

- `Skybox...`: submenu for skybox actions against `assets/skyboxes/manifest.ron` or
  `$ELODIN_ASSETS_DIR/skyboxes/manifest.ron`.
  - `Generate Skybox...`: prompt for a description, generate a new skybox via Blockade, add it to
    the manifest, and activate it.
  - `Clear Skybox`: shown when a skybox is active; removes it from viewports and sensor cameras and
    removes the top-level `skybox` node from the current schematic.
  - `Revert to …`: shown after AI generation replaced an active skybox.
  - Cached manifest entries: activate a preset skybox (`name (active)` marks the current one).

Selecting a cached skybox sets `skybox name="..."` on the current schematic and applies it
immediately. Use **Save Schematic** or **Save Schematic To DB** to persist changes.

Skybox generation requires `BLOCKADE_API_KEY` in the editor environment. Generated cubemaps are
resampled locally to the configured face size (default 2048 px per face); Blockade always returns an
~8K equirect source, so higher local presets upscale rather than fetching a higher remote tier.
Generated assets are written next to the manifest used by the editor.

## Simulation

- `Toggle Recording`: start/stop recording on the connected database.

## Time

- `Set Playback Speed`: pick a preset playback speed.
- `Goto Tick...`: jump to a specific tick (pauses playback).
- `Fix Current Time Range`: lock the current selected range as fixed start/end.
- `Set Time Range`: set start/end offsets using `+`, `-`, or `=` formats (e.g. `+5m`, `-10s`,
  `=2023-01-01T00:00:00Z`).

## Presets

- `Save Schematic`: save to the current schematic path (if set).
- `Save Schematic As...`: save to a new schematic file name.
- `Save Schematic To DB`: write the current schematic to DB metadata.
- `Load Schematic`: load a schematic from the default directory or via a file dialog.
- `Clear Schematic`: clear the current schematic.
- `Set Color Scheme Mode`: switch between Dark/Light if the scheme supports it.
- `Set Color Scheme`: select a color scheme preset.

## Help

- `Documentation`: opens the docs site in a browser.
- `Release Notes`: opens the changelog in a browser.
