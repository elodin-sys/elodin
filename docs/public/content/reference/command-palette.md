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

## Overview

The command palette provides fast access to common actions and creation flows. Commands can be
single-step (toggle, set) or multi-step (prompt for input, then execute). Creation commands target
the focused window; when launched from the "+" add-tab menu, they target that tab container.

## Usage

- Open: `Cmd+P` on macOS, `Ctrl+P` on Windows/Linux.
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
- `Create Action`: prompt for a label, then choose a `send_msg` preset or enter a custom Lua
  command to create an action pane.
- `Create Video Stream`: prompt for the message name, then create a video stream pane.
- `Create Hierarchy`: add a hierarchy pane.
- `Create Inspector`: add an inspector pane.
- `Create Schematic Tree`: add a schematic tree pane.
- `Create Dashboard`: add a dashboard pane.
- `Create Data Overview`: add a data overview pane.
- `Create 3D Object`: prompt for an EQL expression, pick a mesh type (GLTF or primitives), then
  optionally enter dimensions and color.

## Viewport

- `Toggle Wireframe`: enable/disable global wireframe.
- `Toggle HDR`: enable/disable HDR.
- `Toggle Grid`: show/hide the infinite grid.
- `Reset Cameras`: reset all viewports or pick a specific viewport to reset.

## Simulation

- `Toggle Recording`: start/stop recording on the connected database.

## Time

- `Set Playback Speed`: pick a preset playback speed.
- `Goto Tick...`: jump to a specific tick (pauses playback).
- `Fix Current Time Range`: lock the current selected range as fixed start/end.
- `Set Time Range`: set start/end offsets using `+`, `-`, or `=` formats (e.g. `+5m`, `-10s`,
  `=2023-01-01T00:00:00Z`).

## Presets

- `Save DB`: create a native DB snapshot (prompts for a directory name).
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
