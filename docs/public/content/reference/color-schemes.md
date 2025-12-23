+++
title = "Color Schemes"
description = "Theme presets and customization"
draft = false
weight = 107
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 7
+++

## Overview

Color schemes define the UI palette. Each preset has a required dark variant and an optional light
variant. Built-ins ship with the app and do not depend on your asset directory.

## Built-in presets

- `default` (dark + light)
- `eggplant` (dark + light)
- `catppuccini-macchiato` (dark + light)
- `catppuccini-mocha` (dark only)
- `catppuccini-latte` (light-preferred; dark uses mocha)
- `matrix` (dark + light)

## Registry and precedence

- Built-ins are always available.
- User presets are loaded from disk and merged with the built-ins.
- If a user preset reuses a built-in name, the user version replaces it.
- If the same user name appears in multiple locations, later sources win (data dir overrides assets).
- Dark is required; light is optional. If light is missing, the UI stays in dark mode and the
  palette marks the light variant as unavailable.
- Names are matched case-insensitively.

## File locations

- Assets: `color_schemes` inside the asset root (`$ELODIN_ASSETS_DIR` if set, otherwise `./assets`).
- User data: `color_schemes` under the app data directory (for example
  `~/Library/Application Support/systems.elodin.editor/color_schemes` on macOS,
  `~/.local/share/systems/elodin/editor/color_schemes` on Linux, or
  `%APPDATA%\\systems\\elodin\\editor\\color_schemes` on Windows). Entries here override assets
  when names match.

## File naming

- Flat files: `<preset>_dark.json` and `<preset>_light.json` directly inside `color_schemes/`.
- Subfolders: `color_schemes/<preset>/dark.json` and `color_schemes/<preset>/light.json`.
- Presets only load when a dark file exists; the light file is optional.

## JSON format

Each file encodes a `ColorScheme` as RGBA byte arrays:

```json
{
  "bg_primary": [31, 31, 31, 255],
  "bg_secondary": [22, 22, 22, 255],
  "text_primary": [255, 251, 240, 255],
  "text_secondary": [109, 109, 109, 255],
  "text_tertiary": [107, 107, 107, 255],
  "icon_primary": [255, 251, 240, 255],
  "icon_secondary": [98, 98, 98, 255],
  "border_primary": [46, 45, 44, 255],
  "highlight": [20, 95, 207, 255],
  "blue": [20, 95, 207, 255],
  "error": [233, 75, 20, 255],
  "success": [136, 222, 159, 255],
  "shadow": [0, 0, 0, 255]
}
```

## Using presets

- KDL: `theme { scheme="eggplant" mode="light" }` pulls from the merged registry. Unknown names
  fall back to `default`.
- Command Palette: `Set Color Scheme` lists every loaded preset; `Set Color Scheme Mode` switches
  between Dark/Light when the preset defines the requested mode.

## Persistence

The current selection is saved as `color_scheme.json` in the app data directory. This file stores
the scheme name, the mode, and the resolved colors so the UI can restore the last selection on
startup.
