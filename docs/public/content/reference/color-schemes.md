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

## Presets

- Built-ins: `default`, `eggplant`, `catppuccini-macchiato`, `catppuccini-mocha`, `catppuccini-latte`, and `matrix` ship with the app; most have dark and light variants, while mocha is dark-only and latte is light-preferred.
- Registry merge: user presets are loaded from disk and merged with the built-ins; if a user preset reuses a built-in name, the user version replaces it.
- Modes: dark is required; light is optional. When a light file is missing, the UI stays in dark mode and the palette entry shows the mode as unavailable.

## File locations

- Assets: `color_schemes` inside the asset root (`$ELODIN_ASSETS_DIR` if set, otherwise `./assets`).
- User data: `color_schemes` under the app data directory (for example `~/Library/Application Support/systems.elodin.editor/color_schemes` on macOS, `~/.local/share/systems/elodin/editor/color_schemes` on Linux, or `%APPDATA%\\systems\\elodin\\editor\\color_schemes` on Windows). Entries here override assets when names match.

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

- KDL: `theme { scheme="eggplant" mode="light" }` pulls from the merged registry; unknown names fall back to `default`.
- Command Palette: `Set Color Scheme` lists every loaded preset; `Set Color Scheme Mode` switches between Dark/Light when the preset defines the requested mode.
