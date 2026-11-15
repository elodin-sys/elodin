+++
title = "Geographic overlays"
description = "Show place names, roads and geographic surfaces from recording assets."
weight = 105

[extra]
toc = true
order = 5
+++

Store geographic JSON files in the simulation's `assets/` folder and declare them
in its schematic:

```python
ui.schematic(
    ui.viewport(),
    map_overlays=[ui.map_overlay("overlays/places.json")],
)
```

Equivalent KDL:

```kdl
map_overlay path="overlays/places.json" visible=#true
viewport
```

- Assets are ingested into the DB and included by `elodin upload`.
- The editor loads overlays through the same asset connection as models and terrain.
- **Overlays on/off** appears when at least one declared asset has loaded with content.
- Missing, invalid, empty or unreferenced JSON files do not create a button.
- Use `visible=False` in Python (`visible=#false` in KDL) to start with an overlay hidden. The button changes runtime visibility only.
- Source attribution links are available in the button's tooltip.
- Overlays work in native and WASM editors, independently of the terrain provider.

## JSON format

Coordinates are latitude and longitude in degrees, followed by altitude in metres
above the WGS84 ellipsoid. The file can contain any combination of labels, lines
and triangle surfaces:

```json
{
  "source": "My geographic dataset",
  "source_url": "https://example.com/dataset",
  "labels": [["Launch site", 28.6084, -80.6043, 10]],
  "water_names": [],
  "lines": [{"kind": "road", "points": [[28.6084, -80.6043, 10], [28.61, -80.60, 10]]}],
  "areas": [{"triangles": [[[28.60, -80.60, 12], [28.61, -80.60, 12], [28.60, -80.61, 12]]]}]
}
```

- Named `places` objects with `name`, `latitude`, `longitude` and `altitude` are also accepted.
- `water_names` gives matching labels a water color; `excluded_names` omits matching labels.
- `sources` may contain additional attribution URLs.
- Line kinds are `road`, `water` and `airport`. Other kinds use the airport style.
- Labels avoid overlapping each other and are tested against scene geometry for occlusion.
- Coordinates are fixed geographic anchors; terrain does not automatically adjust their heights.
- Limits per asset: 16 MiB of JSON, 512 labels and 100,000 geometry vertices.
