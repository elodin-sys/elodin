# Impeller2 KDL Serdes

Serializer/deserializer for `impeller2_wkt::Schematic` to and from KDL.

## Scope

- Canonical user-facing schema docs: `docs/public/content/reference/schematic.md`
- This README is a code-facing quick reference of what the current parser/serializer supports.

## Top-Level Nodes

The parser accepts these root nodes:

- `theme`
- `window`
- Panel roots: `tabs`, `hsplit`, `vsplit`, `viewport`, `graph`, `component_monitor`, `action_pane`, `query_table`, `query_plot`, `inspector`, `hierarchy`, `schematic_tree`, `data_overview`, `dashboard`
- Scene roots: `object_3d`, `line_3d`, `vector_arrow`

## Panel Nodes

- `tabs { ... }`: container of panels.
- `hsplit` / `vsplit`: container of panels. Child `share=<f32>` controls split weight.
- `viewport`: camera panel (`fov`, `active`, `show_grid`, `show_arrows`, `show_view_cube`, `hdr`, `name`, `pos`, `look_at`) and optional local child `vector_arrow` nodes.
- `graph`: positional EQL string + optional `name`, `type` (`line|point|bar`), `lock`, `auto_y_range`, `y_min`, `y_max`, child `color` nodes.
- `component_monitor`: requires `component_name`; optional `name`.
- `action_pane`: requires `name` and `lua`.
- `query_table`: optional `name`, optional positional query string, optional `type` (`eql|sql`).
- `query_plot`: requires `name` and `query`; optional `refresh_interval` (ms), `auto_refresh`, `color`, `type` (`eql|sql`), `mode` (`timeseries|xy`), `x_label`, `y_label`.
- `inspector`, `hierarchy`: no properties.
- `schematic_tree`, `data_overview`: optional `name`.
- `dashboard`: Bevy UI-style panel tree (`dashboard { node ... }`).
- `video_stream`: panel form `video_stream <msg_name> [name=...]` is supported when nested inside panel containers.
  Top-level `video_stream` is currently not accepted by the root node parser.

## Viewport Flags

`viewport` supports these display toggles:

- `show_grid` (default `#false`)
- `show_arrows` (default `#true`)
- `show_view_cube` (default `#true`)

Example:

```kdl
viewport name="Fin Orientation"
         show_grid=#true
         show_view_cube=#false
         pos="drone.world_pos"
         look_at="drone.world_pos + (0,0,0,0, 1,0,0)"
```

## Scene Nodes

- `object_3d <eql> { ... }`: one mesh child is required.
  - `glb path=... [scale=1.0] [translate="(x,y,z)"] [rotate="(deg_x,deg_y,deg_z)"]`
  - `sphere radius=...`
  - `box x=... y=... z=...`
  - `cylinder radius=... height=...`
  - `plane [size=10.0] [width=size] [depth=size]`
  - `ellipsoid [scale="(1, 1, 1)"]`
  - mesh nodes support optional `color` and optional `emissivity` (clamped to `[0.0, 1.0]` on serialization).
- `line_3d <eql> [line_width=1.0] [color] [perspective=#true]`
- `vector_arrow <vector-eql> [origin] [scale=1.0] [name] [body_frame|in_body_frame=#false] [normalize=#false] [show_name|display_name=#true] [arrow_thickness=0.1] [label_position] [color]`

## Defaults And Aliases

- `window` parse aliases:
  - path: `path`, `file`, or `name`
  - title: `title` or `display`
- `vector_arrow` parse aliases:
  - `in_body_frame` -> `body_frame`
  - `display_name` -> `show_name`
- color child spelling accepts both `color` and `colour`
- `label_position` supports:
  - proportion in `[0, 1]` (number or string)
  - absolute string with meters suffix, for example `"0.30m"`

## Serialization Notes

- Many default scalar properties are omitted (for example `viewport fov=45.0`, `show_arrows=#true`).
- Several nodes always serialize explicit color children, including `line_3d`, `vector_arrow`, `query_plot`, and mesh materials.
