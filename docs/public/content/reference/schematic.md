+++
title = "Schematic KDL"
description = "Schematic KDL reference"
draft = false
weight = 106
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 6
+++

## Glossary

- Top-level nodes: `panel` variants, `object_3d`, `line_3d`, `vector_arrow`, `window`.
- EQL: expressions are evaluated in the runtime EQL context. Vector-like fields expect 3 components; `world_pos` is a 7-component array (quat + position).
- Colors: `color r g b [a]` or named (`black`, `white`, `blue`, `red`, `orange`, `yellow`, `yalk`, `pink`, `cyan`, `gray`, `green`, `mint`, `turquoise`, `slate`, `pumpkin`, `yolk`, `peach`, `reddish`, `hyperblue`); alpha optional. Colors can be inline or in `color`/`colour` child nodes. Defaults to white when omitted unless noted.

### window
- `path`/`file`/`name`: optional secondary schematic file. Relative paths resolve against the parent schematic directory (or CWD). If absent, the entry configures the primary window instead of loading a secondary file.
- `title`/`display`: optional window title.
- `screen`: optional zero-based display index.
- `rect { x y width height }`: optional child; four percentages (clamped 0–100). Used for placement of primary or secondary windows. Secondary windows default to `DEFAULT_SECONDARY_RECT` when unset.

### panel containers
- `tabs { ... }`: children are panels. No extra props.
- `hsplit` / `vsplit`: children are panels. Child `share=<f32>` controls the weight within the split. `active` (bool) is parsed but not currently used. Optional `name`.

### panel content
- `viewport`: `fov` (default 45.0), `active` (bool, default false), `show_grid` (default false), `show_arrows` (default true), `hdr` (default false), `name` (optional label), camera `pos`/`look_at` (optional EQL). Vector arrows can also be declared directly inside the viewport node; those arrows are treated as part of that viewport’s layer and respect its `show_arrows`/`show_grid` settings, allowing you to build a local triad tied to the viewport camera.
- `graph`: positional `eql` (required), `name` (optional), `type` (`line`/`point`/`bar`, default `line`), `lock` (default false), `auto_y_range` (default true), `y_min`/`y_max` (default `0.0..1.0`), child `color` nodes (optional list; otherwise palette).
- `component_monitor`: `component_name` (required).
- `action_pane`: positional `label` (required), `lua` script (required).
- `query_table`: positional `query` (defaults to empty), `type` (`eql` default, or `sql`).
- `query_plot`: positional `label` (required), `query` (required), `refresh_interval` in ms (default 1000), `auto_refresh` (default false), `color` (default white), `type` (`eql` default, or `sql`).
- `schematic_tree`: no properties.
- `dashboard`: layout node (Bevy UI style). Key properties: `label` (optional), `display` (`flex` default, or `grid`/`block`/`none`), `box_sizing` (`border-box` default or `content-box`), `position_type` (`relative` default or `absolute`), `overflow` (per-axis; defaults visible), `overflow_clip_margin` (visual_box + margin, defaults content-box / 0), sizing (`left`/`right`/`top`/`bottom`/`width`/`height`/`min_*`/`max_*` accept `auto`, `px`, `%`, `vw`, `vh`, `vmin`, `vmax`; default `auto`), `aspect_ratio` (optional f32), alignment (`align_items`/`justify_items`/`align_self`/`justify_self`/`align_content`/`justify_content`, all default to `default` variants), flex (`flex_direction`, `flex_wrap`, `flex_grow` default 0, `flex_shrink` default 1, `flex_basis` default `auto`, `row_gap`/`column_gap` default `auto`), `children` (nested dashboard nodes), colors via `bg`/`background` child (default transparent), `text` (optional), `font_size` (default 16), `text_color` child (default white), spacing via `margin`/`padding`/`border` children with `left`/`right`/`top`/`bottom`.

### object_3d
- Positional `eql`: required. Evaluated to a `world_pos`-like value to place the mesh.
- Mesh child (required, exactly one):
  - `glb`: `path` (required), `scale` (default 1.0), `translate` `(x,y,z)` (default 0s), `rotate` `(deg_x,deg_y,deg_z)` in degrees (default 0s).
  - `sphere`: `radius` (required); `color` (default white).
  - `box`: `x`, `y`, `z` (all required); `color` (default white).
  - `cylinder`: `radius`, `height` (both required); `color` (default white).
  - `plane`: `width`/`depth` (default `size` if set, else 10.0); optional `size` shorthand; `color` (default white).
  - `ellipsoid`: `scale` (EQL string, default `"(1, 1, 1)"`), `color` (default white).

  Mesh nodes support an optional `emissivity=<value>` property (0.0–1.0) to make the material glow (e.g., `sphere radius=0.2 emissivity=0.25 { color yellow }`).

### line_3d
- Positional `eql`: required; expects 3 values (or 7 where the last 3 are XYZ).
- `line_width`: default 1.0.
- `color`: default white.
- `perspective`: default true (set false for screen-space lines).

### vector_arrow
- `vector`: EQL expression yielding a 3-component vector (required).
- `origin`: EQL for arrow base; `world_pos` or 3-tuple (optional).
- `scale`: numeric multiplier (default 1.0).
- `normalize`: `#true`/`#false`; normalize before scaling (default false).
- `body_frame` / `in_body_frame`: apply origin rotation to the vector (default false).
- `color`: arrow color (default white).
- `name`: label text; used for legend/overlay (optional).
- `show_name`: show/hide overlay label (default true).
- `arrow_thickness`: numeric thickness multiplier with 3-decimal precision (default `0.1`).
- `label_position`: proportionately 0.0–1.0 along the arrow (0=base, 1=tip) for
   label anchor, or absolutely by specifying a number in a string with an 'm'
   suffix, .e.g., "0.3m" for 0.3 meters from origin (default "0.1m").

## Schema at a glance

Legend: parentheses group alternatives; `|` means “or”; square brackets `[...]` are optional; curly braces `{...}` repeat; `*` is zero-or-more, `+` is one-or-more; angle brackets `<...>` mark positional args.

```kdl
schematic =
  ( window
  | panel
  | object_3d
  | line_3d
  | vector_arrow
  )*

window = "window"
       [path|file|name=string]
       [title=string]
       [screen=int]
       [rect x y w h]

panel =
  viewport
  | graph
  | component_monitor
  | action_pane
  | query_table
  | query_plot
  | schematic_tree
  | dashboard
  | split
  | tabs

split = ("hsplit" | "vsplit")
      [active=bool]
      [name=string]
      { panel [share=float] }+

tabs = "tabs" { panel }+

viewport = "viewport"
         [fov=float]
         [active=bool]
         [show_grid=bool]
         [show_arrows=bool]
         [hdr=bool]
         [name=string]
         [pos=eql]
         [look_at=eql]
         { vector_arrow }

graph = "graph" eql
      [name=string]
      [type=line|point|bar]
      [lock=bool]
      [auto_y_range=bool]
      [y_min=float]
      [y_max=float]
      { color }*

component_monitor = "component_monitor"
                  [component_name=string]

action_pane = "action_pane"
            <label>
            [lua=string]

query_table = "query_table"
            [query=string]
            [type=eql|sql]

query_plot = "query_plot"
           <label>
           [query=string]
           [refresh_interval=ms]
           [auto_refresh=bool]
           [color]
           [type=eql|sql]

schematic_tree = "schematic_tree"
dashboard      = "dashboard" { dashboard_node }+

object_3d = "object_3d"
          <eql>
          { glb
          | sphere
          | box
          | cylinder
          | plane
          | ellipsoid
          }
          [emissivity=float]

line_3d = "line_3d"
        <eql>
        [line_width=float]
        [color]
        [perspective=bool]

vector_arrow = "vector_arrow"
             <vector-eql>
             [origin=eql]
             [scale=float]
             [normalize=bool]
             [body_frame|in_body_frame=bool]
             [color]
             [name=string]
             [show_name=bool]
             [arrow_thickness=float]
             [label_position=0..1]

color = "color"
      ( r g b [a]
      | name [alpha]
      )
```

## Examples

Minimal viewport + graph:

```kdl
viewport name="Main"
         fov=45.0
         active=#true
         show_grid=#true
         pos="drone.world_pos"
         look_at="(0, 0, 0)"
graph "drone.altitude"
      name="Altitude"
      auto_y_range=#true
```

Vector arrow with custom color and label:

```kdl
vector_arrow
  "drone.vel_x,drone.vel_y,drone.vel_z"
  origin="drone.world_pos"
  scale=1.5
  name="Velocity"
  normalize=#true
  body_frame=#true
  arrow_thickness=1.500
  label_position=0.9 {
  color 64 128 255
}
```
