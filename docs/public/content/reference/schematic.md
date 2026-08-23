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

- Top-level nodes: `coordinate`, `theme`, `timeline`, `telemetry_mode`, `skybox`, `environment`, `panel` variants, `object_3d`, `line_3d`, `vector_arrow`, `world_mesh`, `window`.
- EQL: expressions are evaluated in the runtime EQL context. Vector-like fields expect 3 components; `world_pos` is a 7-component array (quat + position).
- Colors: `color r g b [a]` or named (`black`, `white`, `blue`, `red`, `orange`, `yellow`, `yalk`, `pink`, `cyan`, `gray`, `green`, `mint`, `turquoise`, `slate`, `pumpkin`, `yolk`, `peach`, `reddish`, `hyperblue`); alpha optional. Colors can be inline or in `color`/`colour` child nodes. Defaults to white when omitted unless noted.
- Booleans: KDL booleans are `#true`/`#false`. A bare `True` is a *string*, not a boolean — most flags silently fall back to their default if given one. Viewport flags (`hdr`, `show_grid`, `active`, ...) leniently accept `True`/`"true"` (case-insensitive), but prefer the `#` forms everywhere.
- Coordinate frames: `ENU` (East-North-Up), `NED` (North-East-Down), `ECEF` (Earth-Centered Earth-Fixed). Bevy uses a Y-up right-handed system; the `frame` attribute handles the conversion.

### coordinate
- Optional top-level node that sets the global coordinate frame for the schematic.
- `frame`: `"ENU"` (default), `"NED"`, or `"ECEF"`.
- Elements (`viewport`, `object_3d`, `line_3d`, `vector_arrow`, `world_mesh`) that don't specify their own `frame` attribute inherit this global frame.
- Example: `coordinate frame="NED"` sets the entire schematic to use North-East-Down coordinates.

### theme
- Optional top-level node that sets the session UI appearance.
- `mode`: `"dark"` (default) or `"light"`; drives window decorations and picks the dark/light variant of the color scheme. If a preset does not ship a light variant, the theme stays in dark.
- `scheme`: name of a color preset. Built-ins are `default`, `eggplant`, `catppuccini-macchiato`, `catppuccini-mocha`, `catppuccini-latte`, and `matrix`; user presets are picked up from any `color_schemes` folder in the asset directory or data directory. Unknown names fall back to `default`. If a user preset shares a name with a built-in, the user version wins. See [color-schemes](/reference/color-schemes) for the file layout.
- Applies to the whole session; a secondary file can set its own `mode` for its windows, but the active scheme stays the one from the primary schematic.
- Controls both egui styling (palette) and the window decoration theme (Dark/Light).

### timeline
- Optional top-level node that configures playback globally for the editor session.
- `played_color`: named color or tuple string, default `yellow`. Used by the LIVE badge, the timeline playhead cursor, and the played segment of 3D trails.
- `future_color`: named color or tuple string, default `white`. Used by the timeline latest/end cursor and the 3D trail segment that lies ahead of the current playback position.
- `follow_latest`: bool, default `#false`. When omitted, the editor keeps the default start-from-beginning playback behavior. Set it to `#true` to switch to LIVE automatically once the connected stream proves that it is still advancing.
- `range`: optional string preset for the visible time window on load. Accepted values: `full`, `last_5s`, `last_15s`, `last_30s`, `last_1m`, `last_5m`, `last_15m`, `last_30m`, `last_1h`, `last_6h`, `last_12h`, `last_24h`, or `last_<N>s` / `<N>s` for a custom trailing duration. When omitted, the editor uses full range. Trailing presets (`last_*`) always end at the playhead (`min(LastUpdated, CurrentTimestamp)`), so recordings show the last N seconds of what you are watching. `full` still spans the whole database in normal mode; `--replay` separately progressive-reveals the timeline bar as the playhead advances.
- Applies to the primary schematic only; secondary schematic files do not override the global timeline behavior.
- These settings are also editable from the Timeline inspector, opened with the gear button in the timeline bar.

### telemetry_mode
- Optional top-level boolean node: `telemetry_mode #true`.
- When enabled, graphs load locked (X-synced), singleton graph tab chrome is stripped, plot padlocks are replaced by a muted overlay title, and locked graphs hide X-axis tick labels (notches remain). Viewport multi-tabs keep their titles.
- Unlock individual graphs from the graph inspector **Lock** checkbox.
- Serialized only when `#true`.

### skybox
- Optional top-level node that activates a cached skybox by manifest name.
- `name`: required manifest entry name. Entries are read from `assets/skyboxes/manifest.ron`, or from `$ELODIN_ASSETS/skyboxes/manifest.ron` when that environment variable is set.
- When a simulation records into an Elodin DB, the manifest and active cubemap are copied under `{db}/assets/skyboxes/`; see [DB Asset Server](/reference/db-asset-server).
- Applies to the whole schematic: editor viewports and sensor cameras use the same active skybox. Overlay cameras such as the ViewCube keep the normal dark/light UI background.
- Example: `skybox name="desert_night"`.

### environment
- Optional top-level node for cinematic scene lighting. Without it the editor renders as always (baked image-based lighting, no sun, theme background).
- `sun` child (optional): spawns a real directional sun with a visible sun disk. All placement attributes are optional. A bare `sun` — or `earth` with no `sun` at all — tracks the real sun for the playhead's UTC timestamp (live, replay, and scrubbing; ~0.05°). Most explicit wins:
  - `direction`: tuple string, direction *toward* the sun in the schematic coordinate frame (e.g. ECEF). Pins the sun. Example: `direction="(1, 0, 0)"` puts local noon on the prime-meridian equator.
  - `azimuth` / `elevation` (degrees): sun direction in the rendered Y-up world frame. These transcribe 1:1 from pyrotechnique scene values. Setting either one pins the sun; a missing angle uses the old defaults (`azimuth=0`, `elevation=45`).
  - `illuminance`: lux, default `100000` (direct sunlight). Pair with a viewport `ev100` — at the editor's default exposure a full sun blows out the frame.
  - `shadows`: `#true`/`#false`, default `#true`. Enables shadow maps on the sun.
- `ambient` child (optional): `scale` (required) multiplies the editor's baked image-based lighting intensity. `1.0` = unchanged; near-zero (e.g. `0.02`) keeps shadows black on airless bodies.
- `sky` child (optional): `color` (named color or tuple string) sets the main viewport clear color — `"black"` for space scenes. Omit to keep the theme background. An active `skybox` draws over it. With `earth`, `sky color` instead renders as an unlit dome that only regular viewports see (a flat daytime sky, e.g. `"(135,206,235)"`); the cinematic viewport keeps its atmosphere over black space, and pane backgrounds outside the 3D scene keep the theme.
- `atmosphere` child (optional): Bevy's procedural planetary atmosphere (earth scattering medium) — physical blue sky, horizon haze, aerial perspective, altitude darkening, and the limb from space. Requires an active `sun` and an HDR viewport with a sunlit `ev100` (~13-15).
  - `origin`: tuple string, planet center in the schematic coordinate frame [m]. Default `(0, 0, 0)` — the Earth's center in an ECEF scene. In local ENU/NED scenes leave it at the origin and the default radii place the surface `inner_radius` below.
  - `inner_radius` / `outer_radius`: meters from the planet center; default Bevy's earth values (`6360000` / `6460000`). In an ECEF scene set `inner_radius` to the launch site's geocentric radius (the WGS84 radius varies with latitude) so the horizon haze sits at the actual local surface.
  - `ground_albedo`: tuple string, average surface color for multiscattering. Default `(0.3, 0.3, 0.3)`.
  - Sun, atmosphere entity, ambient IBL scale, and sky clear color are scene-global. Bevy still opts each camera into atmosphere rendering via `AtmosphereSettings`; Elodin attaches that to **one** active main viewport at a time (Bevy 0.19 fatally fails wgpu validation when several views carry it). The sky hands off when you switch tabs; a side-by-side multi-viewport layout gets the procedural sky in one pane only (warn once). The atmosphere entity is grid-cell anchored so it survives floating-origin rebases.
- `earth` child (optional): the editor's built-in cinematic Earth for **ECEF** scenes. One word spawns everything from assets embedded in the editor binary: a true-scale WGS84 globe at the ECEF origin (8K day/night/cloud textures), a raymarched atmosphere, the Milky Way skybox, a GPU-particle star field, night city lights, and green/red airglow shells, plus earthshine on the craft. Haze density, star/skybox visibility, city-light intensity, and airglow follow the cinematic camera and sun, so one schematic works from the launch pad to LEO.
  - Requires `coordinate frame="ECEF"` and exactly one `viewport cinematic=#true`. That camera owns HDR, fixed exposure, atmosphere, the Milky Way skybox, the schematic sun, and the Earth render layer (globe, stars, city lights, airglow, and earthshine). Regular viewports keep the editor's IBL, clear color (or `sky color`), and do not see Earth visuals or the sun — at their LDR default exposure a 100 klx sun would blow every sunlit surface out to white. A `sun` is implied when omitted (ephemeris, 100 klx, shadows).
  - Keep `show_view_cube=#true` (the default) on the cinematic viewport: the pane currently renders black without its view-cube overlay camera.
  - Parse errors (not warnings): more than one `cinematic=#true` viewport; any `hdr=#true` viewport in the same schematic as a cinematic viewport; `earth` without a cinematic viewport; `cinematic=#true` without `earth`. Validation is per KDL document (multi-window schematics are not cross-checked).
  - Exposure is fixed at the viewport's authored `ev100` (cinematic default `13.5`). The night look comes from range-limited earthshine and the house OLD SCHOOL bloom (intensity `0.365`) over a dim night map.
  - Optional children tune the particle fields. Omit a child or property to keep the house look. Density is clamped to `0.05…2.0`. The cinematic viewport inspector edits the same knobs live (density/brightness apply on slider release; size and city height apply immediately).
    - `stars`: `density` (scale, default `0.05`; 800k dim + 40k bright + 400k milky way at 100%), `size` (scale, default `0.40`), `brightness` (scale, default `1.88`; rebuilds the HDR gradient).
    - `city_lights`: `density` (scale, default `0.05`; 1.5M at 100%), `size` (screen pixels, default `1.0`), `height` (metres above the ellipsoid, default `0`, clamped `0…100000`), `brightness` (scale, default `0.05`). The house look keeps these minimal and lets the night map plus bloom carry the night side.
    - `airglow`: `density` (scale, default `0.55`; 520k green + 340k red at 100%), `size` (scale, default `1.05`), `brightness` (scale, default `1.45`).
    - `night_map`: `brightness` (scale on the globe's emissive night texture, default `0.05`, clamped `0…4`). This is the continuous sheet under the city-light particles — `0` leaves only particles.
  - Supersedes the `atmosphere` child — the built-in Earth owns its own atmosphere (raymarched, surface radius tracked from the camera's latitude). Remove `atmosphere` when enabling `earth` (it is ignored with a warning). Remove any `object_3d` Earth GLB and `world_mesh "globe"` too; the built-in globe replaces them.
  - Implementation note: the atmosphere is parented to the rotated Earth root, so its local rotation cancels the parent's rotation. This keeps Bevy's atmosphere frame globally unrotated and its limb concentric with the globe.
  - Native editor builds only (the embedded Earth assets and GPU particles are excluded from wasm).
- Example (lunar scene):

```kdl
environment {
    sun azimuth=320.0 elevation=32.0 illuminance=130000.0 shadows=#true
    ambient scale=0.02
    sky color="black"
}
```

- Example (Earth launch site, ECEF):

```kdl
environment {
    sun azimuth=146.6 elevation=55.4 illuminance=100000.0 shadows=#true
    ambient scale=0.05
    atmosphere origin="(0, 0, 0)" inner_radius=6373250.0 outer_radius=6473250.0 ground_albedo="(0.20, 0.22, 0.18)"
}
```

- Example (built-in cinematic Earth, pad to LEO):

```kdl
coordinate frame="ECEF"
environment {
    ambient scale=0.05
    earth
}
viewport name="Chase" cinematic=#true
```

The playhead's UTC timestamp aims the sun. Pin it with `sun direction="(x, y, z)"` or `sun azimuth=… elevation=…` if the lighting should stay put.

### window
- `path`/`file`/`name`: optional secondary schematic file. Relative paths resolve against the parent schematic directory (or CWD). If absent, the entry configures the primary window instead of loading a secondary file.
- `title`/`display`: optional window title.
- `screen`: optional zero-based display index.
- `rect { x y width height }`: optional child; four percentages (clamped 0–100). Used for placement of primary or secondary windows. Secondary windows default to `DEFAULT_SECONDARY_RECT` when unset.

### panel containers
- `tabs { ... }`: children are panels. No extra props.
- `hsplit` / `vsplit`: children are panels. Child `share=<f32>` controls the weight within the split. `active` (bool) is parsed but not currently used. Optional `name`.

### panel content
- `viewport`: `fov` (default 45.0), optional `near`/`far` frustum-presentation distances (defaults `near=0.05`, `far=5.0`; used by `create_frustum`, coverage, and projection overlays; `near` additionally pins the real camera near clip, while perspective rendering and zoom have no far limit), optional `aspect` (if omitted, ratio is derived from viewport size), `active` (bool, default false), `show_grid` (default false), `show_arrows` (default true), `create_frustum` (default false; creates that viewport camera frustum), `show_frustums` (default false; shows frustums created by other viewports on this viewport), `frustums_color` (default `yellow`), `projection_color` (default `white`; colors this viewport's source frustum 2D projection in target viewports), `frustums_thickness` (default `0.006` world units), `show_view_cube` (default true), `view_cube_frame` (optional; `ENU`, `NED`, or `ECEF`; selects which shared ViewCube mesh/labels this viewport shows — falls back to the viewport's `frame`, then ENU), `effects` (bool, default true; when true the viewport camera includes the thruster-particle render layer so KDL `thruster` jets are visible — set `effects=#false` to hide particles in that pane without cloning emitters), `hdr` (default false; enables the HDR render path and is required for bloom; mutually exclusive with `cinematic`), `cinematic` (default false; owns the cinematic Earth camera: inserts HDR, zeros Bevy ambient light, defaults `ev100` to 13.5 and bloom to the old-school preset at intensity 0.365, attaches atmosphere and the Milky Way skybox, and includes the Earth render layer — required when `environment { earth }` is set, forbidden alongside any `hdr=#true` viewport, and at most one per schematic), `ev100` (optional camera exposure in EV100; sunny daylight is ~13-15 — required to balance an [`environment`](#environment) sun, since the default physical-camera exposure is ~EV 8.6), `name` (optional label), `frame` (optional; `ENU`, `NED`, or `ECEF`; inherits from global `coordinate` if omitted), camera `pos` (optional EQL; a 7-vector pose, since a viewport without `look_at` rides its attitude), `look_at` (optional EQL; a bare 3-vector position or a 7-vector pose whose tail 3 elements are the position — aiming only needs a point), `smoothing` (optional, default 0; low-pass time constant in seconds for the follow camera — with `0` the camera pose is applied straight from telemetry, with e.g. `smoothing=0.3` it eases toward its `pos`/`look_at` targets, which cancels high-frequency noise from jittery data; the camera still snaps instantly on seeks or jumps larger than the framing distance, so genuine motion has no lag, and the lag is bounded to a few percent of the framing distance, which also makes the filter fade out as the subject speeds up — full strength while it is slow or parked, all but transparent in flight, so a climbing subject stays centred instead of sliding toward the edge of frame). Note that the smoothed camera steadies the *world* in frame — the followed entity is still drawn from raw telemetry, so its own noise becomes visible against that steady background rather than disappearing. Smoothing relocates the shake rather than removing it, and that trade is what makes a `line_3d` trail readable: the trail is a *history* of world positions while the camera follows the *newest* sample, so only its tip shares the camera's noise. A jittery follow input therefore slides the entire trail, the grid and the terrain across the screen every frame even though their geometry is exactly right, and smoothing is the only thing that can stop that sliding — it never alters the trail's shape, which is data. Measured on a degraded-sensor recording at 35 m framing, static world geometry travels ~23 px per frame at `smoothing=0` versus ~1 px at `smoothing=1` while the vehicle is parked, in exchange for ~1.3 degrees of wobble on the followed entity — which is just the sensor noise over the framing distance, and is therefore the floor rather than a tuning artifact. Once it is flying the filter has largely faded out by design, so the two settings converge: steadying the background there would cost framing accuracy, and a subject drifting off centre is far more objectionable than shimmer in a scene that is already streaming past. With an already-clean follow input it does nothing, which is why the default is `0`. Vector arrows can also be declared directly inside the viewport node; those arrows are treated as part of that viewport’s layer and respect its `show_arrows`/`show_grid` settings, allowing you to build a local triad tied to the viewport camera. An `up` (default depends on frame: `(0,0,1)` for ENU, `(0,0,-1)` for NED) specifies a direction vector in the frame coordinates for the camera. When `frame` is set, the ViewCube and grid axis colors adjust to match the coordinate system (e.g., NED swaps X/Z axis colors) unless `view_cube_frame` overrides the cube. An optional `bloom` child node tunes the glow post-process — see [viewport bloom](#viewport-bloom).
  When `near` is omitted, the camera controller derives it from the current anchor distance and the inspector shows that live value; editing it pins the displayed value.
- `graph`: positional `eql` (required), `name` (optional), `type` (`line`/`point`/`bar`, default `line`), `lock` (default false), `auto_y_range` (default true), `y_min`/`y_max` (default `0.0..1.0`), child `color` nodes (optional list; otherwise palette).
- `component_monitor`: `component_name` (required), `name` (optional).
- `geo_position_gauge`: positional `eql` (required; a 3-vector position, or a 7-vector pose whose tail 3 elements are the position; named `eql=` also accepted), `name` (optional), `source` (`ECEF`/`NED`/`ENU`; inherits from global `coordinate` if omitted, else `ENU`), `display` (`ECEF`/`NED`/`ENU`/`LLA`, default `NED`). Reads a position in `source` and shows it converted to `display` as three labelled values.
- `orientation_gauge`: positional `eql` (required; a 4-vector `[x,y,z,w]` quaternion, or a 7-vector pose whose head 4 elements are the quaternion; named `eql=` also accepted), `name` (optional), `source` (`ECEF`/`NED`/`ENU`; inherits from global `coordinate` if omitted, else `ENU`), `display` (`ECEF`/`NED`/`ENU`, default `NED`; the triad the gimbal draws), optional `reference x y z w` child (body→`source` quaternion shown as neutral by the gimbal; identity when omitted). Renders the attitude as a 3D gimbal.
- `action_pane`: `name` (required pane title), `lua` script (required).
- `query_table`: `name` (optional), positional `query` (defaults to empty), `type` (`eql` default, or `sql`).
- `query_plot`: `name` (required pane title), `query` (required), `refresh_interval` in ms (default 1000), `auto_refresh` (default false), `color` (default white), `type` (`eql` default, or `sql`), `mode` (`timeseries` default, or `xy` for numeric X-axis labels), `x_label` (optional X-axis label for XY mode), `y_label` (optional Y-axis label).
- `data_overview`: `name` (optional pane title).
- `schematic_tree`: `name` (optional pane title). (Hierarchy/Inspector sidebars are implicit and not serialized.)
- `sensor_view`: positional `camera_name` (required; the full sensor camera name in `"entity.camera"` format, e.g., `"drone.scene_cam"`), `name` (optional display label). Displays raw RGBA frames from a sensor camera registered via `world.sensor_camera()`. Frames are rendered continuously by the headless render server at the camera's configured `fps` and pushed to the DB; the panel reads them by timestamp. Unlike `video_stream` (which decodes H.264), `sensor_view` displays raw pixel data directly — no codec involved. Sensor cameras registered with `create_frustum=True` are also available as frustum sources in 3D viewports with `show_frustums=#true`, including coverage and 2D projection overlays. Ellipsoid debug objects are hidden from sensor cameras by default; pass `show_ellipsoids=True` to `world.sensor_camera()` to render them.
- `video_stream`: positional `msg_name` (required; the message name matching the `elodinsink` `msg-name` property), `name` (optional display label; defaults to `"Video Stream <msg_name>"`). Displays an H.264 video stream received by Elodin DB. The video source can be a GStreamer pipeline using `elodinsink`, an OBS Studio SRT stream via a receiver pipeline, or any source that sends H.264 NAL units to Elodin DB. See the [OBS Studio Integration](#obs-studio-integration) section below.
- `dashboard`: layout node (Bevy UI style). Key properties: `name` (optional), `display` (`flex` default, or `grid`/`block`/`none`), `box_sizing` (`border-box` default or `content-box`), `position_type` (`relative` default or `absolute`), `overflow` (per-axis; defaults visible), `overflow_clip_margin` (visual_box + margin, defaults content-box / 0), sizing (`left`/`right`/`top`/`bottom`/`width`/`height`/`min_*`/`max_*` accept `auto`, `px`, `%`, `vw`, `vh`, `vmin`, `vmax`; default `auto`), `aspect_ratio` (optional f32), alignment (`align_items`/`justify_items`/`align_self`/`justify_self`/`align_content`/`justify_content`, all default to `default` variants), flex (`flex_direction`, `flex_wrap`, `flex_grow` default 0, `flex_shrink` default 1, `flex_basis` default `auto`, `row_gap`/`column_gap` default `auto`), `children` (nested dashboard nodes), colors via `bg`/`background` child (default transparent), `text` (optional), `font_size` (default 16), `text_color` child (default white), spacing via `margin`/`padding`/`border` children with `left`/`right`/`top`/`bottom`.

### viewport bloom
- Optional `bloom` child of `viewport`. Spreads pixels brighter than 1.0 ("white") into a soft halo — the glow effect for emissive/`glow` materials. Requires HDR (`hdr=#true` or `cinematic=#true`); without HDR the bloom pass does not run.
- `preset`: starting point for all values.
  - `natural` (default): energy-conserving mix, no threshold. The whole frame gets a subtle filmic softness; very bright pixels halo. Redistributes light, never adds it.
  - `old_school`: additive with a threshold — only hot pixels glow, everything else stays crisp. Adds light. Defaults: intensity `0.05`, threshold `0.6`, softness `0.2`.
- `intensity`: halo strength (0–1). `natural` default `0.15`. Above ~0.5 reads as fog.
- `threshold`: brightness cutoff in multiples of white; only the excess above it blooms. Set just above your lit scene whites (~1.0–1.2) so geometry stays crisp and only `glow`/emissive surfaces halo.
- `threshold_softness`: 0–1 knee blend. ~0.25+ avoids popping as moving pixels cross the threshold.
- Unset properties inherit the preset; omitting the node entirely uses `natural` defaults.
- Tuning model: halo energy ≈ `(pixel luminance − threshold) × intensity`. Widen the gap (raise the material's `glow`, or lower `threshold`), then scale `intensity` to taste.

```kdl
viewport hdr=#true {
    bloom preset="old_school" intensity=0.35 threshold=0.65 threshold_softness=0.6
}
```

### object_3d
- Positional `eql`: required. Evaluated to a `world_pos`-like value to place the mesh.
- `frame`: optional; `ENU`, `NED`, or `ECEF`. Specifies the coordinate frame for interpreting position. Inherits from global `coordinate` if omitted.
- `frame_orientation`: optional; `ENU`, `NED`, or `ECEF`. Coordinate frame used for orientation (`GeoRotation`). Falls back to `frame`, then the global default, when omitted.
- `orientation`: optional; `relative` (default) or `absolute`. Both compose the object's attitude with the frame→Bevy basis (`bevy_R * att`). An identity attitude therefore aligns mesh local axes with the schematic frame (ENU: +X east, +Y north, +Z up) — not with the editor's Bevy Y-up axes. Use `absolute` when you want to be explicit about frame-aligned geometry (compasses, multi-frame scenes).
- Mesh child (required, exactly one):
  - `glb`: `path` (required), `scale` (default 1.0), `translate` `(x,y,z)` (default 0s), `rotate` `(deg_x,deg_y,deg_z)` in degrees (default 0s). On DB record, local paths are stored as `db:…` and served over HTTP on replay; see [DB Asset Server](/reference/db-asset-server). Material overrides (both open-ended strengths, default 0.0 = use the GLB's own materials):
    - `emissivity`: brightens the surface — boosts the model's emissive by `4 × emissivity`, modulated by its base-color texture so the pattern still shows.
    - `glow` + `glow_color` (named color or tuple string, default white): view-dependent fresnel rim, in multiples of white — lights the silhouette so bloom can halo it. Only reads as a glow on an `hdr=#true` viewport; pair with [viewport bloom](#viewport-bloom) and keep `glow` above the bloom `threshold`.
    - `animate` child nodes (optional, multiple): For rigged GLB models, animate specific joints/bones.
      - `joint`: required string; the exact name of the joint/bone in the GLB file.
      - `rotation_vector`: required EQL expression; must evaluate to a 3-element vector `(x, y, z)` where:
        - The vector direction is the rotation axis.
        - The vector magnitude is the rotation angle in degrees.
      - Example: `animate joint="Root.Fin_0" rotation_vector="(0, rocket.fin_deflect, 0)"`
  - `sphere`: `radius` (required); `color` (default white).
  - `box`: `x`, `y`, `z` (all required); `color` (default white).
  - `cylinder`: `radius`, `height` (both required); `color` (default white).
  - `plane`: `width`/`depth` (default `size` if set, else 10.0); optional `size` shorthand; `color` (default white). Horizontal ground rectangle in the object frame (XY, normal +Z / up).
  - `ellipsoid`: `color` (default white), `show_grid` (default `#false`). Renders in regular viewports only — the cinematic viewport hides ellipsoids (like sensor cameras without `show_ellipsoids`).
    - Physical measure
      - `scale`: runtime EQL string that must evaluate to at least 3 values in meters. It can be a literal tuple, a vector component, indexed component values, scalar-vector math, or values from multiple components.
        - Examples:
          ```kdl
          ellipsoid scale="(1, 1, 1)"
          ellipsoid scale="vehicle.pos_std_var"
          ellipsoid scale="2.0 * vehicle.pos_std_var"
          ellipsoid scale="1.91*(vehicle.pos_std_var[1], vehicle.pos_std_var[0], vehicle.pos_std_var[2])"
          ellipsoid scale="3.0 * (vehicle.pos_std_var[1], 0.0, target.pos_std_var[2])"
          ```
    - Error measure
      - `error_covariance_cholesky` as an alternative to specifying the scale
        and rotation, one can specify the lower triangle cholesky L of the
        error covariance matrix P = LL^T. Example `"(a,b,c,d,e,f)"` which
        describes a matrix that looks like this:
        | a 0 0 |
        | b c 0 |
        | d e f |
      - `error_covariance` as an alternative to Cholesky, specify the
        symmetric covariance matrix P directly as a 6-pack. Example
        `"(a,b,c,d,e,f)"` which describes:
        | a b c |
        | b d e |
        | c e f |
        The editor Cholesky-decomposes P at runtime. When both
        `error_covariance_cholesky` and `error_covariance` are set, Cholesky
        takes precedence based on field presence (even if the Cholesky
        expression fails to compile).
      - `error_confidence_interval` (default `70`) the percentage that if this
        were repeated 100 times, we would expect that in 70 cases, the true
        value would be within the bounds. In practice this means that the
        larger the error confidence interval, the larger the ellipsoid.

  Non-GLB mesh nodes support an optional `emissivity=<value>` property (0.0–1.0) to make the material glow (e.g., `sphere radius=0.2 emissivity=0.25 { color yellow }`).

  Mesh nodes and `icon` nodes both support an optional `visibility_range` child node that controls at what camera distances the element is rendered:
  - `visibility_range`: child node with `min` (default 0) and `max` (default infinity) properties. The element is visible when the camera distance is between `min` and `max`. Both mesh and icon are visible at all distances by default; `visibility_range` is purely opt-in.
  - `fade_distance` (icon only, default 0): world-unit distance over which the icon fades in at the `min` boundary and fades out at the `max` boundary. For example, `min=50 fade_distance=50` means the icon starts appearing at distance 50 (alpha=0) and reaches full opacity at distance 100. Mesh nodes use hard visibility cutoffs at their `min`/`max` boundaries.
  - Ranges can overlap (both mesh and icon visible simultaneously) or have gaps.
- `icon` child (optional): Displays a fixed-size billboard icon at the object's position. Each viewport camera independently evaluates whether to show the icon based on its own distance. The icon always faces the camera and maintains a constant screen pixel size.
  - Source (exactly one required):
    - `builtin`: name of a [Material Icons](https://fonts.google.com/icons?icon.set=Material+Icons) glyph (snake_case). Supported names include: `satellite_alt`, `satellite`, `rocket_launch`, `rocket`, `flight`, `flight_takeoff`, `public`, `language`, `circle`, `fiber_manual_record`, `star`, `star_outline`, `location_on`, `place`, `adjust`, `gps_fixed`, `my_location`, `explore`, `navigation`, `near_me`, `diamond`, `hexagon`, `change_history`, `lens`, `panorama_fish_eye`, `radio_button_unchecked`, `brightness_1`, `flare`, `wb_sunny`, `bolt`.
    - `path`: path to a custom PNG image file (loaded from the assets folder). Persisted into the DB like GLB paths when recording; `builtin` icons are not copied.
  - `color` child node: tint color for the icon using the standard `color r g b [a]` format or named colors (default white). See Colors in the glossary above.
  - `visibility_range` child node: `min`, `max`, and `fade_distance` in world units (see above).
  - `size`: desired screen pixel size of the icon (default 32).
- `thruster` children (optional, multiple): GPU exhaust particles attached to the object. Declare one item per nozzle; there is no bank shorthand yet.
  - `position`: required `(x, y, z)` nozzle position in the object body frame.
  - `direction`: optional `(x, y, z)` exhaust direction for **scalar** `intensity`. Omit when `intensity` is a 3-vector (vector mode).
  - `intensity`: required EQL expression.
    - **Vector mode** (no `direction`): a single 3-component EQL vector drives **both** the exhaust direction (opposite the vector) **and** the intensity (its length × `scale`, clamped to `0..1`). Accepts any vector expression, e.g. `lander.main_thrust_viz`, `(0, 0, lander.main_thrust_viz[2])`, or `k * (0, 0, -1)`.
    - **Scalar mode** (with `direction`): a single number, clamped to `0..1`; `direction` fixes where the plume points. Use opposite-signed expressions on paired nozzles for forward/reverse, or for thrust-vector-control attach the thruster to an animated mesh and drive a scalar.
  - `name`: optional debug/display name.
  - `effect`: which particle effect to render. Either a built-in preset — `plume` (default; large hot exhaust) or `cold_gas` (small attitude-jet puff) — or a **hanabi `.effect` asset path** (detected by the `.effect` suffix), e.g. `effect="effects/apollo-lander/descent_plume.effect"`. File effects are authored/tuned externally (e.g. in pyrotechnique) and resolve exactly like `glb` paths: on DB record, local paths are rewritten to `db:…` and served by the [DB Asset Server](/reference/db-asset-server). Texture slots inside the effect bind by slot-name convention: `mask` uses a built-in soft-circle sprite, `smoke` loads `db:textures/smoke_puff.png`, anything else loads `db:textures/soft_circle.png`.
  - `body_frame`: bool, default `#false`. Rotates `direction` or the vector with the object.
  - `scale`: vector-mode multiplier mapping the EQL vector's magnitude onto `0..1` (default `1.0`).
  - `emission_rate`: particles per second at intensity `1.0`. For presets, defaults to `400.0`. For `.effect` files, omit it to use the spawn rate authored inside the file (recommended — the tuning already happened in the authoring tool); setting it overrides the authored rate.
  - `cutoff`: intensity threshold below which the emitter is hidden, default `0.02`.
  - `effect` child nodes (optional, repeated): additional **effect layers** rendered from the same emitter — `effect "<path>.effect"` with a positional path. All layers share the node's position/direction/intensity, so one thruster node replaces duplicate emitter declarations. The standard recipe for volumetric plumes is a velocity-stretched core (the `effect=` property) plus a camera-facing halo layer: stretched sprites foreshorten into a flat fan wherever their divergence points at the viewer, and the billboarded halo is what holds the plume's volume from every angle. `emission_rate` overrides and the `light` child apply to the primary effect only; layers always use the rates authored in their files.
  - `light` child node (optional): a dynamic light at the nozzle whose luminous power tracks the same intensity signal as the particles. Additive plume particles emit no light of their own, so this is what illuminates the nozzle, vehicle structure, and ground. It is emitter-level configuration, deliberately **not** part of the `.effect` file (that is a pure bevy_hanabi asset) — the schema mirrors pyrotechnique's `LightConfig` so tuned values port 1:1.
    - `color`: required `(r, g, b)` linear RGB, 0-1 per channel.
    - `intensity`: required peak luminous power in **lumens** at intensity `1.0`. Illuminance at distance d is `lm / (4π d²)` lux; megalumens are normal for an engine that must read against a ~100 klx sun.
    - `range`: meters beyond which the light has no effect (default `30.0`).
    - `offset`: meters down the exhaust axis from the thruster `position` (default `0.0`). Emitters usually sit inside the nozzle; hang the light at/below the exit plane so it doesn't blast the bell interior at point-blank range.
    - `spot_angle`: full cone angle in degrees for a spot light aimed down the exhaust axis; omit for an omnidirectional point light.
    - `shadows`: bool, default `#false`. Shadow-casting is expensive — keep it to one or two lights per scene.

```kdl
thruster name="DPS" effect="effects/apollo-lander/descent_plume.effect" body_frame=#true \
         position="(0, 0, -1.9)" direction="(0, 0, -1)" intensity="lander.main_thrust_viz[2]" {
    effect "effects/apollo-lander/descent_glow.effect"
    light color="(1.0, 0.95, 0.88)" intensity=3000000.0 range=40.0 offset=0.8 shadows=#true
}
```

  - World-fixed effects (ground dust, pad smoke): attach the thruster to an `object_3d` with a fixed pose. The particles simulate in the emitter's local frame, so a static emitter gives world-fixed particles that survive floating-origin rebasing:

```kdl
object_3d "(0,0,0,1, 0,0,0)" frame=ENU {
    sphere radius=0.02
    thruster name="ground_dust" effect="effects/apollo-lander/ground_dust.effect" \
             body_frame=#true position="(0, 0, 0)" direction="(0, -1, 0)" \
             intensity="lander.dust_viz[0]" cutoff=0.01
}
```

  - **Anchored trails** (persistent smoke columns behind a moving vehicle): a `.effect` that declares the vec3 properties `spawn_origin` and `spawn_axis` is automatically re-homed from the vehicle onto a **world-fixed anchor entity** frozen at the vehicle's position when the effect loads; every frame the runtime feeds the live nozzle pose through those properties, so particles spawn at the moving nozzle but hang in world space — the launch-trail look — while remaining floating-origin-safe (`SimulationSpace::Global` is not, and is never used). Authoring lives in pyrotechnique (`exhaust_smoke` is the reference); declare the thruster on the vehicle like any other, and the anchor management is automatic. Don't put a `light` child on a trail node — it would stay at the anchor, not the nozzle.
  - **Throttle-driven visuals**: a `.effect` that declares a scalar `intensity` property receives the node's live 0..1 intensity as a shader uniform each frame (in addition to the spawn-rate scaling every effect gets). Authors wire it into velocity/color expressions so throttle changes plume *length and brightness*, not just particle density — the falcon9 `merlin_core`/`merlin_flame` effects are the reference (a 1-engine landing burn renders a short dim plume, not a thin full-length one).

```kdl
object_3d "(0,0,0,1, vehicle.position[0], 0, 0)" {
    sphere radius=0.25 { color 80 170 255 }
    thruster name="forward" body_frame=#true position="(-0.35, 0, 0)" direction="(-1, 0, 0)" intensity="vehicle.specific_force[0] / 20.0"
    thruster name="reverse" body_frame=#true position="(0.35, 0, 0)" direction="(1, 0, 0)" intensity="vehicle.specific_force[0] / -20.0"
}
object_3d lander.world_pos {
    thruster name="DPS" body_frame=#true position="(0, -0.55, 0)" intensity=lander.main_thrust_viz
}
```

### line_3d
- Positional `eql`: required; expects 3 values (or 7 where the last 3 are XYZ).
- `frame`: optional; `ENU`, `NED`, or `ECEF`. Specifies the coordinate frame for interpreting the line points. Inherits from global `coordinate` if omitted.
- `line_width`: default 1.0.
- The trail is split into a *played* segment (up to the current playback time) and a *future* segment (ahead of it). When no `future_color` is set, the future segment is dimmed by a default fade so it reads as lighter than the played segment.
- `color`: color of the played segment. When omitted, falls back to the timeline `played_color` (default `yalk`). It does not affect the future segment.
- `future_color`: color of the future segment, independent of `color`. Its alpha sets the future opacity and is used as-is (no extra fade). When omitted, falls back to the timeline `future_color` (default `white`, faded).
- `perspective`: default true (set false for screen-space lines).

### vector_arrow
- `vector`: EQL expression yielding a 3-component vector (required).
- `origin`: EQL for arrow base; `world_pos` or 3-tuple (optional).
- `frame`: optional; `ENU`, `NED`, or `ECEF`. Specifies the coordinate frame for interpreting the vector and origin. Inherits from global `coordinate` if omitted.
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

### world_mesh
- Positional `region`: required terrain region identifier, e.g. `"death_valley"` or `"globe"`.
- `lod_count`: optional LOD depth override.
- `translate`: optional `(x,y,z)` offset in the selected coordinate frame, in meters.
- `frame`: optional; `ENU`, `NED`, or `ECEF`. Specifies the coordinate frame for interpreting terrain axes and translation. Inherits from global `coordinate` if omitted.
- `visible`: bool, default `#true`.

## Schema at a glance

Legend: parentheses group alternatives; `|` means “or”; square brackets `[...]` are optional; curly braces `{...}` repeat; `*` is zero-or-more, `+` is one-or-more; angle brackets `<...>` mark positional args.

```kdl
schematic =
  ( coordinate
  | theme
  | timeline
  | telemetry_mode
  | skybox
  | window
  | panel
  | object_3d
  | line_3d
  | vector_arrow
  | world_mesh
  )*

coordinate = "coordinate"
           frame=ENU|NED|ECEF

theme = "theme"
      [mode=dark|light]
      [scheme=string]

timeline = "timeline"
         [played_color=color_name_or_tuple]
         [future_color=color_name_or_tuple]
         [follow_latest=bool]
         [range=string]

When omitted, `follow_latest` defaults to `#false`.

telemetry_mode = "telemetry_mode" bool

skybox = "skybox"
       name=string

environment = "environment"
            { [sun [azimuth=float] [elevation=float] [direction=tuple] [illuminance=float] [shadows=bool]]
              [ambient scale=float]
              [sky color=color_name_or_tuple]
              [atmosphere …]
              [earth { [stars [density=float] [size=float] [brightness=float]] [city_lights [density=float] [size=float] [height=float] [brightness=float]] [airglow [density=float] [size=float] [brightness=float]] [night_map [brightness=float]] }] }

window = "window"
       [path|file|name=string]
       [title=string]
       [screen=int]
       [rect x y w h]

panel =
  viewport
  | graph
  | component_monitor
  | geo_position_gauge
  | orientation_gauge
  | action_pane
  | query_table
  | query_plot
  | data_overview
  | schematic_tree
  | sensor_view
  | video_stream
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
         [near=float]
         [far=float]
         [aspect=float]
         [active=bool]
         [show_grid=bool]
         [show_arrows=bool]
         [create_frustum=bool]
         [show_frustums=bool]
         [frustums_color=color_name_or_tuple]
         [projection_color=color_name_or_tuple]
         [frustums_thickness=float]
         [hdr=bool]
         [cinematic=bool]
         [ev100=float]
         [name=string]
         [frame=ENU|NED|ECEF]
         [view_cube_frame=ENU|NED|ECEF]
         [pos=eql]
         [look_at=eql]
         [smoothing=float]
         { [bloom [preset=natural|old_school] [intensity=float] [threshold=float] [threshold_softness=float]] vector_arrow }

graph = "graph" eql
      [name=string]
      [type=line|point|bar]
      [lock=bool]
      [auto_y_range=bool]
      [y_min=float]
      [y_max=float]
      { color }*

component_monitor = "component_monitor"
                  [name=string]
                  [component_name=string]

geo_position_gauge = "geo_position_gauge" eql
                   [name=string]
                   [source=ECEF|NED|ENU]
                   [display=ECEF|NED|ENU|LLA]

orientation_gauge = "orientation_gauge" eql
                  [name=string]
                  [source=ECEF|NED|ENU]
                  [display=ECEF|NED|ENU]
                  [{ reference x y z w }]

action_pane = "action_pane"
            [name=string]
            [lua=string]

query_table = "query_table"
            [name=string]
            [query=string]
            [type=eql|sql]

query_plot = "query_plot"
           [name=string]
           [query=string]
           [refresh_interval=ms]
           [auto_refresh=bool]
           [color]
           [type=eql|sql]
           [mode=timeseries|xy]
           [x_label=string]
           [y_label=string]

data_overview = "data_overview"
              [name=string]

schematic_tree = "schematic_tree"
               [name=string]

sensor_view = "sensor_view" <camera_name>
            [name=string]

video_stream = "video_stream" <msg_name>
             [name=string]

dashboard      = "dashboard" { dashboard_node }+

object_3d = "object_3d"
          <eql>
          [frame=ENU|NED|ECEF]
          [frame_orientation=ENU|NED|ECEF]
          [orientation=relative|absolute]
          { glb { animate }*
          | sphere
          | box
          | cylinder
          | plane
          | ellipsoid
          }
          [emissivity=float]
          { [visibility_range] }
          [icon]
          { thruster }*

animate = "animate"
        joint=string
        rotation_vector=eql

icon = "icon"
     (builtin=string | path=string)
     [size=float]
     { [visibility_range] [color] }

thruster = "thruster"
          position=tuple3
          [direction=tuple3]
          intensity=eql
          [name=string]
          [effect=("plume"|"cold_gas"|"<path>.effect")]
          [body_frame=bool]
          [scale=float]
          [emission_rate=float]
          [cutoff=float]
          { ("effect" string)* [light] }

light = "light"
      color=tuple3
      intensity=float
      [range=float]
      [offset=float]
      [spot_angle=float]
      [shadows=bool]

visibility_range = "visibility_range"
                 [min=float]
                 [max=float]
                 [fade_distance=float]

line_3d = "line_3d"
        <eql>
        [frame=ENU|NED|ECEF]
        [line_width=float]
        [color]
        [future_color]
        [perspective=bool]

vector_arrow = "vector_arrow"
             <vector-eql>
             [frame=ENU|NED|ECEF]
             [origin=eql]
             [scale=float]
             [normalize=bool]
             [body_frame|in_body_frame=bool]
             [color]
             [name=string]
             [show_name=bool]
             [arrow_thickness=float]
             [label_position=0..1]

world_mesh = "world_mesh"
           <region>
           [lod_count=int]
           [translate=(x,y,z)]
           [frame=ENU|NED|ECEF]
           [visible=bool]

color = "color"
      ( r g b [a]
      | name [alpha]
      )
```

## OBS Studio Integration

The `video_stream` panel can display live video from OBS Studio. There are two integration paths:

### SRT Receiver (Recommended)

OBS Studio has built-in SRT (Secure Reliable Transport) support. A GStreamer receiver pipeline on the Elodin server demuxes the MPEG-TS stream and forwards H.264 frames to Elodin DB.

**OBS configuration**: Settings -> Stream -> Custom -> `srt://ELODIN_IP:9000?mode=caller`

**Elodin-side receiver pipeline**:

```bash
gst-launch-1.0 \
    srtsrc uri="srt://0.0.0.0:9000?mode=listener" ! \
    tsdemux ! \
    h264parse config-interval=-1 ! \
    queue ! \
    elodinsink db-address=127.0.0.1:2240 msg-name="obs-camera"
```

A convenience script and full example are provided in `examples/video-stream/`.

### obs-gstreamer Direct Pipeline (Alternative)

If the [obs-gstreamer](https://github.com/fzwoch/obs-gstreamer) plugin is installed on the OBS machine, H.264 can be piped directly into `elodinsink` without an intermediate process.

**OBS output pipeline** (configured in obs-gstreamer settings):

```
video. ! h264parse config-interval=-1 ! elodinsink db-address=ELODIN_IP:2240 msg-name="obs-camera" audio. ! fakesink
```

This requires both `obs-gstreamer` and `elodinsink` to be installed on the OBS machine.

### Recommended OBS Encoder Settings

| Setting | Value |
|---|---|
| Encoder | x264 (Software) or NVENC (Hardware) |
| Rate Control | CBR |
| Bitrate | 2500–6000 kbps |
| Keyframe Interval | 2 seconds |
| Profile | Baseline or Main (High also works) |
| Tune | `zerolatency` |

> **Important**: Use H.264, not H.265/HEVC. Elodin's video decoder only supports H.264.

## Examples

Minimal viewport + graph:

```kdl
theme mode="light" scheme="matrix"

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

NED coordinate frame with objects:

```kdl
coordinate frame="NED"

viewport name="Main"
         show_grid=#true
         pos="(0,0,0,1, 0,10,0)"
         look_at="(0,0,0,0, 0,0,0)"

object_3d rocket.world_pos {
    glb path="rocket.glb"
}

line_3d rocket.world_pos line_width=2.0 {
    color white
}

vector_arrow "rocket.velocity" origin="rocket.world_pos" scale=0.5 {
    color cyan
}
```

In this example, all elements use the NED (North-East-Down) coordinate frame. The camera is positioned 10 meters East and looks at the origin. Individual elements can override the global frame with their own `frame` attribute.

Sensor camera panel (from `world.sensor_camera(..., fps=...)`; frames render automatically):

```kdl
sensor_view "drone.scene_cam" name="Forward Camera"
```

Viewport + sensor cameras side by side:

```kdl
hsplit {
    viewport name="3D View" show_grid=#true
    vsplit {
        sensor_view "drone.scene_cam" name="RGB Camera"
        sensor_view "drone.thermal_cam" name="Thermal"
    }
}
```

Video stream panel (e.g. from OBS Studio):

```kdl
video_stream "obs-camera" name="OBS Camera"
```

Viewport + video stream side by side:

```kdl
hsplit {
    viewport name="3D View" show_grid=#true share=0.6
    video_stream "obs-camera" name="OBS Camera" share=0.4
}
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

Rigged GLB model with animated joints:

The `rotation_vector` is an angle-axis: the direction encodes the axis, and the
magnitude encodes the angle in degrees.

```kdl
object_3d rocket.world_pos {
    glb path="rocket.glb"
    animate joint="Root.Fin_0" rotation_vector="(0, rocket.fin_deflect[0], 0)"
    animate joint="Root.Fin_1" rotation_vector="(0, rocket.fin_deflect[1], 0)"
    animate joint="Root.Fin_2" rotation_vector="(0, rocket.fin_deflect[2], 0)"
    animate joint="Root.Fin_3" rotation_vector="(0, rocket.fin_deflect[3], 0)"
}
```

Distance icon with independent visibility ranges:

```kdl
object_3d satellite.world_pos {
    glb path="satellite.glb" {
        visibility_range max=500.0
    }
    icon builtin="satellite_alt" {
        visibility_range min=500.0
        color 76 175 80
    }
}
```

Icon with fade-in (both mesh and icon visible by default, icon fades in over 50 units starting at distance 200):

```kdl
object_3d drone.world_pos {
    glb path="drone.glb"
    icon path="drone-icon.png" size=48 {
        visibility_range min=200.0 fade_distance=50.0
        color 0 188 212
    }
}
```

Overlapping ranges (both mesh and icon visible between 400 and 600 units):

```kdl
object_3d rocket.world_pos {
    glb path="rocket.glb" {
        visibility_range max=600.0
    }
    icon builtin="rocket_launch" {
        visibility_range min=400.0
        color 244 67 54
    }
}
```

Browse all available built-in icon names at [Material Icons](https://fonts.google.com/icons?icon.set=Material+Icons) (use the snake_case version of the icon name, e.g. "Satellite Alt" becomes `satellite_alt`).
