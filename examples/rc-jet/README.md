# BDX RC Jet

The Elite Aerosports BDX is a real 2.65 m sport jet — same airframe people
trailer to the desert, fuel, and fly. This example is a 6-DOF Elodin
simulation of **that** airplane, sitting on **that** kind of field, so
the next session can be a comparison instead of a cold start.

The aero is not a hand-tuned coefficient card. It is a hashed package from
the [open-air](https://github.com/elodin-sys/open-air) pipeline: a traced
reconstruction of the BDX through OpenVSP, OpenAeroStruct (VLM + wingbox),
and a VSPAERO cross-check. Reference geometry, mass and fuel, the
longitudinal linearization and drag polar, the propulsion map, and the
render mesh all come from that package. A few things the pipeline will not
invent — inertia, lateral/directional derivatives, servo limits — are
labeled class-D fallbacks, logged at startup, and meant to be replaced the
first time we weigh, swing, and identify the real jet.

You fly it now, from a radio (or the keyboard), in a rotating WGS84 world
anchored on the Mojave RC field (35.350664 N, 117.809027 W). Chase and the
onboard views sit on the real DEM; the sibling render-server draws
cinematic FPV (globe, atmosphere, daylight) and a Boson+ 640-style LWIR
camera as `sensor_view` frames. The physics pad and the mesh meet at the
same dirt.

Today the model is **analysis-correlated**, not flight-validated. That is
the invitation: take the BDX out, log the same channels we already plot
(attitude, airspeed, throttle, GPS / geodetic altitude), and hang truth
data on this sim.

## Where the numbers come from

Aircraft data has exactly two sources, and every constant belongs to one of
them:

1. **The generated aero package** at `model/elodin_package/` — produced by
   the [open-air](https://github.com/elodin-sys/open-air) aero-design
   pipeline from a traced reconstruction of the 2.65 m BDX (OpenVSP geometry,
   OpenAeroStruct VLM + wingbox analysis, VSPAERO cross-check).
   `elodin_model.json` is the schema-versioned entry point and SHA-256
   manifest; `bdx_model.py` validates all of it before the world is created
   and refuses to run on any mismatch. Reference geometry (S 1.332 m²,
   MAC 0.518 m), mass and fuel state, the longitudinal linearization and
   drag polar, the propulsion map, trim rows, and the render mesh all come
   from here. Evidence classes and allowances: `model/elodin_package/provenance.md`.
2. **Class-D fallbacks** at `class_d_fallbacks.py` — whitepaper-estimate
   placeholders for what the package deliberately publishes as null: the
   lateral-directional/rate/control derivatives, the inertia tensor, servo
   limits, and the spool time constant. Scenarios opt in explicitly and every
   selected fallback is logged at startup. These are **not** BDX truth; they
   are replaced by regenerating the package once measured data exists
   (see `model/elodin_package/integration_guide.md` §6).

Do not restate aircraft constants in Python — the loader is the single
source (guide §9.1).

## The world

The simulation runs in a **rotating ECEF frame** (WGS84): gravity is
point-mass gravitation plus centrifugal and Coriolis terms, the atmosphere
and ground contact key off geodetic altitude, and the aircraft spawns at a
real location — the Mojave RC field (35.350664 N, 117.809027 W, field
elevation 589.274 m), the center of the `mojave_rc_field` terrain region.

The schematic (`bdx.kdl`) keeps editor 3D panes as normal viewports
(aircraft + Mojave mesh, drone-like lighting). Cinematic Earth — globe,
atmosphere, and the 100 klx sun — is owned by `world.sensor_camera(...,
cinematic=True)` and rendered only in the sibling render-server. The
`sensor_view` tile displays those DB frames; it does not load Earth in
the editor.

`bdx.ir_cam` emulates the customer's Boson+ 640 `22640A018-6IARX`: 640×512
at 60 Hz with an 18° horizontal f/1.0 lens, white-hot AGC, DDE, optical
softening, and detector noise. It uses a final render-server sensor-output
pass over the normal terrain camera. The target drone is tagged at 18 °C,
so its mask renders as the cold dark silhouette seen in the reference
flight recording. `bdx.fpv_cam` remains the independent cinematic RGB
camera.

Close-up terrain is the geo-anchored `mojave_rc_field` planar
`world_mesh` (`frame="ENU"`).

The package GLB is already Elodin body (X forward, Y left, Z up). The editor
lifts every glTF as Y-up (`Rx(+90°)`); `bdx.kdl` applies `rotate="(-90, 0, 0)"`
to cancel that so the mesh is not rolled onto its side. The hashed package
GLB is not rewritten.

To fetch and preprocess the terrain atlas (writes
`assets/terrains/planar/mojave_rc_field/`):

```bash
./scripts/prepare_editor_terrain_region.sh mojave_rc_field
```

Without the atlas the terrain falls back to a grid. Cinematic Earth for
FPV is embedded in the render-server (not the editor). Physics uses the
pad's ellipsoid height (589.274 m); the mesh is shifted so its DEM
surface (621.5 m orthometric at the centre) meets that pad.

This example ships binary geometry under `model/elodin_package/` via
Git LFS — run `git lfs pull` if the loader reports manifest hash mismatches.

## Quick start

```bash
elodin editor examples/rc-jet/main.py
```

The RC controller starts automatically (FrSky-style gamepad or keyboard) and
sends `bdx.control_commands` at 60 Hz.

### LWIR reference validation

The customer Boson recording can be reduced to reproducible reference frames
and metrics without extra Python packages:

```bash
uv run python scripts/boson_ref/extract_frames.py ai-context/bdx/mnt/cvapp/data
uv run python scripts/boson_ref/compare_rgba.py /tmp/bdx.ir_cam.rgba
```

The second command expects one raw 640×512 RGBA frame exported from
`bdx.ir_cam`. The same acceptance ranges run in pytest when
`ELODIN_LWIR_FRAME` points to that file.

### Control mapping (Mode 2)

| Gamepad | Keyboard | Control |
|---------|----------|---------|
| Left Stick Y | W/S | Throttle (idles at the engine's 18% floor) |
| Left Stick X | Q/E or A/D | Rudder (Q/A = yaw left, E/D = yaw right) |
| Right Stick Y | Up/Down | Elevator (pull back / down-arrow = nose up) |
| Right Stick X | Left/Right | Aileron (right = roll right) |

Mode 1: `cargo run -p rc-jet-controller -- --mode1`.

### Scenarios

Selected via `ELODIN_RC_JET_SCENARIO` (default `demo`):

| | `validation` | `demo` |
|---|---|---|
| Purpose | Regression anchors; CI | Interactive flying |
| Site | Death Valley floor (−60 m) so the package cruise row stays above ground | Mojave RC field (589.274 m), the `mojave_rc_field` mesh center |
| Condition | Package cruise trim row verbatim: 300 m MSL, 37.83 m/s, α 2.67°, throttle 0.2125 | ~300 m AGL over the pad (889 m MSL) at the same TAS; `ELODIN_RC_JET_ALTITUDE_M` / `ELODIN_RC_JET_SPEED_MPS` re-solve the equilibrium |

Both scenarios start from a **solved trim** — there is no hand-tuned
"trimmed" coefficient anywhere, and a scenario refuses to spawn if no valid
equilibrium exists at its condition. `ELODIN_RC_JET_HEADING_DEG` sets the
initial heading (default 350°). `validation` stays on the Death Valley pad
because the package cruise altitude is 300 m MSL — underground at the
Mojave field.

## Expected behavior (package anchors, not marketing)

From `elodin_model.json .performance_anchors` — the test suite reads these
from the package rather than repeating them:

- **Cruise:** ~37.8 m/s at ~21% throttle, α ≈ 2.7° (300 m).
- **Dash:** ~85 m/s level at full throttle (100 m) — note this is *below*
  the manufacturer's ">200 mph" (89.4 m/s) claim with the provisional 200 N
  engine model; treat max speed as unvalidated.
- **Stall:** ~15 m/s under the documented section-CLmax assumption
  (class C — a consequence of stated assumptions, not a validated figure).
- Validity envelope (guide §5): attached flow |α| ≤ 12°, tabulated α
  −2° to 8°, M ≤ 0.3, and Re/m near the single tabulated condition
  (2.53×10⁶). Outside any bound the physics keeps integrating unclamped
  and the `bdx.aero_valid` telemetry flag drops to 0
  (`flag_invalid_do_not_clamp`).

Fuel burns per the package propulsion map (2.21 kg aboard at spawn), total
mass updates as it burns, and an empty tank is a flameout. Thrust acts
0.044 m above the CG, so full throttle pitches the nose down slightly.

## Module architecture

```
examples/rc-jet/
├── main.py              # scenario select, ECEF world, loads bdx.kdl
├── bdx.kdl              # editor layout (mojave_rc_field + GLB + sensor views)
├── bdx_model.py         # package loader: schema/identity/frames/SHA-256 validation
├── class_d_fallbacks.py # labeled class-D placeholder set (opt-in, logged)
├── scenario.py          # site + scenario + numerics (no aircraft data)
├── trim.py              # level-flight equilibrium solver
├── frames.py            # WGS84 geodesy + rotating-frame helpers
├── atmosphere.py        # ISA troposphere (geodetic altitude)
├── aero.py              # package aero evaluation + the one frame adapter
├── propulsion.py        # map interpolation, spool, fuel, thrust line
├── actuators.py         # rate-limited servo dynamics
├── ground.py            # geodetic ground contact
├── telemetry.py         # lat/lon/alt + local-ENU derived components
├── sim.py               # archetype + system composition (SemiImplicit, 300 Hz)
├── model/elodin_package/  # vendored open-air package (hash-verified unit)
├── tests/               # loader, sign-battery, trim, and closed-loop tests
└── controller/          # Rust RC controller (gamepad/keyboard → control_commands)
```

Integration uses the semi-implicit integrator at 300 Hz (documented as such;
the earlier README claimed RK4 in error).

### Frame conventions (the one adapter)

Coefficients are evaluated in the **standard aerospace frame** (X fwd,
Y right, Z down; +δe TE-down, +δa roll-right, +δr TE-left) and converted to
Elodin body axes (X fwd, Y left, Z up) in exactly one place
(`aero.adapter_body_wrench`): β, q̂, r̂ negate on the way in; F_y, τ_y, τ_z
negate on the way out. The sign battery in `tests/test_physics.py` pins
every direction so the conversion can never silently rot.

## Tests

```bash
pytest examples/rc-jet/tests -o 'pythonpath='   # inside `nix develop .#run`
```

Covers: loader rejection rules (schema/identity/frames/hash/path/symlink),
GLB contract, the sign battery, no-clamp/no-floor guards, wind invariance,
trim and dash reproduction against package anchors, propulsion monotonicity,
fuel-flow integration, thrust-line moment sign, validity flagging, ECEF
altitude hold, and ground rest.

## Fidelity roadmap

The package format is stable while evidence improves (handoff §17): measured
weight/CG, bifilar inertia, hinge geometry + throws (enables the open-air
`flight_dynamics` stage to publish measured lateral/control derivatives),
and an identified engine deck each land as *regenerate package → re-vendor*,
with no loader changes. Until then this example is **analysis-correlated**,
not flight-correlated, and describes itself accordingly.

## References

- `BDX_Simulation_Whitepaper.md` — original model-structure design document
  (numeric sections superseded by the generated package; see banner within)
- `model/elodin_package/integration_guide.md` — the package's own contract
- `ai-context/bdx/Elodin_RC_Jet_Improvement_Guide.md` — the campaign guide
- [Elodin Documentation](https://docs.elodin.systems)
