# Falcon 9 Launch-to-Landing SITL (ECEF)

> **Status: built and calibrated (first pass).** The full stack is
> implemented — 1000 Hz ECEF plant, Rust flight software over UDP, truth
> replay, Monte Carlo calibration — and flies the complete CRS-12 RTLS
> mission autonomously to a propulsive touchdown near LZ-1. Achieved-parity
> numbers and the remaining gaps are in
> [Calibration Results](#calibration-results).

A faithful reproduction of a real Falcon 9 return-to-launch-site (RTLS)
mission — liftoff, ascent, MECO, stage separation, flip, boostback burn,
entry burn, aerodynamic descent, and landing burn to a precision touchdown —
flown autonomously by an external flight-software process in closed loop with
an Elodin 6-DOF plant, simulated end-to-end in the WGS84 **ECEF frame**.
Structure and workflow follow
[`examples/apollo-lander`](../apollo-lander/README.md).

The vehicle model is built entirely from public Falcon 9 data. Where a
parameter is not published (and many are not), it is an explicit calibration
parameter — never a made-up constant disguised as fact.

> For the full math and physics foundation — frames and geodesy, the
> rotating-frame equations of motion, every effector and sensor model, the
> guidance laws, and the calibration methodology with derivations and
> references — see [`WHITEPAPER.md`](WHITEPAPER.md).

## Project Goal

Reproduce a recorded Falcon 9 booster flight, not just a plausible one:

1. **Build** a 6-DOF plant from public Falcon 9 characteristics and an
   external flight-software process that flies the full mission closed-loop.
2. **Run** the SITL simulation and collect telemetry in Elodin-DB.
3. **Compare** simulated telemetry against the recorded flight — trajectory
   profiles, event timing, and touchdown state — through an observation model
   (compare like-for-like observables, not raw truth vs. filtered display).
4. **Calibrate** with `elodin monte-carlo`: score every run against the real
   flight, then iteratively narrow the unpublished dynamics and control
   parameters until the simulation reproduces the recorded mission.

## Reference Mission

The calibration target is **CRS-12** (2017-08-14, booster B1039): the
surveyed mission class with complete public stage-1 telemetry from liftoff
through touchdown *and* a known payload mass. **CRS-11** (2017-06-03, B1035)
— same pads, same profile — is vendored as a held-out validation flight: a
calibrated model should reproduce it without retuning. Sources, quality
notes, and selection rationale live in [`data/README.md`](data/README.md).

- **Launch:** Kennedy Space Center LC-39A (28.6084 N, −80.6043 E)
- **Landing:** Landing Zone 1 (28.4858 N, −80.5444 E), ~15 km south

## Mission Profile (RTLS)

| # | Phase | Primary control |
|---|---|---|
| 1 | Liftoff / vertical ascent | 9 engines + TVC |
| 2 | Pitchover and gravity turn | TVC |
| 3 | Ascent through Max-Q to MECO | TVC (throttle bucket at Max-Q) |
| 4 | MECO + stage separation | Valve sequencing; stage-2 mass departs |
| 5 | Flip to retrograde | Cold-gas RCS |
| 6 | Boostback burn (3 engines) | TVC + RCS roll |
| 7 | Ballistic coast to apogee and descent | RCS attitude hold |
| 8 | Entry burn (3 engines) | TVC + RCS |
| 9 | Aerodynamic descent | Grid fins + RCS |
| 10 | Landing burn (1 engine, hoverslam) | TVC + grid fins |
| 11 | Touchdown at LZ-1 | Ground contact, shutdown + purge |

Recorded CRS-12 event times (webcast-observed, seconds after liftoff):
Max-Q 64, MECO 147, boostback 166–212, apogee 250 (118 km), entry burn
370–384, landing burn 433, touchdown 466.

Every burn requires a relight (TEA-TEB ignition) and ends with a cutoff and
engine purge, so ignition/shutdown sequencing is exercised four times per
flight.

## Closed-Loop SITL Boundary

The flight software sees only simulated sensor interfaces, commands only
simulated actuators and valves, and is never bypassed by the plant. The
external flight-software process owns:

- 6-DOF state estimation (IMU + GPS + radar altimeter fusion)
- Mission phase management and burn sequencing
- Guidance: ascent profile, boostback targeting, entry-burn timing, landing
  guidance to LZ-1
- TVC, throttle, grid-fin, and RCS command generation
- Propulsion valve sequencing, ignition/cutoff logic, post-cutoff purge
- Tank pressure monitoring

The plant owns physics and devices: rigid-body dynamics, gravity, atmosphere
and aerodynamics, engine thrust and propellant depletion, actuator and valve
dynamics, sensor models, and ground contact.

## Vehicle Model (public characteristics)

The model targets the **2017-era vehicle actually flown**: CRS-12 was the
maiden full Block 4, CRS-11 a Full Thrust "Block 3" — engine thrust and
propellant load sit *below* the Block 5 figures most spec sheets quote.
Values marked *est.* are public estimates or explicit calibration parameters
(priors in `spec.toml`); the rest are published figures.

### Stage 1 mass properties

| Parameter | Value |
|---|---:|
| Propellant load (LOX + RP-1) | ~395,000–400,000 kg *est.* (Block 5: 411,000) |
| Dry mass with legs and fins | ~23,000–27,000 kg *est.* |
| Stage 2 + payload departing at separation | ~116,000 kg + Dragon *est.* |
| CG / inertia vs. propellant state | calibration model |

### Main propulsion — 9 × Merlin 1D (sea level)

| Parameter | Value |
|---|---:|
| Thrust per engine, sea level | ~760–780 kN *est.* (Block 5: 845) |
| Vacuum/SL thrust ratio | ~1.08 |
| Specific impulse | ~282 s SL / ~311 s vac |
| Throttle floor | ~57% *est.* |
| Ignition | TEA-TEB (relights limited to 3 engines) |
| Burn usage | 9 ascent / 3 boostback / 3 entry / 1 landing |
| Tank pressurization | heated helium |

Single-engine minimum throttle still exceeds the empty booster's weight —
the landing burn is a hoverslam (no hover authority).

### TVC

| Parameter | Value |
|---|---:|
| Gimbal | per-engine, 2-axis |
| Gimbal range | ~±5 deg *est.* |
| Slew rate / bandwidth | calibration parameter |

### Cold-gas RCS (nitrogen)

| Parameter | Value |
|---|---:|
| Thruster pods | interstage-mounted, used for flip, coast attitude, roll |
| Count / thrust / layout | calibration parameters *est.* |
| Valve open/close response | ~ms-class, calibration parameter |

### Grid fins

| Parameter | Value |
|---|---:|
| Count | 4, titanium |
| Role | pitch/yaw/roll authority during aerodynamic descent |
| Aero model, actuator rate | calibration parameters |

### Feed system and valve sequencing (representative)

Pump-fed engines with helium tank pressurization and nitrogen purge. The
flight software sequences a representative valve set: helium pressurant
infill/vent per tank, main propellant valves, TEA-TEB igniter isolation, and
nitrogen purge. After every engine cutoff the purge is commanded open for a
fixed duration (*est.* 5 s). Tank pressures (pump-fed, low-pressure tanks,
~3–4 bar *est.*) are simulated, monitored by the flight software, and flagged
on excursion.

### Sensors available to the flight software

- **IMU** — primary input for state estimation, realistic rate/noise model
- **GPS** — position/velocity aiding
- **Radar altimeter** — terminal descent altitude
- **Tank / engine-inlet pressures** — simulated feedback

Sensor outputs are produced at realistic sample rates through realistic
interfaces rather than handing the flight software truth state.

## Truth Data and Calibration

The `data/` folder vendors the selected truth sources (see
[`data/README.md`](data/README.md)). Each Monte Carlo run scores the
simulation against the recorded flight:

- **Trajectory RMSE** — altitude and speed vs. the webcast telemetry profile,
  compared through an observation model that reproduces the display's
  filtering and quantization
- **Event timing** — MECO, boostback start/end, entry start/end, landing
  start, touchdown vs. recorded mission times
- **Touchdown state** — position error at LZ-1, vertical/horizontal speed,
  tilt, propellant remaining
- **Physics plausibility** — entry-burn duration and deceleration profile
  consistent with published supersonic-retropropulsion analyses

The calibration loop is apollo-lander's: run a campaign, read the report,
narrow the unpublished-parameter ranges in `spec.toml`, repeat until the
recorded flight is reproduced within thresholds.

## Acceptance Criteria

- One autonomous end-to-end SITL run: liftoff through touchdown at LZ-1 with
  the flight software never reading truth state.
- Simulated trajectory, event times, and touchdown state match the reference
  mission within thresholds set from truth-data uncertainty.
- Ignition, cutoff, and purge sequencing observed on all four burns; tank
  pressures held in band.
- Monte Carlo campaign reports (success rate, dispersion, best-fit run) and
  per-run logs, as in apollo-lander.

## ECEF Design Constraints

- **Rotating ECEF end-to-end (decided).** The world frame *is* WGS84 ECEF,
  rotating with the Earth: the plant integrates with explicit Coriolis and
  centrifugal terms, the landing target and atmosphere are frame-fixed, and
  state velocity is directly the ground-relative speed the webcast displays.
  Earth GLB at the origin, geodetic launch/landing sites, KDL
  `coordinate frame="ECEF"` (see [`examples/geo-frames`](../geo-frames/main.py)).
  Derivation and magnitudes in [`WHITEPAPER.md`](WHITEPAPER.md) §4–5.
  Close-up rendering at ECEF magnitudes is handled by the editor's
  floating-origin system (`big_space`: entities are rebased into grid cells
  and rendered camera-relative), so f64 world coordinates never hit f32
  precision limits in the viewport.
- **Architecture (per apollo-lander):** JAX systems composed into
  `el.six_dof`; `post_step` only ferries telemetry and commands; external
  flight software as a Rust process in `controller/` closed over UDP,
  launched via an `s10` recipe; KDL schematic with trails, phase display, and
  telemetry graphs; Monte Carlo campaign + CI smoke test.

## Open Items

- Aerodynamic model across ascent, retropropulsion, and descent (drag,
  normal force, grid-fin effectiveness)
- RCS layout, thrust, and moment arms (unpublished — calibration)
- Engine transient model (ignition ramp, shutdown, throttle response)
- Flight-computer interface definition (message set, rates, units)
- Launch-day atmosphere (standard atmosphere vs. sounding data)
- Second-stage representation after separation (mass event vs. ghost entity —
  the booster GLB renders as a full stack throughout; a visual split at MECO
  is future work)
- Touchdown envelope thresholds (derive from truth-data uncertainty)
- ~~Booster / pad / landing-zone GLB assets~~ booster GLB + tangent ground
  discs landed with the cinematic port; a modeled pad/LZ remains open

## Cinematic Visuals (pyrotechnique port)

The scene renders the pyrotechnique-authored falcon9 effects live from sim
telemetry (design record: `docs/design-thruster-effects-port.md` §10):

- **Merlin cluster plume** — `merlin_core` + `merlin_flame` layers on one
  `thruster` node (bell-exit emitter at −1.2 m), intensity from
  `booster.thrust_viz`. Tuned in pyrotechnique against `ascent.jpeg` for a
  near-parallel body-aligned column.
- **Persistent exhaust trail** — `exhaust_smoke` uses the anchored-trail
  contract (`spawn_origin`/`spawn_axis` on a world-fixed anchor).
- **Pad + landing clouds** — world-fixed `pad_smoke` at LC-39A / LZ-1.
- **RCS darts** — falcon9 `rcs_dart` (8 jets from `booster.rcs_levels`).
- **Environment** — sun + Bevy atmosphere, single Chase viewport (multi-view
  atmosphere is a Bevy 0.19 limit), ASDS barge GLB at LZ-1, pad disc at LC-39A.

Wall-clock pacing is on by default. Monte Carlo sets `ELODIN_MONTE_CARLO_CONTEXT`
and runs flat out. Particle integration follows the editor playhead when
replaying (`Time<EffectSimulation>`).

```sh
elodin editor examples/falcon9/main.py

# Lean kinematic visual checks (no FSW); screenshots under /tmp.
# Delay must be on the editor process; scenarios start at mission t0.
ELODIN_VIZCHECK_SCENARIO=plume-close ELODIN_SCREENSHOT_DELAY=10 \
  elodin editor examples/falcon9/visual_check.py
ELODIN_VIZCHECK_SCENARIO=rcs-flip ELODIN_SCREENSHOT_DELAY=8 \
  elodin editor examples/falcon9/visual_check.py
ELODIN_VIZCHECK_SCENARIO=barge ELODIN_SCREENSHOT_DELAY=10 \
  elodin editor examples/falcon9/visual_check.py
```

## Layout

```text
examples/falcon9/
  README.md          # this file
  WHITEPAPER.md      # math & physics foundation (frames, dynamics, effectors, guidance)
  main.py            # entry point: params, SITL bridge, result emission
  sim.py             # Elodin plant: components + JAX systems, truth ghost, scoring
  constants.py       # every shared constant (WGS84, vehicle, mission, rates)
  frames.py          # WGS84 geodesy + rotating-frame terms (pure JAX)
  atmosphere.py      # U.S. Standard Atmosphere 1976 (pure JAX)
  propulsion.py      # engine/tank/actuator/mass-stack physics (pure JAX)
  aero.py            # aero tables, plume dominance, grid fins (pure JAX)
  rcs.py             # cold-gas thruster geometry + allocation (pure JAX)
  reference.py       # truth profiles from data/ (stdlib only; sanity_check)
  sensors.py         # IMU/GPS/altimeter/pressure + webcast display model
  controller/        # flight software (Rust): estimator, phases, guidance
  falcon9.kdl        # cinematic ECEF schematic (GLB booster, effects, atmosphere, graphs)
  visual_check.py    # lean kinematic visual checks (plume / rcs / barge)
  visual_check.kdl   # schematic for visual_check.py
  spec.toml          # calibration priors     campaign.toml   # campaign config
  spec.ci.toml       # CI single sample       campaign.ci.toml
  calibrate.py       # rank / narrow / best-json calibration loop tooling
  hooks/             # score, report, ci_score, ci_gate, shared mc_metrics
  test_*.py          # frames, ladder, propulsion, aero, sensors
  data/              # vendored truth data + provenance README
```

## Run

From the repository root inside `nix develop` (`just install` first):

```sh
# Watch the calibrated mission in the editor (ECEF scene, truth ghost, graphs)
elodin editor examples/falcon9/main.py

# Headless single run (prints the scored result)
uv run python examples/falcon9/main.py run

# Fly the CRS-11 holdout with the frozen CRS-12-calibrated vehicle
ELODIN_FALCON9_MISSION=crs11 uv run python examples/falcon9/main.py run

# Monte Carlo calibration campaign (24 LHS samples)
elodin monte-carlo run examples/falcon9/main.py \
  --campaign examples/falcon9/campaign.toml \
  --spec examples/falcon9/spec.toml \
  --out dbs/falcon9-campaign

# Calibration loop tooling
python examples/falcon9/calibrate.py rank dbs/falcon9-campaign
python examples/falcon9/calibrate.py narrow dbs/falcon9-campaign \
  examples/falcon9/spec.toml /tmp/spec-next.toml

# Unit + physics-ladder tests
uv run python -m pytest -q examples/falcon9/

# CI smoke (one truncated deterministic campaign)
scripts/test-falcon9-monte-carlo.sh
```

Stale `falcon9-fsw` processes hold the campaign UDP ports; `pkill -f
falcon9-fsw` if a campaign refuses to start.

## Calibration Results

Seventeen 24-run LHS campaign rounds (fixed seed, narrow-around-best loop via
`calibrate.py`) against the recorded CRS-12 flight. The calibrated best-fit
parameters are baked into `main.py` as the defaults, so a plain run flies the
calibrated vehicle. Nominal calibrated run vs the parity targets:

| Metric | Target | Achieved (calibrated nominal) |
|---|---:|---:|
| Event: MECO | ±3 s | **+0.2 s** |
| Event: entry-burn ignition | ±3 s | **+2.7 s** |
| Event: landing-burn ignition | ±3 s | −7.2 s |
| Event: touchdown | ±3 s | **−0.4 s** |
| Speed RMSE (display space) | ≤ 15 m/s | 51 m/s (3.1% of peak; best run 42) |
| Altitude RMSE (display space) | ≤ 150 m | 1,355 m (best run 310) |
| Touchdown vertical / lateral | ≤ 2 / ≤ 1 m/s | 3.3 / 4.7 m/s |
| Touchdown position error | ≤ 500 m | 1.53 km (best campaign run 66 m) |
| Purge after every cutoff | 4 | **4** |
| Descent q̄ peak (recon ~60 kPa) | 40–120 kPa | **82 kPa** |

Where the remaining residual lives (measured per segment): ascent tracks the
recorded profile at ~8 m/s RMSE; the recovery segments (boostback shaping,
coast arc, entry-burn placement) carry ~45–75 m/s and dominate. The **CRS-11
holdout** (frozen vehicle, mission gates re-derived from its own profile)
completes the full mission and lands intact ~1 km from LZ-1 with all event
structure preserved, but its trajectory RMSE degrades ~4× — the recovery
*guidance shape* is partially overfit to CRS-12's loft. Both honest gaps
point at the same next lever: replace the parametric boostback/entry gates
with recovery-profile tracking (the ascent already does this), and refine the
engines-first aero tables. The campaign report
(`post_campaign/falcon9_report.txt`) and `calibrate.py` make each iteration a
five-minute loop.
