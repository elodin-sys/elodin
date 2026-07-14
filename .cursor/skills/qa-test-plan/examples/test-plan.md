# QA Test Plan: Examples happy-path & feature test (all `examples/`)

> Exercises the **real README happy path** of every `examples/` project and verifies the specific features each is designed to demonstrate — not just "does it compute."
> Evidence artifacts + shared helpers live in `.cursor/skills/qa-test-plan/examples/`.
> Authored per `.cursor/skills/qa-test-plan/`. Area prefix `EX` (Examples), one case per example directory.

## Plan Header

| Field | Value |
|-------|-------|
| Release / milestone | Examples happy-path & feature test (all `examples/`) |
| Git commit | `822eb89a9` |
| Branch | `main` |
| Date started | `<yyyy-mm-dd — fill at run time>` |
| Environment | `<OS, machine, display yes/no, GPU yes/no — fill at run time>` |
| Executor | `<agent model / human name>` |
| Status | NOT STARTED |

## Method: how these cases verify the intended experience

Bench mode (`… bench --ticks N`) only proves the math compiles. It does **not** confirm the intended developer experience, so it is used only for the two examples whose *documented* purpose is a CI compute check (`linalg`, `stablehlo`). Every other case drives the real happy path and verifies the feature behind it:

- **Real run path.** Simulations are launched exactly as the README intends — `elodin run examples/<x>/main.py` (the headless form of `elodin editor …`), the `elodin monte-carlo run` campaign, the SITL build+run, or the standalone client — so `post_step`/`pre_step` hooks, s10-managed sidecars (external controllers, render-server, C/C++ clients, GStreamer), and external-control components all actually execute.
- **Live telemetry verification.** While the sim runs, a probe connects to the live Elodin-DB (`127.0.0.1:2240`) and confirms the *intended* components/messages exist and evolve correctly (e.g. the rocket climbs, the MEKF attitude estimate tracks truth, reaction wheels spin, truth-replay ghosts populate, the C++ log client's messages arrive). This is the data layer behind the UX and is objective and repeatable.
- **Visualization (pixels) is a manual step.** This plan's authoring host has a display (`:0` / wayland) and a GPU (`/dev/dri/card1`), and the editor launches and renders — but its compositor does **not** expose a screen-capture protocol (`grim` fails with "compositor doesn't support the screen capture protocol"), so editor *rendering* cannot be screenshotted/asserted by an agent here. Each sim case therefore carries a **Manual visualization** block: the exact `elodin editor …` command and the specific scene a human should see. Where the headline feature *is* the rendering (video tiles, frustum-coverage overlays) or needs hardware/a browser/a large download, the whole case is `manual`.

### Shared helpers (in this plan folder)

- `.cursor/skills/qa-test-plan/examples/probe.py` — connects to a live DB, lists components, and prints per-component sample counts + first/last/min/max per element, plus message-log counts (`msg:<name>` args).
- `.cursor/skills/qa-test-plan/examples/run_probe.sh` — launches `elodin run <target>` in its own session, waits, runs `probe.py`, then **hard-kills the process group** and waits for port 2240 to free. Killing the group is mandatory: `elodin run` spawns an s10-managed python child with an *instant restart* policy, so killing only the child respawns it and the port never frees.

Usage (inside `nix develop`, from repo root):
`bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/ball/main.py 14 ball.world_pos ball.wind`

## Execution Rules

1. Run every command from the repository root, inside the Nix shell (`nix develop --command <cmd>`).
2. Execute cases in Summary order, one at a time. Do not parallelize — every live-run case binds port 2240 exclusively.
3. Check **Requires** first. If a required case did not PASS, mark this case BLOCKED and record which requirement failed.
4. A case is **PASS** only when every **Pass criteria** item is verified true. Never infer success from absence of errors alone.
5. Record **Evidence** (exit codes, matching output lines, artifact paths) before writing Result.
6. On **FAIL**: save output to `.cursor/skills/qa-test-plan/examples/<case-id>-fail.log`, diagnose briefly in Notes, continue. Never fix code mid-run.
7. Stop the whole run only if `SDK-001` (the build) fails; mark the rest BLOCKED.
8. `manual` cases: the agent marks them SKIPPED and lists them for a human. Perform the documented steps by hand.
9. `agent+visual` cases need a display/GPU; on a machine lacking one, mark BLOCKED with the reason.
10. Kill any background process a case started before moving on (each live-run case already does this via `run_probe.sh`'s group-kill; if you run commands by hand, `pkill -9 -f "elodin run"` and confirm `ss -ltn | grep :2240` is empty). Use `/tmp/qa-run/<case-id>/` for scratch and clean it up.
11. After each case: update the case block (Result, Evidence, Notes), then its Summary row. Fill the run-summary footer at the end.

> **Author-validation note.** Each agent case below carries an `AUTHOR-VALIDATED` line recording the real output observed on 2026-07-10 @ `822eb89a9` (Linux, display + GPU present). Treat those numbers as the expected-good baseline; small run-to-run variation in sample counts and float values is normal.

### Result states

| State | Meaning |
|-------|---------|
| PASS | All pass criteria verified true |
| FAIL | At least one criterion false, or the steps errored |
| BLOCKED | Could not attempt (unmet requirement, missing display/GPU/hardware) |
| SKIPPED | Deliberately not run (manual-only this pass) |

## Summary

| # | ID | Name | Area | Priority | Mode | Result |
|---|----|------|------|----------|------|--------|
| 1 | SDK-001 | Toolchain build and install | SDK | P0 | agent | |
| 2 | EX-001 | ball — gravity/drag/wind/bounce | Examples | P0 | agent | |
| 3 | EX-002 | three-body — graph-query gravity edges | Examples | P1 | agent | |
| 4 | EX-003 | n-body — solar system + truth ghosts | Examples | P1 | agent | |
| 5 | EX-004 | rocket — lookup-table aero + fin PID | Examples | P1 | agent | |
| 6 | EX-005 | drone — cascade PID + MEKF | Examples | P1 | agent | |
| 7 | EX-006 | cube-sat — MEKF/LQR/reaction wheels/EGM08 | Examples | P1 | agent | |
| 8 | EX-007 | linalg — LAPACK ops (CI bench) | Examples | P1 | agent | |
| 9 | EX-008 | stablehlo — StableHLO op coverage (CI bench) | Examples | P2 | agent | |
| 10 | EX-009 | geo-frames — ENU/NED/ECEF + ECEF orbit | Examples | P2 | agent | |
| 11 | EX-010 | ellipsoid — sensor-camera frustum demo | Examples | P2 | agent | |
| 12 | EX-011 | rc-jet — fixed-wing aero/turbine/servo | Examples | P2 | agent | |
| 13 | EX-012 | frames — frame-independence verification | Examples | P1 | agent | |
| 14 | EX-013 | cube-sat-pysim — `to_jax` stepping | Examples | P2 | agent | |
| 15 | EX-014 | db-client — standalone `elodin.db` client | Examples | P1 | agent | |
| 16 | EX-015 | monte-carlo — campaign runner + hooks | Examples | P1 | agent | |
| 17 | EX-016 | apollo-lander — SITL Monte Carlo campaign | Examples | P1 | agent | |
| 18 | EX-017 | betaflight-sitl — quad physics + FCU lockstep | Examples | P1 | agent | |
| 19 | EX-018 | crazyflie-edu — C SITL lockstep | Examples | P2 | agent | |
| 20 | EX-019 | sensor-camera — RGB/thermal camera render | Examples | P1 | agent+visual | |
| 21 | EX-020 | logstream — C++ log ingestion | Examples | P2 | agent | |
| 22 | EX-021 | video-stream — H.264 GStreamer pipeline | Examples | P2 | manual | |
| 23 | EX-022 | voyager — SPICE truth trajectories | Examples | P2 | manual | |
| 24 | EX-023 | rocket-barrowman — Streamlit design suite | Examples | P2 | manual | |

---

## Test Cases

### SDK

#### - [ ] SDK-001 — Toolchain build and install

- **Priority:** P0 | **Mode:** agent | **Requires:** none
- **Description:** The full toolchain (Python SDK, editor, database) builds from source and installs runnable binaries. Every example case depends on this.
- **Expected duration:** up to 60 min cold, ~5 min warm

**Steps**

```bash
nix develop --command just install
nix develop --command elodin --version
nix develop --command elodin-db --version
```

**Pass criteria**

- [ ] `just install` exits 0
- [ ] `elodin --version` and `elodin-db --version` each exit 0 and print a version string

**Result:**
**Evidence:**
**Notes:**

---

### Examples

#### - [ ] EX-001 — ball: gravity / drag / wind / bounce

- **Priority:** P0 | **Mode:** agent | **Requires:** SDK-001
- **Description:** The canonical intro sim (README: `elodin editor examples/ball/main.py`). A steel ball is released at z = 6 m and must fall under gravity, be perturbed by sampled wind, decelerate via aerodynamic drag, and bounce off the ground — `@el.map` effectors composed into `six_dof`. Verifies the core author→run→telemetry loop end to end.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/ball/main.py 14 ball.world_pos ball.wind ball.world_vel
```

**Pass criteria**

- [ ] Probe prints `N_COMPONENTS` ≥ 8 and the `COMPONENTS` line includes `ball.world_pos`, `ball.world_vel`, `ball.wind`, `ball.force`
- [ ] `ball.world_pos SAMPLES` > 1000
- [ ] `ball.world_pos FIRST` z (element index 6) ≈ 6.0 and `LAST` z ≈ 0 (ball fell to the ground); `MIN`/`MAX` z show it stayed within [~0, 6] (bounced, did not fall through)
- [ ] `ball.wind SAMPLES` > 0 (wind sampled each tick)
- [ ] No Python traceback / Rust panic in output

**Manual visualization:** `nix develop --command elodin editor examples/ball/main.py` → a single sphere falls, visibly bounces, leaving a position trail and a velocity arrow in the 3D viewport; the graph panel shows `ball.world_pos` decaying.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 11 components; `ball.world_pos` 1202 samples, z 6.0 → −0.009 (min ~0, bounced), `ball.wind` 1202 samples; editor launched and set frame ENU without panic.

---

#### - [ ] EX-002 — three-body: graph-query gravity edges

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Demonstrates the `GraphQuery`/`edge_fold` pattern (README: `elodin editor examples/three-body/main.py`). A custom `GravityEdge` component + `GravityConstraint` archetype apply Newton's law across three bodies `a`,`b`,`c` seeded for a stable periodic orbit. Verifies graph systems produce a bounded, orbiting solution (not a diverging/NaN one).
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/three-body/main.py 12 a.world_pos b.world_pos c.world_pos
```

**Pass criteria**

- [ ] `COMPONENTS` includes `a.world_pos`, `b.world_pos`, `c.world_pos` and the gravity edges (`a_>_b.gravity_edge`, etc.)
- [ ] Each of `a/b/c.world_pos SAMPLES` > 800
- [ ] Positions stay bounded — every `MIN`/`MAX` for the x,y elements (indices 4,5) is within about [−2, 2] (stable orbit, no divergence)
- [ ] `FIRST` ≠ `LAST` for each body (they actually move)
- [ ] No traceback / panic

**Manual visualization:** `elodin editor examples/three-body/main.py` → three colored spheres (yellow/pink/cyan) orbit around a common center with a trailing line.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 33 components incl. all gravity edges; a/b/c each 1168 samples; positions bounded within ±0.9; FIRST≠LAST (orbiting).

---

#### - [ ] EX-003 — n-body: solar system + truth-replay ghosts

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Solar-system gravity (Sun + 9 planets) with RK4 and **truth-replay ghosts** driven from bundled `planets_truth.csv` via `post_step` (README: run in the editor and compare sim vs truth). Verifies that (a) all planets simulate and (b) `post_step` truth injection populates `truth_<planet>` — the feature bench mode cannot reach. Uses `DBNAME` (not `ELODIN_DB_PATH`) and a bounded tick override.
- **Expected duration:** < 3 min

**Steps**

```bash
mkdir -p /tmp/qa-run/EX-003
nix develop --command bash -c 'ELODIN_NBODY_MAX_TICKS=240 DBNAME=/tmp/qa-run/EX-003/db bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/n-body/main.py 22 earth.world_pos truth_earth.truth_world_pos sun.world_pos'
rm -rf /tmp/qa-run/EX-003
```

**Pass criteria**

- [ ] `COMPONENTS` includes `sun`, `mercury`…`pluto` world_pos and at least one `truth_<planet>.truth_world_pos` (truth ghosts registered)
- [ ] `earth.world_pos SAMPLES` > 0 and `FIRST` ≠ `LAST` (Earth advances along its orbit)
- [ ] `truth_earth.truth_world_pos SAMPLES` > 0 (post_step truth injection ran)
- [ ] `sun.world_pos` stays ≈ origin
- [ ] No traceback / panic

**Manual visualization:** `elodin editor examples/n-body/main.py` (let it run) → planets orbit the Sun with ghost markers/trails from the truth CSV overlaid; `accuracy_report.py` after an export quantifies error.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 270 components incl. sun/mercury…pluto + `truth_<planet>.truth_world_pos` for each; earth advanced along its orbit arc; truth ghosts populated (post_step confirmed).

---

#### - [ ] EX-004 — rocket: lookup-table aero + fin PID

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Guided 6DOF rocket (README: `elodin editor examples/rocket/main.py`): Mach/altitude atmosphere, angle-of-attack aero lookup tables, motor thrust-curve interpolation, a fin pitch PID, and an `external_control` `fin_control_trim`. Verifies powered ascent and that the aero/guidance component family is produced.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/rocket/main.py 14 rocket.world_pos rocket.world_vel rocket.mach
```

**Pass criteria**

- [ ] `COMPONENTS` includes `rocket.world_pos`, `rocket.aero_coefs`, `rocket.mach`, `rocket.thrust`, `rocket.pitch_pid`, `rocket.fin_control_trim`, `rocket.angle_of_attack`
- [ ] `rocket.world_pos` altitude (element index 6) `MAX` is many hundreds of metres above the `FIRST` value (rocket climbed)
- [ ] `rocket.world_vel SAMPLES` > 500 and shows non-zero vertical speed
- [ ] No traceback / panic

**Manual visualization:** `elodin editor examples/rocket/main.py` → the rocket lifts off and pitches over; EQL-derived graphs (`angle_of_attack`, `mach`, velocity norm) plot live.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 29 components incl. aero_coefs/mach/thrust/pitch_pid/fin_control_trim; altitude climbed 1.0 → 1196 m; vertical speed reached ~190 m/s.

---

#### - [ ] EX-005 — drone: cascade PID + MEKF

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Quad-X drone (README: `elodin editor examples/drone/main.py`) with motor thrust curves, a rate/angle **cascade PID**, an **MEKF** state estimator, and simulated IMU/magnetometer sensors — fully autonomous. Verifies the estimator + controller + motor allocation all run and keep the vehicle controlled.
- **Expected duration:** < 3 min

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/drone/main.py 22 drone.world_pos drone.attitude_estimate drone.motor_rpm
```

**Pass criteria**

- [ ] `COMPONENTS` includes `drone.attitude_estimate`, `drone.motor_rpm`, `drone.rate_pid_state`, `drone.gyro`, `drone.accel`, `drone.magnetometer`
- [ ] `drone.world_pos SAMPLES` > 1000
- [ ] `drone.attitude_estimate` stays near-upright (the scalar/quaternion `w` element remains close to 1, e.g. `MIN` > 0.9) — controller held attitude, estimator did not diverge
- [ ] `drone.motor_rpm` shows non-zero values
- [ ] No traceback / panic

**Manual visualization:** `elodin editor examples/drone/main.py` (+ the bundled KDL panels) → the quad flies its trajectory; rate-control and motor panels show live PID/RPM signals.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 39 components incl. attitude_estimate/motor_rpm/rate_pid_state/gyro/accel/magnetometer; 1560 samples; attitude quaternion w in [0.97, 1.0] (upright, estimator stable).

---

#### - [ ] EX-006 — cube-sat: MEKF / LQR / reaction wheels / EGM08

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** LEO OreSat ADCS (README: `elodin editor examples/cube-sat/main.py`): sun sensors + magnetometer + gyro feeding an **MEKF**, an **LQR** attitude controller allocating to three **reaction wheels** (with Stribeck friction), under high-fidelity **EGM08** gravity. Verifies the estimator tracks truth and the wheels/controller drive the attitude maneuver.
- **Expected duration:** < 4 min
- **Note:** EGM08 downloads spherical-harmonic coefficients from `assets.elodin.systems` on first use (network required on a cold cache).

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/cube-sat/main.py 30 ore_sat.world_pos ore_sat.att_est
```

**Pass criteria**

- [ ] `COMPONENTS` includes `ore_sat.att_est`, `ore_sat.P`, `ore_sat.control_force`, `ore_sat.radius`, `rw_1.rw_speed`…`rw_3.rw_speed`, `css_0.css_value`…`css_5.css_value`, `ore_sat.gyro_omega`, `ore_sat.mag_value`
- [ ] `ore_sat.att_est SAMPLES` > 1000 and the estimate `LAST` quaternion is close to the true attitude (`ore_sat.world_pos` orientation elements 0–3) — MEKF tracking
- [ ] `ore_sat.world_pos` position elements (4–6) change over time (orbiting)
- [ ] No traceback / panic

**Manual visualization:** `elodin editor examples/cube-sat/main.py` → the CubeSat detumbles/points using its reaction wheels against the Earth globe.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 91 components incl. att_est/P/control_force/radius, rw_1..3, css_0..5; att_est LAST [0.0003,−0.0198,0.7115,0.7024] ≈ true attitude quat [0.0012,−0.0204,0.7117,0.7022] (MEKF tracks); satellite orbited.

---

#### - [ ] EX-007 — linalg: LAPACK-backed ops (CI bench)

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** This example's documented purpose *is* a CI compute check — it validates LAPACK-backed linear algebra (cholesky, solve, inv, qr, svd, det, slogdet, eigh) and multi-size Kalman/EKF filters on the Cranelift backend. The README's happy path is bench mode; there is no editor scene. Verifies the LAPACK FFI path compiles and runs without error.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash -c 'uv run python examples/linalg/main.py bench --ticks 200 2>&1 | tee /tmp/qa-linalg.log; exit ${PIPESTATUS[0]}'
```

**Pass criteria**

- [ ] Command exits 0
- [ ] Output contains the timing block (`tick time`, `build time`, `real_time_factor`)
- [ ] No Python traceback / Rust panic

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: exit 0, `real_time_factor: 238.449`, no errors.

---

#### - [ ] EX-008 — stablehlo: StableHLO op coverage (CI bench)

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Also a CI op-coverage validator: drives a broad set of StableHLO/CHLO ops (math, sort, shape, control flow incl. `while_loop`/`switch`, bitwise, linalg, convert, cholesky + triangular solve) through JAX → StableHLO → the Cranelift JIT. Happy path is bench; no editor scene. Verifies the JIT handles the op catalog.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash -c 'uv run python examples/stablehlo/main.py bench --ticks 100 2>&1 | tee /tmp/qa-stablehlo.log; exit ${PIPESTATUS[0]}'
```

**Pass criteria**

- [ ] Command exits 0
- [ ] Output contains the timing block (`real_time_factor`)
- [ ] No Python traceback / Rust panic

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: exit 0, `real_time_factor: 2664.109`, no errors.

---

#### - [ ] EX-009 — geo-frames: ENU/NED/ECEF markers + ECEF orbit

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Multi-frame visualization (README: editor): NED origin (lat/lon/alt anchor), an ENU marker, ECEF equator/pole markers, and a `post_step`-driven circular **ECEF orbit** line, with WGS84 ENU→ECEF conversion. Verifies the frame math and the orbit writer produce the intended geodetic markers.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/geo-frames/main.py 18 ned_origin.world_pos ecef_orbit_line.world_pos earth.world_pos
```

**Pass criteria**

- [ ] `COMPONENTS` includes `ned_origin`, `enu_far_east`, `ecef_far_up`, `ecef_north_pole`, `ecef_south_pole`, `ecef_orbit_line`, `earth`
- [ ] `ecef_orbit_line.world_pos` position elements sweep a large circle (x,y `MIN`/`MAX` ≈ ±Earth-radius scale, ~±7.5e6) — the orbit animates
- [ ] `ned_origin.world_pos` orientation elements change over time (marker rotates) while `earth.world_pos` stays at origin
- [ ] No traceback / panic

**Manual visualization:** `elodin editor examples/geo-frames/main.py` → labeled ENU/NED/ECEF markers on an Earth globe with a circling orbit line.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 69 components incl. all frame markers; `ecef_orbit_line` swept x,y ∈ ±7,578,137; `ned_origin` quaternion rotated; earth fixed at origin.

---

#### - [ ] EX-010 — ellipsoid: sensor-camera frustum demo

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** A drone with a mounted `sensor_camera` orbits inside an ellipsoid (README: editor) to demonstrate **frustum ∩ ellipsoid coverage %** and the **2D far-plane projection** — both computed by the editor's frustum plugin. Agent scope verifies the simulation substrate: the `pre_step` scripted drone orbit animates and the entities exist. The coverage/projection/camera overlays themselves are an editor+GPU feature (Manual visualization).
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/ellipsoid/main.py 16 drone.world_pos ellipsoid.world_pos
```

**Pass criteria**

- [ ] `COMPONENTS` includes `drone.world_pos` and `ellipsoid.world_pos`
- [ ] `drone.world_pos SAMPLES` > 1000 and `FIRST` ≠ `LAST` (pre_step orbit animation ran)
- [ ] `ellipsoid.world_pos` stays fixed
- [ ] No traceback / panic

**Manual visualization (headline feature):** `nix develop --command elodin editor examples/ellipsoid/main.py` → two viewports; toggle **COVERAGE** and **PROJ. 2D** in the viewport inspector and confirm a frustum-vs-ellipsoid coverage percentage updates and a 2D projection mesh renders on the far plane as the drone orbits.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9 (sim substrate): 15 components; `drone.world_pos` 3287 samples, orientation swept (animated orbit); `ellipsoid` fixed at origin. Coverage/projection overlays not agent-checkable here — editor-computed (see Manual visualization).

---

#### - [ ] EX-011 — rc-jet: fixed-wing aero / turbine / servo

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** A 6DOF fixed-wing BDX jet (README: `elodin editor examples/rc-jet/main.py`): polynomial stability-derivative aero, turbine spool/thrust dynamics, rate-limited servo control surfaces, Death Valley terrain, an external Rust RC controller (s10 cargo recipe), and an FPV `sensor_camera`. Agent scope verifies the flight physics and that the controller recipe builds. RC gamepad/keyboard input and the FPV camera are Manual (hardware/GPU/display).
- **Expected duration:** < 3 min (first run compiles the Rust controller)

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/rc-jet/main.py 35 bdx.world_pos bdx.world_vel bdx.mach
```

**Pass criteria**

- [ ] `COMPONENTS` includes `bdx.aero_coefs`, `bdx.alpha`, `bdx.beta`, `bdx.mach`, `bdx.spool_speed`, `bdx.thrust`, `bdx.control_surfaces`, `bdx.control_commands`, plus `target.world_pos`
- [ ] `bdx.world_pos SAMPLES` > 2000 and horizontal position advances hundreds of metres (jet flies forward)
- [ ] `bdx.world_vel` forward speed is in a plausible cruise band (tens of m/s, not NaN/diverging)
- [ ] No traceback / panic
- [ ] Run log (`/tmp/qa-run-probe.log`) shows the Rust controller recipe compiling/launching (`Compiling …` / cargo output)

**Manual visualization:** `elodin editor examples/rc-jet/main.py` with a gamepad (or keyboard) → fly the jet over Death Valley terrain; the FPV `sensor_view` tile shows the onboard camera.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: 24 components incl. aero_coefs/alpha/beta/mach/spool_speed/thrust/control_surfaces/control_commands + target; ~9200 samples; flew (0,0,4500)→(2024,1417,3854), cruise ~57–73 m/s; controller recipe observed compiling.

---

#### - [ ] EX-012 — frames: frame-independence verification

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** A self-verifying harness (`world.build()` + `exec.run()` + `exec.history()`, not `World.run()`; ignores `bench`) that checks gravity equivalence in ENU vs NED, inertial-frame equivalence (ECI vs GCRF), and energy conservation, printing an explicit pass/fail summary. Verifies the coordinate-frame system is internally consistent.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash -c 'uv run python examples/frames/main.py 2>&1 | tee /tmp/qa-frames.log; exit ${PIPESTATUS[0]}'
```

**Pass criteria**

- [ ] Command exits 0
- [ ] Output contains `3/3 tests passed` and all three `✅ TEST PASSED` lines
- [ ] No traceback / panic

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: exit 0, all three tests passed (`3/3 tests passed`).

---

#### - [ ] EX-013 — cube-sat-pysim: `to_jax` RL-style stepping

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Exercises the pure-JAX `w.to_jax()` API (README: `uv run python examples/cube-sat-pysim/main.py`): steps the CubeSat attitude sim 500× via `sim.step(1)`, collects `att_est` with `sim.get_state`, and plots it. Ignores `bench`; run directly with a non-interactive matplotlib backend so `plt.show()` cannot block.
- **Expected duration:** < 2 min
- **KNOWN ISSUE:** at `822eb89a9` this script reproducibly raises `IndexError: Too many indices: array is 1-dimensional, but 2 were indexed` at `plt.plot(att_est[:, i]…)`, independent of backend. Until fixed this case is expected to **FAIL** — that is the regression this test surfaces. Record it and continue.

**Steps**

```bash
nix develop --command bash -c 'MPLBACKEND=Agg uv run python examples/cube-sat-pysim/main.py 2>&1 | tee /tmp/qa-csp.log; exit ${PIPESTATUS[0]}'
```

**Pass criteria**

- [ ] Command exits 0
- [ ] No Python traceback (in particular no `IndexError` at the plotting step)

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: reproducibly **FAILS** — exit 1, `IndexError: Too many indices: array is 1-dimensional, but 2 were indexed` at `main.py` `plt.plot(att_est[:, i]…)`. The `to_jax` stepping/`get_state` portion runs; only the plotting shape handling is broken.

---

#### - [ ] EX-014 — db-client: standalone `elodin.db` client

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** *Not* an Elodin simulation — the `elodin.db` client showcase (README headless form: `--no-editor --duration N`): an embedded `edb.Server`, multi-rate `table_writer` writes of a synthetic figure-8 flight, a derived write-back stream (`nav.speed`), message-log events, and the full read API (`latest`/`time_series`/`sql`/`get_msgs`). Verifies the standalone client read/write paths.
- **Expected duration:** < 2 min

**Steps**

```bash
nix develop --command bash -c 'uv run python examples/db-client/main.py --no-editor --duration 5 2>&1 | tee /tmp/qa-dbclient.log; exit ${PIPESTATUS[0]}'
```

**Pass criteria**

- [ ] Command exits 0
- [ ] Output's read-back summary lists components including `drone.world_pos` and derived `drone.nav.speed`, a `time_series … samples` line with a non-zero count, and an `sql over …` result row
- [ ] No Python traceback / Rust panic

**Manual visualization:** `uv run python examples/db-client/main.py` (no flags) → the editor opens with `crazyflie.glb` flying a figure-8 and graphs of every written signal.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: exit 0; read-back showed `time_series drone.nav.speed: 498 samples`, `time_series drone.battery.voltage: 51 samples`, and `sql over drone_battery_voltage: [{'n': 51, 'mean_v': 4.18…}]`.

---

#### - [ ] EX-015 — monte-carlo: campaign runner + hooks

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** The minimal `elodin monte-carlo` campaign (README: `elodin monte-carlo run … --campaign campaign.toml --spec spec.toml`): a saturated-PD point-mass plant driven by an external Python controller over a UDP bridge (s10 `PyRecipe.process`, no build step), with LHS sampling, per-run scoring (`hooks/score.py`), and aggregate reporting (`hooks/report.py`). The full spec is 100 samples; this case uses a **truncated 2-run plan** to bound runtime while exercising the sampler, worker, SITL bridge, hooks, and report writers.
- **Expected duration:** < 4 min

**Steps**

```bash
rm -rf /tmp/qa-run/EX-015 && mkdir -p /tmp/qa-run/EX-015
head -n 3 examples/monte-carlo/plan.csv > /tmp/qa-run/EX-015/plan.csv
nix develop --command elodin monte-carlo run examples/monte-carlo/main.py \
  --campaign examples/monte-carlo/campaign.toml \
  --plan /tmp/qa-run/EX-015/plan.csv --workers 1 --out /tmp/qa-run/EX-015/out
wc -l /tmp/qa-run/EX-015/out/results.csv
test -f /tmp/qa-run/EX-015/out/summary.json && echo "SUMMARY OK"
rm -rf /tmp/qa-run/EX-015
```

**Pass criteria**

- [ ] `elodin monte-carlo run` exits 0 and prints a campaign summary (`finished monte-carlo campaign: ok=…`)
- [ ] `results.csv` has 3 lines (header + 2 runs); `SUMMARY OK` printed
- [ ] The run output includes the report hook's `Final position error` section
- [ ] No Python traceback / Rust panic

**Manual note:** the full documented run is `elodin monte-carlo run examples/monte-carlo/main.py --campaign … --spec examples/monte-carlo/spec.toml --out dbs/monte-carlo-demo` (100 LHS samples) followed by inspecting `campaign_summary.txt`.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: exit 0, `ok` runs, `results.csv` 3 lines, `summary.json` written, report printed `Final position error mean: 5.153 m`.

---

#### - [ ] EX-016 — apollo-lander: SITL Monte Carlo campaign

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** The flagship SITL + Monte Carlo example: an Apollo 11 powered-descent 6DOF sim driven by an external **Rust LGC** guidance controller over a UDP bridge, scored against reconstructed truth telemetry. The CI helper `scripts/test-apollo-monte-carlo.sh` runs one truncated deterministic campaign (`campaign.ci.toml` builds the Rust controller via its `[[build]]` step; `ELODIN_APOLLO_MAX_TICKS=600`; `--workers 1`) — a bounded, terminating end-to-end exercise of JAX physics, the SITL bridge, campaign hooks, and reporting.
- **Expected duration:** 5–20 min (cold: release `cargo build` of the controller)

**Steps**

```bash
nix develop --command bash scripts/test-apollo-monte-carlo.sh
```

**Pass criteria**

- [ ] Script exits 0
- [ ] Output contains `apollo-lander monte-carlo CI passed`
- [ ] The campaign summary reports `ok=1 failed=0` (the run completed and scored)
- [ ] No Python traceback / unhandled Rust panic

**Manual visualization:** `elodin editor examples/apollo-lander/main.py` → the LM descends from radar lock to touchdown with the truth-ghost trail, DPS thrust/RCS arrows, and telemetry graphs.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: exit 0, `apollo-lander monte-carlo CI passed`, `finished monte-carlo campaign: ok=1 failed=0 invalid=0`; Rust controller built, SITL post_step timing reported.

---

#### - [ ] EX-017 — betaflight-sitl: quad physics + FCU lockstep

- **Priority:** P1 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Couples Elodin quad physics to real Betaflight firmware over an 8 kHz UDP lockstep. The full lockstep needs the large Betaflight **git submodule** + a compiled `betaflight_SITL.elf` + the s10-managed FCU process (Manual — see below). The agent case validates the Elodin half that the whole example rests on: `sim.py` compiles and runs the quad 6DOF physics + sensor models standalone (a bounded 200-tick free-fall compile test).
- **Expected duration:** < 3 min

**Steps**

```bash
nix develop --command bash -c 'uv run python examples/betaflight-sitl/sim.py 2>&1 | tee /tmp/qa-bf.log; exit ${PIPESTATUS[0]}'
```

**Pass criteria**

- [ ] Command exits 0
- [ ] Output contains `All physics systems compiled successfully!` and `Physics test complete!`
- [ ] No Python traceback / Rust panic

**Manual (full Betaflight lockstep — human, needs the external submodule):**
1. `git submodule update --init --recursive --depth 1` (fetches `examples/betaflight-sitl/betaflight`).
2. `nix develop --command bash examples/betaflight-sitl/build.sh` → builds `betaflight/obj/main/betaflight_SITL.elf`.
3. `nix develop --command uv run python examples/betaflight-sitl/main.py run` (terminates via `interactive=False` + `max_ticks`).
4. Confirm the log shows the Betaflight SITL process starting and the FDM/RC/motor UDP bridge (ports 9001–9004) exchanging — the quad arms and motors respond. Editor: `elodin editor examples/betaflight-sitl/main.py` to watch it fly.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9 (physics half): exit 0, printed `All physics systems compiled successfully!` / `Physics test complete!`. Full lockstep NOT run here — the Betaflight submodule is not initialized (`examples/betaflight-sitl/betaflight/` empty); it is the Manual procedure above.

---

#### - [ ] EX-018 — crazyflie-edu: C SITL lockstep

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Shares one C control source (`user_code.c`) between a SITL binary and real firmware (README labs). The agent case validates the SITL path end to end: the student C binary **builds**, then `elodin run` starts it as a UDP-lockstep subprocess (via `post_step`) and the control loop produces motor telemetry. Keyboard-armed flight and real-hardware HITL are Manual.
- **Expected duration:** < 3 min

**Steps**

```bash
nix develop --command sh -c '
  bash examples/crazyflie-edu/sitl/build.sh || exit 1
  test -x examples/crazyflie-edu/sitl/sitl_main || exit 1
  bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/crazyflie-edu/main.py 22 crazyflie.world_pos crazyflie.motor_pwm crazyflie.motor_command
'
rm -f examples/crazyflie-edu/sitl/sitl_main
```

**Pass criteria**

- [ ] `build.sh` prints `Build complete` and produces executable `sitl/sitl_main`
- [ ] `COMPONENTS` includes `crazyflie.motor_command`, `crazyflie.motor_pwm`, `crazyflie.motor_rpm`, `crazyflie.gyro`, `crazyflie.accel`, `crazyflie.is_armed_control`
- [ ] `crazyflie.world_pos SAMPLES` > 5000 (500 Hz lockstep advanced), i.e. the C binary connected and the UDP lockstep ran
- [ ] No traceback / panic

**Manual (flight & HITL):** `elodin editor examples/crazyflie-edu/main.py` → arm via keyboard and drive the motors to hover/fly. HITL (hardware): flash the shared code via `firmware/deploy.sh`, then `elodin editor examples/crazyflie-edu/main.py --hitl` with a Crazyradio + real Crazyflie.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: `Build complete` → `sitl_main` built; run produced 33 crazyflie.* components incl. motor_command/motor_pwm/motor_rpm/gyro/accel/is_armed_control; `crazyflie.world_pos` 9628 samples (SITL lockstep confirmed; vehicle idle/disarmed without keyboard input, as expected).

---

#### - [ ] EX-019 — sensor-camera: RGB/thermal camera render

- **Priority:** P1 | **Mode:** agent+visual | **Requires:** SDK-001
- **Description:** Entity-mounted RGB + thermal `sensor_camera`s rendered by a **headless GPU render-server** (auto-started as an s10 sibling); frames are pushed into the DB and the sim's `post_step` `_verify()` checks frame counts / FPS / historical `read_msg` at the final tick. Requires a GPU render-server, so it is `agent+visual` (BLOCKED on a machine without a working GPU). The `ELODIN_SENSOR_CAMERA_*` env + `run` make it terminating.
- **Expected duration:** < 5 min

**Steps**

```bash
mkdir -p /tmp/qa-run/EX-019
nix develop --command sh -c '
  timeout 200 env ELODIN_SENSOR_CAMERA_DB=/tmp/qa-run/EX-019/db ELODIN_SENSOR_CAMERA_MAX_TICKS=600 \
    elodin run examples/sensor-camera/main.py > /tmp/qa-run/EX-019/sc.log 2>&1
  grep -E "first frame seen|verification:" /tmp/qa-run/EX-019/sc.log
  pkill -9 -f "sensor-camera" 2>/dev/null; pkill -9 -f "render-server" 2>/dev/null
'
grep -q "verification: PASS" /tmp/qa-run/EX-019/sc.log && echo "VERIFY PASS" || echo "VERIFY MISSING"
rm -rf /tmp/qa-run/EX-019
```

**Pass criteria**

- [ ] Log shows both cameras delivering real frames (`[scene_cam] first frame seen …: <N> bytes, nonzero=<M>` with M > 0; `[thermal_cam] first frame seen …` with nonzero > 0)
- [ ] Log contains `verification: PASS` (per-camera FPS-ratio OK + historical-read OK); `VERIFY PASS` printed
- [ ] No Python traceback / Rust panic
- (BLOCKED if no GPU render-server is available)

**Manual visualization:** `elodin editor examples/sensor-camera/main.py` → `sensor_view` panels show the live RGB and thermal (iron-bow) camera feeds.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9 (GPU `/dev/dri/card1`): `[scene_cam] first frame … 1228800 bytes nonzero=1228800`, `[thermal_cam] … 65536 bytes nonzero=29371`, `verification: PASS` (scene_cam ~19.7 fps, thermal ~13 fps, historical reads OK). The `run` server does not self-exit after verify, so it is wrapped in `timeout`; the PASS line prints before the timeout.

---

#### - [ ] EX-020 — logstream: C++ log ingestion

- **Priority:** P2 | **Mode:** agent | **Requires:** SDK-001
- **Description:** Demonstrates C++ structured-log ingestion into the DB with a live `log_stream` editor panel (README: `elodin editor examples/logstream/main.py`). Running it auto-builds and runs the C++ `log-client` via s10 against the live DB alongside a ball sim. Verifies the C++ client compiles, connects, and streams `fsw.log` messages into the DB — the feature behind the panel.
- **Expected duration:** < 3 min (first run compiles the C++ client)

**Steps**

```bash
nix develop --command bash .cursor/skills/qa-test-plan/examples/run_probe.sh examples/logstream/main.py 28 ball.world_pos msg:fsw.log
```

**Pass criteria**

- [ ] `MSG fsw.log COUNT` > 0 (C++ log client compiled, connected, and streamed structured log messages)
- [ ] `MSG fsw.log TAIL` shows human-readable log lines (e.g. state/telemetry strings)
- [ ] `ball.world_pos SAMPLES` > 0 (the companion physics sim runs)
- [ ] No Python traceback / Rust panic
- [ ] Run log (`/tmp/qa-run-probe.log`) shows `log-client … compiled ->` and `DB is up, starting log client`

**Manual visualization:** `elodin editor examples/logstream/main.py` → a `Flight Software Log` panel scrolls live `fsw.log` entries with level filtering.

**Result:**
**Evidence:**
**Notes:** AUTHOR-VALIDATED 2026-07-10 @822eb89a9: `MSG fsw.log COUNT 52`, tail included `[cycle 1] Touchdown detected, impact: 3.2g` and `State: RECOVERY`; C++ client log showed `compiled -> …/elodin-log-client` then `DB is up`.

---

#### - [ ] EX-021 — video-stream: H.264 GStreamer pipeline

- **Priority:** P2 | **Mode:** manual | **Requires:** SDK-001
- **Description:** Ingests H.264 video into the DB via the custom `elodinsink` GStreamer plugin (test-pattern, plus optional OBS SRT / RTSP) alongside a rolling-ball sim, shown in editor `video_stream` tiles (README: `elodin editor examples/video-stream/main.py`). The headline path needs the GStreamer plugin build + a real-time (non-terminating) sim + the editor to decode/display the video — not agent-verifiable headlessly, so `manual`. The plugin **build** is an agent-checkable prerequisite (below).
- **Expected duration:** manual

**Agent prerequisite check (validated):**

```bash
nix develop --command cargo build --release --manifest-path fsw/gstreamer/Cargo.toml
ls target/release/libgstelodin.so
```

- [ ] Builds `target/release/libgstelodin.so` (exit 0)

**Manual (full pipeline — human, needs display):**
1. `nix develop --command elodin editor examples/video-stream/main.py` (s10 auto-runs `stream-video.sh`: `videotestsrc → x264enc → h264parse → elodinsink msg-name="test-video"`).
2. In the editor, confirm the **Test Pattern** `video_stream` tile decodes and shows the moving GStreamer test pattern, alongside the rolling-ball 3D scene. (Headless confirmation of `msg:test-video` in the DB is possible but the intended deliverable is the decoded video in the tile.)

**Result:**
**Evidence:**
**Notes:** Prerequisite AUTHOR-VALIDATED 2026-07-10 @822eb89a9: `libgstelodin.so` built (cargo release, 1m01s). Live pipeline not agent-validated — the s10 group front-loads the plugin build and the pipeline is a non-terminating real-time stream whose payoff is the decoded video in the editor tile (needs display); left `manual`.

---

#### - [ ] EX-022 — voyager: SPICE truth trajectories

- **Priority:** P2 | **Mode:** manual | **Requires:** SDK-001
- **Description:** Voyager 1/2 heliocentric gravity against **SPICE-kernel** truth ephemerides for planets and ghost probes (README: download kernels, then editor). Requires downloading large NASA SPICE kernels (`de440.bsp` ≈ 100+ MB) + `spiceypy`, and drives all truth updates from `pre_step` in a non-terminating `run` — not agent-verifiable headlessly, so `manual`. (Also documented WIP: simulated probes do not yet reach Saturn.)
- **Expected duration:** manual

**Manual (human, needs network + display):**
1. `cd examples/voyager && ./download_spice_data.sh` (fetches `naif0012.tls`, `de440.bsp`, Voyager_1/2 BSPs into `nasa_spice_data/`).
2. `nix develop --command elodin editor examples/voyager/main.py`.
3. Confirm the Sun/planets render with SPICE-driven truth positions and Voyager probe trajectories advancing (probes are expected to diverge before Saturn — known WIP).

**Result:**
**Evidence:**
**Notes:** MANUAL — SPICE data is not present (`examples/voyager/nasa_spice_data/` absent) and the run is a non-terminating editor session driven by `pre_step`. Not agent-validated by design.

---

#### - [ ] EX-023 — rocket-barrowman: Streamlit design suite

- **Priority:** P2 | **Mode:** manual | **Requires:** SDK-001
- **Description:** A Streamlit rocket-design suite (Barrowman CP/CG solver, ThrustCurve.org motor search, AI natural-language builder, ISA/NRLMSISE/weather atmospheres) with optional Elodin 3D playback (README: `run_streamlit.sh` or `python main.py`). Requires the Streamlit stack + a browser (and network APIs); `main.py` runs the solver then opens the editor for playback — not agent-verifiable headlessly, so `manual`.
- **Expected duration:** manual

**Manual (human, needs browser):**
1. `nix develop --command bash examples/rocket-barrowman/run_streamlit.sh` (installs Streamlit deps, serves the UI).
2. Open the printed local URL; load the default **Calisto** design and run a simulation.
3. Confirm a flight summary (apogee, stability, etc.) and a 3D trajectory render without error; optionally launch the Elodin editor playback from the UI.

**Result:**
**Evidence:**
**Notes:** MANUAL — Streamlit UI requires a browser and the primary `main.py` path opens the editor for playback (non-terminating). Not agent-validated by design.

---

## Run Summary

| Metric | Count |
|--------|-------|
| Total | 24 |
| PASS | |
| FAIL | |
| BLOCKED | |
| SKIPPED | |

**Notable issues:** (author pre-run) EX-013 cube-sat-pysim reproducibly FAILS at its matplotlib plotting step (`IndexError`); the `to_jax` stepping itself works.

**Follow-up items:** EX-017 full Betaflight lockstep, EX-021 live video pipeline, EX-022 voyager, and EX-023 rocket-barrowman require a human (external submodule / display / large download / browser) and are documented as `manual`.
