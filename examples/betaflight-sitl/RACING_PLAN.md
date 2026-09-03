# Vision-Guided Gate Racing Plan

**Document status:** Authoritative living specification  
**Last verified against repository:** 2026-09-03
**Current resume point:** Package D — Manual piloting, control seam, and ANGLE mode

## 1. Purpose and authority

This document defines the work required to evolve `examples/betaflight-sitl`
from its current scripted takeoff demonstration into an opt-in, vision-guided
gate-racing example.

It is intentionally self-contained. A coding agent should be able to receive the
repository and this document, select the next incomplete package, implement it,
verify it, and leave a useful handoff. No issue discussion, private repository,
chat history, or unwritten project knowledge is required. External library and
Elodin API documentation may be consulted as normal reference material, but it
must not supply missing product requirements.

If this document and another planning artifact disagree, this document governs
the racing work. Existing repository instructions such as `AGENTS.md` still
govern commands and repository-wide development practices.

## 2. Instructions for coding agents

### 2.1 Before starting a package

An agent must:

1. Read this document completely and read applicable repository instructions.
2. Inspect the current implementation; do not trust the status section blindly.
3. Confirm that every listed prerequisite is present.
4. Run the current baseline and the prerequisite packages' verification commands.
5. Check that the package requirements do not contradict the implementation,
   tests, or another section of this document.
6. Tell the operator which package it intends to implement and report any blocker
   before making broad changes.

Only one package should normally be implemented per change. A package may be
split if it proves too large, but combining packages requires operator approval.

### 2.2 When the plan and reality disagree

Do not silently assume that either the code or this document is correct. Report:

- the conflicting statements or assumptions;
- evidence from code, APIs, or test results;
- affected contracts and packages;
- whether unaffected work can safely continue;
- reasonable options and their consequences;
- a recommended resolution.

Ask the operator for explicit approval before changing a material design decision.
A useful request looks like:

> The package assumes X, but the repository demonstrates Y. This affects A, B,
> and C. Options are ... I recommend ... May I update the plan and implement
> that change?

### 2.3 Changes requiring approval

Approval is required before changing:

- the final goal, goals, or non-goals;
- package scope, order, or dependencies;
- public data contracts or configuration semantics;
- coordinate-frame, motor-order, or RC conventions;
- truth-isolation and scoring rules;
- required acceptance criteria;
- backward-compatibility guarantees;
- normal versus GPU CI expectations;
- which capabilities are deferred;
- the document-governance rules in this section.

Never weaken an acceptance criterion merely because an implementation failed to
meet it. Never add a workaround solely to avoid raising a design concern.

### 2.4 Routine updates allowed during authorized work

An agent may update the following without a separate design decision when doing
so does not change their meaning:

- package status and linked implementation commit or PR;
- verified commands, test names, and file paths;
- factual descriptions of merged behavior;
- observed performance and known limitations;
- the current resume point;
- typos and clarifications.

These updates must land with the implementation that made them true.

### 2.5 Recording an approved design change

After approval, the agent must:

1. Update the affected architecture or contract sections.
2. Update every downstream package brief affected by the change.
3. Add a concise entry to the decision log: what changed, why, which assumption
   was wrong, and which packages are affected.
4. Implement and test against the revised design.
5. Update current status and the resume point.

Do not rewrite completed history to imply that a replacement design was always
intended. Preserve a short explanation of superseded decisions so future agents
can understand the code.

## 3. Current baseline

The current example contains:

- six-degree-of-freedom quadrotor physics in `sim.py`;
- reproducible multi-rate IMU, barometer, and magnetometer simulation in
  `sensors.py`;
- FDM, RC, and motor UDP packet handling in `comms.py`;
- a real Betaflight `2026.6.1` SITL binary built from the pinned submodule;
- an event-driven lockstep patch applied by `build.sh`;
- an 8 kHz physics/gyro/PID loop by default;
- an s10 recipe that supervises Betaflight;
- a scripted sequence in `main.py`: boot, arm, fixed throttle, then disarm;
- a DB recording and editor schematic with the drone and telemetry graphs.

The default 8 kHz build busy-waits in Betaflight and consumes approximately one
CPU core to avoid scheduler wakeup latency. The build-time
`VIRTUAL_GYRO_SAMPLE_RATE_HZ` and Python `simulation_rate` must remain equal.

The example does **not** currently contain a camera, gates, course selection,
referee, guidance interface, ANGLE-mode guidance, racing controller, perception,
race tests, or race CI. The README refers to `test_comms.py`, but that file is
not present; Package A must repair this documentation or supply its supported
replacement.

### 3.1 Current conventions

These are established by the existing implementation and must not change without
an approved design update:

| Item | Convention |
|---|---|
| Elodin world | ENU: +X forward/east, +Y left/north, +Z up |
| Elodin body | FLU: +X forward, +Y left, +Z up |
| `world_pos` | `[qx, qy, qz, qw, x, y, z]`, quaternion scalar-last |
| `world_vel` | `[wx, wy, wz, vx, vy, vz]`, angular then linear |
| FDM quaternion | `[w, x, y, z]`; current Gazebo-bridge conversion negates y/z |
| Betaflight sensor body frame | FRD |
| FDM world position/velocity | ENU |
| Motor packet order | `[BR, FR, BL, FL]` |
| Motor spins | `[-1, +1, +1, -1]` for `[BR, FR, BL, FL]` |
| RC channels | AETR: roll 0, pitch 1, throttle 2, yaw 3, AUX1 arm 4, AUX2 mode 5 |
| RC range | 1000–2000 microseconds; 1500 centered |
| Physics/PID rate | 8 kHz default, one FDM packet per lockstep iteration |

### 3.2 Baseline setup and manual verification

From the repository root, with the Elodin Python environment installed:

```bash
git submodule update --init --recursive --depth 1
(cd examples/betaflight-sitl && ./build.sh)
./examples/betaflight-sitl/init_eeprom.py
elodin run examples/betaflight-sitl/main.py
```

A healthy current run prints:

```text
SUCCESS: SITL integration working! Drone took off!
```

Package A makes this message part of a real C0 result: unmet lockstep,
motor-response, or takeoff criteria return a failing process status.

## 4. Final core outcome

The completed core plan provides an opt-in example in which one simulated drone:

1. boots and arms a real Betaflight SITL controller;
2. takes off and hovers;
3. observes orange training gates through a forward FPV sensor camera;
4. acquires and holds a standoff position in front of the next gate;
5. aligns, passes through the gate, and reacquires the next gate;
6. completes three gates on a straight course;
7. lands or safely disarms; and
8. emits a machine-readable race result with per-gate pass times.

The same mission must first work using ground-truth state and gate poses, then
work using rendered frames without exposing truth or gate poses to vision
guidance. Before either autonomous path, a manual mode must let a pilot assess
basic controllability through the same RC boundary using a gamepad or keyboard.
Betaflight remains responsible for attitude stabilization, rate PID, and motor
mixing. Manual and Python guidance initially command RC sticks in ANGLE mode.

Completion of the core plan means:

- the existing scripted takeoff remains available and is still the default;
- a pilot can safely arm, fly, land, and disarm manually with a gamepad or
  keyboard, independently of autonomous guidance;
- the truth-guided straight course completes repeatably in headless CI;
- the vision-guided straight course completes in three consecutive qualified
  runs, with GPU CI when suitable infrastructure exists;
- every layer has an offline or truth-based test that permits failures to be
  attributed to control, geometry, detection, tracking, or integration.

## 5. Goals, non-goals, and deferred work

### 5.1 Goals

- Preserve the existing Betaflight SITL integration example.
- Provide an opt-in manual piloting mode for controllability testing and
  guidance-independent debugging.
- Add realistic asynchronous 30 FPS camera input without blocking physics.
- Make course geometry and pass scoring reusable and independently tested.
- Keep control, camera geometry, detection, and tracking separately testable.
- Mechanically prevent truth from leaking into vision guidance.
- Use measurable package acceptance criteria and machine-readable outcomes.
- Support long pauses by maintaining a verified resume point in this document.

### 5.2 Non-goals for the core plan

- Photorealism, domain randomization, or sim-to-real validation.
- Reinforcement learning or end-to-end learned control.
- Multi-drone racing or collision/penalty rules for touching gate frames.
- Betaflight-native position hold.
- Direct motor control as a product mode.
- A global visual-inertial odometry system.
- Optimizing lap time beyond completing the training course reliably.

### 5.3 Deferred until the core plan demonstrates a need

- ACRO/rate-mode guidance and advanced attitude controllers.
- CNN or other learned gate detectors.
- Spec-size gates, sharp turns, slaloms, vertical courses, and procedural courses.
- MPCC, SE(3), INDI, or other advanced control schemes.
- A large flight-reporting or leaderboard framework.
- Converting the simulation world to NED/FRD.

Deferred work must not be introduced as part of Packages A–L without approval.

## 6. System architecture and invariants

### 6.1 Processes

The running system always has the simulation and Betaflight processes and may
add independently supervised render-server, manual-controller, and editor
processes:

```text
Manual controller (manual mode only)
    ↕ semantic pilot commands through Elodin DB
Simulation: Python physics + sensors + post_step + Elodin DB
    ↕ UDP FDM/RC/motors
Betaflight SITL

Render server (camera enabled only)
    ↕ camera frames through Elodin DB message logs
Simulation

Optional editor
    ↕ telemetry and schematic through Elodin DB
Simulation
```

Registering a sensor camera causes s10 to add the render server. The renderer is
asynchronous and may deliver fewer frames than requested on a slow GPU. It must
never backpressure or become part of the Betaflight lockstep exchange.

### 6.2 Per-tick ordering

After Package D, `post_step` must use this order:

1. Read the current sensor/state components in one batch where practical.
2. Send FDM plus the **previously computed** RC command to Betaflight.
3. Block for one motor response and write `drone.motor_command`.
4. Read the latest latency-adjusted camera frame without blocking.
5. Run the selected command source—manual, scripted, truth, or vision—and retain
   its RC command for the next physics tick.
6. Run referee scoring from ground truth and emit telemetry/results.

A command selected on tick N therefore reaches Betaflight on tick N+1. At the
8 kHz default this is 125 microseconds. Tests comparing command and response
must account for that one-tick delay.

Perception runs only when a fresh camera frame is available. A slow detector may
reduce wall-clock realtime factor, but must not alter simulation-time ordering or
starve Betaflight of FDM packets.

### 6.3 Runtime configuration

The following environment variables form the core user-facing configuration:

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `RACE_GUIDANCE` | `scripted`, `manual`, `truth`, `vision` | `scripted` | Selects the RC command source |
| `RACE_COURSE` | `none`, `single`, `c1_straight` | `none` | Selects course geometry |
| `RACE_CAMERA` | `0`, `1` | `0` | Enables the FPV camera independently for bring-up |

`manual` starts or connects to the manual controller and does not require a
course or camera. `vision` requires the camera and a non-`none` course; the
program must either enable the camera explicitly with a clear startup message or
reject the invalid combination. `truth` requires a non-`none` course. Unknown
values must fail at startup. Running with no environment variables must preserve
the current scripted behavior and must not require a manual controller or GPU
render server.

New knobs may be added in configuration dataclasses, but avoid an expanding set
of environment variables for controller gains and detector internals.

## 7. Stable domain contracts

These contracts describe required information flow. Exact Python organization
may evolve, but changing field meaning or truth boundaries requires approval.

### 7.1 RC command

The selected command source outputs six integer PWM values:

```text
roll, pitch, throttle, yaw, arm, mode
```

They map to RC channels 0–5 in AETR/AUX order. All outputs must be clamped to
1000–2000. Roll, pitch, and yaw center at 1500. `arm >= 1700` requests arming.
`mode >= 1700` requests ANGLE mode after Package D configures AUX2. Scripted mode
may retain its current flight mode to preserve behavior; truth and vision modes
use ANGLE initially.

The empirically observed signs and stick-to-angle behavior must be recorded by
Package D in this document and encoded in one tested conversion helper. Do not
infer the signs from comments alone.

#### Manual input provider

`RACE_GUIDANCE=manual` uses an s10-managed controller modeled on the RC-jet
controller pattern: a separate process polls a gamepad and keyboard at about
100 Hz and writes semantic pilot input through Elodin DB. It must feed the same
RC conversion and clamping helper used by autonomous guidance; it must not write
motor commands or bypass Betaflight.

Semantic input consists of normalized roll, pitch, and yaw in `[-1, 1]`, throttle
in `[0, 1]`, explicit arm/disarm state, and ANGLE-mode state. Mode 2 is the
default gamepad layout (left throttle/yaw, right pitch/roll), with optional Mode
1. Keyboard control provides W/S throttle, Q/E or A/D yaw, and arrow-key
pitch/roll. Arm/disarm and mode changes require dedicated, documented controls;
throttle position alone must never arm the vehicle.

The manual command transport must carry a heartbeat or equivalent freshness
signal. Missing input, no display/device, controller disconnect, or a stale
heartbeat must result in a documented safe command: disarmed, minimum throttle,
centered axes. A headless host must not panic or accidentally arm. Raw semantic
input and resulting `drone.rc_command` must be recorded so pilot action can be
compared with plant response.

### 7.2 Guidance update

A guidance object is stateful for one run and receives one update per physics
tick. The update contains:

- simulation time and tick;
- gyro and accelerometer samples;
- barometer and magnetometer values plus freshness when available;
- latest FPV RGBA frame or `None`, plus a nominal sample time and freshness flag;
- race progress: last gate passed and next gate index;
- public course rules: gate count and inner-opening size;
- optional truth state and gate poses, supplied only in truth mode.

Arrays passed to guidance must be copies or otherwise safe from later mutation.
Guidance must not read `StepContext`, DB components, global course objects, or
renderer state directly. The simulation adapter owns all I/O and constructs the
mode-appropriate update.

### 7.3 Truth isolation

In `RACE_GUIDANCE=vision`:

- drone `world_pos` and `world_vel` are absent from the guidance update;
- gate centers, normals, and yaws are absent;
- synthetic detections are absent;
- importing or reaching the referee/course truth through globals is prohibited;
- camera calibration, gate count, gate inner size, IMU, barometer, magnetometer,
  and ordered pass events are allowed.

The referee always uses truth for scoring, including in vision mode. Package J
must include a test that constructs a vision update and proves truth and gate
poses are unavailable.

### 7.4 Camera

The core FPV camera contract is:

| Property | Value |
|---|---|
| DB name | `drone.fpv` |
| Resolution | 640×360 |
| Format | RGBA uint8, shape `(360, 640, 4)` |
| Rate | 30 frames per simulation second |
| Look direction | body +X |
| Mount | `[0.08, 0.0, 0.02]` metres in body FLU |
| Tilt | 0 degrees for the core course |
| Near/far | no clipping of gates from 0.1–40 m; far at least 100 m |
| Projection | vertical FoV chosen so `fx = fy = 320`, `cx = 320`, `cy = 180` |
| Read latency | 33,000 microseconds by default, using timestamped `read_msg` |

At 640×360, `fx = fy = 320` corresponds to vertical FoV approximately 58.72
degrees and horizontal FoV 90 degrees. Package G must derive and test this
rather than duplicating unexplained constants.

A frame sample is offered to guidance at most once per nominal 30 Hz camera
period. The current `read_msg` API returns only the payload, not the DB message's
actual capture timestamp, so the adapter records the requested sample time
(`ctx.timestamp - latency`) and must not invent a capture time. A repeated
payload is valid sample-and-hold behavior when the renderer lags. Until the
renderer produces a frame, guidance receives `None`. A future timestamp-aware
API would be a material contract change and must follow Section 2.

### 7.5 Courses and gates

A gate has:

- ordered integer index;
- center in ENU metres;
- yaw about world +Z;
- square inner-opening size in metres.

Yaw 0 means the gate plane normal points world +X. The drone approaches from the
negative local-X side and passes in the positive local-X direction. Gates are
static scene entities and must not participate in rigid-body integration.

Core courses are:

| Course | Geometry | Inner opening |
|---|---|---|
| `none` | no gates | — |
| `single` | gate at `(10, 0, 1.8)`, yaw 0 | 2.5 m |
| `c1_straight` | gates at `(10, 0, 1.8)`, `(20, 0, 1.8)`, `(30, 0, 1.8)`, yaw 0 | 2.5 m |

Render gates as four saturated-orange, matte bars around the opening. Exact bar
thickness is implementation detail, but the physical inner opening used by the
referee and PnP must be 2.5 m.

### 7.6 Referee and race result

Only the next gate in sequence can count. For each physics tick:

1. Transform previous and current drone positions into the next gate's local
   frame.
2. Require forward crossing: `previous_x < 0 <= current_x`.
3. Interpolate the crossing fraction at local `x = 0`.
4. Test interpolated local y/z against `±inner_size/2`.
5. On success, record gate index and simulation time exactly once.

Tests must cover centered, edge-inside, edge-outside, backward, yawed, and fast
diagonal crossings. The end-of-run output contract is:

```text
[RACE] course=<name> gates_passed=<passed>/<total> lap_time=<seconds-or-na> status=<COMPLETE|INCOMPLETE> pass_times=[...]
```

Guidance may receive the ordered pass event but never the crossing position or
other hidden truth in vision mode.

### 7.7 Detection and tracking

The detector consumes one nominally fresh RGBA frame sample and returns zero or
more candidates.
Each candidate contains:

- nominal frame sample timestamp (the latency-adjusted DB query time);
- four inner-opening corners ordered top-left, top-right, bottom-right,
  bottom-left in pixel coordinates;
- confidence in `[0, 1]`;
- gate-relative pose from PnP when valid;
- reprojection error or equivalent quality measure.

The camera frame follows OpenCV axes: +Z forward, +X image-right, +Y image-down.
Exactly one tested helper converts camera-relative vectors to body FLU. Detection
and tracking modules must not know world gate poses.

The initial detector is classical: orange-color thresholding, contour/ring
geometry, quadrilateral corners, and planar PnP using the public 2.5 m opening.
The tracker maintains a gate-relative position and velocity estimate, rejects
outliers, exposes staleness, and can coast briefly after detections disappear.
Bearing must be trusted more than monocular depth. Specific filter matrices and
controller gains are implementation details established through Package I tests.

### 7.8 Telemetry

Add telemetry incrementally as the producing package lands. The core DB component
names and meanings are:

| Component | Meaning |
|---|---|
| `drone.manual_control` | Latest normalized pilot axes, arm/mode state, and freshness/heartbeat used by manual mode |
| `drone.rc_command` | Six commanded PWM values in Section 7.1 order |
| `drone.guidance_mode` | Numeric mission phase, with the name/number mapping documented in code |
| `drone.last_gate_passed` | Last ordered gate index, initially `-1` |
| `drone.gate_pass_times` | Three simulation-time pass values for the core course; unset entries use a documented sentinel |
| `drone.gate_det` | Latest `[azimuth_rad, elevation_rad, range_m, confidence]`; invalid values use a documented sentinel |
| `drone.gate_rel_est` | Latest body-FLU `[px, py, pz, vx, vy, vz]` tracker estimate |

The referee owns race-progress components. Guidance and perception own their own
outputs. Telemetry must be observational: another subsystem must not read these
components as a shortcut around its declared contract.

## 8. Mission behavior

The complete core mission uses these observable phases:

```text
ARM → TAKEOFF → HOVER → ACQUIRE → HOLD → COMMIT → ACQUIRE ... → FINISH
                                      ↘ SEARCH ↗
```

- **ARM:** low throttle, request arm and ANGLE mode after boot grace.
- **TAKEOFF:** climb to a safe hold altitude near gate-center height.
- **HOVER:** stabilize before beginning the gate mission.
- **ACQUIRE:** establish a usable target; truth mode uses the known next gate,
  while vision mode requires confirmed detections/tracking.
- **HOLD:** maintain a target 4 m in front of the next gate along its normal.
- **COMMIT:** begin only after alignment; pass forward through the gate while
  steering on the current/coasted target estimate.
- **SEARCH:** on target loss outside a committed pass, hold safely and perform a
  bounded reacquisition behavior.
- **FINISH:** after all ordered passes, land or settle safely and disarm.

An excessive-tilt recovery may be added if needed to meet acceptance criteria,
but is not a substitute for fixing unstable control. Phase transitions must be
explicit, debounced where sensor frames are involved, and available as telemetry.

Truth mode and vision mode share mission semantics and RC conversion. They may
use different state providers, but Package J should reuse the controller proven
by Package E rather than introducing an unrelated controller.

## 9. Testing and determinism policy

- Pure geometry, packet, guidance, detector, and tracker tests must not require
  Elodin, Betaflight, an editor, or a GPU unless the tested API makes that
  unavoidable.
- Fixed sensor/synthetic inputs must produce deterministic results.
- Truth-mode integration runs must be repeatable for a fixed configuration.
- Rendered pixels are not required to be bit-identical across GPU vendors.
  Vision integration is judged by outcomes and tolerances, not frame hashes.
- Normal CI should not require a GPU. GPU-backed vision integration belongs on
  a qualified runner or nightly job.
- Every integration scenario must return a nonzero process status on failed
  acceptance criteria; grepping a warning from a successful process is not a
  test.
- Headless `elodin run` executes a generated recipe once and propagates
  simulation failures. Interactive `elodin editor` retains watch-and-reload
  behavior so source errors can be corrected without closing the editor.

The standard pure-test command, created by Package A and extended thereafter, is:

```bash
python3 -m pytest examples/betaflight-sitl/tests -q
```

## 10. Work packages

Each package is approximately one focused engineer-week. The estimate is a scope
limit, not a promise. If acceptance cannot be reached without broadening scope,
follow the change protocol rather than silently extending the package.

### [x] A — Baseline safety net

**Objective:** Make the current SITL boundary and scripted takeoff a trustworthy
base for later work.

**Prerequisites:** Current baseline in Section 3.

**Required work:**

- Add `examples/betaflight-sitl/tests/` and the standard pytest command.
- Add packet pack/unpack tests and golden canonical FDM cases for level rest and
  pure roll, pitch, and yaw rates.
- Test FLU/FRD conversion, scalar-last/scalar-first quaternion conversion,
  native motor order, and motor spin/position consistency.
- Convert the current headless takeoff result into a true pass/fail exit status.
- Assert nonzero lockstep steps, motor response, and at least 0.1 m takeoff.
- Ensure headless `elodin run` propagates simulation failure while preserving
  the editor's interactive watch-and-reload behavior.
- Repair the README's nonexistent `test_comms.py` instructions.
- Ensure running with no race environment variables remains unchanged.

**Out of scope:** Camera, gates, ANGLE mode, controller tuning, or racing.

**Acceptance:**

```bash
python3 -m pytest examples/betaflight-sitl/tests -q
elodin run examples/betaflight-sitl/main.py
```

Both commands exit zero; deliberately breaking a golden conversion or takeoff
criterion makes the corresponding command fail. The headless run prints one
stable machine-readable C0 result in addition to human diagnostics.

**Handoff:** Mark A complete, record exact commands and measured runtime, and set
the resume point to B, C, or D.

### [ ] B — FPV camera vertical slice

**Objective:** Prove independent camera production, consumption, display, and
recording without introducing racing behavior.

**Prerequisites:** Current baseline; A is recommended but not technically
required.

**Required work:**

- Implement the camera contract in Section 7.4 behind `RACE_CAMERA=1`.
- Add `sensor_view "drone.fpv"` and a frustum-enabled viewport.
- Add a neutral ground plane or other fixed visual reference so frames are not
  uniformly empty.
- Read frames non-blockingly with the specified latency at most once per nominal
  camera period; record requested sample times without claiming they are actual
  renderer capture timestamps.
- At shutdown, report first-frame time, frame count, and observed simulated FPS.
- Document video export for the generated DB.

**Out of scope:** Gates, OpenCV, guidance, or image-based decisions.

**Acceptance:** A headless camera-enabled run receives correctly shaped RGBA
frames, observes at least 15 FPS after warmup on the development GPU, does not
change the 8 kHz lockstep ordering, and produces an exportable nonempty video.
The default camera-disabled run does not start or require the render server.

**Handoff:** Record the DB path, export command, observed FPS, and any GPU-specific
limitations.

### [ ] C — Course and referee vertical slice

**Objective:** Provide independently tested course geometry, rendering, pass
scoring, and race results without autonomous guidance.

**Prerequisites:** Current baseline. This package can proceed independently of B
and D.

**Required work:**

- Implement the gate/course/referee contracts in Sections 7.5 and 7.6.
- Add `none` and `single` course selection; reserve `c1_straight` for Package F.
- Render the orange gate in the editor schematic.
- Track only ordered forward crossings and emit the race result contract.
- Add pure geometry tests for all required crossing cases.
- Ensure the referee can read truth regardless of guidance mode while exposing
  only the allowed progress event to guidance.

**Out of scope:** Camera detection, autonomous steering, or gate collisions.

**Acceptance:** Pure tests pass; `RACE_COURSE=single` visibly renders one static
gate and emits exactly one well-formed final race result. Synthetic referee tests
prove valid, missed, backward, yawed, and diagonal behavior.

**Handoff:** Record final gate coordinate/yaw conventions and the race result
example.

### [ ] D — Manual piloting, control seam, and ANGLE mode

**Objective:** Replace hardcoded RC construction with a minimal, testable command
boundary; add a safe manual piloting option; preserve default behavior; and prove
ANGLE-mode conventions before autonomous guidance.

**Prerequisites:** A.

**Required work:**

- Implement the RC, manual-input, and guidance-update contracts in Sections 7.1
  and 7.2.
- Move the existing timed script behind `RACE_GUIDANCE=scripted` without changing
  its default behavior.
- Add `RACE_GUIDANCE=manual` using an s10-managed gamepad/keyboard controller
  modeled on `examples/rc-jet/controller`; reuse or factor its proven input
  patterns where practical without coupling the two vehicle command schemas.
- Support Mode 2 by default and optional Mode 1, explicit arm/disarm and ANGLE
  controls, deadbands, clamping, smoothing where justified, and stale-input
  failsafe behavior.
- Record semantic manual input and resulting RC commands in the DB.
- Refactor `post_step` to the ordering in Section 6.2.
- Configure and verify AUX2 as ANGLE mode in `init_eeprom.py` while retaining all
  existing required EEPROM settings.
- Implement semantic-input-to-RC conversion, RC clamping, and AETR/AUX channel
  filling in one tested path shared by manual and autonomous sources.
- Add deterministic injected-input tests for throttle, roll, pitch, yaw, arm,
  mode, disconnect, and stale heartbeat behavior.
- Add an opt-in physical sign audit: from a controlled airborne condition, apply
  bounded roll and pitch commands separately and assert observed world-direction
  responses. The audit may use a short, purpose-built initial condition and
  approximate hover throttle; it must not require implementing Package E's
  position controller.
- Record the measured stick signs and encode them in one tested location.

**Out of scope:** Position hold, gate logic, camera input, autonomous flight, or
vision.

**Acceptance:** Default C0 still passes. Unit tests cover input mapping,
deadbands, clamping, channel order, one-tick command delay, and safe behavior
when the controller is absent or stale. A deterministic injected-input run proves
all four control-axis responses and ANGLE engagement. A documented manual
qualification session demonstrates that an operator can arm, take off, command
roll/pitch/yaw/throttle, land, and disarm using a supported gamepad or keyboard.

**Handoff:** Update Section 7.1 with measured signs, document controls and
failsafe timing, and record both automated and manual qualification commands.

### [ ] E — Truth-guided single-gate hold

**Objective:** Prove mission phases and control laws independently of perception.

**Prerequisites:** C and D complete.

**Required work:**

- Implement `RACE_GUIDANCE=truth` with truth state and gate poses supplied only
  through the guidance update.
- Implement ARM, TAKEOFF, HOVER, ACQUIRE, and HOLD phases.
- Use Betaflight ANGLE mode for horizontal attitude, yaw rate, and throttle.
- Hold the target point 4 m before the `single` gate along its normal.
- Emit phase and RC telemetry usable in the editor and DB.
- Keep control gains in a focused configuration object rather than environment
  variables.

**Out of scope:** Passing a gate, multi-gate sequencing, camera, or detector.

**Acceptance:** In three consecutive headless runs, the drone enters HOLD and
maintains 3D RMS position error no greater than 0.30 m from the standoff target
for a continuous 10 simulated seconds, remains armed and controlled, and does
not cross the gate. The referee measures the metric; guidance does not self-grade.

**Handoff:** Record gains, run command, measured RMS values, and any operating
envelope discovered.

### [ ] F — Truth-guided straight course

**Objective:** Complete the core mission and three-gate course using trustworthy
state, before integrating perception.

**Prerequisites:** E.

**Required work:**

- Add `c1_straight` exactly as defined in Section 7.5.
- Add COMMIT, repeated ACQUIRE/HOLD, and FINISH behavior.
- Use the referee's ordered events for scoring and race progress.
- Land or safely settle and disarm after completion.
- Allow a course-specific simulation cap up to 60 simulated seconds.
- Add a headless truth-course scenario with a nonzero exit on incomplete status.

**Out of scope:** Camera use, detector/tracker input, lap-time optimization, or
harder courses.

**Acceptance:** Three consecutive clean-start headless runs emit
`gates_passed=3/3 status=COMPLETE` within 60 simulated seconds and finish safely.
For a fixed configuration, pass times remain within a documented small tolerance;
exact bitwise equality is not required if Betaflight prevents it.

**Handoff:** Record the three pass-time sets, time cap, completion command, and
known control limitations.

### [ ] G — Vision contracts and synthetic geometry

**Objective:** Establish and prove camera/detection geometry without depending on
rendered image detection or flight behavior.

**Prerequisites:** B.

**Required work:**

- Implement the camera calibration, candidate detection, corner-ordering, and
  camera-to-body contracts in Sections 7.4 and 7.7.
- Implement truth-based projection of a gate's inner corners for tests and
  offline synthetic evaluation only.
- Add configurable deterministic pixel noise/dropout for synthetic sequences.
- Round-trip projected corners through planar PnP and the body transform.
- Ensure no production vision guidance path can obtain the truth inputs used by
  the synthetic generator.

**Out of scope:** HSV detection, tracking, live guidance, or dataset training.

**Acceptance:** Deterministic tests cover centered, translated, yawed, and ranged
gates; projection followed by PnP recovers bearing within 0.25 degrees and
translation within tolerances justified in the test for noise-free fixtures.
Tests fail on wrong corner order, intrinsics, tilt sign, or camera/body transform.

**Handoff:** Record exact matrix conventions, tolerances, and fixture-generation
command.

### [ ] H — Classical detector, offline

**Objective:** Detect rendered training gates and estimate their relative pose
without introducing closed-loop flight risk.

**Prerequisites:** B, C, and G.

**Required work:**

- Implement saturated-orange thresholding, contour/ring filtering, inner-quad
  extraction, required corner ordering, confidence, and planar PnP.
- Return zero or more candidates through the Section 7.7 contract.
- Run only on nominally fresh 30 Hz frame samples supplied by the adapter.
- Add fixed checked-in fixtures or deterministic fixture generation plus an
  offline evaluation command over recorded camera data.
- Report detection rate by range, bearing error against synthetic truth, PnP
  reprojection error, and processing time.
- Document the OpenCV dependency and keep OpenCV types inside the detector.

**Out of scope:** Tracker, controller, live RC output, CNN, or partial-gate neural
inference.

**Acceptance:** On the controlled core-course fixture set from 4–20 m, at least
90% of frames with a fully visible gate produce a valid candidate and median
bearing error is at most 2 degrees. Accuracy is a deterministic test. Processing
median should be reported and targeted at 5 ms or less on the development host,
but hardware-dependent timing is not a normal CI failure.

**Handoff:** Record fixture provenance, metrics, command, thresholds, and known
failure ranges.

### [ ] I — Gate-relative tracker, offline

**Objective:** Produce a stable relative target through depth noise, outliers,
and short close-range detector loss.

**Prerequisites:** G. This package can proceed in parallel with H.

**Required work:**

- Implement a gate-relative position/velocity tracker in body FLU.
- Consume only Section 7.7 detections plus allowed IMU/time information.
- Model bearing as more reliable than monocular depth.
- Reject implausible innovations, expose confidence/staleness, and reset after a
  documented rejection streak.
- Coast for a bounded interval while rotating the remembered target consistently
  with body motion.
- Expose a geometric passed indication when the tracked gate moves behind the
  vehicle, without using a blind timer as the sole signal.

**Out of scope:** Image processing, world localization, live flight, or tuning on
hidden world gate poses.

**Acceptance:** Deterministic synthetic tests demonstrate convergence, bounded
response to alternating depth errors, outlier rejection, stale-state behavior,
short coast-through, reset, and passed indication. The tracker produces no NaNs
or unbounded velocity under any fixture.

**Handoff:** Record state definition, noise assumptions, staleness limits, and
all test commands.

### [ ] J — Vision-guided single-gate hold

**Objective:** Close the flight loop on rendered vision while using truth only as
an independent grader.

**Prerequisites:** E, H, and I.

**Required work:**

- Implement `RACE_GUIDANCE=vision` and enforce Section 7.3 truth isolation.
- Connect fresh camera-frame samples to detector, tracker, and the mission/RC
  control structure proven in E. Replace truth position errors with allowed
  gate-relative tracker errors and barometric/visual vertical information rather
  than creating a second unrelated controller.
- Take off using allowed inertial/barometric information, acquire the `single`
  gate, and hold the 4 m standoff.
- Add debounced acquisition and safe behavior for stale/lost tracks.
- Emit phase, detection, tracker, and RC telemetry without exposing hidden truth
  to guidance.
- Add a test that a vision guidance update contains no truth or gate poses.

**Out of scope:** Passing the gate, multi-gate selection, CNN, or global pose
estimation.

**Acceptance:** In three consecutive qualified GPU runs, vision guidance enters
HOLD and remains within 0.50 m 3D RMS of the truth-scored standoff target for a
continuous 10 simulated seconds. During that interval, detector-or-tracker target
availability is at least 90%, and guidance has no truth access.

**Handoff:** Record run DBs, RMS/tracking metrics, detector loss behavior, and
whether any approved contract updates were required.

### [ ] K — Vision-guided straight course

**Objective:** Complete the three-gate core course using rendered frames and
allowed sensor/race context only.

**Prerequisites:** F and J.

**Required work:**

- Add vision COMMIT, bounded coast-through, SEARCH, and reacquisition behavior.
- Handle multiple nearly collinear visible gates without target flip-flop using
  target hysteresis and ordered race progress.
- Cross-check tracker passage behavior against referee events in telemetry while
  keeping referee truth out of guidance.
- Finish and safely disarm after all three gates.
- Add a vision-course scenario with nonzero exit on incomplete status.

**Out of scope:** Harder layouts, smaller gates, aggressive lap times, rate mode,
CNN, or recovery through arbitrary crashes.

**Acceptance:** Three consecutive clean-start qualified GPU runs complete all
three ordered gates within 60 simulated seconds and emit `status=COMPLETE`.
Guidance receives no world pose or gate poses. Failures to detect, track, search,
or pass are distinguishable in telemetry.

**Handoff:** Record pass times, run DBs, target-selection behavior, known failure
modes, and the exact qualification command.

### [ ] L — CI and release-quality cleanup

**Objective:** Make completed core behavior maintainable and understandable to a
new user.

**Prerequisites:** F for normal CI; K for vision CI and final completion.

**Required work:**

- Run pure tests and C0 scripted integration in normal CI.
- Run truth C1 in normal CI if runtime is acceptable; otherwise define an
  approved scheduled tier while retaining a per-change lower-cost truth check.
- Run vision C1 on a qualified GPU/nightly runner; do not pretend CPU-only CI
  verifies rendered vision.
- Verify integration commands fail correctly on scenario failure.
- Update the example README with concise setup, modes, courses, result format,
  troubleshooting, and links to this plan.
- Remove obsolete instructions and ensure all shipped documentation is public
  and self-contained.
- Review this document for actual contracts, mark core status, and set the next
  resume point to optional work or maintenance.

**Out of scope:** Adding new racing capabilities.

**Acceptance:** CI tiers and ownership are documented and green; a new user can
build, initialize, run C0, run truth C1, and understand how qualified vision C1
is run using only repository documentation.

**Handoff:** Mark L and the core plan complete, summarize remaining limitations,
and list optional work as separately approved proposals.

## 11. Dependency graph and interruption points

```text
A ──> D ──> E ──> F ──────────────> K ──> L
          ▲                       ▲
C ───────┘                        │
                                  │
B ──> G ──> H ──┐                 │
         └─> I ─┴─> J ────────────┘
```

C is also required by H. B, C, and most of A can be implemented independently.
The control path (D–F) and perception path (B, G–I) remain separate until J.

Useful stopping points are:

- after A: trustworthy existing SITL example;
- after B: camera-equipped SITL example;
- after C: reusable course and scoring harness;
- after D: manually pilotable Betaflight vehicle with verified RC conventions;
- after E: truth-guided single-gate position hold;
- after F: complete autonomous truth race;
- after H/I: offline validated perception stack;
- after J: vision-guided position hold;
- after K: complete vision-guided straight race.

No stopping point should leave the default example broken or require an unmerged
branch for later resumption.

## 12. Current status and resume point

### Package status

| Package | Status | Evidence / notes |
|---|---|---|
| A | Complete | 24 pure tests pass; C0 returned 0 with 119,995 simulation-loop lockstep responses, max motor 0.574, and 56.836 m takeoff rise in 20 wall-clock seconds. Deliberate 100 m criterion returned 1. Shared headless propagation fix: `bd3aa4b9`. |
| B | Not started | Platform camera API exists; no Betaflight integration |
| C | Not started | No course or referee code |
| D | Not started | RC remains hardcoded; no manual mode; AUX2 ANGLE not configured |
| E | Blocked by C, D | No truth guidance |
| F | Blocked by E | No course controller |
| G | Blocked by B | No racing camera geometry contract in code |
| H | Blocked by B, C, G | No detector |
| I | Blocked by G | No tracker |
| J | Blocked by E, H, I | No vision guidance |
| K | Blocked by F, J | No vision race |
| L | Blocked by F/K | No race CI or final docs |

### Resume point

Implement **Package D — Manual piloting, control seam, and ANGLE mode**.

Packages B and C remain valid independent alternatives if camera or course work
is prioritized first. Package D is the recommended continuation because A now
provides its required protocol, convention, and C0 safety net.

Package A verification from the repository root, with the Elodin Python
environment active and the current release CLI on `PATH`:

```bash
python3 -m pytest examples/betaflight-sitl/tests -q
elodin run examples/betaflight-sitl/main.py
```

The verified C0 result was:

```text
[C0] lockstep_steps=119995 motor_response=true max_motor=0.574 takeoff_delta_m=56.836 status=PASS
```

The pure suite passed 24 tests in 0.05 seconds and C0 completed in 20 wall-clock
seconds. Raising the takeoff criterion temporarily to 100 m emitted
`status=FAIL` and returned process status 1; the required 0.1 m criterion was
then restored.

## 13. Decision log

| Date | Decision | Reason and affected packages |
|---|---|---|
| 2026-09-03 | Run headless recipes once while retaining watched recipes in the editor. | Package A exposed that a failed simulation child was logged and then waited for source reload, so `elodin run` could not return nonzero. The approved shared fix (`bd3aa4b9`) makes headless execution one-shot without changing interactive editor recovery. This enables failure contracts in A, F, K, and L. |
| 2026-09-01 | Keep the current ENU/FLU world, Gazebo-bridge conventions, native motor order, and 8 kHz lockstep. | These are the implemented baseline; changing them is not required for racing. A–L rely on them. |
| 2026-09-01 | Preserve scripted takeoff as the default and make other control modes opt-in. | Allows every package to merge independently without replacing the reference SITL example prematurely. |
| 2026-09-01 | Add manual piloting through the same semantic-input-to-RC boundary before autonomy. | Separates vehicle controllability and Betaflight integration from guidance behavior; D provides gamepad/keyboard control and safe stale-input handling. |
| 2026-09-01 | Prove control with truth before integrating perception. | Separates controller failures from detector/tracker failures. E/F precede J/K. |
| 2026-09-01 | Use ANGLE-mode RC first. | Retains Betaflight stabilization and limits initial control scope. D–K rely on it. |
| 2026-09-01 | Use a level 640×360 camera and 2.5 m orange training gates for the core course. | Provides controlled geometry for incremental bring-up. B, C, G–K rely on it. |
| 2026-09-01 | Start with classical vision and a gate-relative tracker; defer CNN and global VIO. | Makes perception inspectable and testable offline. H–K rely on it. |
| 2026-09-01 | Keep this file self-contained and require approval for material plan changes. | Work is expected to span agents and long interruptions. All packages rely on the governance rules. |

## 14. Glossary

- **Betaflight SITL:** The real Betaflight flight-controller software compiled to
  run as a host process against simulated sensors and motors.
- **FDM packet:** Flight-dynamics-model packet sent from Elodin to Betaflight.
- **Lockstep:** One FDM packet triggers one Betaflight gyro/PID/mixer iteration
  and one motor response before the simulation proceeds.
- **ANGLE mode:** Betaflight self-leveling mode where roll/pitch sticks represent
  attitude targets rather than raw body-rate targets.
- **Manual mode:** An opt-in gamepad/keyboard command source used to pilot the
  simulated vehicle through the same Betaflight RC path as autonomous guidance.
- **FPV camera:** Forward-facing camera used as the vision sensor.
- **PnP:** Perspective-n-point pose recovery from known 3D geometry and observed
  image points.
- **Gate-relative tracker:** Filter estimating gate position and velocity relative
  to the drone, without estimating a global drone pose.
- **Truth:** Simulated world state or hidden gate pose unavailable to vision
  guidance.
- **Referee:** Trusted host-side scorer that uses truth to evaluate ordered gate
  crossings and metrics.
- **Qualified GPU run:** A run on hardware/environment known to support the
  render server at the required camera rate; rendered pixels need not be
  bit-identical to another GPU.
