# Vision-Guided Gate Racing Plan

**Document status:** Authoritative living specification  
**Last verified against repository:** 2026-10-28 (clean G1 source reconstruction; untested)
**Current resume point:** E1 complete; G1 implementation complete on `g1-clean`, pending user diff approval and fresh qualification. F implementation ready; H/I and F qualification wait for G1.

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

The example now additionally contains:

- Package A's pure protocol/convention tests and machine-readable C0 result;
- Package B's opt-in FPV camera (`RACE_CAMERA=1`); the default scripted run
  still does not require a GPU render server;
- Package C's opt-in `none`/`single` course geometry, four-bar gate rendering,
  ordered truth referee, fixed-width race telemetry, final race result, and
  deterministic live positive-crossing qualification;
- Package D's command seam, manual-controller integration, ANGLE configuration,
  one-tick RC latch, automated physical sign audit, and operator-qualified
  keyboard and gamepad control paths;
- Package E's opt-in truth-guided single-gate ARM/TAKEOFF/HOVER/ACQUIRE/HOLD
  controller, independent 10-second standoff hold grader, and phase/RC telemetry;
  and
- Package E1's shared mission controller behind immutable navigation/target
  inputs, with truth guidance reduced to a per-tick adapter. The initial E1
  exact-equivalence qualification is preserved as historical evidence; the
  maintained characterization is now a compact 8-case/78-update digest set.

The camera is independently opt-in for scripted, manual, and truth guidance; it
is observational in truth mode and never replaces copied truth state.

The example does **not** currently contain vision guidance or
perception/tracking, the three-gate course, autonomous gate-passing/COMMIT
behavior, race completion tests, or race CI.

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

### 3.3 Simulated aids and interface findings

A review of the example at `a218f7a6` and the pinned Betaflight `6dbc4218`
on 2026-10-21 established these baseline findings. G1's timing and GPS changes
are complete; remaining findings still constrain later packages.

- **Camera time.** G1 replaces `frame_sample_time` (query-time microseconds)
  with selected image-message time in simulation seconds; see Section 7.4.
- **Truth-derived aids inside Betaflight.** `FEATURE_GPS` is in the SITL
  default features, and the GPS provider defaults to `GPS_VIRTUAL`, fed at
  10 Hz from FDM truth position and velocity. With a fix, Betaflight's Mahony
  filter corrects yaw from course over ground and initializes yaw from it once
  ground speed exceeds its minimum. In ANGLE mode this reaches only
  Betaflight's yaw estimate, not leveling. The FDM sends ENU metres where the
  virtual GPS expects longitude and latitude in degrees, so its position is not
  a valid GPS trajectory. The virtual magnetometer is synthesized from the
  truth quaternion but is unused because `trust_mag` defaults off. The virtual
  barometer uses truth altitude but is unused because no altitude or position
  mode is configured. The build does not define `SITL_ATTITUDE_DIRECT`, so
  Betaflight runs its own attitude estimator on the simulated IMU.
- **Scene.** With the camera enabled, the visual world is a uniform 40 × 40 m
  ground plane centred on the origin, the sky, and the orange gate bars. The
  plane ends at x = 20 m. There is too little texture for generic
  feature-based visual-inertial odometry.
- **Sensors.** The configured barometer noise variance is 0.01 m² (σ = 0.1 m),
  now correctly documented in `sensors.py`. The simulated sensors
  are benign compared with hardware: negligible gyro bias drift, no
  accelerometer bias, and a magnetometer aligned with true north. Estimator
  noise parameters must stay explicit so results are not mistaken for
  hardware performance.
- **Close-range visibility.** A centred 2.5 m opening fills the 58.72°
  vertical field of view at 2.22 m, so a detector that needs four corners
  loses the gate for the final ~2.2 m of every pass.

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
- Supplying vision estimates to Betaflight as GPS or through any other
  Betaflight navigation input.
- Direct motor control as a product mode.
- A global or generic feature-based visual-inertial odometry system (for
  example OpenVINS or Basalt). The core course's rendered scene has too little
  texture (Section 3.3).
- A prior gate map in vision mode.
- Optimizing lap time beyond completing the training course reliably.

### 5.3 Deferred until the core plan demonstrates a need

- Strict rendered-pose/message-time alignment and aligned-stream optimization
  (Section 7.4).
- ACRO/rate-mode guidance and advanced attitude controllers.
- CNN or other learned gate detectors.
- Spec-size gates, sharp turns, slaloms, vertical courses, and procedural courses.
- MPCC, SE(3), INDI, or other advanced control schemes.
- A large flight-reporting or leaderboard framework.
- Converting the simulation world to NED/FRD.
- Corner-pixel (tightly coupled) gate updates for partially visible gates.
- Tracking the following gate as a second landmark during COMMIT.
- Gravity-constrained PnP and vision-based gravity updates.
- Gyro-bias estimation and hardware-like sensor noise.
- Drag- or thrust-model velocity aiding during coast-through.
- A GPS-adapter experiment that feeds the navigator's estimate to Betaflight,
  measured against the companion navigator.
- A visual-inertial odometry benchmark on a textured scene.
- Exporting Betaflight's attitude estimate as a cross-check.

Deferred work must not be introduced as part of Packages A–L, E1, or G1
without approval.

## 6. System architecture and invariants

### 6.1 Processes

The running system always has the simulation and Betaflight processes and may
add independently supervised render-server, manual-controller, and editor
processes:

```text
Manual controller (interactive manual mode only; not the automated audit)
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
5. Run the selected command source—manual, scripted, simulation-time audit,
   truth, or vision—and retain its RC command for the next physics tick.
6. Run referee scoring from ground truth and emit telemetry/results.

A command selected on tick N therefore reaches Betaflight on tick N+1. At the
8 kHz default this is 125 microseconds. Tests comparing command and response
must account for that one-tick delay.

Perception runs only when a fresh camera frame is available. A slow detector may
reduce wall-clock realtime factor, but must not alter simulation-time ordering or
starve Betaflight of FDM packets.

Vision guidance accumulates gyro and accelerometer samples on every tick, runs
the detector on fresh frame samples, and steps the attitude filter, navigator,
and mission controller every 16th tick (500 Hz at the 8 kHz default). Its
500 Hz inputs are interval means, which are exactly what the Package H recorder
stores, so live and replayed runs execute the same navigation code. Truth
guidance continues to update on every tick. Lockstep makes estimator cost a
throughput concern rather than a per-tick correctness limit, but it must be
measured and reported.

### 6.3 Runtime configuration

The following environment variables form the core user-facing configuration:

| Variable | Values | Default | Meaning |
|---|---|---|---|
| `RACE_GUIDANCE` | `scripted`, `manual`, `truth`, `vision` | `scripted` | Selects the RC command source |
| `RACE_COURSE` | `none`, `single`, `c1_straight` | `none` | Selects course geometry |
| `RACE_CAMERA` | `0`, `1` | `0` | Enables the FPV camera independently for bring-up |

`manual` starts or connects to the manual controller and does not require a
course or camera. The `RACE_MANUAL_AUDIT=1` qualification override selects an
in-process, simulation-time command source instead and does not start that
controller. `vision` requires `RACE_CAMERA=1` and a non-`none` course; any
other combination fails at startup with exit status 2 and a clear message.
`truth` requires a non-`none` course. Unknown values must fail at startup.
Running with no environment variables must preserve the current scripted
behavior and must not require a manual controller or GPU render server.

Package H adds the `RACE_RECORD=<directory>` tooling switch for camera-enabled
runs. It writes the offline replay bundle described in Package H and does not
change guidance behavior.

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

The Package D physical audit measured these semantic signs in the live SITL:
positive roll is right-wing-down (`+X` FLU rotation), positive pitch is
nose-down/forward (`+Y` FLU rotation), and positive yaw is nose-right/clockwise
(`-Z` FLU rotation). Roll and pitch map directly above PWM center; right yaw
maps below PWM center. A bounded normalized `+0.25` roll or pitch command
(`1625` PWM) produced approximately 5 degrees of ANGLE-mode attitude before the
audit reversed it; the peak rates were 0.446 rad/s roll and 0.444 rad/s pitch.
The tested `semantic_to_rc` helper in `controls.py` encodes these signs and maps
full normalized axis travel to 1000–2000 PWM.

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

#### Automated physical audit

`RACE_GUIDANCE=manual RACE_MANUAL_AUDIT=1` replaces the interactive manual
source with an in-process `AuditGuidance`. It keys every command phase directly
to `GuidanceUpdate.sim_time`: five simulated seconds safe/disarmed, bounded
armed throttle and positive/negative roll, pitch, and yaw pulses, then a safe
disarmed ending. It does not start the external Rust controller. Its semantic
commands use the common conversion and one-tick RC latch before reaching
Betaflight, and `AxisAudit` assesses only the command actually exchanged and
the measured physical response. This keeps the qualification deterministic
when lockstep runs slower than wall-clock real time.

### 7.2 Guidance update

A guidance object is stateful for one run and receives one update per physics
tick. The update contains:

- simulation time and tick;
- gyro and accelerometer samples;
- barometer and magnetometer values plus freshness when available;
- latest FPV RGBA frame or `None`, plus its nominal image/render time
  (`frame_time`, simulation seconds; Section 7.4) and a freshness flag;
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
  and ordered pass events are allowed;
- the adapter constructs only the base `GuidanceUpdate`; `TruthGuidanceUpdate`,
  `TruthState`, and `GatePose` values are never built;
- `gate_detector.py`, `navigation.py`, and `vision_guidance.py` must not import
  `course`, `referee`, `referee_audit`, `hold_referee`, `sim`, `main`,
  `synthetic_vision`, or `replay`. `test_vision_isolation.py` covers each
  module as it lands.

The referee always uses truth for scoring, including in vision mode. Package J
must include a test that constructs a vision update and proves truth and gate
poses are unavailable.

Isolation also covers the flight controller. Package G1 disables Betaflight's
GPS feature (`feature -GPS`) and verifies `trust_mag = OFF` in every mode. That
removes the only truth-derived aid that reaches Betaflight's estimator in this
configuration (Section 3.3). Betaflight's truth-altitude barometer remains
compiled in but unused, because no altitude or position mode is configured. No
mode that consumes position or altitude may be enabled without an approved
design. Vision results are labeled by sensor set: "camera, IMU, barometer, and
magnetometer on the companion; referee pass events used for gate sequencing".
They must not be described as sensor-only autonomy.

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
period. `FpvCamera` queries `read_msg_at` at `ctx.timestamp - latency` and
keeps that requested time internally for its own accounting.
`GuidanceUpdate.frame_time` is the selected DB image-message timestamp T,
converted by integer timestamp delta to simulation seconds on `sim_time`.
It is nominal image/render time, not verified exact pose/exposure time.
`frame_fresh` is true only when `read_msg_at` selects a timestamp different
from the previous sample. A repeated timestamp is sample-and-hold: the held
frame and its original `frame_time` may still be supplied, with
`frame_fresh=False`. Shutdown accounting reports every renderer frame
(`total_frames`) separately from the frames inside the post-warmup window used
for observed FPS. Until the renderer produces a frame, guidance receives
`None` with `frame_time=None`. Missing/malformed samples and an unsampled
camera period also carry no frame/time; frame copying is unchanged.

**Known limitation (accepted 2026-10-27):** Scene-time alignment is approximate:
the renderer uses an independently phased, throttled pose cache. One reviewed
capture showed body state about 22 ms earlier than message T, not a universal
offset or sub-frame guarantee; phase/backlog can add delay. Sparse unchanged
state's last-change T is not actual image time. Future fusion must not assume
exact scene rewind or compensate with a constant offset. Revisit alignment for
timing-linked gate bias, innovations/rejections, vision hold/pass failures, or
higher dynamics/backlog—not only ACRO. Strict alignment is deferred, outside G1
acceptance.

Before G1, `frame_sample_time` carried query time in DB-timestamp microseconds.

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
referee and PnP must be 2.5 m. With the camera enabled, the neutral ground plane
must extend past the last gate of the selected course. Package F enlarges it
for `c1_straight`, for example to 80 × 80 m.

### 7.6 Referee and race result

Only the next gate in sequence can count. For each physics tick:

1. Transform previous and current drone positions into the next gate's local
   frame.
2. Require forward crossing: `previous_x < 0 <= current_x`.
3. Interpolate the crossing fraction at local `x = 0`.
4. Test interpolated local y/z against `±inner_size/2`.
5. On success, record gate index and simulation time exactly once.

The pass timestamp is linearly interpolated between the previous and current
simulation sample times using the same plane-crossing fraction; it never uses
wall-clock time. Lap time runs from simulation time zero through the final
ordered crossing. An enabled course emits exactly one final result on complete,
incomplete, or interrupted simulation shutdown after execution starts.

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

- nominal image/render time (`GuidanceUpdate.frame_time`, simulation seconds),
  retained unchanged as `GateCandidate.sample_time` (Section 7.4);
- four inner-opening corners ordered top-left, top-right, bottom-right,
  bottom-left in pixel coordinates;
- `corner_order_ambiguous`, a heuristic warning set when the two best square
  labellings are within 10°. An unset flag does not guarantee the gate-frame
  orientation: perspective can still mislabel corners. Gate position, bearing,
  range, and the crossing normal are unaffected by a cyclic relabelling. The
  in-plane gate orientation is not reliable;
- confidence in `[0, 1]`;
- gate-relative pose from PnP when valid;
- reprojection error or equivalent quality measure.

The camera frame follows OpenCV axes: +Z forward, +X image-right, +Y image-down.
Exactly one tested helper converts camera-relative vectors to body FLU. Detection
and tracking modules must not know world gate poses.

The initial detector is classical: orange-color thresholding, contour/ring
geometry, quadrilateral corners, and planar PnP using the public 2.5 m opening.
It lives in `gate_detector.py`, and OpenCV types never leave that module.

The gate tracker is the gate navigator of Section 7.10. It estimates the
vehicle's motion and the target gate's centre in a gravity-aligned local frame,
rejects outliers, exposes staleness, and coasts on the IMU after detections
disappear. It publishes the body-FLU gate-relative estimate of Section 7.8.
Bearing must be trusted more than monocular depth. Filter tuning and controller
gains are implementation details established through Package I tests.

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
| `drone.gate_det` | Latest associated candidate `[azimuth_rad, elevation_rad, range_m, confidence]` in body FLU, written on every fresh frame; invalid: range `-1`, confidence `0` |
| `drone.gate_rel_est` | Body-FLU `[px, py, pz, vx, vy, vz]`: the target gate centre relative to the vehicle, `R_BL (g − p)`, and the physical relative velocity, `−R_BL v`. This is not the time derivative of the body-frame vector, which adds `−ω × r`. Invalid: zeros, with track code 0 in `drone.nav_gate` |
| `drone.nav_est` | Navigator `[x, y, z, vx, vy, vz, qx, qy, qz, qw]` in the local frame L (Section 7.10); invalid while `drone.nav_diag` ready is 0 |
| `drone.nav_gate` | Target gate `[x, y, z, yaw, track_code, age_s]` in L. Track codes: 0 none, 1 tentative, 2 tracking, 3 coasting, 4 lost |
| `drone.nav_diag` | `[ready, last_nis, rejected_streak, relative_sigma_horizontal_m, step_compute_us]` |

Navigation telemetry is written in one batch at 50 Hz; `drone.gate_det` is
written on every fresh frame. The referee owns race-progress components.
Guidance and perception own their own outputs. Telemetry must be observational:
another subsystem must not read these components as a shortcut around its
declared contract.

### 7.9 Mission controller inputs

One mission controller in `guidance.py` serves truth and vision guidance. It
consumes a navigation state, a gate target, and the race-progress fields of the
guidance update (`last_gate_passed`, `next_gate_index`, `gate_count`). It never
reads truth types directly.

| Type | Field | Meaning |
|---|---|---|
| `NavState` | `position`, `velocity` | Body origin and its velocity in a gravity-aligned frame: ENU world in truth mode, local frame L in vision mode |
| | `yaw` | Heading of body +X about +Z in that frame |
| | `yaw_rate` | Body z gyro rate |
| `GateTarget` | `index` | Ordered gate index |
| | `center`, `yaw` | Gate centre and yaw in the same frame as `NavState` |
| | `inner_size` | Public inner opening |
| | `track` | `TrackState` (`TENTATIVE`, `TRACKING`, `COASTING`, `LOST`); always `TRACKING` in truth mode |
| | `age` | Seconds since the last visual update; always 0 in truth mode |

The controller uses only differences between `NavState` and `GateTarget`, plus
yaw. The frame origin therefore does not matter, and a constant attitude-filter
yaw error rotates the state, the target, and the heading used to map
accelerations to sticks together. `TruthGuidance` is a thin adapter that fills
both inputs from `TruthGuidanceUpdate`; vision guidance fills them from the
navigation estimate.

### 7.10 Navigation estimate

Package I's attitude filter and gate navigator run inside vision guidance. They
never see pixels, `StepContext`, or truth.

**Frames and units.** The local navigation frame L has its origin at the body
origin where the navigator initializes on the pad. +Z is up (gravity-aligned);
+X is east and +Y north, with heading from the magnetometer. In this simulator
L's axes match Elodin ENU up to the attitude filter's yaw error, but guidance
never compares L with ENU. `R_LB` maps body FLU vectors into L. Quaternions are
scalar-last. Gravity in L is `[0, 0, −9.81]` m/s². The accelerometer reads body
specific force, `+9.81` m/s² on z at rest. `CameraMount` in `vision.py` remains
the only body–camera transform. All guidance times are simulation seconds on
the `sim_time` clock.

**Attitude filter.** A complementary filter on `q_LB` integrates the mean gyro
rate over each 2 ms step and applies slow corrections. Accelerometer tilt
correction applies only when specific force is near `g` and the rotation rate is
small, because during sustained tilt the accelerometer measures thrust and drag,
not gravity. Magnetometer yaw correction uses the tilt-compensated field. The
filter initializes from the averaged gravity direction and field after a
stationary interval following the spawn drop.

**Navigator.** A 12-state linear Kalman filter over `x = [p, v, b, g]` in L:
body-origin position and velocity, a slowly varying acceleration error
(accelerometer bias plus attitude leakage), and the target gate's centre.
Prediction uses `a = R_LB f̄ + g_L − b`. Because attitude is a known input from
the attitude filter, the translational filter is linear: no Jacobians,
quaternion error states, or reset steps. A separate scalar filter estimates the
gate's yaw. Updates use the Joseph form.

**Measurements.**

- Gate position from PnP, applied at nominal image/render time (Section 7.4)
  using a short ring buffer of past `(t, p, R_LB)`. Its noise is anisotropic:
  bearing-limited across the line of sight and depth-limited, growing with
  range squared, along it. Frames older than the buffer or newer than the
  current step are dropped and counted. A candidate whose gate normal is not
  nearly horizontal in L is rejected, because the gates are upright.
- Barometer altitude relative to a pad reference.
- Zero velocity while the mission controller reports the vehicle on the pad.
  The controller asserts this until it first requests arming; the navigator
  never infers it from sensors, because steady cruise looks stationary.

**Gate lifecycle.** The target is `GuidanceUpdate.next_gate_index`; a change
clears the gate block. Track states are EMPTY, TENTATIVE, TRACKING, COASTING,
and LOST. The largest valid candidate initializes the gate, keeping the
cross-covariance with the vehicle position. Once initialized, the candidate with
the smallest Mahalanobis innovation is accepted inside a χ²(3) gate, so farther
gates seen through the current one are not mistaken for it. A reset requires a
streak of fresh frames whose candidates are all rejected while the gate is
predicted fully in view; expected close-range invisibility never counts toward a
reset. The geometric passed indication is the estimated gate plane falling
behind the vehicle. It is published for comparison with referee events; the
mission sequences gates on the referee's ordered pass events.

**Outputs.** `NavEstimate` carries time, a ready flag, position, velocity,
`q_LB`, yaw rate, acceleration bias, position sigma, and an optional
`GateEstimate`: index, centre, yaw, track state, last update time,
relative-position sigma, yaw sigma, last NIS, and rejection streak. Starting
parameters are listed with Package I; they are tuning, not contract.

## 8. Mission behavior

The course selects a mission profile. `single` runs the **hold** profile
(Packages E and J); `c1_straight` runs the **race** profile (Packages F and K).
The complete core mission uses these observable phases:

```text
ARM → TAKEOFF → HOVER → ACQUIRE → HOLD → COMMIT → ACQUIRE ... → FINISH
                           ↘ SEARCH ↗   (vision mode only)
```

`drone.guidance_mode` codes 0–6 keep their current values. Packages F and K add
codes 7–9.

| Code | Phase | Behavior | Exit |
|---|---|---|---|
| 0 | DISABLED | Scripted and manual modes | — |
| 1 | ARM | After the boot grace (in vision mode, also once the navigator is ready and the target is TRACKING), request arm and ANGLE at low throttle | After the settle time: TAKEOFF. Arm timeout: FAULT |
| 2 | TAKEOFF | Hold the launch XY; ramp the altitude setpoint to gate-centre height | Within tolerance for the dwell: HOVER |
| 3 | HOVER | Hold the launch XY at gate height | After the dwell: ACQUIRE |
| 4 | ACQUIRE | Fly to the standoff 4 m before the target gate along its normal, facing the gate. In vision mode, hold the current position until the target is TRACKING | Within tolerance for the dwell: HOLD. Vision mode, 3 s without TRACKING: SEARCH |
| 5 | HOLD | Hold the standoff with the braking slab active | Hold profile: stays. Race profile: after 1 s with yaw error under 5°, the target TRACKING, and (vision mode) horizontal relative σ under 0.10 m: COMMIT |
| 6 | FAULT | Truth mode: hold the last valid state. Vision mode: brake, descend at 0.4 m/s, then disarm | — |
| 7 | COMMIT | Braking off. Position target 2 m beyond the gate centre along its normal, at a bounded commit speed. Continues through COASTING | Referee pass event: ACQUIRE for the next gate, or FINISH after the last. More than 1.5 m past the plane without a pass event, or LOST: FAULT |
| 8 | SEARCH | Vision mode only, from ACQUIRE or HOLD when the target is LOST. Hold the estimated position and yaw toward the last estimated bearing, within ±30° | TRACKING: ACQUIRE. After 6 s: FAULT |
| 9 | FINISH | Brake to zero velocity, then descend at 0.4 m/s to 0.15 m above the pad | Disarm |

Until Package K adds SEARCH, vision mode enters FAULT wherever this table says
SEARCH. In the hold profile any gate crossing remains a fault, as Package E
requires. In the race profile crossings are expected and are confirmed by
referee pass events. Phase transitions must be explicit, debounced where sensor
frames are involved, and available as telemetry. An excessive-tilt recovery may
be added if needed to meet acceptance criteria, but is not a substitute for
fixing unstable control.

The race profile must finish inside the 60 s cap with margin. Package E's hold
profile reaches HOLD at about 27.6 s, so the race profile needs its own faster
climb and transit configuration. A starting budget is a 0.4 m/s climb,
1.2–1.5 m/s transit and commit speeds, and 1 s holds: about 11 s to HOVER, about
10 s per gate, and about 5 s for FINISH. The hold profile keeps Package E's
qualified configuration.

Truth mode and vision mode share mission semantics, one mission controller
(Section 7.9), and RC conversion. They differ only in the adapter that fills the
controller's inputs.

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
- Recorded replay bundles (Package H) run the detector and navigator on the CPU.
  Small checked-in fixtures, about 2 MB in total at most, may run in normal CI;
  larger recordings stay local. Truth in a bundle is stored separately and is
  read only by the evaluator.
- Before vision guidance gains control authority, the detector and navigator
  run in shadow mode during truth-guided camera runs (Packages J and K). They
  publish navigation telemetry with no control authority, so estimator error is
  measured against truth at the time the controller would have used it.
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

### [x] B — FPV camera vertical slice

**Objective:** Prove independent camera production, consumption, display, and
recording without introducing racing behavior.

**Prerequisites:** Current baseline; A is recommended but not technically
required.

**Required work:**

- Implement the camera contract in Section 7.4 behind `RACE_CAMERA=1`.
- Add `sensor_view "drone.fpv"`. The camera frustum stays available in the
  inspectors and is hidden by default.
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

**Verified 2026-09-26 after merging `main` (`84b7692f`) and the PR 853 review fixes
(WSL2, NVIDIA GeForce RTX 3050 6GB Laptop GPU):**

- `python3 -m pytest examples/betaflight-sitl/tests -q` → 109 passed.
- Default: `RACE_CAMERA=0 elodin run examples/betaflight-sitl/main.py` →
  `FPV camera: disabled`, no render-server, C0 `status=PASS`,
  lockstep_steps `119995`, exit 0 (DB `betaflight_db022`).
- Camera: `RACE_CAMERA=1 elodin run examples/betaflight-sitl/main.py` →
  `drone.fpv` 640×360, `total_frames=437`, `window_frames=389`,
  `observed_sim_fps≈29.92`, `held_samples=0` on this run, `status=PASS`,
  C0 `PASS`, exit 0 (DB `betaflight_db024`).
- Camera plus course: `RACE_CAMERA=1 RACE_COURSE=single elodin run examples/betaflight-sitl/main.py` →
  `[FPV] total_frames=428 window_frames=389 observed_fps=29.92 status=PASS`
  and `[RACE] course=single gates_passed=0/1 status=INCOMPLETE`, exit 0
  (DB `betaflight_db026`). Export:
  `elodin-db export-videos betaflight_db026 --output /tmp/bf_fpv_b026 --fps 30`
  → nonempty 429-frame 640×360 video. The scripted climb does not steer at
  the gate, so the gate is small in the forward view; the schematic test
  proves the FPV pane and the four gate bars are both present.
- Camera plus referee audit:
  `RACE_CAMERA=1 RACE_COURSE=single RACE_REFEREE_AUDIT=1 elodin run examples/betaflight-sitl/main.py`
  → `[FPV] total_frames=55 window_frames=15 observed_fps=30.01 status=PASS`
  and `[C-REFEREE-AUDIT] ... status=PASS`, exit 0 (DB `betaflight_db027`).
  The 2.5 s audit still has 15 post-warmup frames, so no short-window exception
  was added.
- `frame_fresh` is true only for a new selected timestamp. Held frames are
  still offered with `frame_fresh=False` (covered by
  `tests/test_fpv_camera.py`).
- Export:
  `elodin-db export-videos betaflight_db024 --output /tmp/bf_fpv_b024 --fps 30`
  → nonempty `drone.fpv.mp4` (438 frames, 640×360 H.264). The export can be
  one frame ahead of `total_frames` because the render-server may write one
  more frame while it shuts down after the final tick.
- Raising the acceptance floor to 1000 FPS made the same camera run exit 1
  with `reason=fps-below-15`. The shipped floor is 15.
- FPV sampling lives in `fpv_camera.py`. The schematic in `schematic.py`
  injects only the FPV pane, the ground plane, and `sensor_visible=#false` on
  the 10× drone mesh. The frustum is off by default.

**Verified 2026-10-01 after merging `main` (`d8fbf209`) (WSL2, NVIDIA GeForce RTX 3050 6GB Laptop GPU):**

- `python3 -m pytest examples/betaflight-sitl/tests -q` → 109 passed.
  `ruff format --check`, `ruff check`, and `typos -c typos.toml` passed.
- Default: `RACE_CAMERA=0 elodin run examples/betaflight-sitl/main.py` →
  `FPV camera: disabled`, no render-server, C0 `status=PASS`,
  lockstep_steps `119995`, exit 0 (DB `betaflight_db029`).
- Camera plus course: `RACE_CAMERA=1 RACE_COURSE=single elodin run examples/betaflight-sitl/main.py` →
  `[FPV] total_frames=450 window_frames=389 observed_fps=29.92 status=PASS`
  and `[RACE] course=single gates_passed=0/1 status=INCOMPLETE`, C0 `PASS`,
  exit 0 (DB `betaflight_db030`). Export:
  `elodin-db export-videos betaflight_db030 --output /tmp/bf_fpv_fix --fps 30`
  → nonempty 451-frame 640×360 video. Frames at 0.5 s and 3 s show a level
  horizon, the ground plane, and the gate centered ahead. At 8 s the gate is
  larger and still centered. No frame shows the drone model. The mount is
  unchanged.

### [x] C — Course and referee vertical slice

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

Additional positive-path qualification (2026-09-16):

```bash
RACE_COURSE=single RACE_REFEREE_AUDIT=1 \
  elodin run examples/betaflight-sitl/main.py
```

This separate integration fixture leaves the contract course unchanged, starts
the disarmed drone at `(5, 0, 4.9)` m with `+10 m/s` world-X velocity, and ends
at 2.5 simulated seconds before scripted guidance leaves its safe boot phase.
Six-DOF gravity and drag produced one gate-0 pass at interpolated simulation
time `1.085149 s` near `(10.0, 0.0, 1.7958)` m and finished at X=`11.7776 m`;
telemetry read back on the following callback tick matched the event and
retained both unused `-1.0` slots. The process emitted exactly one complete
`[RACE]` line and one passing `[C-REFEREE-AUDIT]` line and returned zero. This
qualifies the live truth/referee/telemetry/result path, not steering or powered
autonomous flight.

### [x] D — Manual piloting, control seam, and ANGLE mode

**Implementation status:** Complete. Code and automated acceptance are complete,
and an operator has qualified both keyboard and gamepad control paths.

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
- Add an opt-in physical sign audit: use a simulation-time guidance source to
  apply bounded roll and pitch commands separately from a controlled airborne
  condition and assert observed world-direction responses. The audit may use a
  short, purpose-built initial condition and approximate hover throttle; it must
  not require implementing Package E's position controller or depend on host
  speed.
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

Automated qualification (2026-09-10):

```bash
python3 -m pytest examples/betaflight-sitl/tests -q
cargo test -p betaflight-sitl-controller
RACE_GUIDANCE=manual RACE_MANUAL_AUDIT=1 elodin run examples/betaflight-sitl/main.py
```

The audit uses an in-process guidance source keyed to `GuidanceUpdate.sim_time`;
it does not start the external manual controller. Its commands traverse the
shared semantic-to-RC conversion, one-tick latch, and Betaflight path before the
physical response is assessed. The pure suite passed 50 tests, the controller
passed 5 tests, and the live injected-input audit reported:

```text
[D-AUDIT] angle=true roll_right_rad_s=0.446 pitch_forward_rad_s=0.444 yaw_right_rad_s=1.480 max_motor=0.391 status=PASS
```

Manual keyboard and gamepad qualification (2026-09-28):

```bash
RACE_GUIDANCE=manual elodin editor examples/betaflight-sitl/main.py
```

The operator tested and signed off on both documented input paths. Each path was
used to arm, take off, command throttle, roll, pitch, and yaw, land, and disarm.
Together with the deterministic physical audit above, this closes Package D.

### [x] E — Truth-guided single-gate hold

**Implementation status:** Complete. The truth-only controller and independent
host hold grader passed three consecutive headless qualifications on 2026-09-29.
This is a hold before one gate, not a gate-pass or race-completion capability.

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

Qualification (2026-09-29, three consecutive clean-start runs from the
repository root, each exit 0):

```bash
RACE_GUIDANCE=truth RACE_COURSE=single \
  ./target/release/elodin run examples/betaflight-sitl/main.py
```

The accepted fixed gains in `TruthGuidanceConfig` (`guidance.py`) are
`takeoff_climb_rate_mps=0.20`, `hover_rc_trim=0.13`, horizontal
`kp_xy=0.7`/`kv_xy=1.5`, altitude `kp_altitude=0.075`/`kd_altitude=0.12`,
and yaw `yaw_kp=0.5`/`yaw_rate_damping=0.10`; the remaining limits and timing
are defined in that config object, not environment variables.
The gate is at `(10,0,1.8)` m ENU (yaw 0, 2.5 m inner opening), and the target
is `(6,0,1.8)` m. The independent referee requires a continuous, eligible
10-simulated-second window of time-weighted 3D RMS ≤0.30 m, with armed/ANGLE,
lockstep and motor response, no forward gate crossing, and no guidance fault.

| Run | Online window (sim s) / RMS (m) | Independent DB window (sim s) / RMS (m) |
|---|---|---|
| 1 | `[27.612500,37.612500]` / 0.075521 | `[27.612500,37.612500]` / 0.075520758 |
| 2 | `[27.629125,37.629125]` / 0.074801 | `[27.629000,37.629000]` / 0.074803460 |
| 3 | `[27.627750,37.627750]` / 0.074188 | `[27.627750,37.627750]` / 0.074187954 |

Each independently reconstructed window is exactly 10 s and contains 80,001
samples; online/offline RMS differences are at most 0.000002460 m. All three
runs reached HOLD, retained armed/ANGLE/lockstep/motor evidence, and had no
fault or crossing (maximum world X 6.315425189, 6.312380604, and 6.314916699
m, respectively; gate plane X=10 m). Each emitted one `[E-HOLD] status=PASS`
and one **expected** incomplete `[RACE]` with `course=single`,
`gates_passed=0/1`, `status=INCOMPLETE`; a failed hold emits
`[E-HOLD] status=FAIL` and exits nonzero. The race result remains incomplete
because Package E deliberately never passes a gate.

Final regressions: `python3 -m pytest examples/betaflight-sitl/tests -q`
(**138 passed**); `cargo test -p betaflight-sitl-controller` (**5 passed**);
`./target/release/elodin run examples/betaflight-sitl/main.py` and
`RACE_COURSE=single ./target/release/elodin run examples/betaflight-sitl/main.py`
(both exit 0 with one `[C0] status=PASS`; the scripted single-course run also
emits one expected incomplete `[RACE]`). Qualified envelope: headless truth +
`single`, 8 kHz Betaflight/physics lockstep, 45 simulated seconds. Camera,
vision, multi-gate flight, and gate passing are not qualified by these runs.

**Merged B + E verification (2026-10-02, rebuilt latest-main binaries/wheel):**
`python -m pytest examples/betaflight-sitl/tests -q` passed **180** pure
tests (including both B and E suites and camera/truth/startup/exit combinations);
`cargo test -p betaflight-sitl-controller` passed **5** tests. Default
camera-off C0 exited 0 with one `[C0] status=PASS` (119,995 lockstep steps).
Three further consecutive unchanged-code, camera-off truth+`single` runs exited
0, with one expected incomplete `[RACE]` and one passing `[E-HOLD]` each:

| Run / DB | Continuous 10 s hold window (sim s) | Online RMS (m) |
|---|---|---:|
| 1 / `betaflight_db007` | `[27.653375,37.653375]` | 0.074574 |
| 2 / `betaflight_db008` | `[27.649250,37.649250]` | 0.073668 |
| 3 / `betaflight_db009` | `[27.657375,37.657375]` | 0.073832 |

All reported armed/ANGLE/lockstep/motor evidence, no fault or crossing, and
exactly 10 s. With `RACE_CAMERA=1`, truth+`single` (`betaflight_db010`) also
exited 0: `[E-HOLD]` RMS 0.071955 m, `[FPV] total_frames=1350
window_frames=1289 observed_fps=29.98 status=PASS`, and one expected incomplete
`[RACE]`. The independent camera still supplies its frame/sample-time/freshness
fields to the immutable truth update; guidance does not use the pixels for
control. `elodin-db export-videos betaflight_db010 --output <out> --fps 30`
produced a nonempty 640×360, 1351-frame `drone.fpv.mp4` (the extra frame can
arrive during renderer shutdown). Camera-enabled scripted C0, Package C
referee audit, and Package D physical audit separately exited 0 with their own
one result line plus one passing `[FPV]` line. These merged runs do not qualify
vision guidance or gate passage.

**Current-main requalification (2026-10-12, base `729022c7`):**
The previously reconciled D-close (`351f1d8c`) and E (`024124cb`) commits
were applied in order without conflicts. The example's behavior and qualified
gains were unchanged. Shared changes since `2fd703f1` include additive Python
UI/window and camera fields, schematic serialization/asset ingest, and
editor/render-server component-filter negotiation. No example API adaptation
was needed; an older binary or wheel was not used for these qualifications.
From the repository root, host Cargo/uv rebuilt:

```bash
cargo build --release -p elodin
maturin build --release --manifest-path=libs/nox-py/Cargo.toml -i "$VIRTUAL_ENV/bin/python"
uv pip install --python "$VIRTUAL_ENV/bin/python" --reinstall-package elodin --no-deps \
  target/wheels/elodin-0.20.1a0-cp310-abi3-manylinux_2_39_x86_64.whl
cargo build --release -p elodin-db --bin elodin-db
python -m pytest examples/betaflight-sitl/tests -q
cargo test --release -p betaflight-sitl-controller
```

The test environment was `/home/ubuntu/py-uv-env`, with `VIRTUAL_ENV` and
`ELODIN_PYTHON` pointing to that environment and its Python respectively.
The release runtime was `0.20.1-alpha.0+172587cf`, built on current main plus
the two applied commits; the installed SDK was `0.20.1a0`. Only the Elodin
package in the shared environment changed (from `0.19.3a0`), not dependencies.
The pinned Betaflight `6dbc4218` was built and ARM/ANGLE EEPROM verified;
only the expected three tracked lockstep-patch files were restored afterward,
leaving the submodule clean and its gitlink unchanged.
Pure tests passed **180**, controller tests **5**, source compilation/live SDK
imports passed, and six invalid startup combinations returned **2**.

All camera-off commands below exited **0**, using
`./target/release/elodin run examples/betaflight-sitl/main.py`:

- Default (`betaflight_db000`): exactly one `[C0] PASS`, 119,995 live lockstep
  responses; no renderer or race/hold/camera result.
- `RACE_COURSE=single` (`betaflight_db001`): one `[C0] PASS` and one expected
  incomplete `[RACE]`.
- `RACE_COURSE=single RACE_REFEREE_AUDIT=1` (`betaflight_db002`): one complete
  `[RACE]` at 1.085149 s and one `[C-REFEREE-AUDIT] PASS`.
- `RACE_GUIDANCE=manual RACE_MANUAL_AUDIT=1` (`betaflight_db003`): one
  `[D-AUDIT] PASS`, measured roll/pitch/yaw 0.469/0.444/1.479 rad/s.

Three consecutive unchanged-code runs with `RACE_GUIDANCE=truth` and
`RACE_COURSE=single` (camera off) each emitted exactly one `[E-HOLD] PASS`
and one expected `gates_passed=0/1 status=INCOMPLETE` `[RACE]`:

| Run / DB | Continuous hold window (sim s) | Online RMS (m) | Independent DB RMS (m) |
|---|---|---:|---:|
| 1 / `betaflight_db004` | `[27.661875,37.661875]` | 0.073873 | 0.073873396 |
| 2 / `betaflight_db005` | `[27.622625,37.622625]` | 0.075235 | 0.075235469 |
| 3 / `betaflight_db006` | `[27.637250,37.637250]` | 0.074413 | 0.074413463 |

Each window is exactly 10 s, with armed/ANGLE/lockstep/motor evidence and no
crossing/fault. Independent CSV reconstruction used 80,001 samples per window,
125 µs maximum sample gaps, and agreed with the printed RMS to <0.0000005 m.
Maximum world X was 6.312389/6.312703/6.311669 m, below the X=10 m gate plane.

Offscreen GPU coexistence used the idle first NVIDIA RTX 3060 (`renderD129`,
Vulkan); no editor GUI was launched. Each command exited **0**, printed one
passing `[FPV]`, and retained its independent scenario results:

| Scenario (`RACE_CAMERA=1`) / DB | Frames total / post-warmup | Simulated FPS | Other results |
|---|---:|---:|---|
| Truth + `single` / `betaflight_db007` | 1350 / 1289 | 29.98 | `[E-HOLD] PASS`, exact 10 s, RMS 0.075011 m; expected incomplete `[RACE]`, no crossing/fault |
| Scripted default / `betaflight_db008` | 450 / 389 | 29.92 | `[C0] PASS` |
| C positive referee audit / `betaflight_db009` | 75 / 14 | 28.01 | `[RACE] COMPLETE`, `[C-REFEREE-AUDIT] PASS` |
| D physical audit / `betaflight_db010` | 720 / 659 | 29.95 | `[D-AUDIT] PASS` |

`elodin-db export-videos betaflight_db007 --output <out> --fps 30` exported a
nonempty 640×360, 1351-frame MP4 (the known late shutdown frame). The ingested
schematic retained the FPV pane, phase/RC graphs, neutral plane, sensor-hidden
drone and all four gate bars; DB telemetry contained phase codes 0–5 and
3,809 RC change rows. This is a data/schematic check, **not** a new editor
visual review or manual hardware sign-off. Combined B/E failure exits remain
covered by the pure exit tests; no live negative criterion was injected.
Evidence logs are named `current-default-c0.log`, `current-scripted-single.log`,
`current-referee-audit.log`, `current-manual-audit.log`, `current-truth-1.log`
through `current-truth-3.log`, and the corresponding `current-*-camera.log`
files. This requalification preserves the earlier history and does not qualify
vision, COMMIT/gate passing, a multi-gate course, or new packages.

### [x] E1 — Mission controller extraction

**Implementation status:** Complete. The initial exact original-controller
replay, independent implementation review, and three consecutive unchanged-source
headless hold qualifications passed on 2026-10-23. The historical qualification
is preserved below; the active characterization baseline is the compact 8-case,
78-update digest set described after it.

**Objective:** Separate Package E's qualified control law and phase machine from
truth types, so truth and vision guidance can share one controller, without
changing behavior.

**Prerequisites:** E.

**Required work:**

- Introduce `NavState`, `GateTarget`, and `TrackState` (Section 7.9) and a
  mission controller in `guidance.py` that consumes them plus the race-progress
  fields only.
- Reduce `TruthGuidance` to an adapter that builds both inputs from
  `TruthGuidanceUpdate` and calls the controller on every tick. It keeps
  latching `gate_poses[0]` exactly as Package E does; retargeting is Package
  F's work.
- Preserve the attributes that `main.py` and the hold-grader wiring use
  (`phase`, `phase_code`, `failure_reason`) and the `TruthGuidanceConfig`
  values.
- Before refactoring, capture golden traces from the current controller:
  deterministic update sequences, scripted or driven by a simple pure-Python
  plant, that cover every phase transition and fault path. Check in the traces,
  or a deterministic generator plus digests.

**Out of scope:** New phases or profiles, gate retargeting, COMMIT, vision
inputs, or gain changes.

**Acceptance:** The refactored controller reproduces the golden traces exactly:
identical phases, fault reasons, and `SemanticControl` values. If a
floating-point reordering is unavoidable, differences must stay within 1e-12 and
be explained in the handoff. The existing guidance tests pass. Three consecutive
clean-start `RACE_GUIDANCE=truth RACE_COURSE=single` runs pass Package E's
acceptance.

**Handoff:** Record the trace-generation command, the type and module locations,
and the E requalification results next to the previous RMS values.

**Verified 2026-10-23, baseline `806b0772`, working branch `bf-e1`:**

`examples/betaflight-sitl/guidance.py` now defines `NavState`, `GateTarget`,
`TrackState` (TENTATIVE/TRACKING/COASTING/LOST, codes 1–4), and the one
`MissionController`. Its normal update consumes navigation state, target,
simulation time, and public race progress only. `TruthGuidance` converts truth
and gyro samples and delegates every tick, preserving first-`gate_poses[0]`
latching and the existing phase/failure/target read surfaces. All configuration
values and control-law expression ordering are unchanged. Track metadata does
not add vision behavior in E1.

A narrow generic `reject_update(sim_time, reason, feedback_nav)` path preserves
legacy invalid-input behavior without adding fields to the navigation contracts:
missing truth uses the last accepted pose with the current valid gyro, while bad
gyro uses current truth with zero rate without accepting a new fault snapshot.
Time errors retain priority, valid updates in FAULT still refresh accepted
feedback, and the first fault reason/target remains latched. No truth types or
simulation I/O enter the mission controller.

At the original E1 acceptance, the original controller's full fixed fixtures
were checked in as `tests/fixtures/guidance_package_e.json`; independent review
requested additional original-source traces in
`guidance_package_e_supplement.json`. Together they covered **39 scenarios /
268 updates**, all phase transitions and ten fault reasons, rotated
control/braking, full quaternions, limits, dwell resets and fault feedback. The
refactor matched phases, reasons and `SemanticControl` values **exactly**,
including signed zero; no floating-point tolerance exception was needed.

The full fixture files, their original capture scripts, source hashes and
byte-for-byte reproduction instructions remain available in commit
`163c052956a15f8d720a93f2e7059f1c52b8ce71`. These are historical acceptance
evidence, not the current test paths. Historical independent regeneration
reproduced both fixtures byte-for-byte; original-only replay passed 41 tests. A
separate deterministic differential probe compared 1,479 sequences / 25,659
updates with exact agreement, including fault bookkeeping and targets.

Verification commands from the repository root, using the installed Elodin SDK
and release runtime:

```bash
python -m pytest examples/betaflight-sitl/tests -q
cargo test --release -p betaflight-sitl-controller
elodin run examples/betaflight-sitl/main.py
RACE_COURSE=single elodin run examples/betaflight-sitl/main.py
RACE_COURSE=single RACE_REFEREE_AUDIT=1 elodin run examples/betaflight-sitl/main.py
RACE_GUIDANCE=manual RACE_MANUAL_AUDIT=1 elodin run examples/betaflight-sitl/main.py
# Run the following three times consecutively, each a fresh process:
RACE_GUIDANCE=truth RACE_COURSE=single elodin run examples/betaflight-sitl/main.py
```

Pure tests passed **300**, including 37 direct generic/controller/adapter tests;
controller tests passed **5**. Targeted `ruff format --check`, `ruff check`, and
`git diff --check` passed. Default and scripted-single C0 both passed with 119,995
lockstep responses; the C referee audit passed at 1.085149 s; the D audit passed
with ANGLE true and roll/pitch/yaw rates 0.470/0.467/1.480 rad/s. All seven final
live processes exited 0, and the reviewed controller source was unchanged
throughout.

| Run / DB | Continuous hold window (sim s) | Independent host-grader RMS (m) |
| --- | --- | ---: |
| 1 / `betaflight_db005` | `[27.700125,37.700125]` | 0.076142 |
| 2 / `betaflight_db008` | `[27.634000,37.634000]` | 0.077087 |
| 3 / `betaflight_db009` | `[27.622750,37.622750]` | 0.072773 |

Each emitted exactly one passing `[E-HOLD]` with a continuous 10 s window,
armed/ANGLE/lockstep/motor evidence, no crossing or fault, and one expected
`gates_passed=0/1 status=INCOMPLETE` `[RACE]`. These are the independent production
host-grader results; no new offline DB reconstruction is claimed. The pre-refactor
baseline truth run also passed (RMS 0.073516 m). Logs and review reports are under
`/home/ubuntu/betaflight-e1`; all three final truth DBs are retained. Camera,
EEPROM/GPS, vision guidance, retargeting, COMMIT and gate passing were not changed
or newly qualified by E1. G1 remains incomplete.

**Current maintained characterization (lean follow-up, 2026-10-23; no new
qualification):** The current suite retains eight representative scenarios / 78
updates, selected from the original expected rows in commit
`163c052956a15f8d720a93f2e7059f1c52b8ce71`:

| Scenario | Updates |
|---|---:|
| `full_mission_and_hold_timeout` | 15 |
| `braking_yaw_saturation` | 11 |
| `yawed_active_latching_braking_crossing` | 5 |
| `full_attitude_and_missing_truth` | 7 |
| `missing_truth_armed` | 8 |
| `gyro_nan_x_armed` | 8 |
| `nonmonotonic_True` | 7 |
| `velocity_and_launch_xy_dwell_interruptions` | 17 |

`tests/test_guidance_golden.py` hashes each scenario's original expected rows
using canonical JSON (`sort_keys=True`, compact separators, ASCII escaping,
`allow_nan=False`, UTF-8); snapshot fields retain phase/code/reason/braking,
target and command, with floating-point values encoded by `float.hex()` including
signed zero. No full traces are copied elsewhere in the current tree. Ordinary
pytest is offline and does not inspect git history, Elodin, Betaflight, or GPU
state. Its `--capture` mode refuses any `guidance.py` or `controls.py` whose SHA256
differs from the original 806b0772 sources, prints digests only, and never edits
the expected map. Safe isolated regeneration is documented in
`tests/fixtures/README.md`.

This approved follow-up reduces ongoing characterization size during active
development; it does not revise E1's original exact-equivalence acceptance or
claim a new live qualification. The full proof, 39/268 trace results, 300-test
result, and three hold RMS results above remain historical evidence from the
initial E1 acceptance. After this cleanup, the current full pure suite passes
268 tests. Ordinary behavioral tests remain in place.


### [ ] F — Truth-guided straight course

**Objective:** Complete the core mission and three-gate course using trustworthy
state, before integrating perception.

**Prerequisites:** E1. G1 must land before F's qualification runs, so that F is
qualified under the final Betaflight configuration.

**Required work:**

- Add `c1_straight` exactly as defined in Section 7.5, and accept
  `RACE_GUIDANCE=truth` with it.
- Enlarge the camera ground plane so it extends past the last gate
  (Section 7.5).
- Add the race profile of Section 8: target the gate named by
  `next_gate_index`, COMMIT (code 7), repeated ACQUIRE/HOLD, and FINISH
  (code 9), selected by course. The `single` hold profile keeps Package E's
  behavior and configuration.
- Use the referee's ordered events for scoring and race progress.
- Land or safely settle and disarm after completion.
- Allow a course-specific simulation cap up to 60 simulated seconds.
- Add a headless truth-course scenario with a nonzero exit on incomplete
  status, printing the phase and failure reason.

**Out of scope:** Camera use, detector/tracker input, lap-time optimization, or
harder courses.

**Acceptance:** Three consecutive clean-start headless runs emit
`gates_passed=3/3 status=COMPLETE` within 60 simulated seconds and finish safely.
For a fixed configuration, pass times remain within a documented small tolerance;
exact bitwise equality is not required if Betaflight prevents it. Package E's
hold qualification still passes on `single`.

**Handoff:** Record the three pass-time sets, time cap, completion command, race
profile configuration, and known control limitations.

### [x] G — Vision contracts and synthetic geometry

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

**Verified 2026-10-05 on `pkg-g-vision-contracts` from `origin/main` (`d6354da1`);
re-verified 2026-10-16 after merging `origin/main` (`c9ae81ad`) and the PR #891
review fixes; corner-order wording rechecked 2026-10-21:**

- `python3 -m pytest examples/betaflight-sitl/tests -q` → 222 passed.
  `ruff format`, `ruff check`, and `typos` passed.
- `K` is `fx = fy = 320`, `cx = 320`, `cy = 180`, derived from `FOCAL_PX` and
  the 640×360 vertical FoV. Horizontal FoV is 90°.
- `R_flu_from_opencv` columns are camera X → body `(0, -1, 0)`, camera Y →
  body `(0, 0, -1)`, camera Z → body `(1, 0, 0)`.
  `R_body_cam = Ry(-tilt) @ R_flu_from_opencv`. Positive `TILT_DEG` pitches the
  optical axis up (nose-up). Positive renderer pitch is nose-down, so
  `sensor_rot_offset` passes `[0, -TILT_DEG, 0]`.
- `order_corners` labels corners with a midline heuristic.
  `corner_order_ambiguous` is a heuristic warning, set when the two best
  labellings are within 10°. An unset flag does not guarantee the gate-frame
  orientation: perspective can still mislabel corners. Gate position, bearing,
  range, and the crossing normal are unaffected by a cyclic relabelling. The
  in-plane gate orientation is not reliable.
- Left for Package H: a 0.2 px quad can solve as a gate about 4 km away at 0 px
  RMS; `_rotation_vector` fails safe within about 1e-6° of 180°; `sample_time`
  is not checked for finiteness; `bearing_body` is measured from the body origin.
- Inner corners are TL, TR, BR, BL. Gate-frame object points, with
  `h = inner_size / 2`, are `(0, +h, +h)`, `(0, -h, +h)`, `(0, -h, -h)`,
  `(0, +h, -h)`.
- Noise-free bearing tolerance is 0.25°. Translation tolerance is
  `1e-6·range + 1e-6` m. Integer-pixel depth tolerance is
  `Z² / (f·S)` m, twice `Z²·0.5/(f·S)`, with `f = 320` and `S = 2.5`.
  The 2 m range does not fit in the 360 px image.
- Fixture command, from the repository root:
  `python3 examples/betaflight-sitl/synthetic_vision.py`.
- The editor schematic is a Python `elodin.ui` builder. Canonical
  `from_kdl(emit_kdl())` matches the previous KDL for every course, audit,
  and camera combination. Named `orange` is the 8-bit color `(255, 128, 0)`.
- PnP is pure NumPy. OpenCV stays in Package H.
- Camera plus course:
  `RACE_CAMERA=1 RACE_COURSE=single elodin run examples/betaflight-sitl/main.py`
  → `[FPV] total_frames=450 window_frames=389 observed_fps=29.92 status=PASS`,
  `[RACE] course=single gates_passed=0/1 status=INCOMPLETE`, C0 `PASS`,
  exit 0 (DB `betaflight_db039`). The scripted climb does not steer at the gate.
- Rendered tilt check, 2026-10-21: same command, video frame 60 at 2.0 s.
  `drone.world_pos` is identity at `(0, 0, 0)`. The gate is `(10, 0, 1.8)`,
  yaw 0, 2.5 m opening. The camera mount is `[0.08, 0, 0.02]`. Tilt 0 reuses
  `betaflight_db047`. Tilt +10° is `betaflight_db048`; `TILT_DEG` was restored
  to 0 afterward.
- Measured inner corners against `project_gate_corners`: the gate-center v
  moves from 123.5 px to 181.0 px, 57.5 px down. Predicted centers are 122.6 px
  and 179.9 px, 57.3 px down. Worst corner error is 1.7 px at tilt 0 and 2.7 px
  at +10°, on the compressed bottom corners.

### [ ] G1 — Sensor timing and truth-boundary fixes

**Status (2026-10-28):** Implementation complete on `g1-clean`, reconstructed
from GitHub `origin/main` `8742b411`; pending user diff approval and fresh
qualification. No builds or tests have run on this clean source.

**Objective:** Supply nominal image/render time on the simulation clock and
remove Betaflight's truth-derived aid. **Prerequisites:** B and G.

- `FpvCamera.poll()` supplies selected image-message time via integer-delta
  conversion as `frame_time`, replacing `frame_sample_time`; query time stays
  internal. `GateCandidate.sample_time` carries the same value (Section 7.4).
- Focused tests cover conversion, held/missing/freshness and copy safety.
- `init_eeprom.py` disables GPS and verifies persisted `trust_mag = OFF` and
  prior settings using exact CLI lines/tokens; importing it does not start SITL.
- The `sensors.py` barometer-noise comment now correctly states σ = 0.1 m.

**Approved scope:** Approximate scene-time alignment is accepted (Section 7.4).
Strict alignment remains a historical FAIL, deferred/non-gating. Its diagnostic
and tests remain unchanged in reference commit `3584748d`, not in this minimal
patch; historical evidence is retained under `/home/ubuntu/betaflight-g1/`.
No timing offset, controller/gain change, renderer/query/stream fix, per-frame
RPC, prototype promotion, or native/build/firmware change is included.

**Qualification pending:** After user diff approval, rebuild matching private
SDK/runtime/DB and default SITL firmware, then check pure/release tests,
persisted EEPROM, default/single C0, C referee, D physical audit, three
consecutive E truth holds and ordinary production FPV. Previous `pkg-g1`
results, including `finish-closure/completion.md` and `origin-sync/`, are
historical only, not qualification of `g1-clean`. Prior keyboard/gamepad sign-off
stands; the automated D audit covers the configuration change. No vision-flight
or racing qualification is claimed.

**Handoff:** Stop for user diff approval before any testing; qualify and integrate
G1 before H/I and F qualification. E1 is complete and available to F/J.

### [ ] H — Classical detector, offline

**Objective:** Detect rendered training gates and estimate their relative pose
without introducing closed-loop flight risk.

**Prerequisites:** B, C, G, and G1.

**Required work:**

- Implement `gate_detector.py`, the only module that imports OpenCV:
  1. Convert RGBA to HSV and threshold saturated orange. The gates use RGB
     `(255, 128, 0)`, hue about 15 on OpenCV's 0–180 scale. Start with hue 5–25,
     saturation at least 120, and value at least 50, then calibrate on recorded
     frames.
  2. Close the mask with a 3 × 3 kernel to seal antialiasing gaps at bar joints.
  3. Run `findContours` with `RETR_CCOMP`. Each hole is a candidate inner
     opening; a farther gate seen through a nearer one is a separate ring with
     its own hole.
  4. Accept a hole only if its polygon approximation is a convex
     quadrilateral, no vertex lies within 2 px of the image border, its area is
     at least 200 px² (rejecting the tiny quads Package G found PnP can place
     kilometres away), and, when the parent ring is not clipped, the ring-to-
     hole area ratio lies between 0.15 and 0.8 (frontal value about 0.35).
  5. Refine corners by fitting a line to each edge and intersecting adjacent
     lines. Confirm any half-pixel contour offset against truth projections.
  6. Order corners with `vision.order_corners` and solve with
     `vision.candidate_from_corners`. Reject candidates with no pose, a
     reprojection RMS above 1.5 px, or a range outside 1–40 m.
  7. Set confidence from opposite-edge agreement, ring-ratio agreement, and
     reprojection quality.
  8. Return candidates largest apparent opening first, through the Section 7.7
     contract.
- Run only on fresh 30 Hz frame samples supplied by the adapter.
- Add `replay.py` with a recorder enabled by `RACE_RECORD=<directory>` in
  camera runs. A compressed bundle holds the 500 Hz navigator inputs (interval
  means of gyro and specific force), barometer and magnetometer samples, fresh
  frames with nominal image/render times (Section 7.4), and, separately, truth
  `world_pos` and `world_vel` at 500 Hz and those nominal times. The evaluator
  passes only sensor fields to the detector and navigator.
- Fixtures: synthetic images made by filling projected outer and inner gate
  squares (test-only code), covering centred, translated, yawed, ranged,
  nested, and clipped gates; plus a small set of recorded renderer frames with
  truth poses from Package E truth runs and, once available, Package F and
  manual runs.
- Add an offline evaluation command that reports detection rate by range,
  bearing error, PnP reprojection error, and processing time.
- Document the OpenCV dependency in the local environment instructions and add
  it to the nix Python environment that CI uses.
- Extend `test_vision_isolation.py` to `gate_detector.py`, and resolve or
  explicitly accept each item Package G left for H.

**Out of scope:** Tracker, controller, live RC output, CNN, or partial-gate neural
inference.

**Acceptance:** On the controlled core-course fixture set from 4–20 m, at least
90% of frames with a fully visible gate produce a valid candidate and median
bearing error is at most 2 degrees. Corner error is at most 0.5 px on clean
synthetic images, and clipped openings are rejected. Accuracy is a deterministic
test. Processing median should be reported and targeted at 5 ms or less on the
development host, but hardware-dependent timing is not a normal CI failure.

**Handoff:** Record fixture provenance, metrics, command, thresholds, recorder
bundle format, and known failure ranges.

### [ ] I — Gate navigator, offline

**Objective:** Produce a stable vehicle-motion and gate-relative target estimate
through depth noise, outliers, and short close-range detector loss.

**Prerequisites:** G and G1. This package can proceed in parallel with H.

**Required work:**

- Implement `navigation.py` with the attitude filter, gate navigator,
  `NavEstimate`, and `GateEstimate` of Section 7.10.
- Consume only Section 7.7 candidates, IMU, barometer, magnetometer, time, the
  on-pad flag, and `next_gate_index`.
- Model bearing as more reliable than monocular depth through the anisotropic
  PnP measurement noise.
- Apply each visual update at nominal image/render time (Section 7.4).
- Reject implausible innovations, expose track state, age, and NIS, and reset
  after a documented rejection streak that excludes expected close-range
  invisibility.
- Coast for a bounded interval on the IMU after detections disappear.
- Expose a geometric passed indication when the estimated gate plane falls
  behind the vehicle, without using a blind timer as the sole signal.
- Provide the body-FLU gate-relative estimate of Section 7.8.
- Build deterministic synthetic trajectories: scripted six-degree-of-freedom
  motion produces exact gyro and specific force with the `sensors.py` noise
  model added, plus barometer, magnetometer, and corners from
  `synthetic_vision` with noise and dropout.
- Evaluate on Package H replay bundles when they are available.
- Extend `test_vision_isolation.py` to `navigation.py`.

Starting parameters (tuning values, not acceptance criteria):

| Group | Parameter | Starting value |
|---|---|---|
| Attitude | Accelerometer and magnetometer correction gains | 1/30 s⁻¹; 1/5 s⁻¹ |
| | Accelerometer gate | within 0.25 m/s² of g, and rotation rate below 0.3 rad/s |
| Navigator | Step rate | 500 Hz (every 16 ticks) |
| | Acceleration and bias process noise | 0.05 m/s²/√Hz; 0.01 m/s²/√s |
| | Corner pixel σ; bearing σ; floor σ | 0.7 px; 0.3°; 0.01 m |
| | Barometer σ; zero-velocity σ | 0.1 m per sample; 0.02 m/s |
| | Association gate; confirmation; reset streak | d² < 16.3; 3 hits within 0.5 s; 10 frames |
| | Tracking freshness; coast limit before LOST | 0.25 s; 2.5 s |
| | Gate-yaw σ | max(1°, 1° × (range / 4 m)²) |
| | Delayed-measurement buffer | 0.5 s |

**Out of scope:** Image processing, world localization, live flight, or tuning on
hidden world gate poses.

**Acceptance:** Deterministic synthetic tests demonstrate convergence from the
pad, normalized innovation squared (NIS) and normalized estimation error
squared within χ² bounds, bounded response to alternating depth
errors, outlier rejection, stale-state behavior, a drift bound over a 2.5 s
coast, reset after a rejection streak and no reset at close range, unbiased
synthetic delayed frames with known fixture times (and measurable bias without
them), a target switch on a progress change, and the passed indication.
Attitude-filter tests
show tilt and yaw offsets decaying toward the accelerometer and magnetometer
references and accelerometer gating during sustained tilt. The navigator
produces no NaNs or unbounded values under any fixture.

**Handoff:** Record state definition, noise assumptions, staleness limits, final
parameters, replay metrics if available, and all test commands.

### [ ] J — Vision-guided single-gate hold

**Objective:** Close the flight loop on rendered vision while using truth only as
an independent grader.

**Prerequisites:** E1, H, and I.

**Required work:**

- First, run the detector and navigator as a shadow observer in
  `RACE_GUIDANCE=truth RACE_CAMERA=1 RACE_COURSE=single` runs. The observer
  consumes only the base `GuidanceUpdate` fields, publishes the `nav_*`,
  `gate_det`, and `gate_rel_est` telemetry, and has no control authority.
- Implement `vision_guidance.py` and `RACE_GUIDANCE=vision`. The adapter
  accumulates IMU samples on every tick, runs the detector on fresh frames,
  steps the attitude filter, navigator, and mission controller at 500 Hz, fills
  the Section 7.9 inputs from `NavEstimate`, and asserts the on-pad flag until
  arming is first requested.
- Enforce Section 7.3 truth isolation and the Section 6.3 startup validation.
- Reuse the E1 mission controller in the hold profile. Do not create a second
  controller.
- Take off using allowed inertial and barometric information, acquire the
  `single` gate, and hold the 4 m standoff. Arming waits until the navigator is
  ready and the target is TRACKING; ACQUIRE holds position until TRACKING; a
  LOST target enters vision-mode FAULT (Section 8).
- Emit phase, detection, navigator, and RC telemetry without exposing hidden
  truth to guidance.
- Add a test that a vision guidance update contains no truth or gate poses, and
  extend `test_vision_isolation.py` to `vision_guidance.py`.

**Out of scope:** Passing the gate, multi-gate selection, SEARCH, CNN, or global
pose estimation.

**Entry gate for closed-loop runs:** Three shadow runs show relative position
error of at most 0.15 m RMS and velocity error of at most 0.10 m/s RMS during
ACQUIRE and HOLD, scored against truth at the time the controller would use the
estimate.

**Acceptance:** In three consecutive qualified GPU runs, vision guidance enters
HOLD and remains within 0.50 m 3D RMS of the truth-scored standoff target for a
continuous 10 simulated seconds. During that interval, target availability,
defined as the fraction of the window in which the track is TRACKING, is at
least 90%, and guidance has no truth access.

**Handoff:** Record shadow and closed-loop run DBs, RMS and tracking metrics,
detector loss behavior, and whether any approved contract updates were
required.

### [ ] K — Vision-guided straight course

**Objective:** Complete the three-gate core course using rendered frames and
allowed sensor/race context only.

**Prerequisites:** F and J.

**Required work:**

- Run the race profile in vision mode: COMMIT continuing through COASTING,
  bounded coast-through, SEARCH (code 8), and reacquisition.
- Hand over to the next gate when `next_gate_index` changes, and handle multiple
  nearly collinear visible gates without target flip-flop through innovation
  gating and ordered race progress.
- Cross-check the navigator's passed indication against referee events in
  telemetry while keeping referee truth out of guidance.
- Finish and safely disarm after all three gates.
- Add a vision-course scenario with nonzero exit on incomplete status. A failed
  run prints the phase and failure reason: no detection, lost track, missed
  gate, or timeout.

**Entry gate for closed-loop runs:** Three shadow runs on Package F's truth
course show the navigator's predicted position at each gate plane, after the
coast through the blind final ~2.2 m, within 0.3 m of truth.

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

- Run pure tests, including the detector and navigator suites with OpenCV in
  the CI Python environment, small replay fixtures, and C0 scripted integration
  in normal CI.
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
A ──> D ──> E ──> E1 ──> F ─────────> K ──> L
          ▲                           ▲
C ───────┘                            │
                                      │
B ──> G ──> G1 ──> H ──┐              │
              └──> I ──┴──> J ────────┘
```

C is also required by H, and E1 by J. G1 must land before F's qualification
runs. E1 and G1 can proceed in parallel, and H and I can proceed in parallel
with each other and with E1 and F. B, C, and most of A can be implemented
independently. The control path (D–F) and perception path (B, G–I) remain
separate until J.

Useful stopping points are:

- after A: trustworthy existing SITL example;
- after B: camera-equipped SITL example;
- after C: reusable course and scoring harness;
- after D: manually pilotable Betaflight vehicle with verified RC conventions;
- after E: truth-guided single-gate position hold;
- after E1: one mission controller behind a navigation-state interface;
- after G1: nominal image/render time (Section 7.4) and no truth-derived Betaflight aid;
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
| A | Complete | 24 pure tests pass; the recorded C0 acceptance run returned 0 with 119,995 simulation-loop lockstep responses, max motor 0.574, and 56.837 m takeoff rise in 29 wall-clock seconds. A deliberate 100 m criterion returned 1. Shared headless propagation fixes: `301ae367` (`#837`) and lifecycle follow-up `36ee3431` (`#838`). |
| B | Complete | `fpv_camera.py` owns sampling, freshness, exact frame accounting, and acceptance. Verified 2026-09-26: `total_frames=437`, `window_frames=389`, `observed_fps=29.92`, exit 0 (DB `betaflight_db024`). Held-frame freshness and window bounds are pure tests. The merged 2026-10-02 B+E truth/camera run also passed with 1350 frames and 29.98 simulated FPS (details under E); scripted and C/D audit camera combinations passed. |
| C | Complete | `none`/`single` startup selection, static four-bar orange rendering, ordered truth referee, Package D progress integration, fixed-width referee telemetry, exactly one final enabled-course result, and an opt-in positive live audit are implemented. The qualification suite passed 96 tests. The live audit used the unchanged vertical 2.5 m gate, reported one complete pass at 1.085149 s, and verified later-tick telemetry readback. Default C0 and ordinary incomplete `single` behavior were preserved. A 1280×720 H.264 recording and crossing frame were reviewed as qualification evidence; a PR should attach or upload those artifacts separately rather than rely on a host-local path. |
| D | Complete | Command seam, one-tick ordering, AUX2 ANGLE configuration, s10 manual controller, 250 ms heartbeat failsafe, DB telemetry, and simulation-time physical sign audit are implemented. The combined automated qualification passed with roll 1.006, pitch 1.011, yaw 1.479 rad/s and accepted max motor 0.391; 5 controller tests also pass. On 2026-09-28, the operator tested and signed off on both keyboard and gamepad paths, including arm, takeoff, throttle/roll/pitch/yaw response, landing, and disarm. |
| E | Complete | Opt-in truth-only single-gate standoff controller and independent 10 s hold grader. Three consecutive original 2026-09-29 runs exited 0 with RMS 0.075521/0.074801/0.074188 m (offline 0.075520758/0.074803460/0.074187954 m), no crossing or fault, and expected incomplete 0/1 race results; 138 pure tests on the original E branch, 5 controller tests, and default/scripted-single C0 regressions passed. The merged 2026-10-02 verification passed 180 pure tests, 5 controller tests, three further truth runs and opt-in B+E camera coexistence (details above). Requalified unchanged on 2026-10-12 atop main `729022c7` with a newly built runtime/SDK, three camera-off passes (0.073873/0.075235/0.074413 m RMS), independent DB agreement, C0/C/D regressions, and all four camera combinations (details above). |
| E1 | Complete | Shared mission controller and immutable navigation/target inputs; thin per-tick truth adapter. Initial acceptance at commit `163c0529`: 39 original-source scenarios / 268 updates exact, 300 pure tests, clean independent review, and three unchanged-source truth holds at 0.076142/0.077087/0.072773 m RMS (historical details under E1). Current maintained characterization: 8 selected cases / 78 updates, each guarded by a canonical SHA-256 digest derived from the original fixture rows; current full pure suite passes 268 tests. This follow-up is test maintenance, not a new qualification. |
| F | Ready for implementation; qualification blocked by G1 | E1 is complete. Three-gate course, race profile, COMMIT/pass sequencing, and safe finish remain; qualification must wait for G1. |
| G | Complete | `vision.py` owns intrinsics, corner order, the camera-to-body helper, and NumPy planar PnP. Positive tilt is nose-up and is registered as negative renderer pitch. `corner_order_ambiguous` flags labellings within 10°. `synthetic_vision.py` is the only truth-projection path. 222 pure tests pass, including centered, translated, yawed, and ranged round trips. |
| G1 | Implementation complete; qualification pending | Clean `g1-clean` reconstruction from GitHub `8742b411`; awaiting user diff approval before builds/tests. Old-source PASS results are historical only; see G1 brief. |
| H | Blocked by G1 qualification/integration | B, C, and G are complete; the offline detector and recorder remain. |
| I | Blocked by G1 qualification/integration | G is complete; the navigator can proceed in parallel with H after G1. |
| J | Blocked by H, I | E1's shared mission controller is complete; detector, navigator, and vision guidance remain. |
| K | Blocked by F, J | No vision race |
| L | Blocked by F/K | No race CI or final docs |

### Resume point

Package E1 is complete. G1 implementation is complete on `g1-clean`, pending
user diff approval and fresh qualification; no tests or builds have run on this
clean source. Package F implementation may proceed, but its qualification runs
and H/I must wait for G1 qualification/integration.

- **E1, mission controller extraction:** the initial 2026-10-23 qualification
  preserved exact original-controller replay and three passing E holds (details
  above). The current maintained baseline is the 8-case/78-update digest set;
  the original 39/268 exact-equivalence evidence remains in commit `163c0529`.
  The shared controller is available to F and J.
- **G1, sensor timing and truth-boundary fixes:** review the clean diff before
  rebuilding or testing. Qualification must use matching fresh GitHub-baseline
  artifacts, GPS off and `trust_mag = OFF`; prior `pkg-g1` PASS results do not
  qualify this reconstruction.

Passing a gate is not yet implemented.

Package C's durable qualification command, output contract, measured result, and
scope are recorded with its work-package acceptance evidence above. User-facing
headless/editor audit and reproducible video-capture instructions, including the
recoverable MPEG-TS fallback, are maintained in `README.md`.

## 13. Decision log

| Date | Decision | Reason and affected packages |
|---|---|---|
| 2026-10-27 | Accept approximate scene-time alignment for G1; defer strict alignment. Selected message T is nominal image/render time on `sim_time`. | Supersedes the exact capture-alignment assumption of 2026-10-22. Keep regressions and independent review; no renderer/query/stream fix or prototype promotion. H/I may proceed after G1 integration using Section 7.4's limitation; F still requires E1. |
| 2026-10-23 | Keep E1's active original-controller characterization to eight representative cases / 78 updates, with digests derived from the original expected rows; retain the full 39-case / 268-update proof in commit `163c0529`. | The exact-equivalence acceptance was already performed and is preserved as historical evidence. This approved maintenance reduction avoids carrying exhaustive golden output during heavy development; behavioral tests and product acceptance remain unchanged. Affects only E1's ongoing characterization artifacts and reproduction instructions. |
| 2026-10-22 | Add Packages E1 (mission controller extraction) and G1 (sensor timing and truth-boundary fixes) instead of folding these prerequisites into F. Keep the existing package letters. | The behavior-preserving refactor, the sensor and configuration fixes, and F's new race behavior are separate concerns; combining them would make a regression hard to attribute. G1 also gates H and I, not only F. Existing commits and PRs refer to packages by letter, so nothing is renumbered. Affects E1, G1, and F–K. |
| 2026-10-22 | Package I is a gate-anchored navigator: a complementary attitude filter plus a 12-state linear Kalman filter in a gravity-aligned local frame, corrected by PnP at the frame capture time, the barometer, and zero velocity on the pad. It still publishes the body-FLU gate-relative estimate. Generic feature-based visual-inertial odometry and Betaflight GPS emulation are non-goals for the core course. | Chosen after a review of three independent research studies on 2026-10-21. The body-FLU tracker left attitude, gravity handling, and rotating-frame terms implicit, and degraded once the gate left view. The rendered scene is too sparse for feature tracking, and a GPS message discards attitude, covariance, and reset information while the pinned firmware's GPS paths have unit and parser defects. Affects §5, §7.7, §7.10, and I–K. |
| 2026-10-22 | `GuidanceUpdate.frame_time`, the capture time in simulation seconds, replaces `frame_sample_time`; `GateCandidate.sample_time` follows. | `frame_sample_time` was query time in DB-timestamp microseconds, while `sim_time` is in seconds. The assumed exact capture alignment is superseded by the 2026-10-27 decision. The 2026-09-26 freshness decision stands. Affects §7.2, §7.4, §7.7, G1, and H–K. |
| 2026-10-22 | Disable Betaflight's GPS feature and verify `trust_mag = OFF` in every mode. Label vision results by sensor set. | The virtual GPS is built from FDM truth and corrects Betaflight's yaw. Keeping truth out of `GuidanceUpdate` was necessary but not sufficient for a truth-free closed loop. The change applies to all modes, so G1 requalifies C0, C, D, and E. Affects §3.3, §7.3, and G1. |
| 2026-10-22 | One mission controller behind `NavState` and `GateTarget` serves truth and vision guidance. The course selects a hold or race profile. Phase codes 7–9 are COMMIT, SEARCH, and FINISH. Vision-mode FAULT brakes, descends, and disarms. | Replaces the earlier expectation that J would adapt E's controller without a defined interface. Affects §7.9, §8, E1, F, J, and K. |
| 2026-10-22 | Add `drone.nav_est`, `drone.nav_gate`, and `drone.nav_diag`, and define invalid values for `drone.gate_det` and `drone.gate_rel_est`. Add OpenCV inside `gate_detector.py` only, the `RACE_RECORD` replay recorder, shadow-mode entry gates for J and K, and a TRACKING-based availability definition for J. Invalid vision startup combinations exit with status 2. | These make detection, navigation, and control failures separately attributable and let the navigator be tuned offline before it gains control authority. The startup rule picks one of the two options §6.3 previously allowed. Affects §6.3, §7.8, §9, and H–L. |
| 2026-10-05 | Port the Betaflight SITL schematic to `elodin.ui` builders. | The KDL string builder was the last example still emitting a schematic by hand. The parsed model matches the previous layout. Affects the editor surface used by B and later packages. |
| 2026-10-16 | Positive `TILT_DEG` stays nose-up and is registered as negative `sensor_camera` pitch. `order_corners` labels corners with a midline heuristic. `corner_order_ambiguous` is a heuristic warning, set when the two best labellings are within 10°. An unset flag does not guarantee the gate-frame orientation: perspective can still mislabel corners. Gate position, bearing, range, and the crossing normal are unaffected by a cyclic relabelling. The in-plane gate orientation is not reliable. | PR #891 review. `coords.md` and §7.1 define positive pitch as nose-down, so the 2026-10-05 nose-up renderer claim was wrong. The ±35° corner-order guarantee did not hold under perspective. Affects G and H. |
| 2026-10-05 | Positive camera tilt pitches the optical axis up. Package G solves planar PnP in NumPy and leaves OpenCV to Package H. | Superseded in part on 2026-10-16: positive renderer pitch is nose-down, not the same sign as `TILT_DEG`. The NumPy solver decision stands. Affects G and H. |
| 2026-10-01 | Hide the frustum by default and set the 10× drone mesh to `sensor_visible=#false` when the FPV camera is on. The mount contract stays `[0.08, 0.0, 0.02]` m, body +X, tilt 0. | The frustum filled the chase view, and the camera sat inside the display mesh. Hiding the mesh from the sensor camera shows the world ahead without changing the camera-to-body geometry Package G tests. Affects B and later G, H, and J. |
| 2026-09-26 | Keep FPV sampling in `fpv_camera.py`. `frame_fresh` means a new `read_msg_at` timestamp, and shutdown FPS uses only timestamps inside the post-warmup window. | Held frames were marked fresh, and the cursor sweep was not an exact frame count. The schematic injects only the FPV fragments. Affects B and later G, H, and J. |
| 2026-09-03 | Run headless recipes once while retaining watched recipes in the editor. | Package A exposed that a failed simulation child was logged and then waited for source reload, so `elodin run` could not return nonzero. The approved shared fixes (`301ae367`, `#837`; lifecycle follow-up `36ee3431`, `#838`) make headless execution one-shot without changing interactive editor recovery, centralize recipe execution dispatch in s10, and add an end-to-end lifecycle CI check. This enables failure contracts in A, F, K, and L. |
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
  to the drone, without estimating a global drone pose. In this plan it is the
  gate navigator.
- **Gate navigator:** Package I's filter estimating vehicle position, velocity,
  and acceleration bias plus the target gate's centre in the local frame L,
  without a gate map.
- **Local navigation frame (L):** Gravity-aligned frame with its origin where
  the navigator initializes on the pad and heading from the magnetometer.
- **Nominal image/render time:** Selected image-message T in simulation seconds;
  `frame_time` and `GateCandidate.sample_time` share this value (Section 7.4).
- **Shadow mode:** Running the detector and navigator during truth-guided
  flight with no control authority, to measure estimator error against truth.
- **Truth:** Simulated world state or hidden gate pose unavailable to vision
  guidance.
- **Referee:** Trusted host-side scorer that uses truth to evaluate ordered gate
  crossings and metrics.
- **Qualified GPU run:** A run on hardware/environment known to support the
  render server at the required camera rate; rendered pixels need not be
  bit-identical to another GPU.
