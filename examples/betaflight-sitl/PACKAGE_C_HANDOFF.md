# Package C Completion Handoff

**Package:** C — Course and referee vertical slice
**Package C implementation:** `a0bbc683` on `package-c-course-referee`
**Referee qualification:** `2a36b035`
**Latest merged main:** `630324df` (merged without conflicts)
**Package D integration on main:** `4fc77694` (`#847`)
**History policy:** The published Package C commit is unchanged; qualification and main synchronization are later commits.

## Completed behavior

- `RACE_COURSE` is parsed independently from `RACE_GUIDANCE`.
- `RACE_COURSE` defaults to `none`; it spawns no gate bars and emits no `[RACE]` result.
- `RACE_COURSE=single` defines one ordered gate at ENU `(10, 0, 1.8)`, yaw `0`, with a `2.5 m` square inner opening.
- The gate is rendered by four `0.2 m` saturated-orange, non-emissive procedural boxes. Each scene entity has only `WorldPos`; recorded poses remain constant and the bars do not match any rigid-body integration query.
- `c1_straight` fails clearly as reserved for Package F. All unknown values fail clearly at startup.
- The referee transforms sampled truth segments into gate-local coordinates, requires exactly `previous_x < 0 <= current_x`, interpolates the plane intersection, applies inclusive local Y/Z opening bounds with only a `1e-9 m` roundoff allowance, scores only the next gate, and records each pass once.
- Pass times are linearly interpolated between sampled **simulation** times using the plane-crossing fraction. They never use wall-clock time. Lap time runs from simulation time zero to the final ordered crossing.
- Enabled courses emit exactly one final result, including normal incomplete completion and interrupted/exceptional simulation shutdown after the simulation bridge starts. Emission is guarded so the CLI's recipe-generation import does not produce a duplicate line.
- Referee telemetry is attached to `drone`:
  - `drone.last_gate_passed`: I64 width 1, initialized to `-1`;
  - `drone.gate_pass_times`: F64 width 3, initialized to `[-1.0, -1.0, -1.0]`.
- Package D ordering is preserved: state read → previous RC/Betaflight exchange → motor write → command-source update/next-tick latch → referee truth scoring. Guidance gets only the prior scoring stage's immutable/public ordered progress and opening/count rules, so a new pass is visible on the next tick. No gate pose or crossing truth enters `GuidanceUpdate`.
- The default schematic/chase expression is unchanged. Only an enabled course opts into fixed course framing.

## Positive referee qualification

`RACE_REFEREE_AUDIT=1` is a narrowly scoped live qualification on the existing
example. It requires `RACE_COURSE=single`, the default `scripted` guidance, and
rejects Package D's manual audit. The controlled fixture changes only the audit
process's initial state and duration:

- position `(5.0, 0.0, 4.9)` m ENU;
- velocity `(10.0, 0.0, 0.0)` m/s in world coordinates;
- duration `2.5` simulated seconds.

The run ends before scripted guidance leaves its five-second safe/disarmed boot
phase. Gravity and quadratic drag carry the drone from the negative local-X side
through the unchanged vertical gate. The production `post_step` truth read,
`world_position_from_transform`, `Referee.observe_truth`, component write, a
later-callback telemetry read, and final `RaceResult` are all exercised. No
teleport, direct event injection, guidance steering, or special course is used.

The pure evaluator requires one gate-0 event, an interpolated pass time in
`0.9–1.3 s`, telemetry gate `0` with matching slot 0 and two `-1.0` slots, a
complete `1/1` result, the expected approach/crossing/departure trajectory, and
exactly one race and audit result line. Its failing fixtures return exit status
1. The live result crossed near `(10.0, 0.0, 1.7958)` m at `1.085149 s`, finished
at X=`11.7776 m`, and the simulation callback interval took about 3.0 wall
seconds:

```text
[RACE] course=single gates_passed=1/1 lap_time=1.085149 status=COMPLETE pass_times=[1.085149]
[C-REFEREE-AUDIT] gate=0 passes=1 telemetry=true result=COMPLETE pass_time=1.085149 status=PASS
```

The editor uses only audit-specific camera framing and substitutes two existing
graph tiles with `last_gate_passed` and `gate_pass_times`; normal no-course and
ordinary course schematics are unchanged.

## Intended files

- `examples/betaflight-sitl/main.py`
- `examples/betaflight-sitl/course.py`
- `examples/betaflight-sitl/referee.py`
- `examples/betaflight-sitl/referee_audit.py`
- `examples/betaflight-sitl/race_runtime.py`
- `examples/betaflight-sitl/tests/test_course.py`
- `examples/betaflight-sitl/tests/test_referee.py`
- `examples/betaflight-sitl/tests/test_referee_audit.py`
- `examples/betaflight-sitl/README.md`
- `examples/betaflight-sitl/RACING_PLAN.md`
- `examples/betaflight-sitl/PACKAGE_C_HANDOFF.md`

The root status also shows `examples/betaflight-sitl/betaflight` as modified. This is the expected setup state from `build.sh` applying `patches/sitl-lockstep-event.patch`; it is not Package C work. Its three modified files remain `src/main/fc/core.c`, `src/main/scheduler/scheduler.c`, and `src/platform/SIMULATOR/include/platform/platform.h`.

## Verification environment

The release CLI was rebuilt after merging `origin/main` at `630324df` and
reports `0.19.3-alpha.0+af1e589a.dirty` (the merge commit plus the expected
dirty Betaflight submodule). Python 3.13.14 and the installed wheel report
`0.19.3-alpha.0+536eb565.dirty`. No wheel rebuild was needed: the merged range
changes only `libs/nox-py/README.md` under `libs/nox-py`, with no Python runtime,
API, or package-version change.

## Verification performed

From the repository root:

```bash
/home/ubuntu/py-uv-env/bin/python -m pytest examples/betaflight-sitl/tests -q
cargo test -p betaflight-sitl-controller
cargo build --release -p elodin
git diff --check
```

Final merged-base results: the Python suite passed all 96 tests in 0.15 seconds;
the Package D controller passed all 5 tests; the release editor build and
`git diff --check` passed.

Current integration form:

```bash
env -u RACE_COURSE -u RACE_GUIDANCE -u RACE_CAMERA -u RACE_MANUAL_AUDIT \
  ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py

RACE_COURSE=single ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py
RACE_COURSE=single RACE_REFEREE_AUDIT=1 \
  ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py
```

Observed merged-base results:

```text
[C0] lockstep_steps=119995 motor_response=true max_motor=0.574 takeoff_delta_m=56.835 status=PASS
[RACE] course=single gates_passed=0/1 lap_time=na status=INCOMPLETE pass_times=[]
[RACE] course=single gates_passed=1/1 lap_time=1.085149 status=COMPLETE pass_times=[1.085149]
[C-REFEREE-AUDIT] gate=0 passes=1 telemetry=true result=COMPLETE pass_time=1.085149 status=PASS
```

Both ordinary commands returned zero. The default emitted one C0 result, zero
race results, and zero audit results. The ordinary single-course run emitted one
C0 result, one expected incomplete race result, and no audit result. The live
audit returned zero with exactly one complete race result and one audit PASS.
`RACE_REFEREE_AUDIT=1` without `RACE_COURSE=single` returned status 1 with a
clear startup error. Unknown and reserved courses retained their status-1 clear
errors.

Package D coexistence was checked with:

```bash
RACE_GUIDANCE=manual RACE_MANUAL_AUDIT=1 RACE_COURSE=single \
  ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py
```

It returned zero with one passing `[D-AUDIT]`, one incomplete `[RACE]` line,
and no Package C audit line.

The merged-base audit recording `betaflight_db005` was exported with current
`elodin-db`. It persisted both the initial values and the later pass update:

```text
drone.last_gate_passed: -1 -> 0
drone.gate_pass_times:  [-1.0, -1.0, -1.0]
                     -> [1.085148620922599, -1.0, -1.0]
```

The four static gate poses were also recorded throughout the integrations; the
ordinary `single` run retained 120,001 samples per bar without adding any gate
to the rigid-body query.

## Rendering and capture evidence

Final artifacts are outside the worktree:

```text
/home/ubuntu/package-c-referee-audit.mp4
/home/ubuntu/package-c-referee-audit.png
```

The MP4 is nonempty (1,470,338 bytes), H.264 Main/yuv420p, 1280×720,
approximately 29.10 FPS, and 5.497944 seconds. The PNG is a nonempty 1280×720
crossing frame (164,254 bytes). Representative frames at 4.0 s (approach),
4.95 s (gate occlusion/crossing), and 5.4 s (departure) were decoded and checked
for nonblank image statistics. Saturated-orange segmentation and the audit's
oblique camera show the full vertical opening; frame occupancy was adequate, so
the suspected size issue was framing rather than a need to alter the 2.5 m
contract gate.

The media predates only main's install-session and bare-timeline serialization
changes. Neither change affects KDL parsing, rendering, camera framing, physics,
or the audit trajectory, so the retained capture remains representative of the
merged code and no GPU recapture was required.

The repository's automated capture script could not run on this host because
Nix is unavailable and its preflight correctly requires Nix's Xwayland/Mesa
environment. Manual Gamescope followed the documented PipeWire/GStreamer flow.
The host's GStreamer 1.24 plugin rejected the documented `target-object` serial
but accepted the same live source through its deprecated numeric `path`. NVENC
initialized and GPU encoder activity reached 3%, but losing the short-lived
Gamescope source before MP4 EOS left a zero-byte file. The reliable x264 fallback
was therefore recorded to recoverable MPEG-TS and stream-copied to the final
fast-start MP4. Gamescope segfaulted during PipeWire teardown only after the
simulation had emitted PASS and the TS was complete. These are capture teardown
issues, not simulation failures; no process remained.

The exact reproduction commands and fallback are in `README.md`.

## Qualification iteration and issues

- Added the qualification without changing Package C's course, score, telemetry, or guidance contracts.
- Analytically screened several initial X distances/speeds against gravity and drag, then live-tested `(5, 0, 4.9)` m at `(10, 0, 0)` m/s. The first live attempt passed at `1.085149 s` and departed past X=10.5 before the 2.5 s cap, so no control tuning was needed.
- Changed only the audit camera from the ordinary course frame to a closer oblique `(3, -4, 3.5)` view looking at `(9, 0, 2)`. The full 2.5 m opening and motion were clear; gate geometry stayed unchanged.
- Verified telemetry only on a callback after the event write rather than assuming same-callback DB visibility.
- Added direct PASS and representative FAIL evaluator tests, including a deliberately broken telemetry criterion producing exit status 1.
- Capture iteration exposed Nix preflight, PipeWire target-selection, NVENC/MP4 finalization, and Gamescope teardown issues. The final recoverable MPEG-TS → MP4 fallback preserved a valid recording and is documented.

## Earlier Package C review fixes

- Resolved all three stash conflicts from the old Package C base (`main.py`, README, and plan) against the landed Package D pipeline rather than selecting either side wholesale.
- Moved referee scoring to Package D's required final post-step stage and wired only public progress into `GuidanceUpdate`.
- Initialized referee history from the configured initial position so a first-tick segment is not silently lost.
- Added monotonic simulation-time validation and sub-tick crossing-time interpolation.
- Added explicit current-endpoint-on-plane, all four inclusive edges, all four just-outside edges, explicit `none`, and non-monotonic-time coverage.
- Fixed duplicate `[RACE]` output discovered under the real CLI's recipe-generation/import lifecycle.
- Added a guarded finalizer for interrupted simulation shutdown while retaining exactly-once output.
- Reconciled documentation with Package D now being on main rather than on a concurrent remote branch.

## Resume point and expected future interaction

Package D still requires operator gamepad/keyboard hardware qualification. After that is recorded, Package E can begin because the C and D code prerequisites are present. Package B remains independently available.

Package F should replace the intentional `c1_straight` startup rejection with the exact three-gate geometry and can use the already reserved three-entry telemetry width. Future truth/vision guidance should continue to receive `RaceProgress` values through `GuidanceUpdate`; it must not import or retain course/referee truth.
