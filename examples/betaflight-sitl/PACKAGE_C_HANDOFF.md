# Package C Completion Handoff

**Package:** C — Course and referee vertical slice
**Branch:** `package-c-course-referee`
**Updated base:** `536eb565574ed41b8cceb6420d721af41e9098d1` (`main` = `origin/main` when verified)
**Package D integration on main:** `4fc77694` (`#847`)
**Commit policy:** No commits were created; Package C remains an uncommitted working-tree change for the developer to commit.

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

## Intended files

- `examples/betaflight-sitl/main.py`
- `examples/betaflight-sitl/course.py`
- `examples/betaflight-sitl/referee.py`
- `examples/betaflight-sitl/race_runtime.py`
- `examples/betaflight-sitl/tests/test_course.py`
- `examples/betaflight-sitl/tests/test_referee.py`
- `examples/betaflight-sitl/README.md`
- `examples/betaflight-sitl/RACING_PLAN.md`
- `examples/betaflight-sitl/PACKAGE_C_HANDOFF.md`

The root status also shows `examples/betaflight-sitl/betaflight` as modified. This is the expected setup state from `build.sh` applying `patches/sitl-lockstep-event.patch`; it is not Package C work. Its three modified files remain `src/main/fc/core.c`, `src/main/scheduler/scheduler.c`, and `src/platform/SIMULATOR/include/platform/platform.h`.

## Verification environment

The pre-existing Python environment held an incompatible old `0.17.4` wheel after main advanced to `0.19.3-alpha.0`. To avoid stale-artifact results, the current wheel was rebuilt and installed:

```bash
/home/ubuntu/py-uv-env/bin/maturin build --release --manifest-path=libs/nox-py/Cargo.toml
uv pip install --python /home/ubuntu/py-uv-env/bin/python --reinstall \
  target/wheels/elodin-0.19.3a0-cp310-abi3-manylinux_2_39_x86_64.whl
```

Both the wheel and release CLI then reported `0.19.3-alpha.0+536eb565.dirty`.

## Verification performed

From the repository root:

```bash
/home/ubuntu/py-uv-env/bin/python -m pytest examples/betaflight-sitl/tests -q
cargo test -p betaflight-sitl-controller
cargo build --release -p elodin
git diff --check
```

Final results are recorded in `RACING_PLAN.md`: the Python suite passed all 81
tests in 0.08 seconds. The Package D controller suite also passed all 5 tests.
The required release build completed successfully, as did `git diff --check`.

Current matching-CLI integration form:

```bash
env -u RACE_COURSE -u RACE_GUIDANCE -u RACE_CAMERA -u RACE_MANUAL_AUDIT \
  ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py

RACE_COURSE=single ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py
```

Observed results:

```text
[C0] lockstep_steps=119995 motor_response=true max_motor=0.574 takeoff_delta_m=56.835 status=PASS
[RACE] course=single gates_passed=0/1 lap_time=na status=INCOMPLETE pass_times=[]
```

Both commands returned zero. The default emitted one C0 result and zero race results. The single-course run emitted one C0 result and exactly one well-formed race result. `INCOMPLETE` is expected because Package C deliberately does not add gate steering.

Startup failures were checked with the same CLI/Python pair:

```text
RACE_COURSE=oval        -> status 1; ERROR: unknown RACE_COURSE='oval' ...
RACE_COURSE=c1_straight -> status 1; ERROR: ... reserved for Package F and is not implemented
```

Package D coexistence was checked with:

```bash
RACE_GUIDANCE=manual RACE_MANUAL_AUDIT=1 RACE_COURSE=single \
  ELODIN_PYTHON=/home/ubuntu/py-uv-env/bin/python \
  ./target/release/elodin run examples/betaflight-sitl/main.py
```

It returned zero with one passing `[D-AUDIT]` and one incomplete `[RACE]` line.

The `betaflight_db012` recording was exported with current `elodin-db`. It showed correctly named, typed fields and initial values:

```text
drone.last_gate_passed = -1
drone.gate_pass_times  = [-1.0, -1.0, -1.0]
```

Each of the four gate `world_pos` series had 120,001 rows but exactly one unique pose, confirming static scene state throughout integration.

## Rendering evidence

Updated main changed editor object/orientation code, so a fresh headless screenshot was generated with the rebuilt editor:

```text
/tmp/package-c-gate-536eb565.png
```

The PNG is nonempty (1280×720, approximately 202 KiB). Runtime logs show valid KDL loading and all four gate entities/components. Saturated-orange pixel segmentation found one dominant 23,828-pixel rectangular ring with a clear inner opening, consistent with all four rendered bars. The installed Gamescope required `/usr/local/lib/x86_64-linux-gnu` for its current pixman and segfaulted during compositor teardown **after** Elodin logged that the screenshot was fully written; no simulation/editor/Betaflight process remained.

## Review fixes made after merging Package D

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
